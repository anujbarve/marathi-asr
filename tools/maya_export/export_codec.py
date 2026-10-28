"""Export the Mimi decoder (codes -> 24 kHz wav) to ONNX, fp32, dynamic frame count, and check parity against
transformers' MimiModel.decode on random codes. Run guarded:

    .venv/bin/python tools/memguard.py --rss-limit-mb 2000 -- \
        .venv-tts/bin/python -m tools.maya_export.export_codec
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import MimiModel

from tools.maya_export.common import ckpt_paths, work_dir

N_Q = 16
N_CODES = 2048
SAMPLES_PER_FRAME = 1920  # 24 kHz / 12.5 Hz
PARITY_TOL = 1e-4


class _Decode(torch.nn.Module):
    def __init__(self, model: MimiModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        return self.model.decode(codes)[0]  # (1, 1, F * 1920)


def _mimi_dir() -> Path:
    from maya3 import hub
    return Path(hub.resolve("maya-research/maya2-native")["mimi_dir"])


def _traceable_sliding_mask(config, inputs_embeds, **_):
    """Explicit causal sliding-window additive mask. transformers' mask helper fails under torch.onnx tracing;
    check_parity compares the exported graph with the unpatched model, so a wrong mask shows up there."""
    t = inputs_embeds.shape[1]
    i = torch.arange(t)[:, None]
    j = torch.arange(t)[None, :]
    allowed = (j <= i) & (j > i - int(config.sliding_window))
    return torch.where(allowed, 0.0, float("-inf"))[None, None].to(inputs_embeds.dtype)


def check_mask(model: MimiModel, lengths=(5, 300)) -> None:
    """The traced graph uses _traceable_sliding_mask. Compare it with transformers' own mask on lengths beyond
    the sliding window (250 frames), where the window actually cuts attention."""
    import transformers.models.mimi.modeling_mimi as M

    for t in lengths:
        emb = torch.zeros(1, t, model.config.hidden_size)
        pos = torch.arange(t)[None]
        ref = M.create_sliding_window_causal_mask(config=model.config, inputs_embeds=emb, attention_mask=None,
                                                  past_key_values=None, position_ids=pos)
        ref = ref.reshape(t, t)
        ref_allowed = ref if ref.dtype == torch.bool else ref == 0
        mine = _traceable_sliding_mask(model.config, emb).reshape(t, t) == 0
        if not torch.equal(ref_allowed, mine):
            raise SystemExit(f"sliding-window mask differs from transformers at T={t}")
    print(f"sliding-window mask matches transformers at T={list(lengths)}")


def export(out_path: Path) -> MimiModel:
    import transformers.models.mimi.modeling_mimi as M

    model = MimiModel.from_pretrained(str(_mimi_dir()), dtype=torch.float32, attn_implementation="eager").eval()
    wrapper = _Decode(model).eval()
    dummy = torch.randint(0, N_CODES, (1, N_Q, 5), dtype=torch.long)
    saved = M.create_sliding_window_causal_mask
    M.create_sliding_window_causal_mask = _traceable_sliding_mask
    try:
        _trace(wrapper, dummy, out_path)
    finally:
        M.create_sliding_window_causal_mask = saved
    return model


def _trace(wrapper: _Decode, dummy: torch.Tensor, out_path: Path) -> None:
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (dummy,),
            str(out_path),
            input_names=["codes"],
            output_names=["wav"],
            dynamic_axes={"codes": {2: "F"}, "wav": {2: "samples"}},
            opset_version=17,
            dynamo=False,
        )


def check_parity(out_path: Path, model: MimiModel, frames=(5, 17, 50)) -> float:
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    worst = 0.0
    rng = np.random.default_rng(0)
    for F in frames:
        codes = rng.integers(0, N_CODES, size=(1, N_Q, F), dtype=np.int64)
        with torch.inference_mode():
            ref = model.decode(torch.from_numpy(codes))[0].float().numpy()
        got = sess.run(None, {"codes": codes})[0]
        assert got.shape == ref.shape == (1, 1, F * SAMPLES_PER_FRAME), (got.shape, ref.shape)
        worst = max(worst, float(np.max(np.abs(got - ref))))
    return worst


def speed(out_path: Path, F: int = 50, reps: int = 20) -> float:
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    codes = np.random.default_rng(1).integers(0, N_CODES, size=(1, N_Q, F), dtype=np.int64)
    sess.run(None, {"codes": codes})
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        sess.run(None, {"codes": codes})
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) / F * 1000.0  # ms per frame


def main() -> None:
    out = work_dir() / "assets" / "v1" / "mimi_decoder.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    model = export(out)
    print(f"exported {out} ({out.stat().st_size / 2**20:.1f} MiB)")
    check_mask(model)
    err = check_parity(out, model)
    print(f"parity max abs error vs MimiModel.decode: {err:.3e} (tol {PARITY_TOL:.0e})")
    ms = speed(out)
    print(f"decode median: {ms:.2f} ms per frame at F=50 (gate <= 8 ms)")
    if err > PARITY_TOL:
        raise SystemExit("parity gate FAILED")


if __name__ == "__main__":
    main()
