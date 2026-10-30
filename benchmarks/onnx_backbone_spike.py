"""THROWAWAY spike: export the Maya2 backbone (Qwen3 LM, KV cache in/out) to ONNX, check parity vs PyTorch,
then dynamic-quantize to int8 and time one decode step. Not wired into the pipeline.

    .venv-tts/bin/python benchmarks/onnx_backbone_spike.py --out-dir <dir> [--quant none|int8]
"""
import argparse
import gc
import time

import numpy as np
import torch
import torch.nn as nn
from transformers import DynamicCache


class BackboneKV(nn.Module):
    """Qwen3 decoder stack with KV cache as flat tensors: (inputs_embeds, k_0..k_L-1, v_0..v_L-1)."""

    def __init__(self, lm_model, n_layers):
        super().__init__()
        self.model = lm_model
        self.n_layers = n_layers

    def forward(self, inputs_embeds, *past):
        k, v = past[: self.n_layers], past[self.n_layers:]
        cache = DynamicCache()
        for i in range(self.n_layers):
            cache.update(k[i], v[i], i)
        out = self.model(inputs_embeds=inputs_embeds, past_key_values=cache, use_cache=True)
        pk = out.past_key_values
        new_k = [pk.layers[i].keys for i in range(self.n_layers)]
        new_v = [pk.layers[i].values for i in range(self.n_layers)]
        return (out.last_hidden_state, *new_k, *new_v)


def empty_past(cfg, batch=1):
    kv = cfg.num_key_value_heads
    hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    z = torch.zeros(batch, kv, 0, hd, dtype=torch.float32)
    return [z.clone() for _ in range(cfg.num_hidden_layers)], [z.clone() for _ in range(cfg.num_hidden_layers)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prompt-len", type=int, default=24)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)

    import os
    os.makedirs(a.out_dir, exist_ok=True)

    from maya3 import hub
    from tts.model.load_s3 import load_model_for_eval
    from scripts import s2_serve_controls  # noqa: F401  (import side effects match the serve path)

    paths = hub.resolve("maya-research/maya2-native")
    cfg_dir = str(paths["ckpt"])
    t0 = time.perf_counter()
    model, _ = load_model_for_eval(str(paths["ckpt"]), device="cpu", dtype=torch.float32,
                                   cfg={"model": {"backbone": str(paths["backbone_dir"]), "attn": "sdpa"},
                                        "data": {"max_tokens": 8192}},
                                   config=None, amortize_frac=1.0, require_anchor=True, verbose=False)
    model.eval()
    lm = model.backbone.lm
    lmcfg = lm.config
    print(f"LOAD fp32 s={time.perf_counter() - t0:.1f} layers={lmcfg.num_hidden_layers} "
          f"d={lmcfg.hidden_size} kv={lmcfg.num_key_value_heads}", flush=True)

    wrap = BackboneKV(lm.model, lmcfg.num_hidden_layers).eval()
    L = lmcfg.num_hidden_layers
    P = a.prompt_len
    emb = torch.randn(1, P, lmcfg.hidden_size) * 0.02
    k0, v0 = empty_past(lmcfg)
    names_in = ["inputs_embeds"] + [f"k_{i}" for i in range(L)] + [f"v_{i}" for i in range(L)]
    names_out = ["hidden"] + [f"nk_{i}" for i in range(L)] + [f"nv_{i}" for i in range(L)]
    dyn = {"inputs_embeds": {1: "t"}, "hidden": {1: "t"}}
    for i in range(L):
        for n in (f"k_{i}", f"v_{i}"):
            dyn[n] = {2: "past"}
        for n in (f"nk_{i}", f"nv_{i}"):
            dyn[n] = {2: "total"}

    # reference: PyTorch prefill then one decode step
    with torch.inference_mode():
        ref_pre = wrap(emb, *k0, *v0)
        dec_emb = torch.randn(1, 1, lmcfg.hidden_size) * 0.02
        ref_dec = wrap(dec_emb, *ref_pre[1:L + 1], *ref_pre[L + 1:])

    onnx_path = os.path.join(a.out_dir, "backbone_fp32.onnx")
    t1 = time.perf_counter()
    with torch.inference_mode():
        torch.onnx.export(
            wrap, (emb, *k0, *v0), onnx_path, input_names=names_in, output_names=names_out,
            dynamic_axes=dyn, opset_version=18, dynamo=False, do_constant_folding=False,
        )
    print(f"EXPORT fp32 s={time.perf_counter() - t1:.1f} path={onnx_path}", flush=True)

    del wrap, model, lm
    gc.collect()
    print("DONE stage=export", flush=True)


if __name__ == "__main__":
    main()
