import unittest

import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3Model

from tools.maya_export import streamed_backbone as SB
from tools.maya_export.common import ckpt_paths

N_LAYERS = 2


def _hf_two_layer(cfg):
    """Upstream transformers Qwen3Model with the first N_LAYERS layers loaded from the checkpoint (fp32)."""
    c = AutoConfig.from_pretrained(cfg.name_or_path)
    c.num_hidden_layers = N_LAYERS
    c.layer_types = c.layer_types[:N_LAYERS]
    c.vocab_size = 200000  # padding_idx needs the real range; embeddings are unused (inputs_embeds)
    c._attn_implementation = "sdpa"
    model = Qwen3Model(c).eval()
    weights = SB._Weights(SB.shard_files(ckpt_paths()["ckpt"]))
    state = {}
    for i in range(N_LAYERS):
        prefix = SB.LAYER_PREFIX.format(i=i)
        for name, _ in model.layers[i].named_parameters():
            state[f"layers.{i}.{name}"] = weights.get(prefix + name).float()
    state["norm.weight"] = weights.get(SB.NORM_KEY).float()
    model.load_state_dict(state, strict=False)
    return model


class TestStreamedBackbone(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.backbone_dir = ckpt_paths()["backbone_dir"]
            cls.shards = SB.shard_files(ckpt_paths()["ckpt"])
        except Exception as e:  # pragma: no cover - checkpoint not cached on this machine
            raise unittest.SkipTest(f"checkpoint not available: {e}")

    def test_prefill_then_steps_match_full_forward(self):
        torch.manual_seed(0)
        stream = SB.StreamedBackbone(self.backbone_dir, self.shards, n_layers=N_LAYERS)
        ref = _hf_two_layer(stream.cfg)
        x = torch.randn(1, 9, stream.cfg.hidden_size) * 0.1

        out = stream.forward(x[:, :6])
        with torch.inference_mode():
            full = ref(inputs_embeds=x[:, :6]).last_hidden_state
        self.assertLess(float((out - full).abs().max()), 1e-4)

        for t in range(6, 9):
            out = stream.forward(x[:, t : t + 1])
            with torch.inference_mode():
                full = ref(inputs_embeds=x[:, : t + 1]).last_hidden_state[:, -1:]
            self.assertLess(float((out - full).abs().max()), 1e-4, f"step {t}")


if __name__ == "__main__":
    unittest.main()
