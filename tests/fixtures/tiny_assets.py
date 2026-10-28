from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np
import onnx
from onnx import helper, TensorProto


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_tiny_backbone_onnx(out_path: Path, d: int = 16, vocab: int = 64, n_layers: int = 2, n_heads: int = 2, head_dim: int = 8) -> None:
    # Inputs:
    # embeds (1, T, 2048) -> here (1, T, d)
    # cos (1, T, head_dim) -> (1, T, 8)
    # sin (1, T, head_dim) -> (1, T, 8)
    # mask (1, 1, T, P+T)
    # past_k_{i}, past_v_{i} ((1, 8, P, 128)) -> ((1, n_heads, P, head_dim))
    # Outputs:
    # hidden (1, T, 2048) -> (1, T, d)
    # cb0_logits (1, T, 2049) -> (1, T, vocab + 1)
    # present_k_{i}, present_v_{i}

    inputs = [
        helper.make_tensor_value_info("embeds", TensorProto.FLOAT, [1, "T", d]),
        helper.make_tensor_value_info("cos", TensorProto.FLOAT, [1, "T", head_dim]),
        helper.make_tensor_value_info("sin", TensorProto.FLOAT, [1, "T", head_dim]),
        helper.make_tensor_value_info("mask", TensorProto.FLOAT, [1, 1, "T", "total_len"]),
    ]
    for i in range(n_layers):
        inputs.append(helper.make_tensor_value_info(f"past_k_{i}", TensorProto.FLOAT, [1, n_heads, "P", head_dim]))
        inputs.append(helper.make_tensor_value_info(f"past_v_{i}", TensorProto.FLOAT, [1, n_heads, "P", head_dim]))

    outputs = [
        helper.make_tensor_value_info("hidden", TensorProto.FLOAT, [1, "T", d]),
        helper.make_tensor_value_info("cb0_logits", TensorProto.FLOAT, [1, "T", vocab + 1]),
    ]
    for i in range(n_layers):
        outputs.append(helper.make_tensor_value_info(f"present_k_{i}", TensorProto.FLOAT, [1, n_heads, "total_len", head_dim]))
        outputs.append(helper.make_tensor_value_info(f"present_v_{i}", TensorProto.FLOAT, [1, n_heads, "total_len", head_dim]))

    # For tiny mock model:
    # hidden = Identity(embeds)
    # cb0_logits = MatMul(hidden, W_cb0) + B_cb0
    # present_k_i = Concat(past_k_i, k_new, axis=2)
    # We construct a simple valid ONNX graph that outputs the right shapes.
    nodes = []
    initializers = []

    # W_cb0: (d, vocab + 1)
    np.random.seed(42)
    w_cb0_val = np.random.randn(d, vocab + 1).astype(np.float32) * 0.1
    # ensure aeos (index vocab) has sensible value
    initializers.append(helper.make_tensor("W_cb0", TensorProto.FLOAT, [d, vocab + 1], w_cb0_val.flatten()))

    nodes.append(helper.make_node("Identity", ["embeds"], ["hidden"]))
    nodes.append(helper.make_node("MatMul", ["hidden", "W_cb0"], ["cb0_logits"]))

    # For past_k, past_v: generate dummy k_new from embeds:
    # embeds (1, T, d) -> slice or reshape to (1, n_heads, T, head_dim)
    # In our case d = 16 = 2 * 8 = n_heads * head_dim!
    # So we can reshape embeds -> (1, T, n_heads, head_dim) then transpose to (1, n_heads, T, head_dim)
    initializers.append(helper.make_tensor("shape_k", TensorProto.INT64, [4], [1, -1, n_heads, head_dim]))
    nodes.append(helper.make_node("Reshape", ["embeds", "shape_k"], ["reshaped_kv"]))
    nodes.append(helper.make_node("Transpose", ["reshaped_kv"], ["k_new"], perm=[0, 2, 1, 3]))

    for i in range(n_layers):
        nodes.append(helper.make_node("Concat", [f"past_k_{i}", "k_new"], [f"present_k_{i}"], axis=2))
        nodes.append(helper.make_node("Concat", [f"past_v_{i}", "k_new"], [f"present_v_{i}"], axis=2))

    graph = helper.make_graph(nodes, "tiny_backbone", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    onnx.save(model, str(out_path))


def _make_tiny_depth_onnx(out_path: Path, d: int = 16, vocab: int = 64, d_mem: int = 16) -> None:
    # DepthFrame (batched, as the real sampled graph):
    # Inputs:
    # h (B, 2048) -> (B, d)
    # cb0 (B,) int64
    # mem (B, R, 1024) -> (B, R, d_mem)
    # noise (B, 15, 2048) float32 -> (B, 15, vocab)
    # Output:
    # codes (B, 16) int64

    inputs = [
        helper.make_tensor_value_info("h", TensorProto.FLOAT, ["B", d]),
        helper.make_tensor_value_info("cb0", TensorProto.INT64, ["B"]),
        helper.make_tensor_value_info("mem", TensorProto.FLOAT, ["B", "R", d_mem]),
        helper.make_tensor_value_info("noise", TensorProto.FLOAT, ["B", 15, vocab]),
    ]
    outputs = [
        helper.make_tensor_value_info("codes", TensorProto.INT64, ["B", 16]),
    ]

    # For tiny test: codes[:, 0] = cb0, codes[:, 1..15] = argmax(noise[:, i]) over the vocab.
    initializers = [helper.make_tensor("unsq_axes", TensorProto.INT64, [1], [1])]
    nodes = [
        helper.make_node("ArgMax", ["noise"], ["deeper_codes"], axis=-1, keepdims=0),
        helper.make_node("Unsqueeze", ["cb0", "unsq_axes"], ["cb0_col"]),
        helper.make_node("Concat", ["cb0_col", "deeper_codes"], ["codes"], axis=1),
    ]

    graph = helper.make_graph(nodes, "tiny_depth", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    onnx.save(model, str(out_path))


def _make_tiny_mimi_onnx(out_path: Path) -> None:
    # MimiDecoder:
    # Input: codes (1, 16, F) int64
    # Output: wav (1, 1, F * 1920) float32
    inputs = [
        helper.make_tensor_value_info("codes", TensorProto.INT64, [1, 16, "F"]),
    ]
    outputs = [
        helper.make_tensor_value_info("wav", TensorProto.FLOAT, [1, 1, "samples"]),
    ]

    # In ONNX helper:
    # Shape of codes -> shape (3,)
    # Gather index 2 -> F
    # Expand 1920 samples per frame.
    # We can do: Cast(codes, FLOAT), ReduceMean along axis 1 -> (1, 1, F)
    # Then Tile by [1, 1, 1920] -> (1, 1, F * 1920)
    initializers = [
        helper.make_tensor("tile_repeats", TensorProto.INT64, [3], [1, 1, 1920]),
    ]
    nodes = [
        helper.make_node("Cast", ["codes"], ["codes_float"], to=TensorProto.FLOAT),
        helper.make_node("ReduceMean", ["codes_float"], ["mean_frames"], axes=[1], keepdims=1),
        helper.make_node("Tile", ["mean_frames", "tile_repeats"], ["wav"]),
    ]

    graph = helper.make_graph(nodes, "tiny_mimi", inputs, outputs, initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)
    onnx.save(model, str(out_path))


def make_tiny_assets(assets_dir: Path, aeos_bias: float = 0.0) -> Path:
    assets_dir = Path(assets_dir)
    assets_dir.mkdir(parents=True, exist_ok=True)

    d = 16
    vocab = 64
    n_layers = 2
    n_heads = 2
    head_dim = 8
    d_mem = 16
    n_q = 16
    n_codes = vocab
    ecapa_dim = 16

    files_sha = {}

    # 1. embed_bf16.bin
    np.random.seed(123)
    embeds_fp32 = np.random.randn(vocab, d).astype(np.float32)
    # Convert fp32 to bf16 uint16
    embeds_bf16 = (embeds_fp32.view(np.uint32) >> 16).astype(np.uint16)
    embed_bytes = embeds_bf16.tobytes()
    (assets_dir / "embed_bf16.bin").write_bytes(embed_bytes)
    files_sha["embed_bf16.bin"] = _sha256(embed_bytes)

    # 2. prompt.npz
    # anchor_fc1_w, anchor_fc1_b, anchor_fc2_w, anchor_fc2_b, anchor_null, aeos, plus audio_in/<name>
    # Note: audio_in parameters: table (n_q * n_codes, d), aeos_in (d,), norm.weight (d,)
    prompt_dict = {
        "anchor_fc1_w": np.random.randn(32, ecapa_dim).astype(np.float32),
        "anchor_fc1_b": np.zeros(32, dtype=np.float32),
        "anchor_fc2_w": np.zeros((d, 32), dtype=np.float32),  # zero init as upstream
        "anchor_fc2_b": np.zeros(d, dtype=np.float32),
        "anchor_null": np.random.randn(ecapa_dim).astype(np.float32),
        "aeos": np.random.randn(d).astype(np.float32),
        "audio_in/aeos_in": np.random.randn(d).astype(np.float32),
        "audio_in/norm": np.ones(d, dtype=np.float32),
    }
    np.savez(assets_dir / "prompt.npz", **prompt_dict)
    files_sha["prompt.npz"] = _sha256((assets_dir / "prompt.npz").read_bytes())

    # audio input table, bf16 raw uint16 row-major (same layout as the real export)
    audio_f32 = np.random.randn(n_q * n_codes, d).astype(np.float32)
    audio_bf16 = (audio_f32.view(np.uint32) >> 16).astype("<u2")
    (assets_dir / "audio_in_bf16.bin").write_bytes(audio_bf16.tobytes())
    files_sha["audio_in_bf16.bin"] = _sha256((assets_dir / "audio_in_bf16.bin").read_bytes())

    # 3. voices.json + voices.npz
    voices_json = {
        "Charvi": {"mode": "named", "use_icl": False, "use_mem": False}
    }
    (assets_dir / "voices.json").write_text(json.dumps(voices_json, indent=2))
    files_sha["voices.json"] = _sha256((assets_dir / "voices.json").read_bytes())


    # 4. controls.json
    controls_json = {
        "text_frames_prior": {
            "mr": {"a": 4.0, "b": 0.5, "resid_p99": 10.0}
        },
        "text": {"terminals": "।.!?;।॥", "fps": 12.5, "frame_s": 0.08, "chunk_s": 16.0},
        "guard": {
            "enabled": True,
            "dur_k": 2.2,
            "dur_margin_s": 1.2,
            "hard_cap": 1500,
            "dur_prior_coefs": {"mr": [1.0, 0.05, 0.03, 0.3, 0.3]},
            "dur_prior_default": [1.0, 0.05, 0.03, 0.3, 0.3]
        },
        "sampling": {
            "q_serve": 16,
            "temp": 0.7,
            "top_k": 64,
            "tau": 0.35,
            "hazard": 0.6
        }
    }
    (assets_dir / "controls.json").write_text(json.dumps(controls_json, indent=2))
    files_sha["controls.json"] = _sha256((assets_dir / "controls.json").read_bytes())

    # 5. tokenizer.json (minimal mock tokenizer using tokenizers library)
    from tokenizers import Tokenizer, models, pre_tokenizers
    vocab_dict = {f"t_{i}": i for i in range(vocab - 5)}
    vocab_dict["<|lang:mr|>"] = vocab - 5
    vocab_dict["<|audio|>"] = vocab - 4
    vocab_dict["<|aeos|>"] = vocab - 3
    vocab_dict["[UNK]"] = vocab - 2
    vocab_dict[" "] = vocab - 1
    tok = Tokenizer(models.WordLevel(vocab_dict, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.save(str(assets_dir / "tokenizer.json"))
    files_sha["tokenizer.json"] = _sha256((assets_dir / "tokenizer.json").read_bytes())

    # 6. ONNX models
    _make_tiny_backbone_onnx(assets_dir / "backbone_wo8.onnx", d=d, vocab=vocab, n_layers=n_layers, n_heads=n_heads, head_dim=head_dim)
    # If aeos_bias is requested, we can adjust W_cb0 in backbone or create an alternate backbone
    if aeos_bias != 0.0:
        # Load and modify W_cb0 last column
        m = onnx.load(str(assets_dir / "backbone_wo8.onnx"))
        for init in m.graph.initializer:
            if init.name == "W_cb0":
                arr = onnx.numpy_helper.to_array(init)
                # bias is applied to the output logits, or we add bias node.
                # In tiny_backbone, cb0_logits = hidden @ W_cb0. If we add a large negative bias to index vocab (aeos):
                # Let's add an explicit bias tensor:
                bias_arr = np.zeros((1, 1, vocab + 1), dtype=np.float32)
                bias_arr[0, 0, vocab] = aeos_bias
                bias_init = helper.make_tensor("bias_cb0", TensorProto.FLOAT, [1, 1, vocab + 1], bias_arr.flatten())
                m.graph.initializer.append(bias_init)
                # change cb0_logits node
                for node in m.graph.node:
                    if node.name == "" and node.op_type == "MatMul":
                        node.output[0] = "cb0_logits_pre"
                m.graph.node.append(helper.make_node("Add", ["cb0_logits_pre", "bias_cb0"], ["cb0_logits"]))
                break
        onnx.save(m, str(assets_dir / "backbone_wo8.onnx"))

    files_sha["backbone_wo8.onnx"] = _sha256((assets_dir / "backbone_wo8.onnx").read_bytes())

    _make_tiny_depth_onnx(assets_dir / "depth_frame_fp16.onnx", d=d, vocab=vocab, d_mem=d_mem)
    files_sha["depth_frame_fp16.onnx"] = _sha256((assets_dir / "depth_frame_fp16.onnx").read_bytes())

    _make_tiny_mimi_onnx(assets_dir / "mimi_decoder.onnx")
    files_sha["mimi_decoder.onnx"] = _sha256((assets_dir / "mimi_decoder.onnx").read_bytes())

    # 7. manifest.json
    manifest = {
        "version": 1,
        "files": files_sha,
        "shapes": {
            "d_model": d,
            "vocab": vocab,
            "n_q": n_q,
            "n_codes": n_codes,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "d_mem": d_mem,
            "rope_theta": 10000.0,
        }
    }
    (assets_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    return assets_dir
