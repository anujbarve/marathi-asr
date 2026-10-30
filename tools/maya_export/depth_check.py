"""fp32 depth graphs vs upstream PyTorch depth (light model, exact fp32 weights) on 20 golden frames."""
import numpy as np, onnxruntime as ort, torch
from tools.maya_export.common import build_light_model, work_dir
from tts.model.depth_v2 import GROUP_OF_LEVEL
B = work_dir() / 'depth_build' / 'fp32'
z = np.load(work_dir() / 'golden' / 'ref' / 'conv_01_s0.npz')
codes = z['codes'][:, :20].T.astype(np.int64); h = z['h_frames'][:20].astype(np.float32)
model = build_light_model(); d = model.depth.eval()
mem_t = d.null_mem.detach().float().expand(20, -1, -1).contiguous()
worst_rel = 0.0; worst_abs = 0.0
with torch.inference_mode():
    for L in range(1, 16):
        x = d._prefix(torch.from_numpy(h), torch.from_numpy(codes), L)
        ref = d.heads[L - 1](d._run(x, GROUP_OF_LEVEL[L], mem_t, None)).float().numpy()
        s = ort.InferenceSession(str(B / f'level_{L:02d}.onnx'), providers=['CPUExecutionProvider'])
        got = s.run(None, {'h': h, 'codes': codes, 'mem': mem_t.numpy()})[0]
        del s
        err = np.abs(got - ref)
        worst_abs = max(worst_abs, float(err.max())); worst_rel = max(worst_rel, float(err.max() / np.abs(ref).max()))
print(f'fp32 vs upstream PyTorch depth: WORST abs {worst_abs:.3e} rel {worst_rel:.3e}')
