import math
import unittest
import numpy as np

from src.maya_fast.sampling import gumbel, topk_gumbel_argmax


def _gammp(a: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x < a + 1.0:
        ap = a
        sum_val = 1.0 / a
        del_val = sum_val
        for _ in range(1, 300):
            ap += 1.0
            del_val *= x / ap
            sum_val += del_val
            if abs(del_val) < abs(sum_val) * 1e-12:
                break
        return sum_val * math.exp(-x + a * math.log(x) - math.lgamma(a))
    else:
        b = x + 1.0 - a
        c = 1.0 / 1e-30
        d = 1.0 / b
        h = d
        for i in range(1, 300):
            an = -i * (i - a)
            b += 2.0
            d = an * d + b
            if abs(d) < 1e-30:
                d = 1e-30
            c = b + an / c
            if abs(c) < 1e-30:
                c = 1e-30
            d = 1.0 / d
            del_h = d * c
            h *= del_h
            if abs(del_h - 1.0) < 1e-12:
                break
        q = math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
        return 1.0 - q


def _chi2_pvalue(stat: float, df: int) -> float:
    return max(0.0, min(1.0, 1.0 - _gammp(df / 2.0, stat / 2.0)))


class TestMayaFastSampling(unittest.TestCase):
    def test_topk_gumbel_matches_multinomial(self):
        # numpy: fixed 2048-logit vector, 40 000 draws, chi-square p > 0.01 against softmax(topk_masked(logits) / 0.7)
        np.random.seed(42)
        vocab_size = 2048
        logits = np.random.randn(vocab_size).astype(np.float32)
        temp = 0.7
        top_k = 64

        topk_indices = np.argpartition(logits, -top_k)[-top_k:]
        scaled = logits[topk_indices] / temp
        exp_scaled = np.exp(scaled - np.max(scaled))
        probs = exp_scaled / np.sum(exp_scaled)

        n_draws = 40000
        rng = np.random.default_rng(12345)
        counts = np.zeros(vocab_size, dtype=np.int64)

        for _ in range(n_draws):
            noise = gumbel(rng, (vocab_size,))
            sample = topk_gumbel_argmax(logits, temp=temp, top_k=top_k, noise=noise)
            counts[sample] += 1

        obs = counts[topk_indices].astype(np.float64)
        exp = (probs * n_draws).astype(np.float64)

        chi2_stat = float(np.sum((obs - exp) ** 2 / exp))
        p_val = _chi2_pvalue(chi2_stat, df=top_k - 1)
        self.assertGreater(p_val, 0.01)


if __name__ == "__main__":
    unittest.main()
