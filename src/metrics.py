import jiwer

from src.normalizer import normalize_marathi_text


def _error_rates(ref: str, hyp: str) -> tuple[float, float]:
    """Returns (WER, CER) for normalized strings. An empty reference scores 0.0,
    and an empty hypothesis against a non-empty reference scores 1.0."""
    if not ref:
        return 0.0, 0.0
    if not hyp:
        return 1.0, 1.0
    return float(jiwer.wer(ref, hyp)), float(jiwer.cer(ref, hyp))


def score_transcript(reference: str, prediction: str) -> dict[str, float]:
    """
    Scores a transcript against its reference with strict and lenient normalization.

    Returns keys: wer, cer (strict Marathi normalization), wer_lenient, cer_lenient
    (also folds vowel length and chandrabindu, which are spelled inconsistently).
    """
    wer, cer = _error_rates(
        normalize_marathi_text(reference), normalize_marathi_text(prediction)
    )
    wer_l, cer_l = _error_rates(
        normalize_marathi_text(reference, lenient=True),
        normalize_marathi_text(prediction, lenient=True),
    )
    return {"wer": wer, "cer": cer, "wer_lenient": wer_l, "cer_lenient": cer_l}
