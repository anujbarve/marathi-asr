import re
import unicodedata

# Common Devanagari punctuation and ASCII punctuation to strip
PUNCTUATION_REGEX = re.compile(r'[\.,\?!;:।॥\(\)\[\]\{\}"\'`‘’“”\-—_\\/\|\*~@#\$%\^&+=<>]')
WHITESPACE_REGEX = re.compile(r'\s+')
# Zero-width joiner (U+200D) and non-joiner (U+200C): invisible, and typed inconsistently
ZERO_WIDTH_REGEX = re.compile(r'[‌‍]')
NUKTA = "़"
# Lenient mode: vowel length (ि/ी, ु/ू) and chandrabindu (ँ/ं) are spelled inconsistently
LENIENT_MAP = str.maketrans({"ि": "ी", "ु": "ू", "ँ": "ं"})

def normalize_marathi_text(text: str, lenient: bool = False) -> str:
    """
    Normalizes Marathi Devanagari text for standard ASR evaluation (WER/CER).
    Rules:
    1. Unicode NFC normalization.
    2. Remove zero-width joiner/non-joiner (ZWJ/ZWNJ).
    3. Fold nukta (e.g. ज़ becomes ज).
    4. Lowercase any embedded English/Latin words.
    5. Remove punctuation including Devanagari danda (।) and double danda (॥).
    6. Normalize whitespace (collapse multiple spaces, trim ends).
    If lenient is True, also fold vowel length (ि/ी, ु/ू) and chandrabindu (ँ/ं).
    """
    if not text:
        return ""

    # 1. Normalize unicode
    text = unicodedata.normalize("NFC", text)

    # 2. Strip ZWJ/ZWNJ
    text = ZERO_WIDTH_REGEX.sub("", text)

    # 3. Fold nukta: NFD splits U+0958-U+095F into base + nukta, then drop the nukta
    text = unicodedata.normalize("NFC", unicodedata.normalize("NFD", text).replace(NUKTA, ""))

    # 4. Lenient vowel-length and chandrabindu folding (optional)
    if lenient:
        text = text.translate(LENIENT_MAP)

    # 5. Lowercase ASCII characters
    text = text.lower()

    # 6. Strip punctuation
    text = PUNCTUATION_REGEX.sub(" ", text)

    # 7. Collapse spaces
    text = WHITESPACE_REGEX.sub(" ", text).strip()
    
    return text
