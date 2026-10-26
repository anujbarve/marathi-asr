import re
import unicodedata

# Common Devanagari punctuation and ASCII punctuation to strip
PUNCTUATION_REGEX = re.compile(r'[\.,\?!;:।॥\(\)\[\]\{\}"\'`‘’“”\-—_\\/\|\*~@#\$%\^&+=<>]')
WHITESPACE_REGEX = re.compile(r'\s+')

def normalize_marathi_text(text: str) -> str:
    """
    Normalizes Marathi Devanagari text for standard ASR evaluation (WER/CER).
    Rules:
    1. Unicode NFC normalization.
    2. Lowercase any embedded English/Latin words.
    3. Remove punctuation including Devanagari danda (।) and double danda (॥).
    4. Normalize whitespace (collapse multiple spaces, trim ends).
    """
    if not text:
        return ""
    
    # 1. Normalize unicode
    text = unicodedata.normalize("NFC", text)
    
    # 2. Lowercase ASCII characters
    text = text.lower()
    
    # 3. Strip punctuation
    text = PUNCTUATION_REGEX.sub(" ", text)
    
    # 4. Collapse spaces
    text = WHITESPACE_REGEX.sub(" ", text).strip()
    
    return text
