"""Small text helpers shared by ingestion, chunking and retrieval (English + Hindi)."""

from __future__ import annotations

import re
import unicodedata

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
WORD = re.compile(r"[\wऀ-ॿ]+", re.UNICODE)
CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
HINDI_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")
SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")  # । = Hindi full stop (danda)

# A few domain terms in both languages so keyword search can bridge Hindi questions and English
# documents (and back). Deliberately small and checkable; the embedding model does the rest.
BILINGUAL = {
    "पराली": "paddy straw stubble",
    "पुआल": "straw",
    "धान": "paddy",
    "फसल अवशेष": "crop residue",
    "अवशेष": "residue",
    "जलाना": "burning",
    "जलाने": "burning",
    "अनुदान": "subsidy",
    "सब्सिडी": "subsidy",
    "प्रोत्साहन": "incentive",
    "राशि": "amount",
    "एकड़": "acre",
    "प्रति एकड़": "per acre",
    "मशीन": "machine machinery",
    "किसान": "farmer",
    "जुर्माना": "fine penalty",
    "आवेदन": "apply application",
    "पंजीकरण": "registration",
    "कस्टम हायरिंग": "custom hiring centre",
    "हैप्पी सीडर": "happy seeder",
    "बेलर": "baler",
    "सुपर सीडर": "super seeder",
    "डीकंपोजर": "decomposer",
    "हरियाणा": "haryana",
    "पंजाब": "punjab",
}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).translate(HINDI_DIGITS)
    text = text.replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def is_hindi(text: str, threshold: float = 0.3) -> bool:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return False
    return sum(bool(DEVANAGARI.match(c)) for c in letters) / len(letters) >= threshold


def tokens(text: str) -> list[str]:
    return [t.lower() for t in WORD.findall(normalize(text))]


def expand_bilingual(query: str) -> str:
    """Append English equivalents of known Hindi terms (keyword search can't translate)."""
    extra = [en for hi, en in BILINGUAL.items() if hi in query]
    return query + (" " + " ".join(extra) if extra else "")


CLAUSE_END = re.compile(r"(?<=[;:])\s+|\s+(?=\(?[ivx]{1,4}\)\s)")  # "; " ":" and "(ii) " clause breaks


def sentences(text: str, max_words: int = 40) -> list[str]:
    """Sentences, with long ones (common in legal notices and OCR text) split at clause breaks."""
    lines = [re.sub(r"^\s*#+\s*", "", ln) for ln in text.split("\n")]  # drop markdown heading marks
    out = []
    for part in (p.strip() for block in lines for p in SENTENCE_END.split(block)):
        if len(part.split()) > max_words:
            out.extend(c.strip() for c in CLAUSE_END.split(part))
        else:
            out.append(part)
    return [p for p in out if len(p) > 2]


NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)")


def numbers(text: str) -> set[str]:
    """Numbers as written, without thousands separators ("Rs 1,000" -> "1000")."""
    return {m.replace(",", "") for m in NUMBER.findall(normalize(text))}
