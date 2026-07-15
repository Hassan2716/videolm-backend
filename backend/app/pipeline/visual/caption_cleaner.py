"""
Caption and OCR cleaner — v6 (stable).

Key insight: hallucination and repetition are DIFFERENT problems:
- Repetition: BLIP generates "neural neural neural" → remove duplicates
- Hallucination: BLIP generates plausible-sounding but WRONG text → replace with OCR

Solution:
1. Fix repetition FIRST (multi-pass)
2. Check if result makes sense (word diversity test)
3. If still hallucinating → use OCR to build caption instead
4. NEVER return empty if input had content
"""
import re
from typing import Tuple
from collections import Counter
from loguru import logger

BLIP_ARTIFACTS = ["arafed", "araffe", "araffed", "araffes"]

WIKI_NOISE = [
    r"From Wikipedia,?\s*the free encyclopedia",
    r"\[show\]\s*|\[hide\]\s*|\[edit\]\s*|\[\d+\]\s*",
    r"Article\s+Talk\s+Read",
    r"Read\s+Edit\s+View",
    r"For the journal,?\s*see[^.]+\.",
    r"tang[ij]+ages?[!?,. ]*",
    r"[¥\$€£]\s*[:;]?\s*['\"]?\s*Read",
    r"Y:\s+['\"]?\s*Read\s*;?\s*/",
    r"Ffoth\s+Wikip[a-z]+",
    r"entyctop[eé]dia",
    r"Stata\s+baring[^.]*",
    r"ef\s+[eé]os\s+Fee[.,!]*",
    r"MA\s+eo\b[^a-zA-Z]*",
    r"Leaming\b",
    r"\d+\s+For\s+the\s+journal",
    r"https?://\S+",
]

# OCR tokens that are clearly OCR garbage (not real words)
OCR_JUNK = {
    'llc','lll','sre','srsa','eris','glusteredidata','plt','rt','lit',
    'jest','aed','eee','hse','bess','sah','sica','fae','sel','pel',
    'ree','dayer','wcexte','pfe','ry','ou','cy','cr','bo','sf','ja',
    'ao','wi','pa','sc','ate',
}

# ALL-CAPS labels known to be educational content (keep these)
CAPS_WHITELIST = {
    'ML','AI','SVM','KNN','PCA','OCR','GPU','CPU','RBF','RNN','CNN',
    'NLP','ROC','AUC','DATA','RAW','INPUT','OUTPUT','HIDDEN','LAYER',
    'NODE','NETWORK','DEEP','LEARNING','NEURAL','ERA','CLASSIFICATION',
    'REGRESSION','VISUALIZATION','HEIGHT','SHOE','SIZE','CLASSMATE',
    'KERNEL','FORMULA','OPTIMIZATION','POLYNOMIAL','SIGMOID','DOT',
}


# ── Step 1: Fix repetition (multi-pass, handles 1-4 word phrases) ─────────────
def fix_repetition(text: str) -> str:
    """
    Remove repeated words/phrases. Multiple passes handles nested repetitions.
    Examples:
      "neural neural neural" → "neural"
      "data visual data visual" → "data visual"
      "of the algorithm of the algorithm" → "of the algorithm"
    """
    for _ in range(6):
        prev = text
        # 4-word phrases first (most specific)
        text = re.sub(r'\b(\w+\s+\w+\s+\w+\s+\w+)(\s+\1)+', r'\1', text, flags=re.IGNORECASE)
        # 3-word
        text = re.sub(r'\b(\w+\s+\w+\s+\w+)(\s+\1)+', r'\1', text, flags=re.IGNORECASE)
        # 2-word
        text = re.sub(r'\b(\w+\s+\w+)(\s+\1)+', r'\1', text, flags=re.IGNORECASE)
        # 1-word (3+ repeats)
        text = re.sub(r'\b(\w+)(\s+\1){2,}', r'\1', text, flags=re.IGNORECASE)
        # comma-separated
        text = re.sub(r'\b([\w][\w\s\-]{3,30})(,\s*\1)+', r'\1', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+', ' ', text).strip()
        if text == prev:
            break
    return text


# ── Step 2: Hallucination detection ──────────────────────────────────────────
def _is_hallucinating(text: str) -> bool:
    """
    True if word diversity is too low → BLIP is making things up.
    Legitimate captions have varied vocabulary.
    'An example of the algorithm for the algorithm of the algorithm...' → True
    'A scatter plot showing height vs shoe size' → False
    """
    words = re.findall(r'\b\w+\b', text.lower())
    if len(words) < 7:
        return False

    STOP = {'a','an','the','of','for','in','on','at','to','and','or','is',
            'are','with','by','from','that','this','it','as','be','was',
            'were','its','has','have','one','two','three','each','which'}
    content = [w for w in words if w not in STOP and len(w) > 2]
    if len(content) < 4:
        return False

    counts = Counter(content)
    # Top 2 content words dominate → hallucinating
    top2 = sum(v for _, v in counts.most_common(2))
    return (top2 / len(content)) > 0.55


# ── Step 3: Build better caption from OCR ─────────────────────────────────────
def _caption_from_ocr(original_caption: str, ocr: str) -> str:
    """
    Build an informative caption using OCR key terms when BLIP caption is bad.
    Only replaces caption if the OCR-based version is clearly better.
    """
    if not ocr or len(ocr.strip()) < 5:
        return ""

    # Clean OCR first to get quality terms
    ocr_clean = clean_ocr(ocr, max_chars=400)

    # Extract ALL-CAPS multi-word labels (most reliable in educational diagrams)
    caps_phrases = re.findall(r"\b[A-Z]{2,}(?:\s+[A-Z']{2,})+\b", ocr)
    CAPS_JUNK = {'LLC','LLL','SRE','SRSA','ERIS','GLUSTEREDIDATA','PLT','RT',
                 'LIT','JEST','AED','EEE','HSE','BESS','SAH','SEL','PEL','REE'}
    labels = []
    seen = set()
    for p in caps_phrases:
        tokens = p.split()
        clean_tokens = [t for t in tokens if t not in CAPS_JUNK]
        if len(clean_tokens) >= 2:
            label = ' '.join(clean_tokens).title()
            if label not in seen and len(label) > 4:
                labels.append(label)
                seen.add(label)
        if len(labels) >= 3:
            break

    # Determine visual type from original caption
    cap_lower = original_caption.lower()
    if "neural" in cap_lower or "network" in cap_lower:
        prefix = "Neural network diagram showing"
    elif "diagram" in cap_lower:
        prefix = "Diagram showing"
    elif "chart" in cap_lower or "graph" in cap_lower:
        prefix = "Chart showing"
    elif "plot" in cap_lower or "scatter" in cap_lower:
        prefix = "Scatter plot showing"
    elif "table" in cap_lower:
        prefix = "Table of"
    elif "screen" in cap_lower:
        prefix = "Screenshot of"
    elif "formula" in ocr_clean.lower() or "kernel" in ocr_clean.lower():
        prefix = "Table showing kernel functions and formulas"
    elif "raw data" in ocr_clean.lower() or "visualization" in ocr_clean.lower():
        prefix = "Data visualization showing"
    elif "classmate" in ocr_clean.lower() or "height" in ocr_clean.lower():
        prefix = "Scatter plot showing"
    else:
        prefix = "Educational frame showing"

    if labels:
        return f"{prefix} {' and '.join(labels[:2])}."

    # Fallback: use first meaningful OCR sentence
    if ocr_clean and len(ocr_clean) > 10:
        return f"{prefix}: {ocr_clean[:100].rstrip('.')}."

    return ""


# ── Public API ────────────────────────────────────────────────────────────────
def clean_caption(caption: str, ocr_hint: str = "") -> str:
    """
    Clean a BLIP/BLIP-2 caption.

    Pipeline:
    1. Remove BLIP artifacts
    2. Fix repetition (multi-pass)
    3. Detect if still hallucinating
    4. If hallucinating → replace with OCR-based caption
    5. Safety: never return empty if input had content
    """
    if not caption:
        return ""

    original = caption.strip()
    text = original

    # Step 1: Remove BLIP artifacts
    for art in BLIP_ARTIFACTS:
        text = re.sub(re.escape(art), " ", text, flags=re.IGNORECASE)

    # Step 2: Fix repetition
    text = fix_repetition(text)
    text = re.sub(r'\s+', ' ', text).strip().rstrip("., ")

    # Step 3: Check if still problematic
    too_short = len(text.split()) <= 4
    hallucinating = _is_hallucinating(text)

    # Step 4: Replace with OCR-based caption if needed
    if (too_short or hallucinating) and ocr_hint:
        ocr_caption = _caption_from_ocr(text, ocr_hint)
        if ocr_caption and len(ocr_caption) > 8:
            logger.debug(f"Caption replaced: '{text[:40]}' → '{ocr_caption[:40]}'")
            return ocr_caption

    # Step 5: Finalise
    if text and len(text) >= 4:
        text = text[0].upper() + text[1:]
        if text[-1] not in '.!?':
            text += '.'
        return text

    # Safety fallback
    return original if len(original) >= 4 else ""


def clean_ocr(ocr_text: str, max_chars: int = 250) -> str:
    """
    Clean OCR output.
    - Removes Wikipedia/web UI noise
    - Filters lines with too many junk tokens
    - Extracts ALL-CAPS diagram labels as fallback
    - Always returns something if input had content
    """
    if not ocr_text:
        return ""

    original = ocr_text.strip()
    text = original

    # Remove web/wiki noise
    for pat in WIKI_NOISE:
        text = re.sub(pat, " ", text, flags=re.IGNORECASE)

    # Fix garbled accented chars
    for old, new in [('é','e'),('è','e'),('ê','e'),('ë','e'),
                     ('à','a'),('â','a'),('ä','a'),('î','i'),
                     ('ï','i'),('ô','o'),('ö','o'),('û','u'),('ü','u')]:
        text = text.replace(old, new).replace(old.upper(), new.upper())

    # Filter lines: keep only lines where <35% of word-tokens are junk
    lines = re.split(r'[\n\r|;]', text)
    good_lines = []
    for line in lines:
        line = line.strip()
        if len(line) < 4:
            continue
        word_tokens = re.findall(r'\b[a-zA-Z]{2,}\b', line)
        if not word_tokens:
            # Line is numbers/math — keep if it has recognisable labels
            if re.search(r'\b(x[12]?|y|height|size|value|node|layer)\b', line, re.I):
                good_lines.append(line)
            continue
        junk_count = sum(1 for t in word_tokens
                         if t.lower() in OCR_JUNK
                         or (len(t) > 2 and not any(c in 'aeiou' for c in t.lower())
                             and t.upper() not in CAPS_WHITELIST))
        if junk_count / len(word_tokens) <= 0.35:
            good_lines.append(re.sub(r'\s+', ' ', line).strip())

    readable = ' '.join(good_lines).strip()

    # If readable is too short, try ALL-CAPS label extraction
    if len(readable) < 10:
        CAPS_JUNK_SET = {'LLC','LLL','SRE','SRSA','ERIS','GLUSTEREDIDATA',
                         'PLT','RT','LIT','JEST','AED','EEE','HSE','BESS',
                         'SAH','SEL','PEL','REE','SC','ATE'}
        phrases = re.findall(r"\b[A-Z][A-Z']{1,}(?:\s+[A-Z']{2,})+\b", text)
        labels = []
        seen = set()
        for p in phrases:
            toks = [t for t in p.split() if t not in CAPS_JUNK_SET]
            if len(toks) >= 2:
                lbl = ' '.join(toks).title()
                if lbl not in seen:
                    labels.append(lbl); seen.add(lbl)
            if len(labels) >= 4:
                break
        if labels:
            readable = ', '.join(labels)

    result = fix_repetition(readable)
    result = re.sub(r'\s+[,:;/!?]+\s+', ' ', result)
    result = re.sub(r'\s+', ' ', result).strip()

    # Trim to max_chars at sentence boundary
    if len(result) > max_chars:
        trunc = result[:max_chars]
        last = max(trunc.rfind('.'), trunc.rfind(','))
        result = result[:last + 1].rstrip() if last > 30 else trunc.rstrip() + '…'

    # Safety: never return empty
    if len(result.strip()) < 4:
        return (original[:max_chars] + '…') if len(original) > max_chars else original

    return result.strip()


def clean_frame_data(caption: str, ocr_text: str) -> Tuple[str, str]:
    """Clean both caption and OCR for a frame. Always returns content."""
    clean_ocr_result = clean_ocr(ocr_text or "", max_chars=250)
    clean_cap = clean_caption(caption or "", ocr_hint=ocr_text or "")
    return clean_cap, clean_ocr_result
