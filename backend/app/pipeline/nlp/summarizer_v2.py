"""
Summarizer v2 — Fixed, grounded, no hallucinations.

Strategy:
1. TextRank (sumy) as PRIMARY — offline, zero network, zero hallucinations
2. BART only if already locally cached
3. Format distinctly per type — SHORT/MEDIUM/DETAILED all produce different output
4. Clean filler before summarizing
5. Truncation protection — always ends at sentence boundary
"""
import re, os
from typing import List, Dict, Optional
from loguru import logger

_model_cache: Dict = {}

# ── Sentence cleaner ──────────────────────────────────────────────────────────
FILLER_PATTERNS = [
    r"(hi|hello|hey)\s+(everyone|guys|there|friends)[^.!?]*[.!?]",
    r"welcome\s+(back\s+)?(to\s+)?[^.!?]*[.!?]",
    r"(don't|do not)\s+forget\s+to\s+(like|subscribe)[^.!?]*[.!?]",
    r"(please\s+)?(like|subscribe|hit\s+the\s+bell)[^.!?]*[.!?]",
    r"in\s+this\s+video\s+(today\s+)?we\s+will[^.!?]*[.!?]",
    r"today\s+we('re| are)\s+going\s+to[^.!?]*[.!?]",
    r"my\s+name\s+is\s+\w+[^.!?]*[.!?]",
    r"i('m| am)\s+\w+\s+and[^.!?]*[.!?]",
]

def _clean_input(text: str) -> str:
    for p in FILLER_PATTERNS:
        text = re.sub(p, " ", text, flags=re.IGNORECASE)
    return re.sub(r'\s+', ' ', text).strip()

def _clean_output(text: str) -> str:
    if not text: return ""
    # Remove model artifacts
    for a in ["arafed","araffe","araffes"]:
        text = text.replace(a, "")
    # Remove consecutive repeated words
    text = re.sub(r'\b(\w+)(\s+\1){2,}', r'\1', text)
    # Remove prompt echoes
    for marker in ["SHORT SUMMARY:","STUDY NOTES:","DETAILED NOTES:",
                   "BULLET SUMMARY:","Segment summary:","Combined summary:"]:
        if marker in text:
            text = text.split(marker, 1)[-1].strip()
    # End at sentence boundary
    last_end = max(text.rfind('.'), text.rfind('!'), text.rfind('?'))
    if last_end > len(text) * 0.6:
        text = text[:last_end + 1]
    return text.strip()

# ── TextRank extractive ───────────────────────────────────────────────────────
_sumy_available: Optional[bool] = None

def _textrank(text: str, n: int) -> str:
    global _sumy_available
    if _sumy_available is None:
        try:
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.summarizers.text_rank import TextRankSummarizer
            _sumy_available = True
        except ImportError:
            _sumy_available = False
            logger.warning("sumy not installed — using fallback sentence extraction. Install with: pip install sumy")
    if _sumy_available:
        try:
            from sumy.parsers.plaintext import PlaintextParser
            from sumy.nlp.tokenizers import Tokenizer
            from sumy.summarizers.text_rank import TextRankSummarizer
            parser = PlaintextParser.from_string(text[:8000], Tokenizer("english"))
            result = TextRankSummarizer()(parser.document, n)
            return " ".join(str(s) for s in result)
        except Exception as e:
            logger.warning(f"TextRank failed: {e}")
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 20]
    return " ".join(sents[:n])

# ── Chunking ──────────────────────────────────────────────────────────────────
def _chunk(text: str, size: int = 400, overlap: int = 30) -> List[str]:
    words = text.split()
    if len(words) <= size: return [text]
    chunks = []
    for i in range(0, len(words), size - overlap):
        c = words[i:i + size]
        if len(c) > 50: chunks.append(" ".join(c))
    return chunks

def _chunk_segments(segments: List[Dict], target: int = 300, overlap: int = 25) -> List[str]:
    if not segments: return []
    result, cur, cs = [], [], None
    for seg in segments:
        words = seg.get("text","").split()
        if cs is None: cs = seg.get("start", 0)
        cur.extend(words)
        if len(cur) >= target:
            result.append(" ".join(cur))
            cur = cur[-overlap:]
            cs = seg.get("start", cs)
    if len(cur) > 40: result.append(" ".join(cur))
    return result

# ── Model registry ────────────────────────────────────────────────────────────
MODEL_REGISTRY = {
    "bart": {
        "hf_id": "facebook/bart-large-cnn",
        "task": "summarization",
        "needs_prefix": False,
        "max_input": 1024,
        "max_output": 142,
        "gen_kwargs": {"do_sample": False, "length_penalty": 2.0},
    },
    "t5": {
        "hf_id": "t5-base",
        "task": "summarization",
        "needs_prefix": True,
        "max_input": 512,
        "max_output": 512,
        "gen_kwargs": {"do_sample": False, "length_penalty": 1.5},
    },
    "pegasus": {
        "hf_id": "google/pegasus-xsum",
        "task": "summarization",
        "needs_prefix": False,
        "max_input": 512,
        "max_output": 128,
        "gen_kwargs": {"do_sample": False, "length_penalty": 0.8, "num_beams": 4},
    },
    "flan": {
        "hf_id": "google/flan-t5-base",
        "task": "summarization",
        "needs_prefix": True,
        "max_input": 512,
        "max_output": 512,
        "gen_kwargs": {"do_sample": False, "length_penalty": 1.5, "num_beams": 4},
    },
}

# ── Availability check (cheap, filesystem-only) ───────────────────────────────
_TOKENIZER_FILES = ("tokenizer.json", "spiece.model", "vocab.json",
                    "sentencepiece.bpe.model", "vocab.txt")

def _cache_complete(hf_id: str) -> bool:
    """Return True only if the HF cache has weights + config + tokenizer files.
    A folder can exist with an incomplete/partial download (e.g. pegasus here),
    which would fail at load time — this catches that before we claim the model runs."""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_slug = "models--" + hf_id.replace("/", "--")
    snap_dir = os.path.join(cache_dir, model_slug, "snapshots")
    if not os.path.isdir(snap_dir):
        return False
    for rev in os.listdir(snap_dir):
        rev_path = os.path.join(snap_dir, rev)
        if not os.path.isdir(rev_path):
            continue
        files = os.listdir(rev_path)
        has_weights = any(f.endswith((".safetensors", ".bin", ".h5")) for f in files)
        has_config = "config.json" in files
        has_tokenizer = any(f in files for f in _TOKENIZER_FILES)
        if has_weights and has_config and has_tokenizer:
            return True
    return False

def model_availability() -> Dict[str, bool]:
    """Report which registered models can genuinely run (complete cache).
    Models that return False will silently fall back to extractive TextRank."""
    result = {}
    for key, cfg in MODEL_REGISTRY.items():
        if key in _model_cache:
            result[key] = True
        else:
            result[key] = _cache_complete(cfg["hf_id"])
    return result

# ── Model loader (lazy, cached) ───────────────────────────────────────────────
def _load_model(model_key: str):
    """Load a HF pipeline for the given model_key. Returns pipeline or None."""
    if model_key in _model_cache:
        return _model_cache[model_key]
    cfg = MODEL_REGISTRY.get(model_key)
    if not cfg:
        logger.warning(f"Unknown model_key: {model_key}, falling back to bart")
        cfg = MODEL_REGISTRY["bart"]
        model_key = "bart"
    try:
        cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
        model_slug = cfg["hf_id"].replace("/", "--")
        cached = os.path.exists(cache_dir) and any(
            model_slug in d for d in os.listdir(cache_dir)
        )
        if not cached:
            logger.info(f"Model {cfg['hf_id']} not cached, skipping (offline mode)")
            return None
        from transformers import pipeline
        pipe = pipeline(cfg["task"], model=cfg["hf_id"], device=-1)
        _model_cache[model_key] = pipe
        logger.info(f"Loaded model: {cfg['hf_id']}")
        return pipe
    except Exception as e:
        logger.warning(f"Failed to load {cfg['hf_id']}: {e}")
        return None

# ── Model-specific abstractive pass ───────────────────────────────────────────
TYPE_PROMPTS = {
    "short":    "Summarize the following in 3-4 concise sentences (about 150 words): ",
    "medium":   "Write a clear summary of the following in about 225 words: ",
    "detailed": ("Write a detailed, comprehensive multi-section summary of the following, "
                 "covering the introduction and context, core concepts and definitions, "
                 "methods and applications, analysis with examples, and conclusions: "),
    "bullets":  "List all the key points of the following as concise bullet points (about 250 words total): ",
    "academic": ("Write an academic summary of the following with an abstract, an "
                 "introduction, and conclusions (about 400 words total): "),
}

# ── Explicit word-count target ranges per summary type ───────────────────────
# Strict ascending order: Short < Medium < Bullets < Academic < Detailed
WORD_RANGES = {
    "short":    (120, 180),
    "medium":   (200, 250),
    "bullets":  (200, 300),
    "academic": (350, 450),
    "detailed": (500, 650),
}

# Per-type TOTAL token budgets for generation (words * ~1.4 for headroom so the
# model can complete sentences naturally before the hard cutoff).
TYPE_MAX = {"short": 250, "medium": 360, "detailed": 920, "bullets": 430, "academic": 640}
TYPE_MIN = {"short": 150, "medium": 250, "detailed": 630, "bullets": 250, "academic": 440}

# Matches instruction fragments that non-instruction-tuned models (T5-base) echo
# back into their output. Sentences containing these are dropped.
_ECHO_MARKERS = re.compile(
    r"(multi-section summary|key points of the following|academic summary"
    r"|concise (?:sentences|bullet points)|summary of the following"
    r"|multi-paragraph summary)",
    re.IGNORECASE,
)


def _scrub_echo(text: str) -> str:
    """Drop sentences that merely parrot the instruction prompt."""
    sents = re.split(r"(?<=[.!?])\s+", text)
    kept = [s for s in sents if s.strip() and not _ECHO_MARKERS.search(s)]
    return " ".join(kept) if kept else text


def _trim_to_word_count(text: str, max_words: int) -> str:
    """Trim text to at most *max_words* words, ending at the last complete sentence."""
    words = text.split()
    if len(words) <= max_words:
        return text
    trimmed = " ".join(words[:max_words])
    last_end = max(trimmed.rfind('.'), trimmed.rfind('!'), trimmed.rfind('?'))
    if last_end > len(trimmed) * 0.5:
        return trimmed[:last_end + 1].strip()
    return trimmed.strip()


def _validate_word_count(text: str, summary_type: str) -> str:
    """Check word count against the target range for *summary_type*.

    Trims over-long output to the last complete sentence within range.
    Under-length output is logged but returned as-is (caller may retry).
    Returns the (possibly trimmed) text.
    """
    lo, hi = WORD_RANGES.get(summary_type, (100, 500))
    wc = len(text.split())
    if wc > hi:
        text = _trim_to_word_count(text, hi)
        new_wc = len(text.split())
        logger.info(f"[validate] {summary_type} was {wc}w → trimmed to {new_wc}w (target {lo}-{hi})")
    elif wc < lo:
        logger.warning(f"[validate] {summary_type} only {wc}w, target {lo}-{hi}w — source may be too short")
    else:
        logger.info(f"[validate] {summary_type} {wc}w ✓ (target {lo}-{hi})")
    return text


def _abstractive(model_key: str, text: str, summary_type: str) -> Optional[str]:
    """Run the selected model over the text, windowing long inputs.

    cfg["max_input"] is the model's INPUT TOKEN capacity (T5/FLAN ≈ 512,
    BART ≈ 1024). We window the input into ~capacity-sized pieces (≈4 chars/token)
    and summarize each, then concatenate. This (a) stops small-context models from
    silently dropping the tail of long transcripts, and (b) lets long types like
    'detailed' produce proportionally longer output for every model instead of a
    single over-compressed ~80-word blob.
    """
    cfg = MODEL_REGISTRY.get(model_key, MODEL_REGISTRY["bart"])
    pipe = _load_model(model_key)
    if not pipe:
        return None
    try:
        prefix = TYPE_PROMPTS.get(summary_type, "summarize: ") if cfg["needs_prefix"] else ""
        char_budget = cfg["max_input"] * 4
        windows = [text[i:i + char_budget] for i in range(0, len(text), char_budget)] or [text]
        windows = [w for w in windows if w.strip()][:6]  # cap total work
        if not windows:
            return None

        total_max = TYPE_MAX.get(summary_type, 360)
        total_min = TYPE_MIN.get(summary_type, 150)
        model_cap = cfg.get("max_output", 512)
        per_max = min(model_cap, max(56, total_max // len(windows)))
        per_min = max(20, total_min // len(windows))

        pieces = [_run_model(model_key, w, per_max, per_min, prefix) for w in windows]
        out = _scrub_echo(" ".join(p for p in pieces if p))
        wc = len(out.split())
        lo, hi = WORD_RANGES.get(summary_type, (100, 500))

        # Retry once with higher per-window budget if too short
        if wc < lo:
            retry_max = min(model_cap, int(per_max * 1.5))
            logger.info(f"[retry] {summary_type} only {wc}w (target {lo}-{hi}), "
                        f"retrying with per_max={retry_max}")
            pieces = [_run_model(model_key, w, retry_max, per_min, prefix) for w in windows]
            out = _scrub_echo(" ".join(p for p in pieces if p))
            wc = len(out.split())
            logger.info(f"[retry] after retry: {wc}w")

        logger.info(
            f"[abstractive] model={model_key} type={summary_type} "
            f"prompt='{prefix[:60]}' windows={len(windows)} per_max={per_max} "
            f"out_words={wc}"
        )
        return out or None
    except Exception as e:
        logger.warning(f"Abstractive summarization with {model_key} failed: {e}")
        return None


def _run_model(model_key: str, text: str, max_len: int, min_len: int,
               prefix: str = "") -> Optional[str]:
    """Single abstractive pass over one input window. Returns scrubbed text or None."""
    cfg = MODEL_REGISTRY.get(model_key, MODEL_REGISTRY["bart"])
    pipe = _load_model(model_key)
    if not pipe:
        return None
    try:
        body = text[: cfg["max_input"] * 4]
        wt = int(len(body.split()) * 1.3)                  # approx input tokens
        model_cap = cfg.get("max_output", 512)
        mx = max(min_len + 8, min(max_len, model_cap, max(wt, min_len + 8)))
        mn = min(min_len, max(4, mx - 1))
        r = pipe(prefix + body, max_length=mx, min_length=mn,
                 truncation=True, **cfg["gen_kwargs"])
        return _scrub_echo(r[0]["summary_text"].strip())
    except Exception as e:
        logger.warning(f"_run_model {model_key} failed: {e}")
        return None


DETAILED_LABELS = ["Introduction & Context", "Core Concepts & Definitions",
                   "Methods, Algorithms & Applications", "Analysis & Examples",
                   "Summary & Conclusions"]


# Per-section (min_words, max_words) budgets for Detailed — 5 sections × ~100-130w = 500-650w
DETAILED_SECTION_BUDGETS = [(80, 130)] * 5


def _detailed_sectioned(model_key: str, sentences: List[str]) -> str:
    """Generate the 5 Detailed sections one model call each, then concatenate.

    This guarantees all 5 sections are produced (not cut off after 2) and makes
    Detailed the longest type — its length is the SUM of 5 per-section summaries
    rather than a single over-compressed pass. Falls back to the raw extractive
    group text for any section whose model call is unavailable/empty.
    """
    sentences = [s for s in sentences if s and s.strip()]
    n = len(sentences)
    ngroups = len(DETAILED_LABELS)
    if n == 0:
        return ""
    size = max(1, (n + ngroups - 1) // ngroups)            # ceil(n / ngroups)
    groups = [sentences[i:i + size] for i in range(0, n, size)][:ngroups]

    sections = []
    for i, grp in enumerate(groups):
        label = DETAILED_LABELS[i] if i < len(DETAILED_LABELS) else f"Section {i + 1}"
        grp_text = " ".join(grp)
        min_w, max_w = DETAILED_SECTION_BUDGETS[i] if i < len(DETAILED_SECTION_BUDGETS) else (80, 130)
        max_tok = int(max_w * 1.4)
        min_tok = int(min_w * 1.3)
        piece = _run_model(model_key, grp_text, max_len=max_tok, min_len=min_tok) or grp_text
        piece = _trim_to_word_count(piece.strip(), max_w)
        if piece:
            sections.append(f"## {label}\n{piece}")
    out = "\n\n".join(sections)
    logger.info(f"[detailed] model={model_key} sections={len(sections)} "
                f"out_words={len(out.split())}")
    return out


ACADEMIC_LABELS = ["Abstract", "Introduction", "Conclusions"]
# Per-section (min_words, max_words): Abstract ~80-100, Intro ~200-250, Concl ~80-100
ACADEMIC_SECTION_BUDGETS = [(70, 100), (180, 250), (70, 100)]


def _academic_sectioned(model_key: str, sentences: List[str]) -> str:
    """Generate the 3 Academic sections one model call each, then concatenate.

    Ensures Abstract, Introduction, and Conclusions are all present and complete.
    Falls back to the raw extractive group text for any section whose model call
    is unavailable/empty.
    """
    sentences = [s for s in sentences if s and s.strip()]
    n = len(sentences)
    if n == 0:
        return ""
    # Allocate sentences: 20% abstract, 50% intro, 30% conclusions
    n_abs = max(2, n // 5)
    n_intro = max(3, n // 2)
    n_conc = max(2, n - n_abs - n_intro)
    groups = [sentences[:n_abs], sentences[n_abs:n_abs + n_intro], sentences[n_abs + n_intro:]]

    sections = []
    for i, grp in enumerate(groups):
        label = ACADEMIC_LABELS[i] if i < len(ACADEMIC_LABELS) else f"Section {i + 1}"
        grp_text = " ".join(grp)
        min_w, max_w = ACADEMIC_SECTION_BUDGETS[i] if i < len(ACADEMIC_SECTION_BUDGETS) else (70, 100)
        max_tok = int(max_w * 1.4)
        min_tok = int(min_w * 1.3)
        piece = _run_model(model_key, grp_text, max_len=max_tok, min_len=min_tok) or grp_text
        piece = _trim_to_word_count(piece.strip(), max_w)
        if piece:
            sections.append(f"**{label}**\n{piece}")
    out = "\n\n".join(sections)
    logger.info(f"[academic] model={model_key} sections={len(sections)} "
                f"out_words={len(out.split())}")
    return out

# ── Format output distinctly per type ────────────────────────────────────────
def _select_parts(parts: List[str], max_words: int, overshoot: int = 30) -> List[str]:
    """Select sentences from *parts* until cumulative word count reaches *max_words*.
    Allows a small overshoot so we don't cut mid-sentence."""
    selected, wc = [], 0
    for p in parts:
        pw = len(p.split())
        if wc + pw > max_words + overshoot:
            break
        selected.append(p)
        wc += pw
    return selected or parts[:3]


def _format(sentences: str, summary_type: str) -> str:
    parts = [s.strip() for s in re.split(r'(?<=[.!?])\s+', sentences) if len(s.strip()) > 10]
    if not parts: return sentences

    lo, hi = WORD_RANGES.get(summary_type, (100, 500))

    if summary_type == "short":
        selected = _select_parts(parts, hi, overshoot=20)
        return " ".join(selected)

    elif summary_type == "bullets":
        selected = _select_parts(parts, hi, overshoot=20)
        return "\n".join(f"• {s}" for s in selected)

    elif summary_type == "medium":
        selected = _select_parts(parts, hi, overshoot=25)
        n = len(selected)
        if n == 0: return " ".join(parts[:5])
        intro  = " ".join(selected[:max(2, n//5)])
        core   = " ".join(selected[max(2,n//5):max(4,3*n//5)])
        detail = " ".join(selected[max(4,3*n//5):max(6,4*n//5)])
        takeaway = " ".join(selected[-2:]) if n >= 4 else ""
        out = f"## Overview\n{intro}\n\n"
        if core: out += f"## Core Concepts\n{core}\n\n"
        if detail: out += f"## Key Details\n{detail}\n\n"
        if takeaway: out += f"## Key Takeaway\n{takeaway}"
        return out.strip()

    elif summary_type == "detailed":
        # Fallback only — normally handled by _detailed_sectioned
        selected = _select_parts(parts, hi, overshoot=40)
        chunk_size = max(3, len(selected) // 4)
        sections = [selected[i:i+chunk_size] for i in range(0, len(selected), chunk_size)]
        labels = DETAILED_LABELS
        out = ""
        for i, sec in enumerate(sections[:5]):
            label = labels[i] if i < len(labels) else f"Section {i+1}"
            out += f"## {label}\n{' '.join(sec)}\n\n"
        return out.strip()

    elif summary_type == "academic":
        # Fallback only — normally handled by _academic_sectioned
        selected = _select_parts(parts, hi, overshoot=40)
        n = len(selected)
        abstract = " ".join(selected[:max(2, n//4)])
        intro    = " ".join(selected[max(2,n//4):max(4, 3*n//4)])
        conc     = " ".join(selected[max(4, 3*n//4):]) if n > 4 else ""
        out = f"**Abstract**\n{abstract}\n\n**Introduction**\n{intro}"
        if conc: out += f"\n\n**Conclusions**\n{conc}"
        return out.strip()

    return sentences

# ── Main public API ───────────────────────────────────────────────────────────
class HierarchicalSummarizer:
    def __init__(self, device: str = "cpu"):
        self.device = device

    def summarize(
        self,
        text: str,
        summary_type: str = "medium",
        model_key: str = "bart",
        segments: Optional[List[Dict]] = None,
    ) -> str:
        if not text or len(text.strip()) < 50:
            return "Insufficient content for summarization."

        text = _clean_input(text)
        logger.info(f"Summarizing {len(text.split())}w → type={summary_type}, model={model_key}")

        # Sentence counts per type (detailed pulls the most extractive content so
        # its 5 sections are each well-supported).
        n_map = {"short":6,"medium":12,"detailed":45,"bullets":14,"academic":20}
        n = n_map.get(summary_type, 12)

        # Chunk into manageable pieces
        chunks = _chunk_segments(segments, 300) if segments else _chunk(text, 400)
        per_n  = max(2, n // max(len(chunks), 1))

        # MAP: summarize each chunk with TextRank (extractive, model-independent)
        chunk_summaries = [_textrank(c, per_n) for c in chunks if c.strip()]
        merged = " ".join(s for s in chunk_summaries if s)

        # DETAILED: generate all 5 sections one model call each — guarantees the
        # full section set and the largest total length of any type.
        if summary_type == "detailed":
            sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', merged)
                     if len(s.strip()) > 10]
            output = _detailed_sectioned(model_key, sents)
            if not output.strip():
                output = _format(merged, "detailed")
            output = _validate_word_count(output, "detailed")
            return _clean_output(output)

        # ACADEMIC: generate 3 sections one model call each
        if summary_type == "academic":
            sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', merged)
                     if len(s.strip()) > 10]
            output = _academic_sectioned(model_key, sents)
            if not output.strip():
                output = _format(merged, "academic")
            output = _validate_word_count(output, "academic")
            return _clean_output(output)

        # REDUCE: run the selected model's abstractive pass
        abs_result = _abstractive(model_key, merged, summary_type)
        if abs_result:
            merged = abs_result
            logger.info(f"{model_key} produced {len(merged.split())}w")
        else:
            logger.info(f"{model_key} unavailable, using extractive-only result")

        # STYLE PASS
        output = _format(merged, summary_type)
        output = _validate_word_count(output, summary_type)
        return _clean_output(output)

    def summarize_all_types(self, text: str, model_key: str = "bart", segments: Optional[List[Dict]] = None) -> Dict[str,str]:
        results: Dict[str, str] = {}
        for st in ["short","medium","detailed","bullets","academic"]:
            results[st] = self.summarize(text, st, model_key, segments)
        return results
