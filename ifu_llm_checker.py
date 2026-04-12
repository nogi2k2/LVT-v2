"""
ifu_llm_checker.py
===================
LLM-based IFU requirement verification using Ollama (local, no API key).

Public API
----------
IFULLMChecker(model="llama3.1:8b")
    .check(requirement_text, ifu_pages)  ->  IFUCheckResult

IFUCheckResult fields
---------------------
satisfied      : bool
confidence     : "high" | "medium" | "low"
reason         : str   — one sentence explanation
evidence_text  : str   — exact text from IFU that satisfies the requirement
evidence_page  : int   — 1-based page number
evidence_para  : str   — full paragraph containing the evidence (for highlighting)
model_used     : str
raw_response   : str
"""

from __future__ import annotations
import json
import re
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class IFUCheckResult:
    satisfied:      bool
    confidence:     str          # "high" | "medium" | "low"
    reason:         str
    evidence_text:  str          # snippet from IFU
    evidence_page:  int          # 1-based, 0 = not found
    evidence_para:  str          # full paragraph for yellow highlight
    model_used:     str
    raw_response:   str = field(default="", repr=False)
    error:          str = ""


# ── Prompt template ────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a medical device regulatory reviewer specialising in
IFU (Instructions for Use) compliance for Philips Respironics accessories.
Your task is to determine whether specific sections of an IFU document
satisfy a given regulatory requirement.

Rules:
- Focus on INTENT, not just keyword matching.
- "Performance specifications" = any section with measurable technical values
  (dimensions, resistance, flow rates, percentages, tolerances).
- Content may be in any language — assess meaning, not language.
- Be conservative: if you are not sure, set confidence to "low".
- Reply ONLY with valid JSON. No preamble, no explanation outside JSON."""

_USER_TEMPLATE = """REQUIREMENT:
{requirement}

IFU PAGES (English section only — pages {page_range}):
{pages_text}

Does any page above satisfy the requirement?

Reply ONLY with this JSON (no markdown, no extra text):
{{
  "satisfied": true or false,
  "confidence": "high" or "medium" or "low",
  "reason": "one sentence explaining your decision",
  "evidence_text": "the exact short phrase from the IFU that satisfies it, or null",
  "evidence_page": the page number (integer) where evidence was found, or 0
}}"""


# ── Pre-filtering: find candidate pages before sending to LLM ─────────────

# Section heading keywords → requirement intent mapping
# Used to quickly narrow 108 pages → 3-5 candidate pages
_SECTION_HINTS = [
    # (requirement keywords, section heading keywords to look for)
    (["performance", "specification", "technical"],
     ["specifications", "specs", "technical data", "performance",
      "spezifikation", "caractéristiques", "specifiche", "especificaciones",
      "especificações", "specificaties", "spesifikasjoner", "specifikationer",
      "spesifikasyonlar", "spesifikasi", "thông số", "사양", "仕様", "规格", "規格"]),

    (["intended use", "intended purpose", "indication"],
     ["intended use", "intended purpose", "indications",
      "verwendungszweck", "uso previsto", "utilização", "beoogd gebruik"]),

    (["warning", "caution", "contraindication"],
     ["warnings", "cautions", "contraindications",
      "avertissements", "warnungen", "advertencias"]),

    (["instruction", "how to use", "setup", "assembly"],
     ["instructions for use", "how to use", "assembly", "bedienungsanleitung",
      "instructions d'utilisation", "istruzioni per l'uso"]),

    (["disposal", "end of life", "discard"],
     ["disposal", "dispose", "discard", "entsorgun"]),

    (["compatibility", "compatible"],
     ["compatibility", "compatible", "kompatibilität"]),

    (["symbol", "glossary"],
     ["symbols", "symbol glossary", "symbols.philips.com", "symbole"]),

    (["contact", "manufacturer", "customer service"],
     ["contact", "customer service", "how to contact", "philips respironics"]),

    (["single use", "do not reuse", "single patient"],
     ["single use", "single patient", "do not reuse", "einmalig"]),

    (["material", "latex", "phthalate"],
     ["material", "latex", "phthalate", "not made with"]),

    (["storage", "environment", "temperature", "humidity"],
     ["storage", "temperature", "humidity", "environment",
      "temperature limit", "humidity limitation"]),

    (["sterile", "sterilization", "sterilisation"],
     ["sterile", "steriliz", "sterilise", "eto", "ethylene oxide"]),
]


def _find_candidate_pages(requirement: str,
                           pages: list[dict]) -> list[dict]:
    """
    Quickly narrow from all pages to the 3-5 most relevant ones.
    Returns pages sorted by relevance score (highest first).
    Only looks at English section (pages 1-3 for this IFU structure).
    """
    req_lower = requirement.lower()

    # Find which hint bucket the requirement falls into
    target_sections = []
    for req_kws, section_kws in _SECTION_HINTS:
        if any(kw in req_lower for kw in req_kws):
            target_sections.extend(section_kws)

    if not target_sections:
        # No hint matched — return first 6 pages (English section)
        return pages[:6]

    scored = []
    for pg in pages:
        text_lower = pg["text"].lower()
        lines      = [l.strip() for l in pg["text"].split("\n") if l.strip()]

        score = 0
        # Bonus if a SHORT line (heading) matches
        for line in lines[:5]:   # headings tend to be near top of page
            if len(line) < 60:
                for kw in target_sections:
                    if kw.lower() in line.lower():
                        score += 10   # strong signal: heading match

        # General text match
        for kw in target_sections:
            if kw.lower() in text_lower:
                score += 2

        if score > 0:
            scored.append((score, pg))

    if not scored:
        return pages[:6]

    scored.sort(key=lambda x: x[0], reverse=True)
    return [pg for _, pg in scored[:5]]


def _parse_llm_json(raw: str) -> dict:
    """Extract and parse JSON from LLM response, handling common issues."""
    # Strip markdown code fences
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    raw = raw.strip("`").strip()

    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Try to extract first JSON object
    m = re.search(r'\{.*?\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass

    # Fallback: parse key-value manually
    result = {}
    for key in ("satisfied", "confidence", "reason", "evidence_text", "evidence_page"):
        m = re.search(rf'"{key}"\s*:\s*(.+?)(?:,|\}})', raw, re.DOTALL)
        if m:
            val = m.group(1).strip().strip('"').strip("'")
            if key == "satisfied":
                result[key] = val.lower() in ("true", "yes", "1")
            elif key == "evidence_page":
                try:
                    result[key] = int(val)
                except ValueError:
                    result[key] = 0
            else:
                result[key] = val
    return result


def _find_evidence_paragraph(page_text: str, evidence_text: str) -> str:
    """
    Find the full paragraph in page_text that contains evidence_text.
    Returns the paragraph for yellow highlighting.
    """
    if not evidence_text or not page_text:
        return ""

    # Split page into paragraphs (blank-line separated or newline separated)
    paragraphs = [p.strip() for p in re.split(r'\n{2,}|\n', page_text) if p.strip()]

    # Find paragraph containing the evidence
    ev_lower = evidence_text.lower()
    for para in paragraphs:
        if ev_lower[:30] in para.lower():
            return para

    # Fuzzy: find paragraph with most words in common
    ev_words = set(ev_lower.split())
    best_para, best_score = "", 0
    for para in paragraphs:
        para_words = set(para.lower().split())
        score = len(ev_words & para_words)
        if score > best_score:
            best_score, best_para = score, para
    return best_para if best_score >= 2 else ""


# ── Main checker class ─────────────────────────────────────────────────────

class IFULLMChecker:
    """
    Uses a local Ollama LLM to check whether IFU pages satisfy a requirement.

    Args:
        model:      Ollama model name (default: "llama3.1:8b")
        host:       Ollama host URL (default: "http://localhost:11434")
        timeout:    Request timeout in seconds (default: 120)
    """

    def __init__(self,
                 model:   str = "llama3.1:8b",
                 host:    str = "http://localhost:11434",
                 timeout: int = 120):
        self.model   = model
        self.host    = host
        self.timeout = timeout
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import ollama
                self._client = ollama.Client(host=self.host)
            except ImportError:
                raise ImportError(
                    "ollama package not installed. Run: pip install ollama")
        return self._client

    def is_available(self) -> tuple[bool, str]:
        """Check if Ollama is running and the model is available."""
        try:
            client = self._get_client()
            models = client.list()
            names  = [m.model for m in models.models]
            if any(self.model in n for n in names):
                return True, f"Model '{self.model}' ready"
            return False, (f"Model '{self.model}' not pulled. "
                           f"Run: ollama pull {self.model}\n"
                           f"Available: {', '.join(names)}")
        except Exception as e:
            return False, f"Ollama not reachable at {self.host}: {e}"

    def check(self,
              requirement: str,
              ifu_pages:   list[dict]) -> IFUCheckResult:
        """
        Check whether the IFU satisfies the requirement.

        Args:
            requirement:  PRD requirement text
            ifu_pages:    list of {"page": int, "text": str}
                          (all pages or English-only subset)

        Returns:
            IFUCheckResult
        """
        # Step 1 — Find candidate pages
        candidates = _find_candidate_pages(requirement, ifu_pages)

        if not candidates:
            return IFUCheckResult(
                satisfied=False, confidence="low",
                reason="No IFU pages provided.",
                evidence_text="", evidence_page=0, evidence_para="",
                model_used=self.model,
                error="No pages to check"
            )

        # Step 2 — Build prompt
        page_range  = f"{candidates[0]['page']}–{candidates[-1]['page']}"
        pages_text  = "\n\n".join(
            f"--- PAGE {p['page']} ---\n{p['text'].strip()}"
            for p in candidates
        )
        user_prompt = _USER_TEMPLATE.format(
            requirement  = requirement,
            page_range   = page_range,
            pages_text   = pages_text,
        )

        # Step 3 — Call LLM
        try:
            client   = self._get_client()
            response = client.chat(
                model    = self.model,
                messages = [
                    {"role": "system",  "content": _SYSTEM_PROMPT},
                    {"role": "user",    "content": user_prompt},
                ],
                options  = {"temperature": 0.1, "num_predict": 300},
            )
            raw = response.message.content.strip()
        except Exception as e:
            logger.exception("Ollama call failed")
            return IFUCheckResult(
                satisfied=False, confidence="low",
                reason=f"LLM call failed: {e}",
                evidence_text="", evidence_page=0, evidence_para="",
                model_used=self.model, error=str(e)
            )

        # Step 4 — Parse response
        parsed = _parse_llm_json(raw)

        satisfied     = bool(parsed.get("satisfied", False))
        confidence    = str(parsed.get("confidence", "low"))
        reason        = str(parsed.get("reason", ""))
        evidence_text = str(parsed.get("evidence_text") or "")
        evidence_page = int(parsed.get("evidence_page", 0))

        # Step 5 — Find full paragraph for yellow highlighting
        evidence_para = ""
        if evidence_page > 0 and evidence_text:
            for pg in candidates:
                if pg["page"] == evidence_page:
                    evidence_para = _find_evidence_paragraph(
                        pg["text"], evidence_text)
                    break

        return IFUCheckResult(
            satisfied=satisfied,
            confidence=confidence,
            reason=reason,
            evidence_text=evidence_text,
            evidence_page=evidence_page,
            evidence_para=evidence_para,
            model_used=self.model,
            raw_response=raw,
        )
