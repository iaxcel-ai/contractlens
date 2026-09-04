"""
Clause segmentation.

WHAT: Contracts and Terms & Conditions documents are usually one big wall of
text. Before we can classify or flag anything, we need to break that text
into individual "clauses" - roughly one rule/promise/condition per chunk.

WHY spaCy: Splitting on periods alone breaks badly on contracts, because they
are full of abbreviations, numbered lists ("Section 3.2"), and sentences that
run for several lines. spaCy's sentence boundary detection uses a trained
model (not just regex) to find real sentence breaks, which gives us a much
cleaner starting point than naive splitting.

After spaCy gives us sentences, we do some extra cleanup that is specific to
contracts:
- Merge numbered-list headers (e.g. "3." or "(a)") into the sentence that
  follows them, so we don't end up with a "clause" that is just "3."
- Drop very short fragments (page numbers, stray headers) that aren't real
  clauses.
"""

import re
import spacy

# Load the small English model once. It's not trained on legal text
# specifically, but sentence boundary detection doesn't need domain
# knowledge - it just needs to know English grammar/punctuation patterns.
_nlp = spacy.load("en_core_web_sm")

# Matches list markers like "1.", "(a)", "iv.", "A." sitting on their own,
# which spaCy sometimes treats as their own "sentence".
_LIST_MARKER_RE = re.compile(r"^\s*(\(?[0-9]{1,3}\)?[.)]|\(?[a-zA-Z]\)?[.)]|\(?[ivxlcdm]+\)?[.)])\s*$", re.IGNORECASE)

# A real clause should have a minimum amount of content. Anything shorter
# is almost always a page number, a header, or a stray fragment.
_MIN_CLAUSE_LENGTH = 20


def _clean_text(raw_text: str) -> str:
    """Normalize whitespace before feeding text into spaCy."""
    # Collapse multiple blank lines / spaces so spaCy doesn't treat
    # line breaks inside a paragraph as sentence breaks.
    text = re.sub(r"[ \t]+", " ", raw_text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()


def segment_into_clauses(raw_text: str) -> list[str]:
    """
    Take raw contract/ToS text and return a list of individual clause
    strings, cleaned up and ready for classification.
    """
    text = _clean_text(raw_text)
    if not text:
        return []

    doc = _nlp(text)
    raw_sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    clauses: list[str] = []
    pending_marker = ""  # holds a list marker like "3." until the next real sentence

    for sentence in raw_sentences:
        if _LIST_MARKER_RE.match(sentence):
            # This "sentence" is just a list marker on its own line -
            # remember it and attach it to the next real sentence instead
            # of treating it as its own clause.
            pending_marker = sentence
            continue

        # Strip a leading list marker that's still attached to the sentence
        # (e.g. "3. You may cancel...") so classification isn't distracted
        # by the numbering.
        sentence = re.sub(r"^\s*(\(?[0-9]{1,3}\)?[.)]|\(?[a-zA-Z]\)?[.)])\s*", "", sentence)

        if pending_marker:
            pending_marker = ""

        if len(sentence) >= _MIN_CLAUSE_LENGTH:
            clauses.append(sentence)

    return clauses
