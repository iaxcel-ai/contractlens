"""
Plain-English rewriting for flagged clauses.

Two things happen here:
1. SUMMARIZATION (Sumy): contract clauses are often one long, dense
   sentence. Sumy's LexRank summarizer scores sentences by how "central"
   they are to the text (similar to how Google ranks web pages) and picks
   out the most important part. For a single clause this mostly helps
   trim filler/legal boilerplate around the core rule when a clause is
   long; for short clauses it just returns the clause as-is.
2. CATEGORY TEMPLATES: summarization alone doesn't turn legal phrasing into
   plain English - it just shortens it. So for each category we also apply
   a short, plain-language explainer template that describes what a flag
   in that category generally MEANS for the user, in everyday words.
"""

import os
import nltk

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

# NLTK's default download folder lives under the user's home directory,
# which on some Windows setups sits on a different drive than the temp
# folder NLTK downloads through - that combination fails outright. Storing
# the tokenizer data inside the project instead sidesteps that and also
# keeps the project self-contained (no dependency on a machine-wide
# download).
_NLTK_DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "nltk_data")
if _NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.append(_NLTK_DATA_DIR)

_summarizer = LexRankSummarizer()

# Plain-language "what this category is about" explainers shown for every
# flagged clause, regardless of the specific wording used.
_CATEGORY_EXPLAINERS = {
    "cancellation": "This clause explains how and when you're allowed to cancel or end the agreement.",
    "auto-renewal": "This clause explains whether your subscription/contract renews automatically, and what notice (if any) you get first.",
    "fees": "This clause explains what you'll be charged and under what conditions.",
    "data sharing": "This clause explains how your personal data may be collected, used, or shared with others.",
    "liability": "This clause explains who is responsible (and who is NOT responsible) if something goes wrong.",
    "dispute resolution": "This clause explains how disagreements between you and the company get resolved (e.g. arbitration vs. court).",
    "other": "This is a general contract clause.",
}


def _summarize(text: str) -> str:
    """Use Sumy's LexRank to pull out the most important sentence(s) of a clause."""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    # Most clauses are one sentence anyway; asking for 2 sentences just
    # means "give me back the whole thing" for short clauses, and trims
    # down genuinely long/multi-sentence clauses.
    summary_sentences = _summarizer(parser.document, sentences_count=2)
    summary = " ".join(str(sentence) for sentence in summary_sentences)
    return summary if summary else text


def rewrite_clause(clause_text: str, category: str, flag_reasons: list[str]) -> str:
    """
    Build a plain-English explanation for a flagged clause:
    category context + summarized clause + the specific reasons it was flagged.
    """
    explainer = _CATEGORY_EXPLAINERS.get(category, _CATEGORY_EXPLAINERS["other"])
    summary = _summarize(clause_text)

    lines = [explainer, f"In short: {summary}"]
    if flag_reasons:
        lines.append("Why it's flagged: " + " ".join(flag_reasons))

    return " ".join(lines)
