"""
Red-flag scoring.

WHY RULE-BASED?

Risk is a policy decision. Rules make the reasoning visible, auditable,
and easy to adjust without retraining a model.
"""

import re

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# DATA MODELS
# ---------------------------------------------------------------------------

@dataclass
class Flag:
    reason: str
    severity: int  # 1 = mild, 2 = moderate, 3 = serious


@dataclass
class ClauseFacts:
    """
    Structured facts extracted from a clause.

    Values can remain None when the clause does not provide enough
    information.
    """

    early_repayment: bool = False
    early_repayment_restricted: bool = False

    fee_present: bool = False
    penalty_present: bool = False
    fee_percentage: float | None = None

    automatic_renewal: bool = False
    renewal_without_notice: bool = False

    notice_period_days: int | None = None
    cancellation_window_days: int | None = None
    cancellation_window_hours: int | None = None

    broad_data_sharing: bool = False
    third_party_count: int = 0

    binding_arbitration: bool = False
    class_action_waiver: bool = False

    unilateral_change: bool = False


@dataclass
class ClauseResult:
    text: str
    category: str
    flags: list[Flag] = field(default_factory=list)
    facts: ClauseFacts | None = None

    @property
    def is_flagged(self) -> bool:
        return len(self.flags) > 0

    @property
    def risk_score(self) -> int:
        return sum(flag.severity for flag in self.flags)

    @property
    def risk_level(self) -> str:

        if not self.flags:
            return "None"

        worst = max(flag.severity for flag in self.flags)

        if worst >= 3:
            return "High"

        if worst == 2:
            return "Moderate"

        return "Low"


# ---------------------------------------------------------------------------
# TERM GROUPS
# ---------------------------------------------------------------------------

EARLY_REPAYMENT_TERMS = [
    "early repayment",
    "early settlement",
    "prepayment",
    "early payoff",
    "pay off early",
    "settle early",
    "ahead of schedule",
    "before maturity",
    "before the maturity date",
    # Same underlying concept ("ending the arrangement before the agreed
    # term"), but subscription/contract phrasing rather than loan phrasing
    # - without these, a fee for cancelling/terminating a subscription
    # early goes undetected entirely (the old plain "early termination
    # fee" keyword rule this replaced covered it; this generalization
    # needs to as well).
    "early termination",
    "cancel early",
    "cancel before",
    "terminate early",
]

FEE_TERMS = [
    "fee",
    "charge",
    "administration fee",
    "administrative fee",
    "processing fee",
    "penalty",
    "cost",
]

RENEWAL_TERMS = [
    "automatically renew",
    "auto-renew",
    "automatically extend",
    "automatically be extended",
    "automatically re-enrolled",
    "renews automatically",
    "renewal",
    "recurring basis",
    "continues until cancelled",
    "continues until canceled",
    "continues indefinitely",
]

NOTICE_TERMS = [
    "notice",
    "notify",
    "notification",
    "reminder",
]

THIRD_PARTY_CATEGORIES = [
    "affiliate",
    "service provider",
    "business partner",
    "credit reference agenc",
    "debt collection agenc",
    "advertiser",
    "marketing partner",
    "data broker",
    "subsidiar",
    "contractor",
]


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------

def _has_any(text: str, phrases: list[str]) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def _extract_percentage(text: str) -> float | None:
    """
    Extract the first percentage mentioned in a clause.

    Examples:
        4%
        4 percent
        4 per cent
    """

    match = re.search(
        r"\b(\d+(?:\.\d+)?)\s*(?:%|percent|per cent)\b",
        text.lower()
    )

    if match:
        return float(match.group(1))

    return None


# Contracts spell out small numbers ("seven days", "three weeks") as often
# as they use digits - a duration extractor that only handles digits misses
# a large fraction of real contract phrasing.
_WORD_NUMBERS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_WORD_NUMBER_ALTERNATION = "|".join(_WORD_NUMBERS)
_WORD_DAY_RE = re.compile(rf"\b({_WORD_NUMBER_ALTERNATION})\b\s*-?\s*days?\b")
_WORD_WEEK_RE = re.compile(rf"\b({_WORD_NUMBER_ALTERNATION})\b\s*-?\s*weeks?\b")


def _extract_days(text: str) -> int | None:
    """
    Extract a day or week duration and convert it to days.

    Checks digit days ("30 days"), then spelled-out days ("thirty days"),
    then weeks in either form ("3 weeks" / "three weeks", converted to
    days) - contracts phrase notice/cancellation windows in all of these.

    This intentionally returns only a general duration. More specialised
    extraction can later distinguish notice periods from renewal periods.
    """

    lowered = text.lower()

    match = re.search(r"\b(\d+)\s*-?\s*days?\b", lowered)
    if match:
        return int(match.group(1))

    word_match = _WORD_DAY_RE.search(lowered)
    if word_match:
        return _WORD_NUMBERS[word_match.group(1)]

    match = re.search(r"\b(\d+)\s*-?\s*weeks?\b", lowered)
    if match:
        return int(match.group(1)) * 7

    word_week_match = _WORD_WEEK_RE.search(lowered)
    if word_week_match:
        return _WORD_NUMBERS[word_week_match.group(1)] * 7

    return None


def _extract_hours(text: str) -> int | None:
    """Extract a '<number> hour(s)' duration, if any - a same-day window
    ("48 hours") reads far more aggressively than the equivalent day count
    ("2 days"), so it's tracked separately rather than folded into days."""
    match = re.search(r"\b(\d+)\s*-?\s*hours?\b", text.lower())
    return int(match.group(1)) if match else None


def _is_negated(text: str, keyword: str) -> bool:
    """
    Detect simple negations near a keyword.

    Examples:
        no penalty
        not result in a penalty
        without an additional fee
    """

    pattern = re.compile(
        rf"\b(no|not|without)\b(?:\s+\S+){{0,5}}\s+{re.escape(keyword)}\b"
    )

    return bool(pattern.search(text.lower()))


def _has_unnegated(lowered: str, terms: list[str]) -> bool:
    """True if any term in `terms` appears in the text AND isn't negated
    near that specific occurrence.

    This matters when a concept can be triggered by several different
    words (e.g. fee_present matches "fee", "charge", or "penalty") - a
    single `not _is_negated(text, "fee")` guard only catches negation of
    the word "fee" itself, and misses "...shall not result in an
    additional penalty" entirely, since "fee" never appears in that
    sentence for the guard to check.
    """
    return any(term in lowered and not _is_negated(lowered, term) for term in terms)


# ---------------------------------------------------------------------------
# FACT EXTRACTION
# ---------------------------------------------------------------------------

def extract_facts(text: str) -> ClauseFacts:
    """
    Extract structured concepts from a clause.

    These facts are deliberately independent of the ML category prediction.
    """

    lowered = text.lower()

    facts = ClauseFacts()

    # Early repayment
    facts.early_repayment = _has_any(
        lowered,
        EARLY_REPAYMENT_TERMS
    )

    if facts.early_repayment:

        if _has_any(lowered, [
            "not permitted",
            "prohibited",
            "cannot",
            "may not",
            "not allowed",
            "restricted",
        ]):
            facts.early_repayment_restricted = True

    # Fees. Negation-aware per matched term (see _has_unnegated) - not just
    # a plain _has_any - because "shall not result in an additional
    # penalty" would otherwise register as a fee being present.
    facts.fee_present = _has_unnegated(
        lowered,
        FEE_TERMS
    )

    facts.penalty_present = (
        "penalty" in lowered
        and not _is_negated(lowered, "penalty")
    )

    facts.fee_percentage = _extract_percentage(text)

    # Automatic renewal
    facts.automatic_renewal = _has_any(
        lowered,
        RENEWAL_TERMS
    )

    if facts.automatic_renewal:

        if _has_any(lowered, [
            "without notice",
            "without further notice",
            "no reminder notice",
            "no notice",
        ]):
            facts.renewal_without_notice = True

    # Notice period
    if _has_any(lowered, [
        "notice",
        "notify",
        "prior written notice",
    ]):
        facts.notice_period_days = _extract_days(text)

    # Cancellation window
    if _has_any(lowered, [
        "cancel",
        "cancellation",
        "terminate",
        "termination",
    ]):
        facts.cancellation_window_days = _extract_days(text)
        facts.cancellation_window_hours = _extract_hours(text)

    # Data sharing
    facts.third_party_count = sum(
        1
        for party in THIRD_PARTY_CATEGORIES
        if party in lowered
    )

    broad_purpose = _has_any(lowered, [
        "commercial purposes",
        "business purposes",
        "any purpose",
        "for any purpose",
        "marketing purposes",
    ])

    facts.broad_data_sharing = (
        facts.third_party_count >= 3
        or broad_purpose
    )

    # Arbitration
    facts.binding_arbitration = _has_any(lowered, [
        "binding arbitration",
        "resolved through arbitration",
        "mandatory arbitration",
    ])

    facts.class_action_waiver = _has_any(lowered, [
        "waive your right to participate in any class action",
        "class action waiver",
        "class or collective action",
        "representative proceeding",
    ])

    # Unilateral changes
    facts.unilateral_change = (
        _has_any(lowered, [
            "may change",
            "may modify",
            "may update",
            "may amend",
        ])
        and _has_any(lowered, [
            "without notice",
            "at any time",
            "sole discretion",
        ])
    )

    return facts


# ---------------------------------------------------------------------------
# CROSS-CATEGORY RISK RULES
# ---------------------------------------------------------------------------

def _check_cross_category_risks(
    text: str,
    facts: ClauseFacts
) -> list[Flag]:

    flags = []

    # Early repayment fee
    #
    # This catches the concept regardless of whether the classifier called
    # the clause "fees", "cancellation", or something else. fee_present is
    # already negation-aware (see _has_unnegated in extract_facts), so no
    # separate negation check is needed here - a prior version checked
    # `not _is_negated(text, "fee")` specifically, which missed clauses
    # where the negated word was "penalty"/"charge"/etc. instead of "fee".

    if (
        facts.early_repayment
        and facts.fee_present
    ):
        flags.append(
            Flag(
                "Charges a fee or financial penalty for ending the agreement or repaying the balance early.",
                2,
            )
        )

    # Early repayment restriction

    if facts.early_repayment_restricted:
        flags.append(
            Flag(
                "Restricts or prevents early repayment, limiting the borrower's ability to settle the balance ahead of schedule.",
                2,
            )
        )

    # Auto-renewal without notice

    if (
        facts.automatic_renewal
        and facts.renewal_without_notice
    ):
        flags.append(
            Flag(
                "Automatically renews or extends without a clear reminder or advance notice.",
                3,
            )
        )

    # Short notice before automatic renewal

    if (
        facts.automatic_renewal
        and facts.notice_period_days is not None
        and facts.notice_period_days < 14
    ):
        flags.append(
            Flag(
                f"Only {facts.notice_period_days} days of notice are provided before automatic renewal or extension.",
                3,
            )
        )

    # Broad data sharing

    if facts.broad_data_sharing:

        if facts.third_party_count >= 3:
            flags.append(
                Flag(
                    f"Allows sharing information with a broad range of third parties ({facts.third_party_count} categories detected).",
                    2,
                )
            )

    # Class action waiver

    if facts.class_action_waiver:
        flags.append(
            Flag(
                "Limits the ability to participate in class or representative legal action.",
                3,
            )
        )

    # One-sided changes

    if facts.unilateral_change:
        flags.append(
            Flag(
                "Allows one party to change terms unilaterally without clear advance notice.",
                2,
            )
        )

    return flags


# ---------------------------------------------------------------------------
# CATEGORY-SPECIFIC RULES
# ---------------------------------------------------------------------------

def _check_cancellation(
    text: str,
    facts: ClauseFacts
) -> list[Flag]:

    flags = []
    lowered = text.lower()

    if _has_any(lowered, [
        "no cancellations",
        "no refunds",
        "no exceptions",
        "cannot be cancelled",
        "cannot be canceled",
        "unable to cancel",
        "will be denied",
        "will not be accepted",
    ]):
        flags.append(
            Flag(
                "Blocks cancellation entirely or provides almost no ability to cancel.",
                3,
            )
        )

    if _has_any(lowered, [
        "forfeit all fees",
        "100% of the remaining",
    ]):
        flags.append(
            Flag(
                "Cancelling triggers a severe financial penalty.",
                3,
            )
        )

    if (
        facts.cancellation_window_hours is not None
        and facts.cancellation_window_hours <= 72
    ):
        # An hours-based window reads as more aggressive than the
        # equivalent day count ("48 hours" hits differently than "2 days"),
        # so it's rated more seriously than the days check below.
        flags.append(
            Flag(
                f"Cancellation window is only {facts.cancellation_window_hours} hour(s) - typical policies allow at least 14-30 days.",
                3,
            )
        )
    elif (
        facts.cancellation_window_days is not None
        and facts.cancellation_window_days < 14
    ):
        flags.append(
            Flag(
                f"Cancellation window is only {facts.cancellation_window_days} days.",
                2,
            )
        )

    return flags


def _check_auto_renewal(
    text: str,
    facts: ClauseFacts
) -> list[Flag]:

    flags = []

    if (
        facts.automatic_renewal
        and facts.notice_period_days is None
        and not facts.renewal_without_notice
    ):
        flags.append(
            Flag(
                "Automatically renews or extends without clearly stating an advance notice period.",
                2,
            )
        )

    return flags


def _check_fees(
    text: str,
    facts: ClauseFacts
) -> list[Flag]:

    flags = []
    lowered = text.lower()

    if _has_any(lowered, [
        "without prior notice",
        "without notice",
        "at our sole discretion",
        "not disclosed",
    ]):
        flags.append(
            Flag(
                "Fees may be changed or applied without clear disclosure or advance notice.",
                3,
            )
        )

    if (
        _has_any(lowered, [
            "hidden",
            "not itemized",
            "without itemization",
        ])
        and not _has_any(lowered, [
            "no hidden fees",
            "not hidden",
        ])
    ):
        flags.append(
            Flag(
                "Language suggests undisclosed or non-itemized charges.",
                3,
            )
        )

    return flags


def _check_data_sharing(
    text: str,
    facts: ClauseFacts
) -> list[Flag]:

    flags = []
    lowered = text.lower()

    if (
        _has_any(lowered, [
            "sell your personal data",
            "sold to third parties",
            "sold to data brokers",
        ])
        and not _has_any(lowered, [
            "do not sell",
            "does not sell",
            "will not sell",
        ])
    ):
        flags.append(
            Flag(
                "Explicitly allows personal data to be sold or transferred to third parties.",
                3,
            )
        )

    if _has_any(lowered, [
        "without your consent",
        "without further consent",
    ]):
        flags.append(
            Flag(
                "Allows data use or sharing without obtaining further consent.",
                3,
            )
        )

    if "opt out" in lowered:
        flags.append(
            Flag(
                "Data sharing is opt-out rather than opt-in.",
                1,
            )
        )

    return flags


def _check_liability(
    text: str,
    facts: ClauseFacts
) -> list[Flag]:

    flags = []
    lowered = text.lower()

    if _has_any(lowered, [
        "not be liable for any damages whatsoever",
        "waive all rights",
        "without exception",
        "howsoever arising",
        "shall bear no responsibility",
    ]):
        flags.append(
            Flag(
                "Liability disclaimer is extremely broad.",
                3,
            )
        )

    if (
        "indemnify and hold us harmless" in lowered
        and _has_any(lowered, [
            "without limitation",
            "any and all claims",
        ])
    ):
        flags.append(
            Flag(
                "Requires broad indemnification with limited boundaries.",
                2,
            )
        )

    return flags


def _check_dispute_resolution(
    text: str,
    facts: ClauseFacts
) -> list[Flag]:

    flags = []

    if (
        facts.binding_arbitration
        and "small claims" not in text.lower()
    ):
        flags.append(
            Flag(
                "Requires binding arbitration without a clear small-claims alternative.",
                1,
            )
        )

    if (
        "arbitration costs" in text.lower()
        and _has_any(text.lower(), [
            "borne entirely by the user",
            "company's legal fees",
        ])
    ):
        flags.append(
            Flag(
                "Places arbitration costs heavily on the user regardless of outcome.",
                3,
            )
        )

    return flags


def _check_other(
    text: str,
    facts: ClauseFacts
) -> list[Flag]:

    return []


# ---------------------------------------------------------------------------
# CATEGORY CHECKER MAP
# ---------------------------------------------------------------------------

_CATEGORY_CHECKERS = {
    "cancellation": _check_cancellation,
    "auto-renewal": _check_auto_renewal,
    "fees": _check_fees,
    "data sharing": _check_data_sharing,
    "liability": _check_liability,
    "dispute resolution": _check_dispute_resolution,
    "other": _check_other,
}


# ---------------------------------------------------------------------------
# MAIN EVALUATION
# ---------------------------------------------------------------------------

def _remove_duplicate_flags(flags: list[Flag]) -> list[Flag]:
    """Remove duplicate flags with identical reasons."""

    unique = []
    seen = set()

    for flag in flags:
        if flag.reason not in seen:
            unique.append(flag)
            seen.add(flag.reason)

    return unique


def evaluate_clause(
    text: str,
    category: str,
    secondary_categories: list[str] | None = None,
) -> ClauseResult:
    """
    Evaluate a clause for risks.

    The primary category determines the main category-specific rules.
    Plausible secondary categories can also contribute rules.

    Cross-category concept rules always run.
    """

    facts = extract_facts(text)

    flags = []

    # Cross-category rules always run.
    flags.extend(
        _check_cross_category_risks(text, facts)
    )

    # Build category list.
    categories_to_check = [category]

    if secondary_categories:
        for secondary in secondary_categories:
            if secondary not in categories_to_check:
                categories_to_check.append(secondary)

    # Run category-specific rules.
    for current_category in categories_to_check:

        checker = _CATEGORY_CHECKERS.get(
            current_category,
            _check_other
        )

        flags.extend(
            checker(text, facts)
        )

    flags = _remove_duplicate_flags(flags)

    return ClauseResult(
        text=text,
        category=category,
        flags=flags,
        facts=facts,
    )


# ---------------------------------------------------------------------------
# DOCUMENT SCORING
# ---------------------------------------------------------------------------

def score_document(
    clause_results: list[ClauseResult]
) -> dict:

    total_severity = sum(
        clause.risk_score
        for clause in clause_results
    )

    flagged_count = sum(
        1
        for clause in clause_results
        if clause.is_flagged
    )

    worst_flag_severity = max(
        (
            flag.severity
            for clause in clause_results
            for flag in clause.flags
        ),
        default=0,
    )

    score = min(
        100,
        max(
            total_severity * 6,
            worst_flag_severity * 15,
        ),
    )

    if score >= 60:
        level = "High Risk"
    elif score >= 25:
        level = "Moderate Risk"
    elif score > 0:
        level = "Low Risk"
    else:
        level = "Minimal Risk"

    return {
        "score": score,
        "level": level,
        "flagged_clauses": flagged_count,
        "total_clauses": len(clause_results),
        "total_severity": total_severity,
    }