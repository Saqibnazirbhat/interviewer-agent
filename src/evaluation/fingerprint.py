"""Authenticity Fingerprinting — behavioral analysis beyond content scoring.

Analyzes patterns that indicate whether a candidate is answering authentically
vs. using external help (copy-paste, AI generation, etc.):

1. Latency patterns — variance in response times (real humans vary; bots are consistent)
2. Vocabulary complexity — Flesch-like readability shifts between answers
3. Topic-switch speed — how fast they respond after a category change
4. Answer length distribution — natural vs. suspiciously uniform lengths

Produces an AuthenticityScore 0-100 where 100 = fully authentic.
"""

import logging
import math
import re
from dataclasses import dataclass, field

logger = logging.getLogger("interviewer.fingerprint")


@dataclass
class AuthenticityFingerprint:
    """Computed authenticity metrics for a single interview."""
    latency_variance: float = 0.0
    latency_cv: float = 0.0  # coefficient of variation
    vocab_complexity_mean: float = 0.0
    vocab_complexity_variance: float = 0.0
    topic_switch_penalty: float = 0.0
    length_uniformity: float = 0.0  # 0=perfectly uniform (suspicious), 1=varied
    overall_score: float = 100.0
    flags: list = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "latency_variance": round(self.latency_variance, 2),
            "latency_cv": round(self.latency_cv, 2),
            "vocab_complexity_mean": round(self.vocab_complexity_mean, 2),
            "vocab_complexity_variance": round(self.vocab_complexity_variance, 2),
            "topic_switch_penalty": round(self.topic_switch_penalty, 2),
            "length_uniformity": round(self.length_uniformity, 2),
            "overall_score": round(self.overall_score, 1),
            "flags": self.flags,
            "details": self.details,
        }


def compute_vocab_complexity(text: str) -> float:
    """Compute a simple vocabulary complexity score for a text.

    Uses average word length + type-token ratio as a proxy for
    Flesch-Kincaid-style complexity. Returns 0-10 scale.
    """
    words = re.findall(r"[a-zA-Z]+", text.lower())
    if len(words) < 3:
        return 0.0

    avg_word_len = sum(len(w) for w in words) / len(words)
    unique_ratio = len(set(words)) / len(words)  # type-token ratio

    # Combine: avg word length (typical 4-7) + TTR (typical 0.3-0.9)
    # Scale to 0-10
    score = (avg_word_len - 3) * 1.5 + unique_ratio * 5
    return max(0.0, min(10.0, score))


def compute_authenticity(responses: list[dict]) -> AuthenticityFingerprint:
    """Analyze a list of interview responses and compute authenticity metrics.

    Args:
        responses: List of response dicts with keys:
            - answer: str
            - time_seconds: float
            - category: str
            - skipped: bool
            - difficulty: str (optional)

    Returns:
        AuthenticityFingerprint with overall_score 0-100.
    """
    fp = AuthenticityFingerprint()

    # Filter to non-skipped answers with content
    answered = [r for r in responses if not r.get("skipped") and r.get("answer", "").strip()]
    total = len(responses)
    if len(answered) < 3:
        # Not enough data for behavioral analysis — cap the score since we
        # genuinely cannot vouch for authenticity with so few data points,
        # regardless of whether the candidate answered every question asked.
        if total > 0:
            participation = len(answered) / total
            # Cap at 50: even 100% participation on 1-2 answers is not enough
            # to run variance/uniformity checks, so we can't award a high score.
            fp.overall_score = round(min(50.0, participation * 50), 1)
            fp.flags.append(f"Only {len(answered)}/{total} questions answered — insufficient data for authenticity analysis")
        else:
            fp.overall_score = 0.0
            fp.flags.append("No answers recorded — cannot assess authenticity")
        fp.details["reason"] = "Too few answers for fingerprinting"
        return fp

    # 1. Latency analysis
    times = [r["time_seconds"] for r in answered if r.get("time_seconds", 0) > 0]
    if len(times) >= 3:
        mean_t = sum(times) / len(times)
        variance_t = sum((t - mean_t) ** 2 for t in times) / len(times)
        std_t = math.sqrt(variance_t)
        cv = std_t / mean_t if mean_t > 0 else 0

        fp.latency_variance = variance_t
        fp.latency_cv = cv
        fp.details["latency_times"] = [round(t, 1) for t in times]
        fp.details["latency_mean"] = round(mean_t, 1)

        # Low CV = suspiciously consistent timing
        # Real humans typically have CV > 0.3
        if cv < 0.15 and len(times) >= 4:
            fp.flags.append("Suspiciously consistent response times")

    # 2. Vocabulary complexity shifts
    complexities = [compute_vocab_complexity(r["answer"]) for r in answered]
    if len(complexities) >= 3:
        mean_c = sum(complexities) / len(complexities)
        var_c = sum((c - mean_c) ** 2 for c in complexities) / len(complexities)

        fp.vocab_complexity_mean = mean_c
        fp.vocab_complexity_variance = var_c
        fp.details["vocab_scores"] = [round(c, 2) for c in complexities]

        # Very low variance = suspiciously uniform vocabulary (AI-like)
        if var_c < 0.3 and mean_c > 5:
            fp.flags.append("Vocabulary complexity is suspiciously uniform and high")

        # Very high complexity on easy questions might indicate AI
        easy_complexities = [
            compute_vocab_complexity(r["answer"])
            for r in answered
            if r.get("difficulty") == "easy" and len(r.get("answer", "")) > 50
        ]
        if easy_complexities and sum(easy_complexities) / len(easy_complexities) > 7:
            fp.flags.append("Unusually complex vocabulary on easy questions")

    # 3. Topic-switch speed
    category_switch_times = []
    for i in range(1, len(answered)):
        if answered[i].get("category") != answered[i - 1].get("category"):
            t = answered[i].get("time_seconds", 0)
            if t > 0:
                category_switch_times.append(t)

    non_switch_times = []
    for i in range(1, len(answered)):
        if answered[i].get("category") == answered[i - 1].get("category"):
            t = answered[i].get("time_seconds", 0)
            if t > 0:
                non_switch_times.append(t)

    if category_switch_times and non_switch_times:
        avg_switch = sum(category_switch_times) / len(category_switch_times)
        avg_noswitch = sum(non_switch_times) / len(non_switch_times)

        # Real humans usually take slightly longer on category switches
        # If switches are faster than same-category, that's suspicious
        if avg_switch < avg_noswitch * 0.7 and len(category_switch_times) >= 2:
            fp.topic_switch_penalty = 1.0
            fp.flags.append("Responds faster on topic switches than same-topic questions")

        fp.details["avg_switch_time"] = round(avg_switch, 1)
        fp.details["avg_same_topic_time"] = round(avg_noswitch, 1)

    # 4. Answer length distribution
    lengths = [len(r.get("answer", "")) for r in answered]
    if len(lengths) >= 3:
        mean_len = sum(lengths) / len(lengths)
        var_len = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
        cv_len = math.sqrt(var_len) / mean_len if mean_len > 0 else 0

        fp.length_uniformity = min(1.0, cv_len)
        fp.details["answer_lengths"] = lengths
        fp.details["length_cv"] = round(cv_len, 3)

        # Very uniform lengths are suspicious
        if cv_len < 0.15 and mean_len > 100 and len(lengths) >= 4:
            fp.flags.append("Answer lengths are suspiciously uniform")

    # 5. Timeout pattern analysis — consistently hitting exactly 2:00 is suspicious
    timeout_count = sum(1 for r in answered if r.get("timing_status") == "timeout")
    early_count = sum(1 for r in answered if r.get("timing_status") == "early")
    total_answered = len(answered)
    if total_answered >= 4:
        timeout_ratio = timeout_count / total_answered
        fp.details["timeout_count"] = timeout_count
        fp.details["timeout_ratio"] = round(timeout_ratio, 2)
        if timeout_ratio >= 0.7:
            fp.flags.append("Ran out of time on most questions — possible external help or disengagement")
        # Consistently using almost all the time (high times near 100s) is also suspicious
        near_limit = sum(1 for r in answered if r.get("time_seconds", 0) >= 90)
        if near_limit >= total_answered * 0.6 and total_answered >= 4:
            fp.flags.append("Consistently used nearly the full time limit")

    # Compute overall score
    score = 100.0

    # Penalty for each flag
    score -= len(fp.flags) * 15

    # Latency CV penalty (low CV = suspicious)
    if fp.latency_cv > 0:
        if fp.latency_cv < 0.15:
            score -= 15
        elif fp.latency_cv < 0.25:
            score -= 5

    # Length uniformity penalty
    if fp.length_uniformity < 0.15 and len(lengths) >= 4:
        score -= 10

    # Topic switch penalty
    score -= fp.topic_switch_penalty * 10

    fp.overall_score = max(0.0, min(100.0, score))
    return fp
