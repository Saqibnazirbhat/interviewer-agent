"""Adaptive difficulty engine — picks the next question based on real-time candidate performance."""


# Difficulty levels in order
DIFFICULTY_ORDER = {"easy": 0, "medium": 1, "hard": 2}
DIFFICULTY_NAMES = ["easy", "medium", "hard"]


class AdaptiveEngine:
    """Tracks running performance and selects optimal next question from the pool.

    Strategy:
    - Maintains a rolling performance score (0-10) from recent answers.
    - Maps performance to a target difficulty band.
    - Picks the next unanswered question closest to that target difficulty,
      while also ensuring category variety (avoids asking the same category twice in a row).
    - First question is always easy/medium to ease the candidate in.
    """

    def __init__(self, questions: list[dict]):
        self.pool = list(questions)  # all generated questions
        self.asked_ids: set[int] = set()
        self.scores: list[float] = []  # per-answer scores (0-10 scale)
        self.last_category: str = ""
        self.question_order: list[dict] = []  # track the order questions were served

    @property
    def performance(self) -> float:
        """Rolling performance — weighted toward recent answers (last 3 weigh more)."""
        if not self.scores:
            return 5.0  # neutral start
        if len(self.scores) <= 3:
            return sum(self.scores) / len(self.scores)
        # Weight recent answers 2x
        recent = self.scores[-3:]
        older = self.scores[:-3]
        weighted = sum(older) + sum(recent) * 2
        count = len(older) + len(recent) * 2
        return weighted / count

    @property
    def target_difficulty(self) -> str:
        """Map performance to a target difficulty level."""
        perf = self.performance
        if perf >= 7.5:
            return "hard"
        elif perf >= 4.5:
            return "medium"
        else:
            return "easy"

    @property
    def remaining(self) -> list[dict]:
        """Questions not yet asked."""
        return [q for q in self.pool if q.get("id") not in self.asked_ids]

    @property
    def is_done(self) -> bool:
        return len(self.remaining) == 0

    def pick_first(self) -> dict:
        """Pick the first question — always easy or medium, to ease the candidate in."""
        candidates = [q for q in self.remaining if q.get("difficulty") in ("easy", "medium")]
        if not candidates:
            candidates = self.remaining
        chosen = candidates[0]
        return self._serve(chosen)

    def record_score(self, score_total: int, max_total: int = 40):
        """Record the score for the most recent answer (normalized to 0-10)."""
        normalized = round(score_total / max(max_total, 1) * 10, 1)
        self.scores.append(min(10.0, max(0.0, normalized)))

    def record_skip(self):
        """Record a skipped answer — treated as a low score for adaptation."""
        self.scores.append(2.0)

    def pick_next(self) -> dict | None:
        """Pick the next question based on current performance.

        Selection logic:
        1. Determine target difficulty from running performance.
        2. Score each remaining question on how well it fits:
           - Difficulty match (primary factor)
           - Category variety bonus (avoid same category twice in a row)
        3. Pick the best-scoring candidate.
        """
        remaining = self.remaining
        if not remaining:
            return None

        target = self.target_difficulty
        target_idx = DIFFICULTY_ORDER.get(target, 1)

        best_q = None
        best_score = -999

        for q in remaining:
            q_diff = DIFFICULTY_ORDER.get(q.get("difficulty", "medium"), 1)

            # Primary: how close is this question's difficulty to the target?
            # 0 = exact match, -1 or -2 = further away
            diff_delta = abs(q_diff - target_idx)
            diff_score = -diff_delta * 10  # heavy penalty for mismatch

            # Bonus: prefer stepping one level at a time (not jumping easy -> hard)
            if len(self.scores) > 0 and diff_delta <= 1:
                diff_score += 3

            # Category variety: bonus if different from last asked
            cat_score = 5 if q.get("category") != self.last_category else 0

            # Slight bonus for questions earlier in the original list (preserve some ordering)
            order_bonus = -self.pool.index(q) * 0.1

            total = diff_score + cat_score + order_bonus
            if total > best_score:
                best_score = total
                best_q = q

        if best_q is None:
            best_q = remaining[0]

        return self._serve(best_q)

    def _serve(self, question: dict) -> dict:
        """Mark a question as asked and return it with its adaptive index."""
        self.asked_ids.add(question["id"])
        self.last_category = question.get("category", "")
        served = dict(question)
        served["adaptive_index"] = len(self.asked_ids) - 1  # 0-based position in interview
        self.question_order.append(served)
        return served

    def get_status(self) -> dict:
        """Return current adaptive state for debugging / UI display."""
        return {
            "questions_asked": len(self.asked_ids),
            "questions_remaining": len(self.remaining),
            "rolling_performance": round(self.performance, 1),
            "target_difficulty": self.target_difficulty,
            "scores": list(self.scores),
        }
