"""N-gram language model with Katz backoff and Jelinek-Mercer smoothing.

The implementation is intentionally minimal: counts are stored in plain
dicts, vocabularies are explicit, and OOV is mapped to a single ``<unk>``
token. ARPA-style export/import uses the well-known ``\\data\\`` /
``\\N-grams:`` blocks (see CMU Sphinx / KenLM docs for the format).

Additional utilities: pruning, vocabulary coverage, linear interpolation
of two models via :class:`MixtureLM`, and EM-based lambda estimation.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from hypofuse.exceptions import LanguageModelError

UNK = "<unk>"
BOS = "<s>"
EOS = "</s>"


@dataclass(frozen=True)
class VocabCoverage:
    """Vocabulary coverage statistics for a token sequence."""

    total: int
    oov: int
    oov_rate: float


@dataclass
class NgramLM:
    order: int
    vocab: frozenset[str]
    counts: dict[int, Counter] = field(default_factory=dict)
    total_unigrams: int = 0
    log_probs: dict[int, dict[tuple[str, ...], float]] = field(default_factory=dict)
    backoff: dict[int, dict[tuple[str, ...], float]] = field(default_factory=dict)
    source: str = "train"

    @classmethod
    def train(cls, sentences: list[list[str]], order: int = 3) -> NgramLM:
        if order < 1:
            raise LanguageModelError("order must be >= 1")
        vocab = {UNK, BOS, EOS}
        for sent in sentences:
            for tok in sent:
                vocab.add(tok)
        padded = [[BOS] * (order - 1) + sent + [EOS] for sent in sentences]
        counts: dict[int, Counter] = {n: Counter() for n in range(1, order + 1)}
        for sent in padded:
            for n in range(1, order + 1):
                for i in range(len(sent) - n + 1):
                    ng = tuple(sent[i : i + n])
                    counts[n][ng] += 1
        total = sum(counts[1].values())
        return cls(order=order, vocab=frozenset(vocab), counts=counts, total_unigrams=total)

    def _map(self, token: str) -> str:
        return token if token in self.vocab else UNK

    def map_sequence(self, tokens: list[str]) -> list[str]:
        return [self._map(t) for t in tokens]

    # ----- probability methods -----

    def prob_katz(self, ngram: tuple[str, ...]) -> float:
        n = len(ngram)
        if n < 1 or n > self.order:
            raise LanguageModelError(f"ngram order out of range: {n}")
        ngram = tuple(self._map(t) for t in ngram)
        counts = self.counts[n]
        if n == 1:
            c = counts.get(ngram, 0)
            return c / self.total_unigrams if c else 1.0 / self.total_unigrams
        c = counts.get(ngram, 0)
        if c > 0:
            return c / self.counts[n - 1].get(ngram[:-1], c)
        prefix = ngram[:-1]
        total = self.counts[n - 1].get(prefix, 0)
        if total == 0:
            return 1.0 / (self.total_unigrams + 1)
        observed = sum(c2 for ng, c2 in counts.items() if ng[:-1] == prefix and c2 > 0)
        beta = total - observed
        if beta <= 0:
            return 1.0 / (self.total_unigrams + 1)
        prefix_prob = self.prob_katz(prefix) * total / max(1, total)
        return beta / total * prefix_prob

    def prob_jelinek(
        self, ngram: tuple[str, ...], lambdas: tuple[float, ...] | None = None
    ) -> float:
        n = len(ngram)
        if n < 1 or n > self.order:
            raise LanguageModelError(f"ngram order out of range: {n}")
        if lambdas is None:
            lambdas = tuple(1.0 / self.order for _ in range(self.order))
        if abs(sum(lambdas) - 1.0) > 1e-6:
            raise LanguageModelError("lambdas must sum to 1")
        ngram = tuple(self._map(t) for t in ngram)
        prob = 0.0
        for k, lam in enumerate(lambdas[:n], start=1):
            if k == 1:
                cond: tuple[str, ...] = ()
                cond_ngram = ngram[-1:]
            else:
                cond = ngram[-k:-1]
                cond_ngram = ngram[-k:]
            counts = self.counts[k]
            num = counts.get(cond_ngram, 0)
            denom = self.counts[k - 1].get(cond, 0) if k > 1 else self.total_unigrams
            if denom == 0:
                denom = 1
            prob += lam * (num / denom)
        return max(prob, 1e-12)

    def perplexity(self, tokens: list[str], method: str = "katz") -> float:
        if not tokens:
            return 1.0
        tokens = self.map_sequence(tokens)
        padded = [BOS] * (self.order - 1) + tokens + [EOS]
        log_sum = 0.0
        n = 0
        for i in range(self.order - 1, len(padded)):
            ng = tuple(padded[i - self.order + 1 : i + 1])
            p = self.prob_katz(ng) if method == "katz" else self.prob_jelinek(ng)
            log_sum -= math.log(max(p, 1e-12))
            n += 1
        return math.exp(log_sum / max(1, n))

    def prob(self, ngram: tuple[str, ...], method: str = "auto") -> float:
        """Dispatch probability computation to the named method.

        ``"auto"`` (the default) reads the ARPA tables when the model was
        loaded from ARPA -- such a model carries no counts -- and falls back
        to Katz backoff otherwise, so rescoring behaves the same for trained
        and loaded models.
        """
        if method == "auto":
            method = "arpa" if self.source == "arpa" else "katz"
        if method == "katz":
            return self.prob_katz(ngram)
        if method == "jelinek":
            return self.prob_jelinek(ngram)
        if method == "arpa":
            return self._prob_arpa(ngram)
        raise LanguageModelError(f"unknown probability method: {method}")

    def _prob_arpa(self, ngram: tuple[str, ...]) -> float:
        """Compute probability using ARPA tables with backoff."""
        n = len(ngram)
        if n < 1 or n > self.order:
            raise LanguageModelError(f"ngram order out of range: {n}")
        ngram = tuple(self._map(t) for t in ngram)
        if not self.log_probs:
            lp, bo = self._compute_arpa_tables()
            return self._arpa_prob_lookup_tables(ngram, lp, bo)
        return self._arpa_prob_lookup_tables(ngram, self.log_probs, self.backoff)

    def _arpa_prob_lookup_tables(
        self,
        ngram: tuple[str, ...],
        lp: dict[int, dict[tuple[str, ...], float]],
        bo: dict[int, dict[tuple[str, ...], float]],
    ) -> float:
        n = len(ngram)
        if n in lp and ngram in lp[n]:
            return 10 ** lp[n][ngram]
        if n == 1:
            # A loaded model carries no counts, so a unigram the ARPA text does
            # not list falls back to its exported <unk> probability and, failing
            # that, to a uniform distribution over the vocabulary.
            unk = lp.get(1, {}).get((UNK,))
            if unk is not None:
                return 10**unk
            return 1.0 / max(len(self.vocab), 1)
        ctx = ngram[:-1]
        shorter = ngram[1:]
        ctx_n = len(ctx)
        alpha_log10 = bo.get(ctx_n, {}).get(ctx, 0.0)
        return (10**alpha_log10) * self._arpa_prob_lookup_tables(shorter, lp, bo)

    # ----- ARPA table computation -----

    def _compute_arpa_tables(
        self,
    ) -> tuple[
        dict[int, dict[tuple[str, ...], float]],
        dict[int, dict[tuple[str, ...], float]],
    ]:
        """Compute ARPA log-probs and backoff weights from counts."""
        lp: dict[int, dict[tuple[str, ...], float]] = {}
        bo: dict[int, dict[tuple[str, ...], float]] = {}
        for n in range(1, self.order + 1):
            lp[n] = {}
            for ng in self.counts[n]:
                p = max(self.prob_katz(ng), 1e-12)
                lp[n][ng] = math.log10(p)
        for n in range(1, self.order):
            bo[n] = {}
            for ctx in sorted(self.counts[n].keys()):
                obs: set[str] = set()
                if n + 1 in self.counts:
                    for ng in self.counts[n + 1]:
                        if ng[:-1] == ctx:
                            obs.add(ng[-1])
                if not obs:
                    bo[n][ctx] = 0.0
                    continue
                num = 1.0 - sum(10 ** lp[n + 1].get((*ctx, w), -12.0) for w in obs)
                shorter_ctx = ctx[1:]
                denom_sum = 0.0
                for w in obs:
                    denom_sum += self._arpa_lookup((*shorter_ctx, w), lp, bo)
                denom = 1.0 - denom_sum
                if abs(denom) < 1e-12 or num <= 0.0:
                    # The observed continuations already consume the mass, so
                    # nothing is left to discount: back off with weight one
                    # rather than a degenerate 1e-12, which would score every
                    # unseen continuation as impossible.
                    bo[n][ctx] = 0.0
                else:
                    bo[n][ctx] = math.log10(min(num / denom, 1.0))
        return lp, bo

    def _arpa_lookup(
        self,
        ngram: tuple[str, ...],
        lp: dict[int, dict[tuple[str, ...], float]],
        bo: dict[int, dict[tuple[str, ...], float]],
    ) -> float:
        n = len(ngram)
        if n in lp and ngram in lp[n]:
            return 10 ** lp[n][ngram]
        if n == 1:
            # A loaded model carries no counts, so a unigram the ARPA text does
            # not list falls back to its exported <unk> probability and, failing
            # that, to a uniform distribution over the vocabulary.
            unk = lp.get(1, {}).get((UNK,))
            if unk is not None:
                return 10**unk
            return 1.0 / max(len(self.vocab), 1)
        ctx = ngram[:-1]
        shorter = ngram[1:]
        ctx_n = len(ctx)
        alpha_log10 = bo.get(ctx_n, {}).get(ctx, 0.0)
        return (10**alpha_log10) * self._arpa_lookup(shorter, lp, bo)

    # ----- ARPA export / import -----

    def to_arpa(self) -> str:
        """Export the model in ARPA format with log10 probabilities.

        Uses ``%.4f`` formatting and tab separators. Backoff weights are
        included for orders below ``self.order``.
        """
        if self.log_probs:
            lp = self.log_probs
            bo = self.backoff
        else:
            lp, bo = self._compute_arpa_tables()
        lines = ["\\data\\"]
        for n in range(1, self.order + 1):
            count = len(lp.get(n, {}))
            lines.append(f"ngram {n}={count}")
        for n in range(1, self.order + 1):
            lines.append(f"\\{n}-grams:")
            for ng in sorted(lp.get(n, {}).keys()):
                log_p = lp[n][ng]
                tokens = " ".join(ng)
                if n < self.order and n in bo and ng in bo[n]:
                    bow = bo[n][ng]
                    lines.append(f"{log_p:.4f}\t{tokens}\t{bow:.4f}")
                else:
                    lines.append(f"{log_p:.4f}\t{tokens}")
        lines.append("\\end\\")
        return "\n".join(lines)

    @classmethod
    def from_arpa(cls, payload: str) -> NgramLM:
        """Parse an ARPA format string into an NgramLM.

        Populates ``log_probs`` and ``backoff`` tables and sets
        ``source="arpa"`` so that ``prob(..., method="arpa")`` uses them.
        """
        order = 0
        log_probs: dict[int, dict[tuple[str, ...], float]] = {}
        backoff_w: dict[int, dict[tuple[str, ...], float]] = {}
        section = None
        for raw_line in payload.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("\\data\\"):
                continue
            if line.startswith("\\") and line.endswith("-grams:"):
                section = int(line.split("-")[0].lstrip("\\"))
                log_probs[section] = {}
                order = max(order, section)
                continue
            if line.startswith("\\end\\"):
                continue
            if line.startswith("ngram "):
                continue
            if section is None:
                continue
            parts = line.split("\t")
            log_p = float(parts[0])
            tokens = tuple(parts[1].split())
            log_probs[section][tokens] = log_p
            if len(parts) > 2:
                bow_str = parts[2].strip()
                if bow_str:
                    if section not in backoff_w:
                        backoff_w[section] = {}
                    backoff_w[section][tokens] = float(bow_str)
        vocab = {UNK, BOS, EOS}
        for n in range(1, order + 1):
            for ng in log_probs.get(n, {}):
                vocab.update(ng)
        return cls(
            order=order,
            vocab=frozenset(vocab),
            log_probs=log_probs,
            backoff=backoff_w,
            source="arpa",
        )

    # ----- utilities -----

    def prune(self, min_count: int) -> NgramLM:
        """Return a new LM with rare n-grams (orders >= 2) removed."""
        new_counts: dict[int, Counter] = {}
        for n in range(1, self.order + 1):
            if n == 1:
                new_counts[n] = Counter(self.counts[n])
            else:
                new_counts[n] = Counter(
                    {ng: c for ng, c in self.counts[n].items() if c >= min_count}
                )
        total = sum(new_counts[1].values())
        return NgramLM(
            order=self.order,
            vocab=self.vocab,
            counts=new_counts,
            total_unigrams=total,
        )

    def coverage(self, tokens: list[str]) -> VocabCoverage:
        """Compute vocabulary coverage for a list of tokens."""
        total = len(tokens)
        oov = sum(1 for t in tokens if t not in self.vocab)
        rate = oov / total if total > 0 else 0.0
        return VocabCoverage(total=total, oov=oov, oov_rate=rate)

    def interpolate(self, other: NgramLM, weight: float) -> MixtureLM:
        """Create a linear interpolation with another model.

        Returns a :class:`MixtureLM` whose probability is
        ``weight * self + (1 - weight) * other``.
        """
        if not 0.0 <= weight <= 1.0:
            raise LanguageModelError("interpolation weight must be in [0.0, 1.0]")
        return MixtureLM(model_a=self, model_b=other, weight=weight)

    def fit_jelinek_lambdas(
        self, held_out: list[list[str]], iterations: int = 5
    ) -> tuple[float, ...]:
        """Estimate Jelinek-Mercer lambdas via EM on held-out data.

        Returns a tuple of lambdas that sum to 1.
        """
        if not held_out:
            return tuple(1.0 / self.order for _ in range(self.order))
        lambdas = [1.0 / self.order] * self.order
        for _ in range(iterations):
            expected = [0.0] * self.order
            for sent in held_out:
                padded = [BOS] * (self.order - 1) + sent + [EOS]
                for i in range(self.order - 1, len(padded)):
                    probs: list[float] = []
                    for k in range(1, self.order + 1):
                        start = i - k + 1
                        if start < 0:
                            probs.append(0.0)
                            continue
                        sub = tuple(self._map(t) for t in padded[start : i + 1])
                        if k == 1:
                            p = self.counts[1].get(sub, 0) / max(self.total_unigrams, 1)
                        else:
                            ctx = sub[:-1]
                            c_ng = self.counts[k].get(sub, 0)
                            c_ctx = self.counts[k - 1].get(ctx, 0)
                            p = c_ng / c_ctx if c_ctx > 0 else 0.0
                        probs.append(p)
                    total_prob = sum(lam * p for lam, p in zip(lambdas, probs, strict=True))
                    if total_prob < 1e-12:
                        continue
                    for k in range(self.order):
                        expected[k] += lambdas[k] * probs[k] / total_prob
            total_expected = sum(expected)
            if total_expected < 1e-12:
                break
            lambdas = [e / total_expected for e in expected]
        return tuple(lambdas)


@dataclass(frozen=True)
class MixtureLM:
    """Linear interpolation of two n-gram language models."""

    model_a: NgramLM
    model_b: NgramLM
    weight: float

    def log_prob(self, ngram: tuple[str, ...]) -> float:
        """Compute log10 of the interpolated probability."""
        p_a = self.model_a.prob_jelinek(ngram)
        p_b = self.model_b.prob_jelinek(ngram)
        p = self.weight * p_a + (1.0 - self.weight) * p_b
        return math.log10(max(p, 1e-12))
