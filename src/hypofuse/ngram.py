"""N-gram language model with Katz backoff and Jelinek-Mercer smoothing.

The implementation is intentionally minimal: counts are stored in plain
dicts, vocabularies are explicit, and OOV is mapped to a single ``<unk>``
token. ARPA-style export/import uses the well-known ``\\data\\`` /
``\1-grams:`` blocks (see CMU Sphinx / KenLM docs for the format).
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field

from hypofuse.exceptions import LanguageModelError

UNK = "<unk>"
BOS = "<s>"
EOS = "</s>"


@dataclass
class NgramLM:
    order: int
    vocab: frozenset[str]
    counts: dict[int, Counter] = field(default_factory=dict)
    total_unigrams: int = 0

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

    def prob_katz(self, ngram: tuple[str, ...]) -> float:
        n = len(ngram)
        if n < 1 or n > self.order:
            raise LanguageModelError(f"ngram order out of range: {n}")
        ngram = tuple(self._map(t) for t in ngram)
        counts = self.counts[n]
        if n == 1:
            c = counts.get(ngram, 0)
            return c / self.total_unigrams if c else 1.0 / self.total_unigrams
        # Katz backoff: use observed count if c >= k, else back off.
        c = counts.get(ngram, 0)
        if c > 0:
            return c / self.counts[n - 1].get(ngram[:-1], c)
        # Compute alpha for backoff.
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
            lambdas = tuple(0.5 / self.order for _ in range(self.order))
        if abs(sum(lambdas) - 1.0) > 1e-6:
            raise LanguageModelError("lambdas must sum to 1")
        ngram = tuple(self._map(t) for t in ngram)
        prob = 0.0
        for k, lam in enumerate(lambdas[:n], start=1):
            if k == 1:
                cond = ()
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

    def to_arpa(self) -> str:
        lines = []
        lines.append("\\data\\")
        for n in range(1, self.order + 1):
            total = sum(c for ng, c in self.counts[n].items() if len(ng) == n)
            lines.append(f"ngram {n}={total}")
        for n in range(1, self.order + 1):
            lines.append(f"\\{n}-grams:")
            for ng, c in sorted(self.counts[n].items()):
                log_p = math.log10(c / self.total_unigrams) if c > 0 else -99.0
                tokens = " ".join(ng)
                if n > 1:
                    bow = math.log10(max(c / max(1, self.counts[n - 1].get(ng[:-1], c)), 1e-12))
                    lines.append(f"{log_p:.4f}\t{tokens}\t{bow:.4f}")
                else:
                    lines.append(f"{log_p:.4f}\t{tokens}")
        lines.append("\\end\\")
        return "\n".join(lines)

    @classmethod
    def from_arpa(cls, payload: str) -> NgramLM:
        order = 0
        counts: dict[int, Counter] = {}
        section = None
        for raw_line in payload.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("\\data\\"):
                continue
            if line.startswith("\\") and line.endswith("-grams:"):
                section = int(line.split("-")[0].lstrip("\\"))
                counts[section] = Counter()
                order = max(order, section)
                continue
            if line.startswith("\\end\\"):
                continue
            if line.startswith("ngram "):
                continue
            if section is None:
                continue
            parts = line.split()
            n = section
            tokens = tuple(parts[1 : 1 + n])
            counts[n][tokens] = float(parts[0])
        vocab = {UNK, BOS, EOS}
        for n in range(1, order + 1):
            for ng in counts[n]:
                vocab.update(ng)
        total = float(sum(c for ng, c in counts[1].items()))
        return cls(order=order, vocab=frozenset(vocab), counts=counts, total_unigrams=total or 1.0)
