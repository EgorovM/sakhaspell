"""Биграммная модель для переранжирования кандидатов.

Зачем она нужна, показал замер потолка: верный вариант исправления уже
присутствует среди кандидатов в 93.8% случаев на опечатках, а первым мы его
ставим только в 80.5%. Тринадцать пунктов теряются в ранжировании, которое
смотрит на цену правки и частоту слова, но не на контекст.

Классический пример — «сйын». Кандидаты «ыйын» и «сайын» оба на расстоянии одной
правки, побеждает «ыйын», потому что вдвое частотнее. В контексте
«бу ___ үлэлээбиттэрин» правильный очевиден, но контекста ранжирование не видит.

Модель намеренно простая: частоты биграмм с откатом на униграммы (stupid
backoff). Для выбора из пяти кандидатов ни сглаживание Кнесера-Нея, ни нейросеть
не нужны, а цена — микросекунды на CPU.
"""
from __future__ import annotations

import gzip
import math
import pathlib

BOS, EOS = "<s>", "</s>"
# Штраф за откат с биграммы на униграмму. Значение из «stupid backoff»
# (Brants et al., 2007) — там показано, что при больших корпусах подбирать его
# точнее смысла нет.
BACKOFF = 0.4


class BigramLM:
    """Частоты биграмм и униграмм с откатом.

    Хранение: внешний словарь по первому слову, внутренний по второму. Так на
    300 тысячах форм выходит примерно вдвое компактнее, чем словарь с кортежами
    в ключах, и поиск идёт по одному слову, а не по паре.
    """

    def __init__(self, uni: dict[str, int], bi: dict[str, dict[str, int]],
                 total: int) -> None:
        self.uni = uni
        self.bi = bi
        self.total = max(total, 1)

    # --- загрузка -----------------------------------------------------------
    @classmethod
    def load(cls, path: str | pathlib.Path, *, min_count: int = 1) -> "BigramLM":
        """Загрузка. `min_count` отсекает редкие биграммы прямо при чтении —
        так можно уменьшить память, не пересобирая модель."""
        path = pathlib.Path(path)
        uni: dict[str, int] = {}
        with _open(path / "unigrams.tsv") as f:
            next(f)
            for line in f:
                w, c = line.rstrip("\n").split("\t")
                uni[w] = int(c)
        bi: dict[str, dict[str, int]] = {}
        with _open(path / "bigrams.tsv") as f:
            next(f)
            for line in f:
                a, b, c = line.rstrip("\n").split("\t")
                if int(c) < min_count:
                    continue
                d = bi.get(a)
                if d is None:
                    d = bi[a] = {}
                d[b] = int(c)
        return cls(uni, bi, sum(uni.values()))

    def __len__(self) -> int:
        return sum(len(d) for d in self.bi.values())

    # --- вероятности --------------------------------------------------------
    def logp_unigram(self, word: str) -> float:
        """Десятичный логарифм вероятности слова без контекста."""
        c = self.uni.get(word, 0)
        # Неизвестному слову даём вероятность реже самого редкого известного,
        # иначе кандидат вне модели получает нулевую и выигрывает у всех.
        return math.log10((c + 0.5) / self.total)

    def logp(self, word: str, prev: str | None = None) -> float:
        """Логарифм P(word | prev) с откатом на униграмму."""
        if prev:
            d = self.bi.get(prev)
            if d:
                c = d.get(word)
                if c:
                    return math.log10(c / max(self.uni.get(prev, c), 1))
        return math.log10(BACKOFF) + self.logp_unigram(word)

    def score(self, word: str, left: str | None = None,
              right: str | None = None) -> float:
        """Оценка слова в контексте: слева P(word|left), справа P(right|word).

        Правый контекст важен не меньше левого: в «бу ___ үлэлээбиттэрин»
        подсказка идёт именно справа.
        """
        s = self.logp(word, left)
        if right:
            s += self.logp(right, word)
        return s


def _open(path: pathlib.Path):
    if path.exists():
        return path.open(encoding="utf-8")
    gz = path.with_suffix(path.suffix + ".gz")
    if gz.exists():
        return gzip.open(gz, "rt", encoding="utf-8")
    raise FileNotFoundError(f"нет ни {path}, ни {gz}")
