"""Спелчекер: слой L1 целиком.

Правила якутской грамматики (`grammar.py`) участвуют здесь двумя способами, и
оба бесплатны по ложным срабатываниям: штрафом кандидату, нарушающему гармонию
гласных, и восстановлением спецбукв за пределами лексикона. Третий способ —
отсев по гармонии в хвосте лексикона — живёт в `Lexicon.check_form` и по
умолчанию выключен, потому что стоит 0.04 пункта ложных срабатываний.


Порядок проверки одного слова определяется не удобством, а тем, какие ошибки
встречаются на самом деле:

  1. слово принято лексиконом — выходим сразу, это 98% токенов;
  2. восстановление ҕҥөһү перебором вариантов — самый массовый реальный случай,
     решается точно и без поиска по расстоянию;
  3. поиск по взвешенному расстоянию — настоящие опечатки;
  4. ничего не нашли — помечаем слово, но подсказок не даём.

Ранжирование кандидатов: цена правки в первую очередь, частота — во вторую.
Частота берётся логарифмом, иначе очень частое слово перебивает правку вдвое
меньшей цены и «көр» превращается в «биэр».
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .errors import restoration_variants
from .fuzzy import Trie
from .grammar import harmony_violations
from .lexicon import Lexicon
from .norm import normalize
from .tokenize import Token, tokenize

# Во сколько сотых оценивается разница в частоте на порядок. Подобрано так, чтобы
# правка ценой 0.35 (замена ҕ→г) не перебивалась ничем, а правки одной цены
# разводились по частоте.
FREQ_WEIGHT = 12
MAX_SEARCH_COST = 220
# Штраф кандидату, нарушающему гармонию гласных. На деноминализации гармонию
# нарушает 33.6% испорченных слов против 0.4% правильных (E15), так что признак
# сильный. Величина в тех же сотых, что и цена правки: 40 — меньше одной обычной
# правки, поэтому гармония разводит кандидатов равной цены, но не перебивает
# заметно более дешёвый вариант.
GRAMMAR_PENALTY = 40
# Если восстановление спецбукв дало кандидата не дороже этого, полный поиск по
# дереву пропускается. 105 — это три дешёвых замены (ҕ→г и т.п.) или одна такая
# замена плюс запас; дороже уже начинается территория настоящих опечаток.
CHEAP_ENOUGH = 105


@dataclass
class Suggestion:
    form: str
    cost: int
    freq: int
    kind: str          # shadow | restore | fuzzy
    penalty: int = 0   # штраф за нарушение правил грамматики

    @property
    def score(self) -> float:
        return (self.cost + self.penalty
                - FREQ_WEIGHT * math.log10(max(self.freq, 1)))


@dataclass
class Issue:
    token: Token
    suggestions: list[Suggestion]
    reason: str

    @property
    def best(self) -> str | None:
        return self.suggestions[0].form if self.suggestions else None


class SpellChecker:
    def __init__(self, lexicon: Lexicon, *, max_suggestions: int = 5,
                 use_tail: bool = True, grammar: bool = True,
                 grammar_penalty: int = GRAMMAR_PENALTY) -> None:
        self.lex = lexicon
        self.max_suggestions = max_suggestions
        self.use_tail = use_tail
        self.grammar = grammar
        self.grammar_penalty = grammar_penalty
        # Внесловарное восстановление отключаемо отдельно от штрафа: это разные
        # применения одного правила, и цена у них разная.
        self.out_of_lex = True
        # Дерево строим по ядру: подсказывать надо только надёжными формами.
        self.trie = Trie.from_freq(lexicon.core)

    # --- одно слово ---------------------------------------------------------
    def suggest(self, word: str) -> list[Suggestion]:
        low = word.lower()
        out: dict[str, Suggestion] = {}

        # 0. известная тень искажения — исправление знаем точно
        origin = self.lex.shadow.get(low)
        if origin:
            out[origin] = Suggestion(origin, 1, self.lex.core.get(origin, 1), "shadow")

        # 1. восстановление специальных букв: перебор вариантов и точный поиск
        # по словарю. Это дёшево — хэш вместо обхода дерева.
        for v in restoration_variants(low):
            if v == low:
                continue
            f = self.lex.core.get(v)
            if f:
                n_changed = sum(a != b for a, b in zip(v, low)) or 1
                out[v] = Suggestion(v, 35 * n_changed, f, "restore")

        # 2. Поиск по расстоянию — самая дорогая часть (обход префиксного дерева).
        # Если восстановление уже дало дешёвого кандидата, полный поиск не нужен:
        # деноминализация несравнимо частотнее опечаток, и найденный по ней
        # вариант всё равно выиграет ранжирование.
        if not out or min(s.cost for s in out.values()) > CHEAP_ENOUGH:
            for c in self.trie.search(low, max_cost=MAX_SEARCH_COST,
                                      limit=self.max_suggestions * 6):
                if c.form == low:
                    continue
                prev = out.get(c.form)
                if prev is None or c.cost < prev.cost:
                    out[c.form] = Suggestion(c.form, c.cost, c.freq, "fuzzy")

        # 3. Восстановление вне лексикона. Форм якутского бесконечно много,
        # ядро покрывает не всё: у 2.6% ошибок бенчмарка верной формы в нём нет,
        # и там лексикону предложить нечего. Если само слово нарушает гармонию,
        # а вариант восстановления её соблюдает, это достаточное основание
        # предложить вариант, даже не найдя его в словаре.
        if self.grammar and self.out_of_lex and not out and harmony_violations(low):
            for v in restoration_variants(low):
                if v != low and not harmony_violations(v):
                    n_changed = sum(a != b for a, b in zip(v, low)) or 1
                    out[v] = Suggestion(v, 35 * n_changed, 1, "grammar")
                    break

        if self.grammar and self.grammar_penalty:
            for s in out.values():
                if harmony_violations(s.form):
                    s.penalty = self.grammar_penalty

        res = sorted(out.values(), key=lambda s: s.score)
        return _restore_case(word, res)[: self.max_suggestions]

    # --- текст --------------------------------------------------------------
    def check(self, text: str) -> list[Issue]:
        text = normalize(text)
        issues: list[Issue] = []
        for t in tokenize(text):
            v = self.lex.check_token(t, use_tail=self.use_tail)
            if v.ok:
                continue
            issues.append(Issue(t, self.suggest(t.text), v.reason))
        return issues

    def correct(self, text: str) -> str:
        text = normalize(text)
        out, prev = [], 0
        for iss in self.check(text):
            if not iss.suggestions:
                continue
            out.append(text[prev:iss.token.start])
            out.append(iss.suggestions[0].form)
            prev = iss.token.end
        out.append(text[prev:])
        return "".join(out)


def _restore_case(src: str, sugg: list[Suggestion]) -> list[Suggestion]:
    if src.islower() or not src:
        return sugg
    if src.isupper():
        fn = str.upper
    elif src[0].isupper():
        fn = str.capitalize
    else:
        return sugg
    return [Suggestion(fn(s.form), s.cost, s.freq, s.kind, s.penalty) for s in sugg]
