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
from .grammar import SAFE_RULES, harmony_violations, violations
from .lexicon import Lexicon
from .lm import BOS, EOS, BigramLM
from .norm import normalize
from .tokenize import Token, tokenize
from .userdict import UserDict

# Во сколько сотых оценивается разница в частоте на порядок. Подобрано так, чтобы
# правка ценой 0.35 (замена ҕ→г) не перебивалась ничем, а правки одной цены
# разводились по частоте.
FREQ_WEIGHT = 12
MAX_SEARCH_COST = 220
# Вес контекстной модели при ранжировании. Логарифм вероятности умножается на
# него и вычитается из цены, как раньше вычиталась частота. Подобран на dev
# (E18): слишком малый не переставляет ничего, слишком большой позволяет
# контексту победить правку вдвое меньшей цены.
LM_WEIGHT = 22
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
    kind: str          # shadow | restore | fuzzy | grammar | userdict
    penalty: int = 0   # штраф за нарушение правил грамматики
    context: float | None = None   # логарифм вероятности в контексте

    @property
    def score(self) -> float:
        base = self.cost + self.penalty
        if self.context is not None:
            return base - LM_WEIGHT * self.context
        return base - FREQ_WEIGHT * math.log10(max(self.freq, 1))


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
                 grammar_penalty: int = GRAMMAR_PENALTY,
                 flag_rules: tuple[str, ...] | None = None,
                 userdict: UserDict | None = None,
                 lm: BigramLM | None = None) -> None:
        self.lex = lexicon
        self.max_suggestions = max_suggestions
        self.use_tail = use_tail
        self.grammar = grammar
        self.grammar_penalty = grammar_penalty
        # Пользовательский словарь идёт впереди всего: человек знает про свои
        # имена и термины больше, чем корпус и грамматика вместе.
        self.userdict = userdict or UserDict()
        # Контекстная модель. Без неё ранжирование работает как раньше, по
        # частоте слова; с ней частота заменяется вероятностью в контексте.
        self.lm = lm
        # Помечать слово, которое словарь принял, но которое нарушает правила.
        # Единственное применение правил, способное поймать ошибку, невидимую
        # для словаря, — и оно не окупается, поэтому выключено по умолчанию.
        #
        # Замер (E17): полный набор правил поднимает детекцию на `real` с 67.4%
        # до 71.8%, но ложные срабатывания растут с 1.79% до 3.53%, а F1 падает
        # на всех задачах, кроме `real`. Дешёвый набор мягче — 2.16% ложных, —
        # но тоже отнимает почти пункт F1 на опечатках.
        #
        # Причина в заимствованиях: правила исконной фонетики они нарушают
        # законно, а отличить заимствование от ошибки без словаря нельзя.
        # Включать имеет смысл на тексте, где заимствований мало — в олонхо,
        # в художественной прозе:
        #     SpellChecker(lex, flag_rules=SAFE_RULES)
        self.flag_rules: tuple[str, ...] | None = flag_rules
        # Внесловарное восстановление отключаемо отдельно от штрафа: это разные
        # применения одного правила, и цена у них разная.
        self.out_of_lex = True
        # Дерево строим по ядру: подсказывать надо только надёжными формами.
        self.trie = Trie.from_freq(lexicon.core)

    # --- одно слово ---------------------------------------------------------
    def suggest(self, word: str, left: str | None = None,
                right: str | None = None) -> list[Suggestion]:
        low = word.lower()
        out: dict[str, Suggestion] = {}

        # 0a. правка, заданная пользователем, — сильнее любой другой
        forced = self.userdict.correction(low)
        if forced:
            return _restore_case(word, [Suggestion(forced, 0, 10 ** 9, "userdict")])

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

        if self.lm is not None and (left or right):
            for s in out.values():
                s.context = self.lm.score(s.form, left, right)

        res = sorted(out.values(), key=lambda s: s.score)
        return _restore_case(word, res)[: self.max_suggestions]

    def _accepted_by_user(self, word: str) -> bool:
        """Принято пользовательским словарём — целиком или по частям.

        Части дефисного сложения проверяются в обоих словарях сразу: человек
        добавил «скайраннинг», и «скайраннинг-куонкурус» должно пройти, хотя
        вторая часть известна только основному лексикону.
        """
        if not self.userdict:
            return False
        if self.userdict.accepts(word):
            return True
        w = word.lower()
        if "-" not in w:
            return False
        parts = [p for p in w.split("-") if p]
        if len(parts) < 2:
            return False
        return (any(self.userdict.accepts(p) for p in parts)
                and all(self.userdict.accepts(p)
                        or self.lex.check_form(p, use_hyphen=False).ok
                        for p in parts))

    # --- текст --------------------------------------------------------------
    def check(self, text: str) -> list[Issue]:
        text = normalize(text)
        issues: list[Issue] = []
        toks = list(tokenize(text))
        # Соседние СЛОВА, а не токены: между словами могут стоять числа и
        # латиница, и для контекста они бесполезны.
        words = [i for i, t in enumerate(toks) if t.is_word]
        pos = {i: k for k, i in enumerate(words)}
        for ti, t in enumerate(toks):
            if t.is_word:
                if self.userdict.correction(t.text.lower()):
                    issues.append(Issue(t, self.suggest(t.text), "userdict"))
                    continue
                if self._accepted_by_user(t.text):
                    continue
            v = self.lex.check_token(t, use_tail=self.use_tail)
            if v.ok:
                if not (self.flag_rules and t.is_word
                        and violations(t.text, self.flag_rules)):
                    continue
                issues.append(Issue(t, self._suggest_at(toks, words, pos, ti), "grammar"))
                continue
            issues.append(Issue(t, self._suggest_at(toks, words, pos, ti), v.reason))
        return issues

    def _suggest_at(self, toks, words, pos, ti) -> list[Suggestion]:
        """Подсказки для токена с учётом соседних слов."""
        t = toks[ti]
        left = right = None
        if self.lm is not None and ti in pos:
            k = pos[ti]
            left = toks[words[k - 1]].text.lower() if k > 0 else BOS
            right = (toks[words[k + 1]].text.lower()
                     if k + 1 < len(words) else EOS)
        return self.suggest(t.text, left, right)

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
