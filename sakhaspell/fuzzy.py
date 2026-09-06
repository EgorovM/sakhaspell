"""Генерация кандидатов: префиксное дерево с поиском по ограниченному расстоянию.

Почему не SymSpell, хотя он быстрее. SymSpell экономит на индексе за счёт усечения
до префикса (обычно 7 символов) — это опирается на допущение, что ошибки лежат
ближе к началу слова. Для якутского допущение неверное: слово агглютинативное,
основная масса вариативности и ошибок — в хвосте аффиксов («оҕолорбутугар»),
а начало как раз стабильно. Полный индекс удалений без усечения для 300k форм
даёт ~17 млн ключей и несколько гигабайт в питоне — на ноутбук это не поставить.

Дерево с построчным динамическим программированием даёт то же расстояние, память
линейна по словарю и делится между общими префиксами, а отсечение по минимуму
строки убирает почти весь перебор.

Расстояние — Дамерау-Левенштейн с перестановкой соседних (частая опечатка при
наборе) и с настраиваемой ценой замены: для якутского замены ҕ↔г, ө↔о, ү↔у,
һ↔с дешевле прочих, потому что это не опечатка, а письмо без раскладки.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# Замены, которые в якутском письме стоят дешевле обычной опечатки: так пишут,
# когда под рукой нет якутской раскладки. Стоимость в сотых, базовая правка = 100.
CHEAP_SUBS: dict[tuple[str, str], int] = {}
# ь вместо һ — самая частая реальная ошибка в вебе (data/errors/pairs_web.tsv:
# уьу→уу, киьини→киһини, эьиги→эһиги, туьэн→түһэн). Мягкий знак стоит на русской
# раскладке и похож на һ начертанием, поэтому им заменяют һ чаще, чем чем-либо ещё.
# ц вместо ҥ — то же самое в OCR (хацалас→хаҥалас, мэцэ→мэҥэ).
for _a, _b in (("ҕ", "г"), ("ҥ", "н"), ("ө", "о"), ("ү", "у"), ("һ", "с"),
               ("һ", "ь"), ("ҥ", "ц"),
               ("һ", "х"), ("һ", "h"), ("ҕ", "g"), ("ҥ", "n"), ("ө", "o"),
               ("ү", "y"), ("ө", "ё"), ("ү", "ю"), ("е", "э"), ("ъ", "ь")):
    CHEAP_SUBS[(_a, _b)] = 35
    CHEAP_SUBS[(_b, _a)] = 35

# Соседи по раскладке ЙЦУКЕН — обычная механическая опечатка.
_ROWS = ("йцукенгшщзхъ", "фывапролджэ", "ячсмитьбю")
KEYBOARD_NEIGHBOURS: dict[str, set[str]] = {}
for _r, _row in enumerate(_ROWS):
    for _i, _c in enumerate(_row):
        n = set()
        if _i:
            n.add(_row[_i - 1])
        if _i + 1 < len(_row):
            n.add(_row[_i + 1])
        for _dr in (-1, 1):
            _rr = _r + _dr
            if 0 <= _rr < len(_ROWS):
                for _j in (_i - 1, _i, _i + 1):
                    if 0 <= _j < len(_ROWS[_rr]):
                        n.add(_ROWS[_rr][_j])
        KEYBOARD_NEIGHBOURS[_c] = n

BASE = 100
NEIGHBOUR_SUB = 70


def sub_cost(a: str, b: str) -> int:
    if a == b:
        return 0
    c = CHEAP_SUBS.get((a, b))
    if c is not None:
        return c
    if b in KEYBOARD_NEIGHBOURS.get(a, ()):
        return NEIGHBOUR_SUB
    return BASE


@dataclass(slots=True)
class _Node:
    kids: dict[str, "_Node"] = field(default_factory=dict)
    freq: int = 0          # >0 => здесь кончается словоформа


@dataclass
class Candidate:
    form: str
    freq: int
    cost: int              # взвешенное расстояние в сотых

    def __repr__(self) -> str:      # pragma: no cover
        return f"{self.form}({self.cost/100:.2f}, f={self.freq})"


class Trie:
    """Префиксное дерево словоформ с поиском по взвешенному расстоянию."""

    def __init__(self) -> None:
        self.root = _Node()
        self.size = 0

    def add(self, word: str, freq: int = 1) -> None:
        node = self.root
        for ch in word:
            nxt = node.kids.get(ch)
            if nxt is None:
                nxt = node.kids[ch] = _Node()
            node = nxt
        if not node.freq:
            self.size += 1
        node.freq = max(node.freq, freq)

    @classmethod
    def from_freq(cls, freq: dict[str, int]) -> "Trie":
        t = cls()
        for w, f in freq.items():
            t.add(w, f)
        return t

    def __contains__(self, word: str) -> bool:
        node = self.root
        for ch in word:
            node = node.kids.get(ch)
            if node is None:
                return False
        return node.freq > 0

    def search(self, word: str, max_cost: int = 200, limit: int = 0) -> list[Candidate]:
        """Все словоформы в пределах max_cost (в сотых). Перестановка соседних
        символов считается одной правкой ценой BASE."""
        n = len(word)
        first = list(range(0, (n + 1) * BASE, BASE))
        out: list[Candidate] = []
        for ch, node in self.root.kids.items():
            self._walk(node, ch, ch, first, None, None, word, max_cost, out)
        out.sort(key=lambda c: (c.cost, -c.freq))
        return out[:limit] if limit else out

    def _walk(self, node: _Node, ch: str, prefix: str, prev: list[int],
              prev2: list[int] | None, prev_ch: str | None,
              word: str, max_cost: int, out: list[Candidate]) -> None:
        n = len(word)
        cur = [prev[0] + BASE]
        for i in range(1, n + 1):
            wc = word[i - 1]
            best = min(
                cur[i - 1] + BASE,                      # вставка в слово
                prev[i] + BASE,                         # удаление из слова
                prev[i - 1] + sub_cost(wc, ch),         # замена
            )
            # перестановка соседних символов
            if (prev2 is not None and prev_ch is not None and i > 1
                    and wc == prev_ch and word[i - 2] == ch):
                best = min(best, prev2[i - 2] + BASE)
            cur.append(best)

        if node.freq and cur[n] <= max_cost:
            out.append(Candidate(prefix, node.freq, cur[n]))

        if min(cur) > max_cost:
            return
        for nch, kid in node.kids.items():
            self._walk(kid, nch, prefix + nch, cur, prev, ch, word, max_cost, out)
