"""Правила якутской фонетики и орфографии из грамматики.

Всё, что здесь описано, взято из грамматических источников, а не выведено из
корпуса. Это принципиально: остальные слои проекта построены на статистике, и
интересно ровно то, добавляет ли книжное знание что-то поверх неё.

Гармония гласных (сингармонизм) в якутском строится на трёх признаках: ряд
(передний/задний), огубленность и степень открытости. Таблица сочетаемости взята
в готовом виде, потому что вывести её из признаков не получается — там есть
несимметричные места, вроде того, что после «о» допускается «у», но не «а»:

    после а аа ы ыы ыа  →  а аа ы ыы ыа
    после э ээ и ии иэ  →  э ээ и ии иэ
    после о оо          →  о оо у уу уо
    после ө өө          →  ө өө ү үү үө
    после у уу уо       →  у уу а аа уо
    после ү үү үө       →  ү үү э ээ үө

Гармония действует только в исконных словах. Русские заимствования её нарушают
законно и массово, поэтому слово с буквой, которая в якутском употребляется лишь
в заимствованиях (в, е, ё, ж, з, ф, ц, ш, щ, ъ, ю, я), из проверки исключается.

Позиционные ограничения на согласные проверены по корпусу и приведены к тому,
что подтверждается данными, — см. `scripts/check_grammar_rules.py` и E15
в docs/experiments.md.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# --- гласные -----------------------------------------------------------------
SHORT = "аыоуэиөү"
LONG = ("аа", "ыы", "оо", "уу", "ээ", "ии", "өө", "үү")
DIPHTHONGS = ("ыа", "уо", "иэ", "үө")
VOWEL_CHARS = set(SHORT) | set("еёюя")     # русские гласные для опознания заимствований

# Ядро слога: долгота и дифтонг — единицы, а не две гласные подряд.
NUCLEI = set(SHORT) | set(LONG) | set(DIPHTHONGS)

# Таблица сочетаемости. Ключ — предыдущее ядро, значение — что за ним допустимо.
_GROUPS: dict[tuple[str, ...], tuple[str, ...]] = {
    ("а", "аа", "ы", "ыы", "ыа"): ("а", "аа", "ы", "ыы", "ыа"),
    ("э", "ээ", "и", "ии", "иэ"): ("э", "ээ", "и", "ии", "иэ"),
    ("о", "оо"):                  ("о", "оо", "у", "уу", "уо"),
    ("ө", "өө"):                  ("ө", "өө", "ү", "үү", "үө"),
    ("у", "уу", "уо"):            ("у", "уу", "а", "аа", "уо"),
    ("ү", "үү", "үө"):            ("ү", "үү", "э", "ээ", "үө"),
}
FOLLOWS: dict[str, frozenset[str]] = {}
for _keys, _vals in _GROUPS.items():
    for _k in _keys:
        FOLLOWS[_k] = frozenset(_vals)

# --- согласные ---------------------------------------------------------------
CONSONANTS = set("бвгҕджзйклмнҥпрсһтфхцчшщ")

# Буквы, которые в якутском письме встречаются только в русских заимствованиях.
# Слово с любой из них не подчиняется гармонии, и требовать её от него нельзя.
LOAN_LETTERS = set("вежзфцшщъюяё")

# Слово не начинается с этих букв. По грамматике сюда же относят «г» и «п»,
# но корпус их у начала слова показывает, поэтому в правило они не вошли.
NO_INITIAL = set("ҕҥй")

WORD_RE = re.compile(r"[а-яёa-zA-ZҔҕҤҥӨөҺһҮү-]+")


@dataclass(frozen=True, slots=True)
class Violation:
    kind: str          # harmony | initial | cluster | vowelless
    position: int      # позиция в слове
    detail: str


def parse_nuclei(word: str) -> list[tuple[int, str]]:
    """Ядра слогов в порядке следования: (позиция, ядро).

    Двухсимвольные ядра разбираются раньше односимвольных, иначе «уо» распадётся
    на «у» и «о» и любое слово с дифтонгом станет нарушением.
    """
    w = word.lower()
    out: list[tuple[int, str]] = []
    i = 0
    while i < len(w):
        if w[i] not in SHORT:
            i += 1
            continue
        pair = w[i:i + 2]
        if pair in LONG or pair in DIPHTHONGS:
            out.append((i, pair))
            i += 2
        else:
            out.append((i, w[i]))
            i += 1
    return out


def is_loanword(word: str) -> bool:
    """Слово содержит буквы, употребляемые только в заимствованиях."""
    return any(c in LOAN_LETTERS for c in word.lower())


def harmony_violations(word: str) -> list[Violation]:
    """Нарушения гармонии гласных.

    Проверяется каждая часть дефисного сложения отдельно. Гармония действует
    внутри слова, а через дефис ряд меняется законно: «дьон-сэргэ» (о→э),
    «күүс-көмө» (үү→ө), «дьиэ-уот» (иэ→уо) — всё это правильные слова, и они
    попадали в нарушители, пока проверка шла по всей строке целиком.

    Для заимствований возвращается пусто: они гармонии не подчиняются.
    """
    if is_loanword(word):
        return []
    out: list[Violation] = []
    offset = 0
    for part in word.lower().split("-"):
        nuclei = parse_nuclei(part)
        for (_, prev), (pos, cur) in zip(nuclei, nuclei[1:]):
            allowed = FOLLOWS.get(prev)
            if allowed is not None and cur not in allowed:
                out.append(Violation("harmony", offset + pos, f"{prev}→{cur}"))
        offset += len(part) + 1
    return out


def positional_violations(word: str) -> list[Violation]:
    """Позиционные ограничения на согласные."""
    w = word.lower().strip("-")
    if not w:
        return []
    out: list[Violation] = []
    if w[0] in NO_INITIAL:
        out.append(Violation("initial", 0, f"слово на «{w[0]}»"))
    return out


def cluster_violations(word: str, max_run: int = 3) -> list[Violation]:
    """Стечения согласных. В исконных словах больше двух подряд не бывает;
    порог здесь мягче на единицу, потому что диграфы «дь» и «нь» на письме
    выглядят как два согласных и раздувают любое стечение."""
    if is_loanword(word):
        return []
    out: list[Violation] = []
    run, start = 0, 0
    for i, c in enumerate(word.lower() + " "):
        if c in CONSONANTS or c == "ь":
            if run == 0:
                start = i
            run += 1
        else:
            if run > max_run:
                out.append(Violation("cluster", start, word[start:start + run]))
            run = 0
    return out


def violations(word: str, *, harmony: bool = True, positional: bool = True,
               clusters: bool = True) -> list[Violation]:
    """Все нарушения правил в слове."""
    out: list[Violation] = []
    if harmony:
        out += harmony_violations(word)
    if positional:
        out += positional_violations(word)
    if clusters:
        out += cluster_violations(word)
    return out


def is_wellformed(word: str, **kw) -> bool:
    return not violations(word, **kw)
