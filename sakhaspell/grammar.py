"""Правила якутской фонетики и орфографии из грамматики.

Всё здесь взято из грамматических источников, а не выведено из корпуса. Это
принципиально: остальные слои проекта построены на статистике, и проверяется
ровно то, добавляет ли книжное знание что-то поверх неё.

Правила оформлены реестром `RULES`, чтобы замер (`scripts/check_grammar_rules.py`)
прогонялся по всем сразу и было видно, какое из них корпус подтверждает, а какое
опровергает. Грамматики упрощают, и часть формулировок на 300 тысячах реальных
словоформ не держится — см. E17 в docs/experiments.md.

Источники: Убрятова и др. «Грамматика современного якутского литературного
языка» (1982), описание фонетики в русской Википедии и на wiki.sakhatyla.ru,
Ivanova, Washington, Tyers «A Free/Open-Source Morphological Analyser and
Generator for Sakha» (LREC 2022).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# --- инвентарь ---------------------------------------------------------------
SHORT = "аыоуэиөү"
LONG = ("аа", "ыы", "оо", "уу", "ээ", "ии", "өө", "үү")
DIPHTHONGS = ("ыа", "уо", "иэ", "үө")
NUCLEI = set(SHORT) | set(LONG) | set(DIPHTHONGS)

CONSONANTS = set("бвгҕджзйклмнҥпрсһтфхцчшщ")
RU_VOWELS = set("еёюя")

# Буквы, которые в якутском письме встречаются только в русских заимствованиях.
# Слово с любой из них живёт по русским правилам, и требовать от него якутских
# закономерностей нельзя.
LOAN_LETTERS = set("вежзфцшщъюяё")

# --- гармония гласных --------------------------------------------------------
# Таблица сочетаемости взята готовой, а не выведена из признаков: вывести
# не получается, там есть несимметричные места — после «о» допускается «у»,
# но не «а», а после «у» наоборот «а», но не «ы».
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

# --- позиционные ограничения -------------------------------------------------
# Согласные, с которых не начинается якутское слово.
NO_INITIAL = set("ҕҥй")
# Согласные, на которые якутское слово не оканчивается.
#
# Грамматика называет здесь б, г, ҕ, д, һ, ч. Корпус подтверждает только ҕ
# (10 форм, 53 вхождения — обрывки слов). Остальное правило опровергает:
# на «һ» оканчиваются законные «тыһ» (14 753 вхождения), «нэһ», «бөһ», «баһ»,
# а на б/г/д/ч — заимствования и аббревиатуры («млрд», «психолог», «куб»,
# «ильич»), которые исконной фонетике не подчиняются. См. E17.
NO_FINAL = set("ҕ")
# Сочетания двух согласных, допустимые на конце слова.
FINAL_CLUSTERS = {"лт", "рт", "нт", "ст", "кт", "мп", "нк"}


@dataclass(frozen=True, slots=True)
class Violation:
    rule: str
    position: int
    detail: str

    def __repr__(self) -> str:      # pragma: no cover
        return f"{self.rule}@{self.position}:{self.detail}"

    # обратная совместимость: раньше поле называлось kind
    @property
    def kind(self) -> str:
        return self.rule


# --- разбор ------------------------------------------------------------------
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


def _parts(word: str) -> list[tuple[int, str]]:
    """Части дефисного сложения с их смещениями.

    Правила действуют внутри слова. Через дефис ряд меняется законно
    («дьон-сэргэ», «күүс-көмө»), и проверять сложение целиком нельзя.
    """
    out, off = [], 0
    for p in word.lower().split("-"):
        if p:
            out.append((off, p))
        off += len(p) + 1
    return out


# --- правила -----------------------------------------------------------------
def rule_harmony(word: str) -> list[Violation]:
    """Гармония гласных: нёбная и губная одной таблицей сочетаемости."""
    if is_loanword(word):
        return []
    out = []
    for off, part in _parts(word):
        nuclei = parse_nuclei(part)
        for (_, prev), (pos, cur) in zip(nuclei, nuclei[1:]):
            allowed = FOLLOWS.get(prev)
            if allowed is not None and cur not in allowed:
                out.append(Violation("harmony", off + pos, f"{prev}→{cur}"))
    return out


def rule_vowel_pairs(word: str) -> list[Violation]:
    """Сочетания гласных: только 8 долгот и 4 дифтонга.

    Любая другая пара разных гласных подряд в исконном слове невозможна:
    «аи», «оэ», «уы» не бывают.
    """
    if is_loanword(word):
        return []
    out = []
    for off, part in _parts(word):
        i = 0
        while i + 1 < len(part):
            a, b = part[i], part[i + 1]
            if a in SHORT and b in SHORT:
                pair = a + b
                if pair not in LONG and pair not in DIPHTHONGS:
                    out.append(Violation("vowel_pairs", off + i, pair))
                    i += 2
                    continue
                i += 2
                continue
            i += 1
    return out


def rule_initial(word: str) -> list[Violation]:
    """Слово не начинается на ҕ, ҥ, й."""
    out = []
    for off, part in _parts(word):
        if part and part[0] in NO_INITIAL:
            out.append(Violation("initial", off, f"слово на «{part[0]}»"))
    return out


def rule_final(word: str) -> list[Violation]:
    """Слово не оканчивается на звонкие б, г, ҕ, д, ж и на һ, ч."""
    if is_loanword(word):
        return []
    out = []
    for off, part in _parts(word):
        if part and part[-1] in NO_FINAL:
            out.append(Violation("final", off + len(part) - 1,
                                 f"слово на «{part[-1]}»"))
    return out


def rule_clusters(word: str, max_run: int = 2) -> list[Violation]:
    """В исконном слове не бывает более двух согласных подряд.

    Диграфы «дь» и «нь» на письме выглядят как согласный плюс мягкий знак,
    поэтому перед подсчётом они схлопываются в один символ — иначе «сылдьар»
    даст мнимое стечение из трёх.
    """
    if is_loanword(word):
        return []
    out = []
    for off, part in _parts(word):
        collapsed = part.replace("дь", "д").replace("нь", "н")
        run, start = 0, 0
        for i, c in enumerate(collapsed + " "):
            if c in CONSONANTS:
                if run == 0:
                    start = i
                run += 1
            else:
                if run > max_run:
                    out.append(Violation("clusters", off + start,
                                         collapsed[start:start + run]))
                run = 0
    return out


def rule_final_cluster(word: str) -> list[Violation]:
    """На конце слова допустимы лишь отдельные сочетания двух согласных."""
    if is_loanword(word):
        return []
    out = []
    for off, part in _parts(word):
        collapsed = part.replace("дь", "д").replace("нь", "н")
        tail = ""
        for c in reversed(collapsed):
            if c in CONSONANTS:
                tail = c + tail
            else:
                break
        if len(tail) >= 2 and tail[-2:] not in FINAL_CLUSTERS:
            out.append(Violation("final_cluster",
                                 off + len(collapsed) - len(tail), tail))
    return out


def rule_soft_sign(word: str) -> list[Violation]:
    """Мягкий знак употребляется только в диграфах «дь» и «нь».

    Это правило прицельно бьёт по самой частой реальной ошибке якутского письма:
    «ь» вместо «һ» (уьу, киьини, эьиги — см. E4).
    """
    if is_loanword(word):
        return []
    out = []
    w = word.lower()
    for i, c in enumerate(w):
        if c == "ь" and (i == 0 or w[i - 1] not in "дн"):
            out.append(Violation("soft_sign", i, "ь не после д/н"))
    return out


def rule_triple_letter(word: str) -> list[Violation]:
    """Один и тот же гласный не идёт трижды подряд: долгота — это ровно два."""
    out = []
    w = word.lower()
    for i in range(len(w) - 2):
        if w[i] == w[i + 1] == w[i + 2] and w[i] in SHORT:
            out.append(Violation("triple_letter", i, w[i] * 3))
    return out


RULES: dict[str, Callable[[str], list[Violation]]] = {
    "гармония гласных": rule_harmony,
    "сочетания гласных": rule_vowel_pairs,
    "начало слова": rule_initial,
    "конец слова": rule_final,
    "стечение согласных": rule_clusters,
    "стечение на конце": rule_final_cluster,
    "мягкий знак": rule_soft_sign,
    "тройная гласная": rule_triple_letter,
}

# Дешёвые правила: срабатывают на правильной форме в 0.2–0.3% случаев против
# 1.1–1.4% у гармонии, а ловят при этом больше неё (E17). Именно ими можно
# помечать слово, которое словарь принял.
SAFE_RULES = ("начало слова", "конец слова", "мягкий знак",
              "тройная гласная", "стечение на конце")

# Полный набор. Ловит вдвое больше, но и цена вдвое выше — годится там, где
# важнее не пропустить ошибку, чем не потревожить пользователя.
ALL_RULES = tuple(RULES)

# Набор по умолчанию для `violations()` и `is_wellformed()` — полный: это
# справочная функция, и урезать её нет причин. На пометку слов в спелчекере
# правила по умолчанию НЕ влияют, см. SpellChecker(flag_rules=...).
DEFAULT_RULES = ALL_RULES


def violations(word: str, rules: tuple[str, ...] = DEFAULT_RULES) -> list[Violation]:
    """Все нарушения выбранных правил."""
    out: list[Violation] = []
    for name in rules:
        out += RULES[name](word)
    return out


def is_wellformed(word: str, rules: tuple[str, ...] = DEFAULT_RULES) -> bool:
    return not violations(word, rules)


# --- обратная совместимость с 0.2.0 -----------------------------------------
harmony_violations = rule_harmony
positional_violations = rule_initial


def cluster_violations(word: str, max_run: int = 2) -> list[Violation]:
    return rule_clusters(word, max_run)
