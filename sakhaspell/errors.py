"""Модель ошибок якутского письма — в обе стороны.

Прямое направление (порча) нужно, чтобы делать обучающие и тестовые данные из
чистого корпуса. Обратное (восстановление) — это рабочий путь спелчекера для
самого массового реального случая: человек набрал текст с русской раскладки и
все пять якутских букв превратились в русские.

Основа — систематические подстановки, а не случайный шум:

    ҕ → г      оҕо → ого
    ҥ → н, нг  саҥа → сана, санга
    ө → о      көр → кор
    ү → у      үөрэх → уорэх
    һ → h, с   киһи → киhи, киси

Эти замены не опечатки: они детерминированы и покрывают целое слово сразу, поэтому
восстановление делается перебором вариантов по позициям, а не поиском по расстоянию.
Расстояние остаётся для настоящих опечаток.
"""
from __future__ import annotations

import itertools
import random
import re

# --- порча: якутское письмо -> письмо без специальных букв -------------------
DENORM_RU = str.maketrans({"ҕ": "г", "ҥ": "н", "ө": "о", "ү": "у", "һ": "с",
                           "Ҕ": "Г", "Ҥ": "Н", "Ө": "О", "Ү": "У", "Һ": "С"})
DENORM_LAT = str.maketrans({"ҕ": "g", "ҥ": "n", "ө": "o", "ү": "y", "һ": "h",
                            "Ҕ": "G", "Ҥ": "N", "Ө": "O", "Ү": "Y", "Һ": "H"})
# ҥ регулярно пишут диграфом
_NG = str.maketrans({"ҥ": "\x00", "Ҥ": "\x01"})


def denormalize(text: str, style: str = "ru", *, ng_digraph: bool = False) -> str:
    """Убирает якутские буквы. style: ru (кириллические замены), lat (h/g/n/o/y),
    mixed (һ→h, остальное кириллицей — как чаще всего пишут на практике)."""
    if ng_digraph:
        text = text.translate(_NG).replace("\x00", "нг").replace("\x01", "Нг")
    if style == "ru":
        return text.translate(DENORM_RU)
    if style == "lat":
        return text.translate(DENORM_LAT)
    if style == "mixed":
        return text.translate(DENORM_RU).replace("с", "с")  # база
    raise ValueError(style)


def denormalize_mixed(text: str, rng: random.Random) -> str:
    """Как пишут на самом деле: часть букв заменена, часть нет, һ то h, то с."""
    h = "h" if rng.random() < 0.5 else "с"
    out = []
    for ch in text:
        low = ch.lower()
        if low not in "ҕҥөүһ" or rng.random() < 0.15:   # 15% букв уцелело
            out.append(ch)
            continue
        rep = {"ҕ": "г", "ҥ": "нг" if rng.random() < 0.3 else "н",
               "ө": "о", "ү": "у", "һ": h}[low]
        out.append(rep.upper() if ch.isupper() else rep)
    return "".join(out)


# --- восстановление: письмо без специальных букв -> варианты -----------------
# Обратная карта. Позиция в слове порождает развилку, слово — декартово произведение
# развилок. Ограничиваем перебор, иначе длинное слово даёт тысячи вариантов.
# ь→һ и ц→ҥ добыты из корпуса (scripts/mine_errors.py), а не придуманы: в вебе
# «ь» вместо «һ» — самая частая ошибка вообще, в OCR так же ведёт себя «ц» вместо «ҥ».
RESTORE: dict[str, tuple[str, ...]] = {
    "г": ("ҕ",), "н": ("ҥ",), "о": ("ө",), "у": ("ү",),
    "с": ("һ",), "h": ("һ",), "x": ("һ",), "х": ("һ",),
    "ь": ("һ",), "ц": ("ҥ",),
    "g": ("ҕ",), "n": ("ҥ",), "y": ("ү",),
}
_NG_RE = re.compile(r"нг")
MAX_VARIANTS = 4096


def restoration_variants(word: str, *, max_variants: int = MAX_VARIANTS) -> list[str]:
    """Все варианты слова с восстановленными якутскими буквами, включая исходный.

    Порядок — от наименьшего числа изменений к наибольшему, чтобы потребитель мог
    остановиться на первом, который принял лексикон.
    """
    low = word.lower()
    slots: list[tuple[str, ...]] = []
    for ch in low:
        alts = RESTORE.get(ch)
        slots.append((ch,) + alts if alts else (ch,))

    n_var = 1
    for s in slots:
        n_var *= len(s)
        if n_var > max_variants:
            # слишком много развилок: восстанавливаем только самые частые буквы
            slots = [s if s[0] in "гоу" else (s[0],) for s in slots]
            break

    seen: set[str] = set()
    out: list[tuple[int, str]] = []
    for combo in itertools.product(*slots):
        v = "".join(combo)
        if v in seen:
            continue
        seen.add(v)
        out.append((sum(a != b for a, b in zip(v, low)), v))
        if len(out) >= max_variants:
            break
    # диграф нг -> ҥ даёт вариант другой длины, добавляем отдельно
    if _NG_RE.search(low):
        v = _NG_RE.sub("ҥ", low)
        if v not in seen:
            out.append((1, v))
    out.sort()
    return [v for _, v in out]


# --- случайные опечатки для синтетики ---------------------------------------
def typo(word: str, rng: random.Random) -> str:
    """Одна механическая опечатка: вставка, удаление, замена или перестановка."""
    from .fuzzy import KEYBOARD_NEIGHBOURS
    if len(word) < 3:
        return word
    op = rng.choice(("del", "ins", "sub", "swap"))
    i = rng.randrange(len(word))
    if op == "del":
        return word[:i] + word[i + 1:]
    if op == "swap" and i + 1 < len(word):
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    neigh = list(KEYBOARD_NEIGHBOURS.get(word[i].lower(), ()) or "аеиоу")
    c = rng.choice(neigh)
    if op == "ins":
        return word[:i] + c + word[i:]
    return word[:i] + c + word[i + 1:]
