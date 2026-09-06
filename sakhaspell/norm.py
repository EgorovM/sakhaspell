"""Слой L0: нормализация письма.

Разбирается с тем, что ломает любую последующую проверку ещё до неё: невидимые
символы, чужие кириллические буквы, похожие на якутские, и латиница внутри
кириллического слова.

Состав карты замен получен из инвентаря 1.65 млрд символов корпуса
(`data/char_inventory.tsv`), а не из алфавита. Что реально встречается:

    ѳ U+0473 фита                27 513   ← вместо ө
    ң U+04A3 казахское эн        16 158   ← вместо ҥ
    ӊ U+04CA эн с хвостом         5 591   ← вместо ҥ
    ҋ U+048B и краткое с хвостом     774   ← вместо й
    ұ U+04B1 казахское у             656   ← вместо ү
    ғ U+0493 казахское гha           142   ← вместо ҕ
    ӧ U+04E7 о с диерезисом           24   ← вместо ө

Все замены один-в-один по длине, поэтому офсеты токенов не разъезжаются. Удаление
невидимых символов длину меняет, для него есть отдельная функция с картой позиций.
"""
from __future__ import annotations

import unicodedata

# --- чужие кириллические буквы, которые в якутском тексте всегда ошибка ------
CONFUSABLES: dict[str, str] = {
    "ѳ": "ө", "Ѳ": "Ө",   # ѳ Ѳ фита
    "ӧ": "ө", "Ӧ": "Ө",   # ӧ Ӧ о с диерезисом
    "ң": "ҥ", "Ң": "Ҥ",   # ң Ң казахское эн с descender
    "ӊ": "ҥ", "Ӊ": "Ҥ",   # ӊ Ӊ эн с хвостом
    "ғ": "ҕ", "Ғ": "Ҕ",   # ғ Ғ казахское гha со штрихом
    "ұ": "ү", "Ұ": "Ү",   # ұ Ұ казахское у со штрихом
    "ҋ": "й", "Ҋ": "Й",   # ҋ Ҋ и краткое с хвостом
}

# --- невидимое ---------------------------------------------------------------
INVISIBLE = {
    "­",  # мягкий перенос — массово приезжает из вёрстки и из OCR
    "﻿", "​", "‌", "‍", "⁠",
}

# --- латинские гомоглифы -----------------------------------------------------
# Применяются ТОЛЬКО внутри слова, которое в остальном кириллическое: изолированно
# «a» может быть законной латинской буквой, а «Ampere» переписывать нельзя.
LATIN_TO_CYR: dict[str, str] = {
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
    "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
    # h и H в якутском письме без раскладки заменяют һ, но это уже не гомоглиф,
    # а орфографическая ошибка — ею занимается модель ошибок, не нормализация.
}

_CYRILLIC_RANGES = ((0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F))


def is_cyrillic(ch: str) -> bool:
    o = ord(ch)
    return any(a <= o <= b for a, b in _CYRILLIC_RANGES)


def fix_confusables(text: str) -> str:
    """Чужие кириллические буквы → якутские. Длина сохраняется."""
    if not any(c in CONFUSABLES for c in text):
        return text
    return "".join(CONFUSABLES.get(c, c) for c in text)


def strip_invisible(text: str) -> tuple[str, list[int]]:
    """Убирает невидимые символы. Возвращает текст и карту: для каждой позиции
    результата — позиция в исходной строке."""
    out, idx = [], []
    for i, ch in enumerate(text):
        if ch in INVISIBLE:
            continue
        out.append(ch)
        idx.append(i)
    return "".join(out), idx


def fix_mixed_script(word: str, *, min_cyr: float = 0.5) -> str:
    """Латинские гомоглифы внутри кириллического слова → кириллица.

    Срабатывает, только если слово уже преимущественно кириллическое и каждая
    латинская буква в нём имеет гомоглиф. «Cаха» с латинской C чинится, «CD-ROM»
    остаётся как есть.
    """
    letters = [c for c in word if c.isalpha()]
    if not letters:
        return word
    cyr = sum(is_cyrillic(c) for c in letters)
    lat = [c for c in letters if not is_cyrillic(c)]
    if not lat or cyr / len(letters) < min_cyr:
        return word
    if not all(c in LATIN_TO_CYR for c in lat):
        return word
    return "".join(LATIN_TO_CYR.get(c, c) for c in word)


def normalize(text: str, *, mixed_script: bool = True) -> str:
    """Полная нормализация L0. Порядок важен: NFC → невидимое → конфузаблы →
    смешанный регистр письма."""
    text = unicodedata.normalize("NFC", text)
    text, _ = strip_invisible(text)
    text = fix_confusables(text)
    if mixed_script:
        # по словам, чтобы решение о латинице принималось в границах слова
        from .tokenize import tokenize
        toks = tokenize(text)
        if any(t.kind == "latin" or t.kind == "word" for t in toks):
            parts, prev = [], 0
            for t in toks:
                if t.kind not in ("word", "latin"):
                    continue
                fixed = fix_mixed_script(t.text)
                if fixed != t.text:
                    parts.append(text[prev:t.start])
                    parts.append(fixed)
                    prev = t.end
            if parts:
                parts.append(text[prev:])
                text = "".join(parts)
    return text


def normalize_for_lexicon(text: str) -> str:
    """Нормализация под сбор словоформ: то же самое плюс схлопывание пробелов."""
    return " ".join(normalize(text).split())
