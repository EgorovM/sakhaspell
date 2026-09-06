"""Токенизация якутского текста с сохранением офсетов.

Офсеты обязательны: спелчекер должен вернуть редактору диапазон для подчёркивания,
а не только исправленное слово. Всё остальное — производное от этого.

Особенности якутского, которые здесь учтены:
  * дефисные сложения продуктивны и очень часты («көрөн-истэн», «оҕо-уруу»), поэтому
    дефис внутри слова не разрывает токен, но части доступны отдельно;
  * ⟨дь⟩ и ⟨нь⟩ — диграфы, но в письме это две обычные буквы, спецобработки не требуют;
  * буквы ҕ ҥ ө һ ү сами по себе достаточны, чтобы отличить якутское слово от русского.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

ALPHABET = "абвгҕдеёжзийклмнҥоөпрсһтуүфхцчшщъыьэюя"
SAKHA_ONLY = "ҕҥөһү"
VOWELS = "аеёиоөуүыэюя"
CONSONANTS = "бвгҕджзйклмнҥпрсһтфхцчшщ"

# Кириллица целиком, а не только якутский алфавит: слово с русской буквой — это
# тоже слово, которое чекер обязан увидеть и оценить, а не молча пропустить.
_WORD_CHARS = r"Ѐ-ӿԀ-ԯ"
_WORD_RE = re.compile(
    rf"[{_WORD_CHARS}]+(?:[-'’][{_WORD_CHARS}]+)*"
)
# Латиница и цифры — отдельными классами, чтобы чекер их пропускал осознанно.
_LATIN_RE = re.compile(r"[A-Za-z]+(?:[-'’][A-Za-z]+)*")
_NUM_RE = re.compile(r"\d+(?:[.,:/-]\d+)*")


@dataclass(frozen=True, slots=True)
class Token:
    text: str
    start: int
    end: int
    kind: str  # word | latin | number | other

    @property
    def is_word(self) -> bool:
        return self.kind == "word"

    def norm(self) -> str:
        return self.text.lower().replace("’", "'")


# Числовая форма с якутским аффиксом — законная письменная норма («1990-с сыл»,
# «5-тэн», «2-с»). Без этого правила число и аффикс распадаются на два токена,
# аффикс попадает в проверку как отдельное слово и даёт ложное срабатывание.
_MASTER_RE = re.compile(
    rf"(?P<word>[{_WORD_CHARS}]+(?:[-'’][{_WORD_CHARS}]+)*)"
    rf"|(?P<latin>[A-Za-z]+(?:[-'’][A-Za-z]+)*)"
    rf"|(?P<number>\d+(?:[.,:/-]\d+)*(?:-[{_WORD_CHARS}]+)*)"
)


def tokenize(text: str) -> list[Token]:
    """Все токены-кандидаты с офсетами в исходной строке."""
    out: list[Token] = []
    for m in _MASTER_RE.finditer(text):
        kind = m.lastgroup or "other"
        out.append(Token(m.group(), m.start(), m.end(), kind))
    return out


def words(text: str) -> list[Token]:
    return [t for t in tokenize(text) if t.is_word]


# --- предложения ------------------------------------------------------------
# Сокращения, после которых точка не заканчивает предложение. Список якутский:
# «б.а.» — быһата аата, «о.д.а.» — уонна да атын, плюс ходовые русские.
_ABBREV = {
    "б.а", "о.д.а", "с", "г", "гг", "в", "вв", "стр", "т", "тт", "и.т.д", "и.т.б",
    "проф", "доц", "акад", "обл", "р", "оз", "им", "тыс", "млн", "млрд", "руб",
}
_SENT_END = re.compile(r"[.!?…]+[\"»')\]]*\s+")
_INITIAL = re.compile(r"\b[А-ЯЁӨҮҔҤҺ]\.\s*$")


def sentences(text: str) -> list[tuple[int, int]]:
    """Границы предложений как (start, end). Консервативно: лучше склеить два
    предложения, чем разрезать одно посреди сокращения — для сбора корпуса
    словоформ склейка безвредна, а разрез плодит обрубки."""
    spans: list[tuple[int, int]] = []
    start = 0
    for m in _SENT_END.finditer(text):
        head = text[start:m.start() + 1]
        tail = head.rstrip(".!?…\"»')]").rsplit(None, 1)
        last = tail[-1].lower().rstrip(".") if tail else ""
        if last in _ABBREV or _INITIAL.search(head):
            continue
        end = m.start() + 1
        if end > start:
            spans.append((start, end))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return [(a, b) for a, b in spans if text[a:b].strip()]


def sakha_letter_share(text: str) -> float:
    """Доля букв ҕҥөһү среди всех букв. В живом якутском тексте это ~7.6%."""
    letters = [c for c in text.lower() if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c in SAKHA_ONLY for c in letters) / len(letters)


def alphabet_share(text: str) -> float:
    """Доля букв, входящих в якутский алфавит, среди всех букв."""
    letters = [c for c in text.lower() if c.isalpha()]
    if not letters:
        return 0.0
    return sum(c in ALPHABET for c in letters) / len(letters)
