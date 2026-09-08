"""Слой L1: акцептор словоформ.

Первичный акцептор спелчекера. Устройство продиктовано разбором непокрытых форм
на отложенном наборе (см. docs/experiments.md):

  * дефисные сложения продуктивны и часты («сайыҥҥыттан-күһүҥҥүттэн»,
    «оҕустарар-кырбаттарар») — их надо принимать по частям, иначе каждое второе
    попадёт в ошибки;
  * числовые формы с аффиксами («1990-с», «5-тэн») — законная письменная норма;
  * хвост частот (форма встречена 1-2 раза в редактируемых источниках) наполовину
    состоит из настоящих редких форм. Подтверждение из OCR-источников отделяет
    настоящее слово от опечатки: опечатка не повторяется в другом корпусе.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field

from .grammar import harmony_violations
from .norm import normalize
from .tokenize import Token, tokenize

# Форма из хвоста принимается, если её независимо подтверждают прочие источники.
CORROBORATION = 3


@dataclass
class Verdict:
    ok: bool
    reason: str
    freq: int = 0


@dataclass
class Lexicon:
    core: dict[str, int] = field(default_factory=dict)
    tail: dict[str, int] = field(default_factory=dict)
    other: dict[str, int] = field(default_factory=dict)
    # тени деноминализации: форма -> правильное написание. Заполняется из
    # shadows.tsv (scripts/find_shadows.py). Эти формы удалены из ядра, потому
    # что они не слова, а систематическая ошибка письма без якутской раскладки.
    shadow: dict[str, str] = field(default_factory=dict)

    # --- загрузка -----------------------------------------------------------
    @classmethod
    def load(cls, path: str | pathlib.Path | None = None, *,
             with_tail: bool = True, with_shadows: bool = True) -> "Lexicon":
        """Загружает лексикон. Без аргумента берётся встроенный в пакет."""
        path = pathlib.Path(path) if path is not None else bundled_path()
        lex = cls()
        lex.core = _read(path / "core.tsv", value_col=1)
        if with_tail:
            lex.tail = _read(path / "tail.tsv", value_col=1)
            # Подтверждающие частоты берутся из самого tail.tsv (столбец freq_other).
            # only_other.tsv сюда не грузится намеренно: он содержит формы, которых
            # нет в редактируемых источниках вообще, а `other` спрашивают только для
            # форм из `tail`. Пересечение этих множеств пусто по построению, так что
            # его 881 тысяча записей и 22 МБ не влияли ни на один вердикт.
            lex.other = _read(path / "tail.tsv", value_col=3)
        sp = _resolve(path / "shadows.tsv")
        if with_shadows and sp is not None:
            with _open(sp) as f:
                header = next(f).rstrip("\n").split("\t")
                i_orig = header.index("origin")
                i_demote = header.index("demote")
                for line in f:
                    c = line.rstrip("\n").split("\t")
                    if len(c) > i_demote and c[i_demote] == "1":
                        lex.shadow[c[0]] = c[i_orig]
            # тень не должна приниматься ни ядром, ни хвостом
            for w in lex.shadow:
                lex.core.pop(w, None)
                lex.tail.pop(w, None)
        return lex

    # --- проверка -----------------------------------------------------------
    def check_form(self, w: str, *, use_tail: bool = True, use_hyphen: bool = True,
                   use_grammar: bool = False, depth: int = 0) -> Verdict:
        w = w.lower()
        if w in self.shadow:
            return Verdict(False, "shadow")
        if w in self.core:
            return Verdict(True, "core", self.core[w])

        if use_hyphen and depth == 0 and ("-" in w or "'" in w or "’" in w):
            parts = [p for p in w.replace("’", "'").replace("'", "-").split("-") if p]
            if len(parts) > 1 and all(
                    self.check_form(p, use_tail=use_tail, use_hyphen=False,
                                    use_grammar=use_grammar, depth=1).ok
                    for p in parts):
                return Verdict(True, "hyphen-parts")

        if use_tail and w in self.tail:
            if self.other.get(w, 0) >= CORROBORATION:
                # Подтверждение другими источниками спасает редкое слово, но
                # заодно спасает и систематическую ошибку. Нарушение гармонии —
                # независимый признак, и он даёт +1.45 пункта F1 на задаче `real`.
                #
                # По умолчанию выключено, потому что обмен не бесплатный:
                # ложные срабатывания на редакционном тексте растут с 1.684% до
                # 1.724%. Дополнительно отвергаются 6 320 форм хвоста, и это в
                # основном законные русские имена с якутскими аффиксами
                # («лидочка», «иисустан», «синагогаларга»), а не ошибки —
                # гармонию они нарушают по праву заимствования. Включать стоит
                # там, где имён мало: в художественном тексте, в олонхо.
                if use_grammar and harmony_violations(w):
                    return Verdict(False, "tail-disharmonic", self.tail[w])
                return Verdict(True, "tail-corroborated", self.tail[w])
            return Verdict(False, "tail-unconfirmed", self.tail[w])

        return Verdict(False, "unknown")

    def check_token(self, t: Token, **kw) -> Verdict:  # noqa: D102
        if t.kind == "number":
            return Verdict(True, "number")
        if t.kind == "latin":
            return Verdict(True, "latin-skip")
        if t.kind != "word":
            return Verdict(True, "other-skip")
        return self.check_form(t.text, **kw)

    def check_text(self, text: str, **kw) -> list[tuple[Token, Verdict]]:
        text = normalize(text)
        return [(t, self.check_token(t, **kw)) for t in tokenize(text)]

    def flagged(self, text: str, **kw) -> list[Token]:
        return [t for t, v in self.check_text(text, **kw) if not v.ok]


def bundled_path() -> pathlib.Path:
    """Каталог со словарём, который едет вместе с пакетом.

    Словарь лежит сжатым (7.5 МБ на 300 тысяч форм с хвостом), поэтому
    `pip install sakhaspell` даёт рабочий чекер без отдельной загрузки данных.
    """
    return pathlib.Path(__file__).resolve().parent / "data"


def _resolve(path: pathlib.Path) -> pathlib.Path | None:
    """Файл как есть или его gzip-версия. В репозитории лежит сжатое."""
    if path.exists():
        return path
    gz = path.with_suffix(path.suffix + ".gz")
    return gz if gz.exists() else None


def _open(path: pathlib.Path):
    if path.suffix == ".gz":
        import gzip
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8")


def _read(path: pathlib.Path, *, value_col: int) -> dict[str, int]:
    out: dict[str, int] = {}
    resolved = _resolve(path)
    if resolved is None:
        return out
    with _open(resolved) as f:
        next(f)
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > value_col:
                try:
                    v = int(parts[value_col])
                except ValueError:
                    continue
                if v:
                    out[parts[0]] = v
    return out
