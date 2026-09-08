"""Полный конвейер: L0 нормализация → L2 тэггер → L1 лексикон.

Порядок слоёв не произвольный.

Тэггер идёт перед лексиконом, а не после, потому что он решает задачу, которую
лексикон решить не может в принципе: 41% ошибок деноминализации — это слова,
которые сами по себе законны, и пословная проверка их не видит (E6). Тэггер
смотрит на контекст и восстанавливает буквы; после него лексикону остаются
настоящие опечатки, с которыми он справляется поиском по расстоянию.

Тэггер необязателен. Без него работает только L1 — медленнее по качеству, но
без GPU и без модели, одним файлом словаря. Это и есть обещанный офлайн-режим.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

from .checker import Issue, SpellChecker, Suggestion
from .lexicon import Lexicon
from .norm import normalize
from .tokenize import Token, tokenize


@dataclass
class Correction:
    """Одна правка с позицией в исходном тексте."""
    start: int
    end: int
    before: str
    after: str
    source: str        # tagger | lexicon
    alternatives: list[str]


class Pipeline:
    def __init__(self, lexicon_dir: str | pathlib.Path | None = None, *,
                 tagger_dir: str | pathlib.Path | None = None,
                 device: str = "cpu", max_suggestions: int = 5,
                 tagger_threshold: float = 0.9) -> None:
        self.lex = Lexicon.load(lexicon_dir)
        self.checker = SpellChecker(self.lex, max_suggestions=max_suggestions)
        self.tagger = None
        self.vocab = None
        self.device = device
        # Порог уверенности тэггера. По умолчанию высокий: правка в правильном
        # слове обходится дороже пропущенной ошибки (см. docs/experiments.md).
        self.tagger_threshold = tagger_threshold
        if tagger_dir:
            self._load_tagger(pathlib.Path(tagger_dir), device)

    def _load_tagger(self, path: pathlib.Path, device: str) -> None:
        import torch
        from .model import CharTagger, TaggerConfig
        from .tagger import CharVocab

        ckpt = torch.load(path / "best.pt", map_location=device, weights_only=False)
        cfg = TaggerConfig(**ckpt["cfg"])
        model = CharTagger(cfg)
        model.load_state_dict(ckpt["model"])
        model.to(device).eval()
        self.tagger = model
        self.vocab = CharVocab.load(path / "vocab.json")

    # --- слои ---------------------------------------------------------------
    def restore(self, text: str) -> str:
        """Только L2: восстановление ҕҥөһү по контексту."""
        if self.tagger is None:
            return text
        return self.tagger.tag([text], self.vocab, self.device,
                               max_len=self.tagger.cfg.max_len - 8,
                               threshold=self.tagger_threshold)[0]

    def check(self, text: str) -> list[Issue]:
        """Только L1: пословная проверка."""
        return self.checker.check(text)

    # --- всё вместе ---------------------------------------------------------
    def correct(self, text: str, *, use_tagger: bool = True) -> str:
        text = normalize(text)
        if use_tagger and self.tagger is not None:
            text = self.restore(text)
        return self.checker.correct(text)

    def corrections(self, text: str, *, use_tagger: bool = True) -> list[Correction]:
        """Правки с позициями в ИСХОДНОМ тексте — для подсветки в редакторе.

        Тэггер работает посимвольно и меняет длину строки только на диграфе
        «нг»→«ҥ», поэтому позиции считаем по словам, а не по символам: слово,
        которое тэггер изменил, сопоставляется исходному по порядку.
        """
        src = normalize(text)
        out: list[Correction] = []

        staged = src
        if use_tagger and self.tagger is not None:
            staged = self.restore(src)
            a = [t for t in tokenize(src) if t.is_word]
            b = [t for t in tokenize(staged) if t.is_word]
            if len(a) == len(b):
                for ta, tb in zip(a, b):
                    if ta.text != tb.text:
                        out.append(Correction(ta.start, ta.end, ta.text, tb.text,
                                              "tagger", []))
            else:
                staged = src        # выравнивание не сошлось — тэггер не применяем

        # лексикон работает по тексту после тэггера, но позиции нужны исходные
        a = [t for t in tokenize(src) if t.is_word]
        b = [t for t in tokenize(staged) if t.is_word]
        by_index = {i: t for i, t in enumerate(a)} if len(a) == len(b) else {}
        applied = {c.start for c in out}
        for i, tb in enumerate(b):
            v = self.lex.check_token(tb)
            if v.ok:
                continue
            sug = self.checker.suggest(tb.text)
            if not sug:
                continue
            ta = by_index.get(i, tb)
            if ta.start in applied:
                # тэггер уже правил это слово; лексикон уточняет результат
                for c in out:
                    if c.start == ta.start:
                        c.after = sug[0].form
                        c.alternatives = [s.form for s in sug[1:]]
                        c.source = "tagger+lexicon"
                continue
            out.append(Correction(ta.start, ta.end, ta.text, sug[0].form,
                                  "lexicon", [s.form for s in sug[1:]]))
        out.sort(key=lambda c: c.start)
        return out
