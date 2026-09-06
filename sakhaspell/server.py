"""HTTP-сервис проверки орфографии.

    uvicorn sakhaspell.server:app --host 0.0.0.0 --port 8080

Настройка через переменные окружения:
    SAKHASPELL_LEXICON   каталог лексикона (по умолчанию data/lexicon)
    SAKHASPELL_TAGGER    каталог тэггера; без него работает только словарь
    SAKHASPELL_DEVICE    cpu | cuda
    SAKHASPELL_THRESHOLD порог уверенности тэггера

Замер bashspell — 115–132 мс на слово в `suggest()` — показывает, где у таких
сервисов узкое место. Поэтому здесь: проверка и подсказки разнесены по разным
ручкам, подсказки считаются только для запрошенных слов, а результаты по словам
кэшируются. Редактор должен звать `/check` на каждый ввод и `/suggest` только
когда пользователь открыл меню исправлений.
"""
from __future__ import annotations

import functools
import os
import pathlib
import time

from .pipeline import Pipeline

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as e:  # pragma: no cover
    raise SystemExit("нужен fastapi: pip install fastapi uvicorn") from e


LEXICON = os.getenv("SAKHASPELL_LEXICON", "data/lexicon")
TAGGER = os.getenv("SAKHASPELL_TAGGER") or None
DEVICE = os.getenv("SAKHASPELL_DEVICE", "cpu")
THRESHOLD = float(os.getenv("SAKHASPELL_THRESHOLD", "0.9"))
MAX_CHARS = 20000

app = FastAPI(title="sakhaspell", description="проверка орфографии якутского языка")
_pipe: Pipeline | None = None


def pipe() -> Pipeline:
    global _pipe
    if _pipe is None:
        _pipe = Pipeline(LEXICON, tagger_dir=TAGGER, device=DEVICE,
                         tagger_threshold=THRESHOLD)
    return _pipe


class TextIn(BaseModel):
    text: str = Field(..., description="текст для проверки")
    use_tagger: bool = Field(True, description="применять контекстную модель")


class WordsIn(BaseModel):
    words: list[str] = Field(..., max_length=500)


@app.on_event("startup")
def _warm() -> None:
    t0 = time.perf_counter()
    p = pipe()
    p.corrections("Оҕо үөрэҕэ", use_tagger=p.tagger is not None)
    print(f"готов за {time.perf_counter()-t0:.1f} с: лексикон {len(p.lex.core)}, "
          f"теней {len(p.lex.shadow)}, тэггер {'есть' if p.tagger else 'нет'}")


@app.get("/health")
def health() -> dict:
    p = pipe()
    return {"ok": True, "lexicon": len(p.lex.core), "shadows": len(p.lex.shadow),
            "tagger": bool(p.tagger), "device": DEVICE}


@app.post("/check")
def check(body: TextIn) -> dict:
    """Полная проверка текста с позициями правок."""
    if len(body.text) > MAX_CHARS:
        raise HTTPException(413, f"не длиннее {MAX_CHARS} символов")
    t0 = time.perf_counter()
    p = pipe()
    corr = p.corrections(body.text, use_tagger=body.use_tagger)
    return {
        "corrections": [{"start": c.start, "end": c.end, "before": c.before,
                         "after": c.after, "source": c.source,
                         "alternatives": c.alternatives} for c in corr],
        "corrected": p.correct(body.text, use_tagger=body.use_tagger),
        "ms": round((time.perf_counter() - t0) * 1000, 2),
    }


@app.post("/spell")
def spell(body: WordsIn) -> dict:
    """Только вердикт по словам, без подсказок. Дёшево — для подсветки."""
    p = pipe()
    return {"ok": {w: _spell_one(w) for w in body.words}}


@app.post("/suggest")
def suggest(body: WordsIn) -> dict:
    """Подсказки по словам. Дорого — звать только по требованию."""
    return {"suggestions": {w: _suggest_one(w) for w in body.words}}


@functools.lru_cache(maxsize=100_000)
def _spell_one(word: str) -> bool:
    return pipe().lex.check_form(word).ok


@functools.lru_cache(maxsize=20_000)
def _suggest_one(word: str) -> tuple[str, ...]:
    return tuple(s.form for s in pipe().checker.suggest(word))
