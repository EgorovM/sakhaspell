"""Слой L2: посимвольный тэггер восстановления якутских букв.

Почему именно так, а не seq2seq.

Разбор потолка (scripts/analyze_ceiling.py) показал: 41% ошибок деноминализации
невидимы для пословной проверки — испорченное слово само является законным
(«киси», «сана», «ого»). Значит нужен контекст. Но переписывать текст целиком
seq2seq-моделью опасно: она меняет и то, о чём её не просили, а обзор
(«GECToR: Tag, Not Rewrite», «Pillars of GEC») прямо рекомендует для продукта
неавторегрессивную разметку.

Задача восстановления ҕҥөһү по своей природе посимвольная: для каждой буквы «г»
надо решить, это «г» или «ҕ». Это диакритизация, а для неё разметка — точное
соответствие задаче, а не компромисс. Модель физически не может изменить символ,
которому предсказала KEEP.

Метки:
    KEEP    оставить символ как есть
    ҕ ҥ ө ү һ   заменить символ на якутскую букву
    DELETE  удалить символ (нужно для диграфа «нг» → «ҥ»: н получает метку ҥ,
            г получает DELETE)

Модель — трансформерный энкодер над символами. Он маленький (около 11 млн
параметров) и работает на CPU, потому что словарь символов — сотни, а не десятки
тысяч, и глубина не нужна: зависимости в этой задаче локальные, в пределах слова
и его соседей.
"""
from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- метки -------------------------------------------------------------------
KEEP, DELETE = 0, 1
TARGETS = ("ҕ", "ҥ", "ө", "ү", "һ")
TAG_TO_CHAR = {i + 2: c for i, c in enumerate(TARGETS)}
CHAR_TO_TAG = {c: i + 2 for i, c in enumerate(TARGETS)}
N_TAGS = 2 + len(TARGETS)

# Символы, которые вообще могут оказаться искажённой якутской буквой. Всё
# остальное получает KEEP всегда, и на нём не считается ни лосс, ни метрика:
# иначе 97% позиций тривиальны и метрика ничего не показывает.
AMBIGUOUS = set("гноусхьцgnhoy")

PAD, UNK, BOS, EOS = 0, 1, 2, 3


class CharVocab:
    """Словарь символов. Строится по корпусу, редкие символы уходят в UNK."""

    def __init__(self, chars: list[str]) -> None:
        self.itos = ["<pad>", "<unk>", "<bos>", "<eos>"] + chars
        self.stoi = {c: i for i, c in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, UNK) for c in text]

    @classmethod
    def from_counts(cls, counts: dict[str, int], min_freq: int = 50) -> "CharVocab":
        chars = sorted((c for c, n in counts.items() if n >= min_freq),
                       key=lambda c: -counts[c])
        return cls(chars)

    def save(self, path) -> None:
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos[4:], f, ensure_ascii=False)

    @classmethod
    def load(cls, path) -> "CharVocab":
        import json
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))


# --- разметка пары (испорченный, эталонный) ----------------------------------
def make_tags(src: str, tgt: str) -> list[int] | None:
    """Метки для src так, чтобы их применение дало tgt.

    Возвращает None, если пара несовместима с набором меток (например, порча
    сделала что-то кроме деноминализации) — такие пары в обучение не идут.
    """
    tags: list[int] = []
    i = j = 0
    while i < len(src) and j < len(tgt):
        a, b = src[i], tgt[j]
        if a == b:
            tags.append(KEEP)
            i += 1
            j += 1
            continue
        # Диграф «нг» -> «ҥ» проверяется раньше обычной замены: иначе «н»->«ҥ»
        # съест один символ, «г» останется без пары и выравнивание развалится.
        if (b.lower() == "ҥ" and a.lower() == "н" and i + 1 < len(src)
                and src[i + 1].lower() == "г"
                and not (j + 1 < len(tgt) and tgt[j + 1].lower() == "г")):
            tags.append(CHAR_TO_TAG["ҥ"])
            tags.append(DELETE)
            i += 2
            j += 1
            continue
        t = CHAR_TO_TAG.get(b.lower())
        if t is not None and _compatible(a, b):
            tags.append(t)
            i += 1
            j += 1
            continue
        return None
    if i != len(src) or j != len(tgt):
        return None
    return tags


_COMPAT = {
    "ҕ": set("гg"), "ҥ": set("нnц"), "ө": set("оo"),
    "ү": set("уy"), "һ": set("схьh"),
}


def _compatible(src_ch: str, tgt_ch: str) -> bool:
    return src_ch.lower() in _COMPAT.get(tgt_ch.lower(), ())


def apply_tags(src: str, tags: list[int]) -> str:
    """Применяет метки к строке. Регистр исходного символа сохраняется."""
    out = []
    for ch, t in zip(src, tags):
        if t == KEEP:
            out.append(ch)
        elif t == DELETE:
            continue
        else:
            rep = TAG_TO_CHAR[t]
            out.append(rep.upper() if ch.isupper() else rep)
    if len(tags) < len(src):
        out.append(src[len(tags):])
    return "".join(out)


# --- модель ------------------------------------------------------------------
@dataclass
class TaggerConfig:
    vocab_size: int = 256
    d_model: int = 384
    n_heads: int = 6
    n_layers: int = 6
    d_ff: int = 1536
    dropout: float = 0.1
    max_len: int = 512


class CharTagger(nn.Module):
    def __init__(self, cfg: TaggerConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=PAD)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout, batch_first=True, norm_first=True,
            activation="gelu")
        self.enc = nn.TransformerEncoder(layer, cfg.n_layers,
                                         norm=nn.LayerNorm(cfg.d_model))
        self.head = nn.Linear(cfg.d_model, N_TAGS)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n = x.shape
        pos = torch.arange(n, device=x.device).unsqueeze(0).expand(b, n)
        h = self.drop(self.emb(x) + self.pos(pos))
        h = self.enc(h, src_key_padding_mask=x.eq(PAD))
        return self.head(h)

    @torch.no_grad()
    def tag(self, texts: list[str], vocab: CharVocab, device: str = "cpu",
            max_len: int = 512, threshold: float = 0.5,
            batch_size: int = 64) -> list[str]:
        """Разметка с порогом уверенности.

        Порог несимметричен и применяется только к правкам: заменить символ
        модель может лишь при вероятности выше threshold, оставить — всегда.
        Асимметрия намеренная. Пропущенная ошибка стоит пользователю ничего, а
        правка в правильном слове стоит доверия ко всему инструменту, поэтому
        цена ошибок разная и порог должен это отражать.
        """
        self.eval()
        out = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            n = min(max(len(t) for t in chunk), max_len)
            ids = torch.full((len(chunk), n), PAD, dtype=torch.long)
            for j, t in enumerate(chunk):
                e = vocab.encode(t[:n])
                ids[j, :len(e)] = torch.tensor(e)
            probs = self(ids.to(device)).softmax(-1).cpu()
            conf, pred = probs.max(-1)
            pred = torch.where(conf >= threshold, pred,
                               torch.full_like(pred, KEEP))
            for j, t in enumerate(chunk):
                head = t[:n]
                out.append(apply_tags(head, pred[j, :len(head)].tolist()) + t[n:])
        return out
