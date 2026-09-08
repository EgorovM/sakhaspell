"""Модель слоя L2: посимвольный тэггер.

Требует torch — ставится как `pip install "sakhaspell[tagger]"`. Разметка,
которую модель предсказывает, и её применение к строке живут в `tagger.py`
и torch не требуют.

Почему разметка, а не порождение. Разбор потолка (scripts/analyze_ceiling.py)
показал: часть ошибок деноминализации невидима для пословной проверки —
испорченное слово само является законным. Значит нужен контекст. Но переписывать
текст целиком seq2seq-моделью опасно: она меняет и то, о чём её не просили,
а обзоры («GECToR: Tag, Not Rewrite», «Pillars of GEC») прямо рекомендуют для
продукта неавторегрессивную разметку. Здесь модель физически не может изменить
символ, которому предсказала KEEP.

Модель маленькая, около 11 млн параметров, и работает на CPU: словарь символов —
сотни, а не десятки тысяч, и зависимости в этой задаче локальные, в пределах
слова и его соседей.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .tagger import KEEP, N_TAGS, PAD, CharVocab, apply_tags

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
