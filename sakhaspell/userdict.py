"""Пользовательский словарь.

В остатке ложных срабатываний почти всё — имена, топонимы и термины: `лидочка`,
`калибулин`, `скайраннинг`, `параолимпийскай`. Ни корпус, ни правила грамматики
их не выучат: имён бесконечно много, и появляются новые. Единственное, что
здесь работает, — дать человеку сказать «это слово правильное».

Хранится простым текстом, по слову в строке, чтобы файл можно было открыть и
поправить руками или положить в репозиторий команды. Пустые строки и строки,
начинающиеся с #, игнорируются.

Кроме списка «принимать», есть список «всегда исправлять»: пара «неверно →
верно», применяемая до всех прочих проверок. Она нужна там, где организация
договорилась о написании термина и хочет его выдерживать.
"""
from __future__ import annotations

import os
import pathlib
from dataclasses import dataclass, field


def default_path() -> pathlib.Path:
    """Обычное место словаря: XDG или домашний каталог."""
    env = os.getenv("SAKHASPELL_USERDICT")
    if env:
        return pathlib.Path(env).expanduser()
    base = os.getenv("XDG_CONFIG_HOME")
    root = pathlib.Path(base).expanduser() if base else pathlib.Path.home() / ".config"
    return root / "sakhaspell" / "userdict.txt"


@dataclass
class UserDict:
    accept: set[str] = field(default_factory=set)
    replace: dict[str, str] = field(default_factory=dict)
    path: pathlib.Path | None = None

    # --- загрузка и сохранение ----------------------------------------------
    @classmethod
    def load(cls, path: str | pathlib.Path | None = None) -> "UserDict":
        p = pathlib.Path(path) if path is not None else default_path()
        d = cls(path=p)
        if not p.exists():
            return d
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "->" in line:
                a, b = line.split("->", 1)
                a, b = a.strip().lower(), b.strip()
                if a and b:
                    d.replace[a] = b
            else:
                d.accept.add(line.lower())
        return d

    def save(self, path: str | pathlib.Path | None = None) -> pathlib.Path:
        p = pathlib.Path(path) if path is not None else (self.path or default_path())
        p.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Пользовательский словарь sakhaspell.",
                 "# Слово в строке — принимать как правильное.",
                 "# «неверно -> верно» — всегда исправлять.",
                 ""]
        lines += sorted(self.accept)
        if self.replace:
            lines.append("")
            lines += [f"{a} -> {b}" for a, b in sorted(self.replace.items())]
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.path = p
        return p

    # --- изменение ----------------------------------------------------------
    def add(self, word: str) -> None:
        self.accept.add(word.strip().lower())

    def remove(self, word: str) -> None:
        self.accept.discard(word.strip().lower())

    def add_replacement(self, wrong: str, right: str) -> None:
        self.replace[wrong.strip().lower()] = right.strip()

    # --- запросы ------------------------------------------------------------
    def accepts(self, word: str) -> bool:
        w = word.lower()
        if w in self.replace:      # слово, которое велено исправлять, не принимаем
            return False
        if w in self.accept:
            return True
        # Дефисное сложение принимается, если приняты все части: пользователь
        # добавил «скайраннинг», и «скайраннинг-клуб» тоже должен пройти.
        if "-" in w:
            parts = [p for p in w.split("-") if p]
            return len(parts) > 1 and all(p in self.accept for p in parts)
        return False

    def correction(self, word: str) -> str | None:
        return self.replace.get(word.lower())

    def __len__(self) -> int:
        return len(self.accept) + len(self.replace)

    def __bool__(self) -> bool:
        return bool(self.accept or self.replace)
