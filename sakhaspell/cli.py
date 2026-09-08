"""Командная строка спелчекера.

    sakhaspell check "Ого уорэгэ кисиэхэ сана кыагы биэрэр"
    sakhaspell fix --file статья.txt --in-place
    sakhaspell repl

Словарь встроен в пакет, указывать его не нужно. Флаг --lexicon пригодится,
только если собран свой.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

from .pipeline import Pipeline


def _force_utf8() -> None:
    """Вывод в UTF-8 независимо от кодировки консоли.

    На Windows консоль по умолчанию не UTF-8, и печать якутских букв валится
    с UnicodeEncodeError ещё до того, как пользователь увидит хоть одну правку.
    Поймано на прогоне CI под windows-latest.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError):
            pass


def _pipeline(args) -> Pipeline:
    return Pipeline(args.lexicon, tagger_dir=args.tagger, device=args.device)


def _read_input(args) -> str:
    if args.text:
        return " ".join(args.text)
    if args.file:
        return pathlib.Path(args.file).read_text(encoding="utf-8")
    return sys.stdin.read()


def cmd_check(args) -> int:
    p = _pipeline(args)
    text = _read_input(args)
    corr = p.corrections(text, use_tagger=not args.no_tagger)
    if args.json:
        print(json.dumps([{"start": c.start, "end": c.end, "before": c.before,
                           "after": c.after, "source": c.source,
                           "alternatives": c.alternatives} for c in corr],
                         ensure_ascii=False, indent=2))
        return 1 if corr else 0
    if not corr:
        print("ошибок не найдено")
        return 0
    for c in corr:
        line = text.count("\n", 0, c.start) + 1
        alts = f"   ещё: {', '.join(c.alternatives)}" if c.alternatives else ""
        print(f"{line}:{c.start}  {c.before}  →  {c.after}   [{c.source}]{alts}")
    print(f"\nвсего правок: {len(corr)}")
    return 1


def cmd_fix(args) -> int:
    p = _pipeline(args)
    text = _read_input(args)
    out = p.correct(text, use_tagger=not args.no_tagger)
    if args.in_place and args.file:
        pathlib.Path(args.file).write_text(out, encoding="utf-8")
        print(f"записано в {args.file}", file=sys.stderr)
    else:
        sys.stdout.write(out if out.endswith("\n") else out + "\n")
    return 0


def cmd_repl(args) -> int:
    p = _pipeline(args)
    print("ввод — строка текста, выход — Ctrl-D")
    for line in sys.stdin:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        corr = p.corrections(line, use_tagger=not args.no_tagger)
        if not corr:
            print("  ок")
            continue
        print("  " + p.correct(line, use_tagger=not args.no_tagger))
        for c in corr:
            alts = f"  ({', '.join(c.alternatives[:3])})" if c.alternatives else ""
            print(f"    {c.before} → {c.after} [{c.source}]{alts}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="sakhaspell",
                                 description="спелчекер якутского языка")
    ap.add_argument("--lexicon", default=None,
                    help="каталог лексикона; по умолчанию встроенный в пакет")
    ap.add_argument("--tagger", default=None, help="каталог обученного тэггера")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-tagger", action="store_true",
                    help="только словарный слой, без нейросети")
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name, fn, doc in (("check", cmd_check, "показать ошибки"),
                          ("fix", cmd_fix, "исправить текст"),
                          ("repl", cmd_repl, "интерактивный режим")):
        s = sub.add_parser(name, help=doc)
        s.set_defaults(fn=fn)
        if name != "repl":
            s.add_argument("text", nargs="*", help="текст; иначе --file или stdin")
            s.add_argument("--file", help="файл со входным текстом")
        if name == "check":
            s.add_argument("--json", action="store_true")
        if name == "fix":
            s.add_argument("--in-place", action="store_true")

    _force_utf8()
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
