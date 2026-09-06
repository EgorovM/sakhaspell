"""Абляции акцептора на отложенном наборе.

Каждое правило стоит либо ложных срабатываний, либо пропущенных ошибок. Чтобы
решать осознанно, меряем вклад каждого по отдельности на редактируемом тексте
(там всё, что не принято, — ложное срабатывание).
"""
import argparse, collections, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sakhaspell.lexicon import Lexicon
from sakhaspell.tokenize import tokenize

CONFIGS = [
    ("только ядро",              dict(use_tail=False, use_hyphen=False)),
    ("+ дефисные сложения",      dict(use_tail=False, use_hyphen=True)),
    ("+ подтверждённый хвост",   dict(use_tail=True,  use_hyphen=True)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", type=pathlib.Path, required=True)
    args = ap.parse_args()

    lex = Lexicon.load(args.lexicon)
    print(f"ядро {len(lex.core)}, хвост {len(lex.tail)}, подтверждающих {len(lex.other)}\n")

    held = collections.defaultdict(list)
    with (args.lexicon / "heldout.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            held[d["tier"]].append(d["text"])

    print(f"{'конфигурация':28s} " + "  ".join(f"{t:>10s}" for t in ("edited", "ocr", "web")))
    results = {}
    for name, kw in CONFIGS:
        row = {}
        for tier in ("edited", "ocr", "web"):
            tok = bad = 0
            for s in held[tier]:
                for t in tokenize(s):
                    if not t.is_word:
                        continue
                    tok += 1
                    if not lex.check_form(t.text, **kw).ok:
                        bad += 1
            row[tier] = bad / max(tok, 1)
        results[name] = row
        print(f"{name:28s} " + "  ".join(f"{row[t]:9.2%}" for t in ("edited", "ocr", "web")))

    print("\nдоля непринятых токенов; на редактируемом тексте это ложные срабатывания")
    (args.lexicon / "acceptor_ablation.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
