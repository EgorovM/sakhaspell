"""Абляция книжных правил на SakhaSpellBench.

Три способа применить гармонию гласных, и у каждого своя цена:

  ранжирование   штраф кандидату, нарушающему гармонию;
  хвост          форма из хвоста, нарушающая гармонию, не принимается;
  вне словаря    если само слово нарушает гармонию, а вариант восстановления
                 нет, предложить его, даже не найдя в лексиконе.

Меряем каждый по отдельности и все вместе. Смотреть надо не только на F1:
правило, поднявшее полноту ценой ложных срабатываний на чистом тексте, — это
ухудшение, как бы ни выглядел F1.

    /usr/bin/python3 scripts/ablate_grammar.py --lexicon data/lexicon \
        --bench data/bench --split dev --limit 400
"""
import argparse, json, pathlib, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sakhaspell.checker import SpellChecker
from sakhaspell.lexicon import Lexicon
from sakhaspell.norm import normalize
from sakhaspell.tokenize import tokenize

TASKS = ["clean", "denorm_full", "denorm_mixed", "real", "typo", "mixed"]

CONFIGS = {
    "без правил":      dict(grammar=False, tail_grammar=False),
    "ранжирование":    dict(grammar=True,  tail_grammar=False, out_of_lex=False),
    "хвост":           dict(grammar=False, tail_grammar=True),
    "вне словаря":     dict(grammar=True,  tail_grammar=False, out_of_lex=True),
    "всё вместе":      dict(grammar=True,  tail_grammar=True,  out_of_lex=True),
}


def evaluate(ch: SpellChecker, path: pathlib.Path, *, tail_grammar: bool,
             limit: int) -> dict:
    tp = fp = fn = det_tp = det_fn = fp_tok = clean_tok = 0
    n = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            src, tgt = normalize(d["src"]), normalize(d["tgt"])
            a = [t.text for t in tokenize(src) if t.is_word]
            b = [t.text for t in tokenize(tgt) if t.is_word]
            if len(a) != len(b):
                continue
            n += 1
            gold = {i: b[i] for i in range(len(a)) if a[i].lower() != b[i].lower()}
            clean_tok += len(a) - len(gold)

            idx = {t.start: i for i, t in enumerate(t for t in tokenize(src) if t.is_word)}
            made, flagged = {}, set()
            for t in tokenize(src):
                v = ch.lex.check_token(t, use_grammar=tail_grammar)
                if v.ok:
                    continue
                i = idx.get(t.start)
                if i is None:
                    continue
                flagged.add(i)
                sug = ch.suggest(t.text)
                if sug:
                    made[i] = sug[0].form

            for i, corr in made.items():
                if i in gold and corr.lower() == gold[i].lower():
                    tp += 1
                else:
                    fp += 1
            fn += len(gold) - sum(1 for i in gold
                                  if i in made and made[i].lower() == gold[i].lower())
            det_tp += len(flagged & gold.keys())
            det_fn += len(gold.keys() - flagged)
            fp_tok += len(flagged - gold.keys())
            if limit and n >= limit:
                break

    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9),
            "detection": det_tp / max(det_tp + det_fn, 1),
            "fpr": fp_tok / max(clean_tok, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", type=pathlib.Path, default=None)
    ap.add_argument("--bench", type=pathlib.Path, required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()

    lex = Lexicon.load(args.lexicon)
    results: dict[str, dict] = {}

    for name, cfg in CONFIGS.items():
        ch = SpellChecker(lex, grammar=cfg["grammar"])
        # «вне словаря» — часть ветки grammar в suggest(); когда он не нужен,
        # отключаем его, оставляя штраф в ранжировании.
        ch.out_of_lex = cfg.get("out_of_lex", True)
        t0 = time.perf_counter()
        row = {}
        for task in TASKS:
            p = args.bench / f"{args.split}.{task}.jsonl"
            if not p.exists():
                continue
            row[task] = evaluate(ch, p, tail_grammar=cfg["tail_grammar"],
                                 limit=args.limit)
        row["_sec"] = round(time.perf_counter() - t0, 1)
        results[name] = row

    hdr = [t for t in TASKS if t != "clean"]
    print(f"\nF1 по задачам\n{'конфигурация':16s} " +
          "  ".join(f"{t[:11]:>11s}" for t in hdr) + f" {'FPR чистый':>11s}")
    for name, row in results.items():
        cells = "  ".join(f"{row[t]['f1']:10.2%}" for t in hdr if t in row)
        print(f"{name:16s} {cells} {row['clean']['fpr']:10.2%}")

    print(f"\nДетекция\n{'конфигурация':16s} " +
          "  ".join(f"{t[:11]:>11s}" for t in hdr))
    for name, row in results.items():
        cells = "  ".join(f"{row[t]['detection']:10.2%}" for t in hdr if t in row)
        print(f"{name:16s} {cells}")

    base = results["без правил"]
    print(f"\nИзменение F1 против «без правил»\n{'конфигурация':16s} " +
          "  ".join(f"{t[:11]:>11s}" for t in hdr) + f" {'FPR':>11s}")
    for name, row in results.items():
        if name == "без правил":
            continue
        cells = "  ".join(f"{row[t]['f1']-base[t]['f1']:+10.2%}" for t in hdr if t in row)
        print(f"{name:16s} {cells} {row['clean']['fpr']-base['clean']['fpr']:+10.2%}")

    (args.bench / f"grammar_ablation.{args.split}.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
