"""Оценка полного конвейера L0+L2+L1 на SakhaSpellBench.

Отличие от eval_bench.py: здесь между нормализацией и словарём работает
посимвольный тэггер. Сравнение с чистым L1 показывает, сколько именно даёт
контекст — и не портит ли он чистый текст, что важнее.

    $E scripts/eval_pipeline.py --lexicon data/lexicon --bench data/bench \
        --tagger runs/tagger_v2 --split dev --device cuda
"""
import argparse, json, pathlib, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sakhaspell.norm import normalize
from sakhaspell.pipeline import Pipeline
from sakhaspell.tokenize import tokenize

TASKS = ["clean", "denorm_full", "denorm_mixed", "real", "typo", "mixed", "asr"]


def words(text: str) -> list[str]:
    return [t.text for t in tokenize(text) if t.is_word]


def evaluate(pipe: Pipeline, path: pathlib.Path, *, use_tagger: bool,
             limit: int = 0, batch: int = 64) -> dict:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break

    tp = fp = fn = 0
    det_tp = det_fn = fp_tokens = clean_tokens = 0
    n_sent = n_tok = 0
    t0 = time.perf_counter()

    for r in rows:
        src, tgt = normalize(r["src"]), normalize(r["tgt"])
        a, b = words(src), words(tgt)
        if len(a) != len(b):
            continue
        n_sent += 1
        n_tok += len(a)
        gold = {i: b[i] for i in range(len(a)) if a[i].lower() != b[i].lower()}
        clean_tokens += len(a) - len(gold)

        idx = {t.start: i for i, t in enumerate(t for t in tokenize(src) if t.is_word)}
        made, flagged = {}, set()
        for c in pipe.corrections(src, use_tagger=use_tagger):
            i = idx.get(c.start)
            if i is None:
                continue
            flagged.add(i)
            made[i] = c.after

        for i, corr in made.items():
            if i in gold and corr.lower() == gold[i].lower():
                tp += 1
            else:
                fp += 1
        fn += len(gold) - sum(1 for i in gold
                              if i in made and made[i].lower() == gold[i].lower())
        det_tp += len(flagged & gold.keys())
        det_fn += len(gold.keys() - flagged)
        fp_tokens += len(flagged - gold.keys())

    dt = time.perf_counter() - t0
    p = tp / max(tp + fp, 1)
    r_ = tp / max(tp + fn, 1)
    return {"sentences": n_sent, "precision": p, "recall": r_,
            "f1": 2 * p * r_ / max(p + r_, 1e-9),
            "detection_recall": det_tp / max(det_tp + det_fn, 1),
            "fpr": fp_tokens / max(clean_tokens, 1),
            "tokens_per_sec": n_tok / max(dt, 1e-9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", type=pathlib.Path, required=True)
    ap.add_argument("--bench", type=pathlib.Path, required=True)
    ap.add_argument("--tagger", type=pathlib.Path, required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=400)
    args = ap.parse_args()

    pipe = Pipeline(args.lexicon, tagger_dir=args.tagger, device=args.device)
    print(f"лексикон {len(pipe.lex.core)}, теней {len(pipe.lex.shadow)}, "
          f"тэггер на {args.device}\n")

    print(f"{'задача':14s} {'режим':10s} {'P':>7s} {'R':>7s} {'F1':>7s} "
          f"{'детекция':>9s} {'FPR':>7s}")
    out = {}
    for task in TASKS:
        p = args.bench / f"{args.split}.{task}.jsonl"
        if not p.exists():
            continue
        for mode, flag in (("L1", False), ("L1+L2", True)):
            m = evaluate(pipe, p, use_tagger=flag, limit=args.limit)
            out[f"{task}/{mode}"] = m
            print(f"{task if mode=='L1' else '':14s} {mode:10s} "
                  f"{m['precision']:7.2%} {m['recall']:7.2%} {m['f1']:7.2%} "
                  f"{m['detection_recall']:9.2%} {m['fpr']:7.2%}")
    dst = args.bench / f"results.pipeline.{args.split}.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {dst}")


if __name__ == "__main__":
    main()
