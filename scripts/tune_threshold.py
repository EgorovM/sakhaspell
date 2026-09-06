"""Подбор порога уверенности тэггера на dev.

Порог задаёт обмен между пропущенными ошибками и правками в правильном тексте.
Обмен несимметричен: пропущенная ошибка стоит пользователю ничего, а лишняя
правка стоит доверия к инструменту. Поэтому выбираем не максимум F1, а
максимум F1 при ограничении на ложные срабатывания.

    $E scripts/tune_threshold.py --lexicon data/lexicon --bench data/bench \
        --tagger runs/tagger_v2 --max-fpr 0.02
"""
import argparse, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sakhaspell.pipeline import Pipeline
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from eval_pipeline import evaluate

GRID = [0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995]
TASKS = ["clean", "denorm_full", "denorm_mixed", "mixed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", type=pathlib.Path, required=True)
    ap.add_argument("--bench", type=pathlib.Path, required=True)
    ap.add_argument("--tagger", type=pathlib.Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--max-fpr", type=float, default=0.02)
    args = ap.parse_args()

    pipe = Pipeline(args.lexicon, tagger_dir=args.tagger, device=args.device)

    print(f"{'порог':>7s} " + "  ".join(f"{t[:11]:>11s}" for t in TASKS)
          + f" {'FPR чист.':>10s}")
    rows = {}
    for th in GRID:
        pipe.tagger_threshold = th
        cells, clean_fpr = [], None
        for task in TASKS:
            p = args.bench / f"dev.{task}.jsonl"
            m = evaluate(pipe, p, use_tagger=True, limit=args.limit)
            if task == "clean":
                clean_fpr = m["fpr"]
                cells.append(f"{m['fpr']:10.2%}")
            else:
                cells.append(f"{m['f1']:10.2%}")
            rows[f"{th}/{task}"] = m
        print(f"{th:7.3f} " + "  ".join(f"{c:>11s}" for c in cells)
              + f" {clean_fpr:10.2%}")

    ok = [(th, rows[f"{th}/denorm_mixed"]["f1"]) for th in GRID
          if rows[f"{th}/clean"]["fpr"] <= args.max_fpr]
    if ok:
        best = max(ok, key=lambda p: p[1])
        print(f"\nлучший порог при FPR <= {args.max_fpr:.1%}: {best[0]} "
              f"(F1 denorm_mixed {best[1]:.2%})")
    (args.bench / "threshold_tuning.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
