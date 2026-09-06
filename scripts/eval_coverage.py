"""Покрытие лексикона на отложенных предложениях = базовая доля ложных срабатываний.

Это главная метрика первого слоя. Всё, что лексикон не принял на заведомо
корректном тексте, спелчекер подчеркнёт зря. Точка отсчёта — apertium-sah:
91.04% наивного покрытия на газетном корпусе (LREC 2022), то есть 8.96% ложных
подчёркиваний.

    /usr/bin/python3 scripts/eval_coverage.py --lexicon data/lexicon --out data/lexicon
"""
import argparse, collections, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sakhaspell.tokenize import tokenize


def load_forms(path: pathlib.Path) -> set[str]:
    forms = set()
    with path.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            forms.add(line.split("\t", 1)[0])
    return forms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    core = load_forms(args.lexicon / "core.tsv")
    tail = load_forms(args.lexicon / "tail.tsv")
    other = load_forms(args.lexicon / "only_other.tsv")
    print(f"ядро {len(core)}, хвост {len(tail)}, только-прочие {len(other)}")

    by = collections.defaultdict(lambda: collections.Counter())
    miss = collections.Counter()
    miss_by_tier = collections.defaultdict(collections.Counter)

    with (args.lexicon / "heldout.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            src, tier = d["source"], d["tier"]
            for t in tokenize(d["text"]):
                if not t.is_word:
                    continue
                w = t.text.lower()
                by[src]["tok"] += 1
                by[tier + "*"]["tok"] += 1
                if w in core:
                    by[src]["core"] += 1
                    by[tier + "*"]["core"] += 1
                elif w in tail:
                    by[src]["tail"] += 1
                    by[tier + "*"]["tail"] += 1
                else:
                    by[src]["miss"] += 1
                    by[tier + "*"]["miss"] += 1
                    miss[w] += 1
                    miss_by_tier[tier][w] += 1

    rows = []
    for k, c in sorted(by.items(), key=lambda kv: -kv[1]["tok"]):
        n = max(c["tok"], 1)
        rows.append((k, c["tok"], c["core"] / n, (c["core"] + c["tail"]) / n, c["miss"] / n))

    print(f"\n{'источник':32s} {'токенов':>9s} {'ядро':>8s} {'ядро+хвост':>11s} {'нет вовсе':>10s}")
    for k, n, a, b, m in rows:
        print(f"{k:32s} {n:9d} {a:8.2%} {b:11.2%} {m:10.2%}")

    with (args.out / "coverage.json").open("w", encoding="utf-8") as f:
        json.dump({"rows": [{"key": k, "tokens": n, "core": a, "core_tail": b, "miss": m}
                            for k, n, a, b, m in rows],
                   "top_missing_edited": miss_by_tier["edited"].most_common(300)},
                  f, ensure_ascii=False, indent=2)

    print("\nчаще всего не покрыто в редактируемых источниках:")
    for w, c in miss_by_tier["edited"].most_common(40):
        print(f"  {c:5d}  {w}")


if __name__ == "__main__":
    main()
