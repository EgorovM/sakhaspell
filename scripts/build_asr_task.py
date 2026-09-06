"""Задача постобработки ASR из настоящих выходов распознавателя.

Всё остальное в бенчмарке — синтетика, пусть и построенная на добытой из
корпуса статистике. Здесь пары настоящие: `hyp` — то, что выдала ASR-модель,
`ref` — эталонная расшифровка. Ошибки ровно такие, какие бывают, в тех
пропорциях, в каких бывают.

Отбираются только пары, где расхождение пословное (одинаковое число слов):
вставки и пропуски слов — это ошибка распознавания речи, а не орфографии, и
спелчекеру их чинить нечем.

    /usr/bin/python3 scripts/build_asr_task.py \
        --preds /home/jovyan/shared-volume/sakha/lm --out data/bench
"""
import argparse, collections, json, pathlib, random, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sakhaspell.norm import normalize
from sakhaspell.tokenize import tokenize

# Файлы с пересчитанными («rescored») предсказаниями — это тот же набор после
# переоценки языковой моделью. Берём обе версии, но помечаем, чтобы одинаковые
# предложения не попали и в dev, и в test.
SKIP = ("tune_dev",)


def words(text: str) -> list[str]:
    return [t.text for t in tokenize(text) if t.is_word]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seen: set[str] = set()
    rows = []
    stats = collections.Counter()
    for p in sorted(args.preds.glob("pred_*.jsonl")):
        if any(s in p.name for s in SKIP):
            continue
        with p.open(encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ref, hyp = d.get("ref"), d.get("hyp")
                if not ref or not hyp:
                    continue
                stats["read"] += 1
                ref, hyp = normalize(ref.strip()), normalize(hyp.strip())
                if ref == hyp:
                    stats["identical"] += 1
                    continue
                a, b = words(hyp), words(ref)
                if len(a) != len(b):
                    stats["length_mismatch"] += 1
                    continue
                if not (4 <= len(a) <= 60):
                    stats["length_range"] += 1
                    continue
                key = ref
                if key in seen:
                    stats["dup"] += 1
                    continue
                seen.add(key)
                n_edits = sum(x.lower() != y.lower() for x, y in zip(a, b))
                # предложение, где расходится больше половины слов, — это не
                # орфографическая правка, а другое распознавание целиком
                if n_edits > len(a) * 0.5:
                    stats["too_different"] += 1
                    continue
                rows.append({"src": hyp, "tgt": ref, "n_edits": n_edits,
                             "source": p.stem})
                stats["kept"] += 1

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    half = len(rows) // 2
    for split, part in (("dev", rows[:half]), ("test", rows[half:])):
        path = args.out / f"{split}.asr.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for i, r in enumerate(part):
                f.write(json.dumps({"id": f"{split}/asr/{i}", **r},
                                   ensure_ascii=False) + "\n")
        n_ed = sum(r["n_edits"] for r in part)
        print(f"  {split}.asr: {len(part)} предложений, {n_ed} правок")

    print("\n" + json.dumps(dict(stats), ensure_ascii=False, indent=2))
    total_words = sum(len(words(r["src"])) for r in rows)
    total_edits = sum(r["n_edits"] for r in rows)
    print(f"\nдоля ошибочных слов у ASR: {total_edits/max(total_words,1):.2%}")


if __name__ == "__main__":
    main()
