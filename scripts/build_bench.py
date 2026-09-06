"""SakhaSpellBench: набор для оценки спелчекера.

Шесть задач, каждая проверяет свой сценарий:

  clean         текст без ошибок. Меряет ложные срабатывания — единственную
                метрику, по которой пользователь выключает спелчекер.
  denorm_full   весь текст набран без якутской раскладки: ҕҥөүһ → гнoус.
  denorm_mixed  как пишут на самом деле: часть букв заменена, һ то h, то с.
  real          ошибки, добытые из корпуса (scripts/mine_errors.py) — не выдумка,
                а то, что люди пишут на самом деле: уьу, киьини, багар.
  typo          механические опечатки по раскладке, 8% слов.
  mixed         опечатки поверх частичной деноминализации.

Предложения берутся из отложенного набора, который не участвовал в сборке
лексикона. Деление на dev и test — чтобы пороги подбирались не на тесте.

    /usr/bin/python3 scripts/build_bench.py --lexicon data/lexicon \
        --errors data/errors --out data/bench
"""
import argparse, difflib, json, pathlib, random, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sakhaspell.errors import denormalize, denormalize_mixed, typo
from sakhaspell.tokenize import tokenize

TYPO_RATE = 0.08
REAL_RATE = 0.35          # доля слов, заменяемых на добытый ошибочный вариант
N_PER_TASK = 3000

# Обратный индекс добытых ошибок: правильная форма -> её реальные искажения.
REAL_ERRORS: dict[str, list[str]] = {}


def load_real_errors(path: pathlib.Path, tiers=("web", "edited")) -> None:
    for tier in tiers:
        p = path / f"pairs_{tier}.tsv"
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            next(f)
            for line in f:
                c = line.rstrip("\n").split("\t")
                if len(c) < 5:
                    continue
                REAL_ERRORS.setdefault(c[1], []).append(c[0])


def corrupt_words(text: str, rng: random.Random, rate: float, fn) -> str:
    """Порча отдельных слов; границы прочих токенов сохраняются."""
    toks = [t for t in tokenize(text) if t.is_word and len(t.text) >= 4]
    if not toks:
        return text
    chosen = [t for t in toks if rng.random() < rate]
    if not chosen:
        chosen = [rng.choice(toks)]
    out, prev = [], 0
    for t in chosen:
        rep = fn(t.text, rng)
        if rep is None or rep == t.text:
            continue
        out.append(text[prev:t.start])
        out.append(rep)
        prev = t.end
    out.append(text[prev:])
    return "".join(out)


def _typo(w: str, rng: random.Random) -> str:
    return typo(w, rng)


def _real(w: str, rng: random.Random) -> str | None:
    variants = REAL_ERRORS.get(w.lower())
    if not variants:
        return None
    v = rng.choice(variants)
    return v.capitalize() if w[:1].isupper() else v


TASKS = {
    "clean":        lambda s, r: s,
    "denorm_full":  lambda s, r: denormalize(s, "ru"),
    "denorm_mixed": lambda s, r: denormalize_mixed(s, r),
    "real":         lambda s, r: corrupt_words(s, r, REAL_RATE, _real),
    "typo":         lambda s, r: corrupt_words(s, r, TYPO_RATE, _typo),
    "mixed":        lambda s, r: corrupt_words(denormalize_mixed(s, r), r, TYPO_RATE, _typo),
}


def diff_words(src: str, tgt: str) -> list[tuple[int, str, str]]:
    a = [t.text for t in tokenize(src) if t.is_word]
    b = [t.text for t in tokenize(tgt) if t.is_word]
    if len(a) == len(b):
        return [(i, x, y) for i, (x, y) in enumerate(zip(a, b)) if x != y]
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        for k in range(max(i2 - i1, j2 - j1)):
            out.append((i1 + k,
                        a[i1 + k] if i1 + k < i2 else "",
                        b[j1 + k] if j1 + k < j2 else ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", type=pathlib.Path, required=True)
    ap.add_argument("--errors", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--n", type=int, default=N_PER_TASK)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.errors:
        load_real_errors(args.errors)
        print(f"добытых ошибок: {len(REAL_ERRORS)} правильных форм с вариантами")

    pool = []
    with (args.lexicon / "heldout.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d["tier"] != "edited":
                continue
            n_words = sum(1 for t in tokenize(d["text"]) if t.is_word)
            if 6 <= n_words <= 40:
                pool.append(d)
    rng = random.Random(args.seed)
    rng.shuffle(pool)
    print(f"пригодных отложенных предложений: {len(pool)}")

    half = len(pool) // 2
    splits = {"dev": pool[:half], "test": pool[half:]}
    counts = {}
    for split, items in splits.items():
        for task, fn in TASKS.items():
            take = items[: args.n]
            path = args.out / f"{split}.{task}.jsonl"
            n_edits = n_changed = 0
            with path.open("w", encoding="utf-8") as f:
                for i, d in enumerate(take):
                    r = random.Random(f"{args.seed}/{split}/{task}/{i}")
                    tgt = d["text"]
                    src = fn(tgt, r)
                    edits = diff_words(src, tgt)
                    n_edits += len(edits)
                    n_changed += bool(edits)
                    f.write(json.dumps({"id": f"{split}/{task}/{i}", "source": d["source"],
                                        "src": src, "tgt": tgt, "n_edits": len(edits)},
                                       ensure_ascii=False) + "\n")
            counts[f"{split}.{task}"] = {"n": len(take), "edits": n_edits,
                                         "sentences_with_edits": n_changed}
            print(f"  {split}.{task}: {len(take)} предл., {n_edits} правок, "
                  f"{n_changed} испорченных предложений")
    (args.out / "meta.json").write_text(
        json.dumps({"counts": counts, "typo_rate": TYPO_RATE,
                    "real_rate": REAL_RATE, "seed": args.seed},
                   ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
