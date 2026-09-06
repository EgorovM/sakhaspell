"""Добыча настоящих ошибок из корпуса.

Синтетическую порчу легко придумать, но она не совпадает с тем, что люди и машины
делают на самом деле. SAGE решает это методом SBSC: снимает статистику ошибок с
параллельного набора и воспроизводит её распределение. Параллельного набора для
якутского нет, зато есть источники разного качества.

Форма, которой нет в ядре лексикона, но которая лежит в одной правке от частой
формы из ядра, — почти наверняка ошибка. Из веб-источников так добываются ошибки
живых людей, из OCR-источников — ошибки распознавания. Это две разные модели
ошибок, и смешивать их нельзя.

    /usr/bin/python3 scripts/mine_errors.py --sent data/sent --lexicon data/lexicon \
        --out data/errors --procs 60
"""
import argparse, collections, json, pathlib, sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sakhaspell.fuzzy import Trie
from sakhaspell.tokenize import tokenize

# Кандидат считается исправлением, если он достаточно частотнее ошибки: настоящее
# слово встречается на порядки чаще своей опечатки.
FREQ_RATIO = 20
MAX_COST = 150          # 1.5 обычной правки: одна опечатка или две дешёвых замены
MIN_WRONG_FREQ = 2      # разовый мусор не берём

_TRIE: Trie | None = None
_CORE: dict[str, int] = {}


def _init(core: dict[str, int]) -> None:
    global _TRIE, _CORE
    _CORE = core
    _TRIE = Trie.from_freq(core)


def _probe(item: tuple[str, int]) -> tuple[str, str, int, int, int] | None:
    w, f = item
    cands = _TRIE.search(w, max_cost=MAX_COST, limit=8)
    for c in cands:
        if c.form == w:
            return None
        if c.freq >= f * FREQ_RATIO:
            return w, c.form, c.cost, f, c.freq
    return None


def count_tier(args: tuple[pathlib.Path, set]) -> collections.Counter:
    path, core = args
    c = collections.Counter()
    with path.open(encoding="utf-8") as f:
        for line in f:
            for t in tokenize(line):
                if t.is_word:
                    w = t.text.lower()
                    if w not in core:
                        c[w] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sent", type=pathlib.Path, required=True)
    ap.add_argument("--lexicon", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--procs", type=int, default=60)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    core: dict[str, int] = {}
    with (args.lexicon / "core.tsv").open(encoding="utf-8") as f:
        next(f)
        for line in f:
            p = line.split("\t")
            core[p[0]] = int(p[1])
    core_set = set(core)
    tiers = json.loads((args.sent / "stats.json").read_text(encoding="utf-8"))["tiers"]
    print(f"ядро: {len(core)} форм", flush=True)

    by_tier: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    paths = [p for p in sorted(args.sent.glob("*.txt")) if p.stat().st_size > 0]
    with ProcessPoolExecutor(max_workers=min(args.procs, len(paths))) as ex:
        for p, c in zip(paths, ex.map(count_tier, [(p, core_set) for p in paths])):
            t = tiers.get(p.stem, "?")
            by_tier[t] += c
            print(f"  {p.stem}: {len(c)} неизвестных форм", flush=True)

    _init(core)
    all_pairs: dict[str, list] = {}
    for tier in ("web", "ocr", "edited"):
        cand = [(w, f) for w, f in by_tier[tier].items() if f >= MIN_WRONG_FREQ]
        print(f"\n{tier}: проверяю {len(cand)} форм", flush=True)
        pairs = []
        with ProcessPoolExecutor(max_workers=args.procs) as ex:
            for i, r in enumerate(ex.map(_probe, cand, chunksize=200), 1):
                if r:
                    pairs.append(r)
                if i % 50000 == 0:
                    print(f"    {i}/{len(cand)}, найдено {len(pairs)}", flush=True)
        pairs.sort(key=lambda r: -r[3])
        all_pairs[tier] = pairs
        with (args.out / f"pairs_{tier}.tsv").open("w", encoding="utf-8") as f:
            f.write("wrong\tright\tcost\tfreq_wrong\tfreq_right\n")
            for w, r, c, fw, fr in pairs:
                f.write(f"{w}\t{r}\t{c}\t{fw}\t{fr}\n")
        print(f"  {tier}: {len(pairs)} пар -> pairs_{tier}.tsv", flush=True)

    summary = {t: {"unknown_forms": len(by_tier[t]),
                   "unknown_tokens": sum(by_tier[t].values()),
                   "pairs": len(all_pairs.get(t, []))}
               for t in by_tier}
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
