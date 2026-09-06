"""Сборка словоформного лексикона — первичного акцептора спелчекера.

Ключевое решение проекта: первичный акцептор — словоформы из корпуса, а не FST.
У apertium-sah точность 98.5%, но наивное покрытие на газетах 91.04% (LREC 2022):
FST-акцептор подчеркнёт 9% правильного текста. Корпусный лексикон такой проблемы
не имеет, зато имеет обратную — в него норовят просочиться ошибки. Отсюда правила
доверия ниже.

Форма попадает в ядро лексикона, если она
  * встречается в редактируемых источниках не реже MIN_FREQ раз И
  * встречается минимум в MIN_SOURCES разных редактируемых источниках,
  * либо встречается в одном источнике не реже SOLO_FREQ раз.

Требование двух независимых источников — главный фильтр от опечаток: одна и та же
опечатка редко повторяется в двух разных редакциях, а настоящее слово повторяется.

    /usr/bin/python3 scripts/build_lexicon.py --sent data/sent --out data/lexicon
"""
import argparse, collections, json, math, pathlib, sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sakhaspell.tokenize import tokenize

MIN_FREQ, MIN_SOURCES, SOLO_FREQ = 3, 2, 20
HELDOUT_EVERY = 200          # каждое N-е предложение — в отложенный набор


def count_source(path: pathlib.Path) -> tuple[str, collections.Counter, collections.Counter, list[str]]:
    """Частоты форм в источнике. Отдельно нижний регистр (для акцептора) и
    исходный (чтобы отличить имена собственные). Плюс отложенные предложения."""
    lower, cased, held = collections.Counter(), collections.Counter(), []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            s = line.rstrip("\n")
            if i % HELDOUT_EVERY == 0:
                held.append(s)
                continue
            for t in tokenize(s):
                if not t.is_word:
                    continue
                lower[t.text.lower()] += 1
                cased[t.text] += 1
    return path.stem, lower, cased, held


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sent", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--procs", type=int, default=18)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    meta = json.loads((args.sent / "stats.json").read_text(encoding="utf-8"))
    tiers = meta["tiers"]
    paths = sorted(p for p in args.sent.glob("*.txt") if p.stat().st_size > 0)
    print(f"источников: {len(paths)}", flush=True)

    per_source: dict[str, collections.Counter] = {}
    cased_all = collections.Counter()
    heldout: dict[str, list[str]] = {}
    with ProcessPoolExecutor(max_workers=min(args.procs, len(paths))) as ex:
        for src, lower, cased, held in ex.map(count_source, paths):
            per_source[src] = lower
            cased_all += cased
            heldout[src] = held
            print(f"  {src}: {sum(lower.values())} токенов, {len(lower)} форм", flush=True)

    edited = [s for s in per_source if tiers.get(s) == "edited"]
    other = [s for s in per_source if tiers.get(s) != "edited"]
    print(f"редактируемых источников: {len(edited)}, прочих: {len(other)}")

    # сводные частоты
    freq_edited, n_src_edited = collections.Counter(), collections.Counter()
    for s in edited:
        c = per_source[s]
        freq_edited.update(c)
        for w in c:
            n_src_edited[w] += 1
    freq_other = collections.Counter()
    for s in other:
        freq_other.update(per_source[s])

    core, tail = {}, {}
    for w, f in freq_edited.items():
        ns = n_src_edited[w]
        if (f >= MIN_FREQ and ns >= MIN_SOURCES) or f >= SOLO_FREQ:
            core[w] = (f, ns)
        else:
            tail[w] = (f, ns)

    with (args.out / "core.tsv").open("w", encoding="utf-8") as f:
        f.write("form\tfreq\tn_sources\tfreq_other\n")
        for w, (fr, ns) in sorted(core.items(), key=lambda kv: -kv[1][0]):
            f.write(f"{w}\t{fr}\t{ns}\t{freq_other.get(w, 0)}\n")
    with (args.out / "tail.tsv").open("w", encoding="utf-8") as f:
        f.write("form\tfreq\tn_sources\tfreq_other\n")
        for w, (fr, ns) in sorted(tail.items(), key=lambda kv: -kv[1][0]):
            f.write(f"{w}\t{fr}\t{ns}\t{freq_other.get(w, 0)}\n")
    # формы, которых нет в редактируемых источниках вообще: сырьё для модели ошибок
    only_other = {w: c for w, c in freq_other.items() if w not in freq_edited}
    with (args.out / "only_other.tsv").open("w", encoding="utf-8") as f:
        f.write("form\tfreq_other\n")
        for w, c in sorted(only_other.items(), key=lambda kv: -kv[1]):
            f.write(f"{w}\t{c}\n")

    hp = args.out / "heldout.jsonl"
    with hp.open("w", encoding="utf-8") as f:
        for src, lines in heldout.items():
            for s in lines:
                f.write(json.dumps({"source": src, "tier": tiers.get(src, "?"),
                                    "text": s}, ensure_ascii=False) + "\n")

    tok_edited = sum(freq_edited.values())
    cov_core = sum(freq_edited[w] for w in core) / max(tok_edited, 1)
    summary = {
        "tokens_edited": tok_edited,
        "tokens_other": sum(freq_other.values()),
        "forms_edited": len(freq_edited),
        "forms_core": len(core),
        "forms_tail": len(tail),
        "forms_only_other": len(only_other),
        "token_coverage_core_on_train": round(cov_core, 5),
        "heldout_sentences": sum(len(v) for v in heldout.values()),
        "params": {"MIN_FREQ": MIN_FREQ, "MIN_SOURCES": MIN_SOURCES,
                   "SOLO_FREQ": SOLO_FREQ},
        "per_source_tokens": {s: sum(c.values()) for s, c in per_source.items()},
        "per_source_forms": {s: len(c) for s, c in per_source.items()},
    }
    (args.out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
