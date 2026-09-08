"""Сборка биграммной модели для переранжирования кандидатов.

Замер потолка показал, где лежит главный незакрытый резерв: верный вариант
исправления уже присутствует среди кандидатов в 93.8% случаев на опечатках,
а первым мы его ставим только в 80.5%. Тринадцать пунктов теряются в
ранжировании, которое смотрит лишь на цену правки и частоту слова.

Пример: «сйын» → кандидаты «ыйын» и «сайын», оба на расстоянии одной правки.
Побеждает «ыйын», потому что вдвое частотнее. В контексте «бу ___ үлэлээбиттэрин»
правильный вариант очевиден, но контекста ранжирование не видит.

Модель намеренно простая. Для переранжирования пяти кандидатов не нужна ни
нейросеть, ни сглаживание Кнесера-Нея: достаточно частот биграмм с откатом на
униграммы. Считается один раз, работает на CPU за микросекунды.

Отсечение по частоте обязательно: без него биграмм получается больше 20 млн,
и модель не влезает в пакет. Порог подобран так, чтобы уложиться в бюджет
и не потерять покрытие — см. E18.

    /usr/bin/python3 scripts/build_lm.py --sent data/sent --lexicon data/lexicon \
        --out data/lm --min-count 3
"""
import argparse, collections, gzip, json, math, pathlib, sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sakhaspell.norm import normalize
from sakhaspell.tokenize import tokenize

# Только редактируемые источники: модель должна знать, как язык выглядит в норме,
# а не как выглядят OCR-ошибки и письмо без раскладки.
EDITED = ["kyym", "eder_saas", "sakha_sire", "uluus_media", "svfu",
          "sakhapechat", "childrens_lib", "sakha_texts_v2"]

BOS, EOS = "<s>", "</s>"


def count_file(path: pathlib.Path):
    uni = collections.Counter()
    bi = collections.Counter()
    with path.open(encoding="utf-8") as f:
        for line in f:
            words = [t.text.lower() for t in tokenize(normalize(line.rstrip("\n")))
                     if t.is_word]
            if not words:
                continue
            seq = [BOS] + words + [EOS]
            uni.update(seq)
            bi.update(zip(seq, seq[1:]))
    return uni, bi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sent", type=pathlib.Path, required=True)
    ap.add_argument("--lexicon", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--min-count", type=int, default=3)
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    paths = [args.sent / f"{s}.txt" for s in EDITED]
    paths = [p for p in paths if p.exists() and p.stat().st_size > 0]
    print(f"источников: {len(paths)}", flush=True)

    uni = collections.Counter()
    bi = collections.Counter()
    with ProcessPoolExecutor(max_workers=min(args.procs, len(paths))) as ex:
        for path, (u, b) in zip(paths, ex.map(count_file, paths)):
            uni += u
            bi += b
            print(f"  {path.stem}: {len(u)} слов, {len(b)} биграмм", flush=True)

    total = sum(uni.values())
    print(f"\nвсего токенов {total}, различных слов {len(uni)}, "
          f"биграмм {len(bi)}", flush=True)

    # Словарь модели ограничен ядром лексикона: биграмма с формой, которой
    # спелчекер всё равно не предложит, в переранжировании бесполезна.
    vocab = None
    if args.lexicon:
        core = set()
        p = args.lexicon / "core.tsv"
        opener = (lambda: gzip.open(p.with_suffix(".tsv.gz"), "rt", encoding="utf-8")) \
            if not p.exists() else (lambda: p.open(encoding="utf-8"))
        with opener() as f:
            next(f)
            for line in f:
                core.add(line.split("\t", 1)[0])
        vocab = core | {BOS, EOS}
        print(f"словарь модели: {len(vocab)} форм", flush=True)

    kept = {}
    for (a, b), c in bi.items():
        if c < args.min_count:
            continue
        if vocab is not None and (a not in vocab or b not in vocab):
            continue
        kept[(a, b)] = c
    print(f"биграмм после отсечения (>= {args.min_count}): {len(kept)}", flush=True)

    # Формат: строки «слово1<TAB>слово2<TAB>частота», сжатые gzip. Сортировка по
    # первому слову даёт длинные общие префиксы и вдвое лучшее сжатие.
    with gzip.open(args.out / "bigrams.tsv.gz", "wt", encoding="utf-8") as f:
        f.write("w1\tw2\tcount\n")
        for (a, b), c in sorted(kept.items()):
            f.write(f"{a}\t{b}\t{c}\n")
    with gzip.open(args.out / "unigrams.tsv.gz", "wt", encoding="utf-8") as f:
        f.write("word\tcount\n")
        for w, c in sorted(uni.items(), key=lambda kv: -kv[1]):
            if c >= args.min_count and (vocab is None or w in vocab):
                f.write(f"{w}\t{c}\n")

    meta = {"tokens": total, "vocab": len(uni), "bigrams_all": len(bi),
            "bigrams_kept": len(kept), "min_count": args.min_count}
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
    for p in sorted(args.out.glob("*")):
        print(f"  {p.name:20s} {p.stat().st_size/1e6:7.2f} МБ")


if __name__ == "__main__":
    main()
