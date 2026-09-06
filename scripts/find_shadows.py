"""Тень искажения: чистка лексикона от систематической ошибки письма.

Правило «форма подтверждена другими источниками» ловит случайные опечатки, но
бессильно против систематической ошибки. Письмо без якутской раскладки — именно
такая: «киси» вместо «киһи» встречается во всех источниках сразу, набирает
частоту и попадает в лексикон. После этого спелчекер никогда её не подчеркнёт,
и 41% ошибок деноминализации становятся невидимы (scripts/analyze_ceiling.py).

Проверяются все способы искажения, найденные в корпусе (scripts/mine_errors.py),
а не только кириллическая деноминализация:

    ҕ → г        оҕо → ого
    ҥ → н, нг, ц саҥа → сана, санга; хаҥалас → хацалас
    ө → о        көр → кор
    ү → у        үлэ → улэ
    һ → с, ь, h  киһи → киси, киьи, киhи

Замены комбинируются: реальная форма «киьини» — это одновременно һ→ь. Поэтому
для каждой правильной формы порождается всё множество её искажений, а не одно.

Форма X — тень формы Y, если X получается искажением Y и Y достаточно частотнее X.
Порог по отношению частот, а не по абсолютной: у частотных слов тень тоже
частотная. Часть пар — законные омографы («тус» существует сам по себе и
одновременно является тенью «түс»), их отделяет тот же порог.

    /usr/bin/python3 scripts/find_shadows.py --lexicon data/lexicon --out data/lexicon
"""
import argparse, collections, itertools, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Все замены, подтверждённые добычей ошибок из корпуса.
SUBS: dict[str, tuple[str, ...]] = {
    "ҕ": ("г",),
    "ҥ": ("н", "нг", "ц"),
    "ө": ("о",),
    "ү": ("у",),
    "һ": ("с", "ь", "h"),
}
SPECIAL = set(SUBS)

# Ниже этого отношения freq(оригинал)/freq(тень) форма считается самостоятельным
# словом, а не ошибкой.
SHADOW_RATIO = 8.0
MAX_VARIANTS = 64          # у формы с 6 спецбуквами вариантов слишком много


def shadows_of(word: str, limit: int = MAX_VARIANTS) -> list[str]:
    """Все искажения формы. Исходная форма в результат не входит."""
    slots = []
    n = 1
    for ch in word:
        if ch in SUBS:
            alts = (ch,) + SUBS[ch]
            n *= len(alts)
            slots.append(alts)
        else:
            slots.append((ch,))
        if n > limit:
            return []
    out = []
    for combo in itertools.product(*slots):
        v = "".join(combo)
        if v != word:
            out.append(v)
    return out


def _read(path: pathlib.Path, value_col: int = 1) -> dict[str, int]:
    d: dict[str, int] = {}
    if not path.exists():
        return d
    with path.open(encoding="utf-8") as f:
        next(f)
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) > value_col:
                try:
                    d[c[0]] = int(c[value_col])
                except ValueError:
                    pass
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--ratio", type=float, default=SHADOW_RATIO)
    args = ap.parse_args()

    core = _read(args.lexicon / "core.tsv", 1)
    tail = _read(args.lexicon / "tail.tsv", 1)
    tail_other = _read(args.lexicon / "tail.tsv", 3)
    print(f"ядро {len(core)}, хвост {len(tail)}")

    # Кандидаты в тени: всё, что вообще может быть предъявлено акцептору, то есть
    # ядро плюс подтверждённая часть хвоста.
    candidates: dict[str, tuple[int, str]] = {}
    for w, f in core.items():
        candidates[w] = (f, "core")
    for w, f in tail.items():
        if w not in candidates and tail_other.get(w, 0) >= 3:
            candidates[w] = (f, "tail")

    # Для каждой правильной формы (со спецбуквами) ищем её тени среди кандидатов.
    best_origin: dict[str, tuple[str, int]] = {}
    for w, f in core.items():
        if not any(c in SPECIAL for c in w):
            continue
        for sh in shadows_of(w):
            if sh in candidates:
                prev = best_origin.get(sh)
                if prev is None or f > prev[1]:
                    best_origin[sh] = (w, f)

    rows = []
    for sh, (origin, of) in best_origin.items():
        sf, where = candidates[sh]
        # частота тени берётся суммарная: в редактируемых источниках она может
        # быть мала, но если её массово пишут в вебе — это всё равно ошибка
        ratio = of / max(sf, 1)
        rows.append((sh, sf, where, origin, of, ratio, ratio >= args.ratio))

    rows.sort(key=lambda r: -r[1])
    drop = [r for r in rows if r[6]]
    keep = [r for r in rows if not r[6]]

    with (args.out / "shadows.tsv").open("w", encoding="utf-8") as f:
        f.write("shadow\tfreq_shadow\twhere\torigin\tfreq_origin\tratio\tdemote\n")
        for sh, sf, wh, o, of, ratio, dp in rows:
            f.write(f"{sh}\t{sf}\t{wh}\t{o}\t{of}\t{ratio:.1f}\t{int(dp)}\n")

    drop_core = [r for r in drop if r[2] == "core"]
    print(f"\nтеней найдено: {len(rows)} (в ядре {sum(r[2]=='core' for r in rows)}, "
          f"в подтверждённом хвосте {sum(r[2]=='tail' for r in rows)})")
    print(f"понижено: {len(drop)}, из них было в ядре {len(drop_core)}")
    print(f"оставлено как омографы: {len(keep)}")

    print(f"\nпонижаем (отношение >= {args.ratio}):")
    for sh, sf, wh, o, of, ratio, _ in drop[:22]:
        print(f"  {sh:16s} {sf:7d} [{wh:4s}]  <- {o:16s} {of:8d}  x{ratio:.0f}")
    print("\nоставляем как самостоятельные слова:")
    for sh, sf, wh, o, of, ratio, _ in keep[:16]:
        print(f"  {sh:16s} {sf:7d} [{wh:4s}]  vs {o:16s} {of:8d}  x{ratio:.1f}")

    (args.out / "shadow_summary.json").write_text(json.dumps({
        "ratio": args.ratio, "shadows": len(rows), "demoted": len(drop),
        "demoted_from_core": len(drop_core), "kept": len(keep),
    }, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
