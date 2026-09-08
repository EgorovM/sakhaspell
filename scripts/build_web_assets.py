"""Сборка данных для браузерной версии.

Страница-демонстрация работает целиком в браузере, без сервера: словарь
скачивается один раз и дальше проверка идёт локально. Поэтому размер данных —
главное ограничение, и он определил формат.

Замеры на нашем словаре из 301 628 форм:

    список по частоте, gzip           1.88 МБ
    список по алфавиту, gzip          1.02 МБ   общие префиксы жмутся
    + фронт-кодирование               0.63 МБ   агглютинация даёт длинные префиксы
    частоты отдельным байтом          0.22 МБ   логарифмические корзины
                                      -------
                                      0.85 МБ

Фронт-кодирование выигрывает именно на якутском: формы одного корня идут в
сортировке подряд и делят до 15 символов префикса («оҕолорбутугар» и
«оҕолорбутуттан»).

    python scripts/build_web_assets.py --lexicon data/lexicon --out docs/data
"""
import argparse, gzip, json, math, pathlib

MAX_PREFIX = 35          # длина префикса кодируется одним символом от '0'
FREQ_SCALE = 16          # log2(freq) * 16 укладывается в байт до freq ~ 2^15


def _open(path: pathlib.Path):
    """Файл как есть или его gzip-версия: в пакете словарь лежит сжатым."""
    if path.exists():
        return path.open(encoding="utf-8")
    gz = path.with_suffix(path.suffix + ".gz")
    if gz.exists():
        return gzip.open(gz, "rt", encoding="utf-8")
    raise FileNotFoundError(f"нет ни {path}, ни {gz}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    freq: dict[str, int] = {}
    with _open(args.lexicon / "core.tsv") as f:
        next(f)
        for line in f:
            c = line.rstrip("\n").split("\t")
            freq[c[0]] = int(c[1])

    shadows: dict[str, str] = {}
    with _open(args.lexicon / "shadows.tsv") as f:
        header = next(f).rstrip("\n").split("\t")
        i_o, i_d = header.index("origin"), header.index("demote")
        for line in f:
            c = line.rstrip("\n").split("\t")
            if len(c) > i_d and c[i_d] == "1":
                shadows[c[0]] = c[i_o]
    for w in shadows:
        freq.pop(w, None)

    forms = sorted(freq)
    print(f"форм: {len(forms)}, теней: {len(shadows)}")

    # фронт-кодирование: сколько символов совпало с предыдущей формой + хвост
    lines, prev = [], ""
    for w in forms:
        n = 0
        for a, b in zip(prev, w):
            if a != b:
                break
            n += 1
        n = min(n, MAX_PREFIX)
        lines.append(chr(48 + n) + w[n:])
        prev = w
    blob = "\n".join(lines).encode()
    (args.out / "forms.txt.gz").write_bytes(gzip.compress(blob, 9))

    ranks = bytes(min(255, int(math.log2(freq[w] + 1) * FREQ_SCALE)) for w in forms)
    (args.out / "freq.bin.gz").write_bytes(gzip.compress(ranks, 9))

    (args.out / "shadows.json.gz").write_bytes(
        gzip.compress(json.dumps(shadows, ensure_ascii=False,
                                 separators=(",", ":")).encode(), 9))

    meta = {"forms": len(forms), "shadows": len(shadows),
            "max_prefix": MAX_PREFIX, "freq_scale": FREQ_SCALE}
    (args.out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                        encoding="utf-8")

    total = sum(p.stat().st_size for p in args.out.glob("*.gz"))
    for p in sorted(args.out.glob("*")):
        print(f"  {p.name:20s} {p.stat().st_size/1024:8.1f} КБ")
    print(f"  {'ИТОГО':20s} {total/1024:8.1f} КБ")


if __name__ == "__main__":
    main()
