"""Инвентарь символов по источникам корпуса.

Нормализацию нельзя писать по учебнику алфавита. В реальных текстах лежат казахские
ҒҢ вместо якутских ҔҤ, латинские гомоглифы и мягкие переносы — сначала смотрим, что
есть, потом решаем, что с этим делать. Разбивка по источникам нужна, чтобы отличить
цифровые тексты от OCR: у них разный мусор.

    /usr/bin/python3 scripts/char_inventory.py --corpus <corpus_sah.jsonl> --out data/
"""
import argparse, collections, json, pathlib, unicodedata

# Якутский алфавит по Убрятовой: 20 общих с русским + 5 своих + два диграфа.
SAKHA_LOWER = set("абвгҕдеёжзийклмнҥоөпрсһтуүфхцчшщъыьэюя")
SAKHA_ONLY = set("ҕҥөһү")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    total = collections.Counter()
    by_source = collections.defaultdict(collections.Counter)
    docs = collections.Counter()

    with args.corpus.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            src, text = d.get("source", "?"), d.get("text", "")
            docs[src] += 1
            c = collections.Counter(text)
            total += c
            by_source[src] += c
            if i % 100000 == 0:
                print(f"  {i} док.", flush=True)

    n = sum(total.values())
    with (args.out / "char_inventory.tsv").open("w", encoding="utf-8") as f:
        f.write("char\tcodepoint\tname\tcount\tshare\tin_alphabet\n")
        for ch, cnt in total.most_common():
            disp = repr(ch)[1:-1] if unicodedata.category(ch)[0] in "CZ" else ch
            f.write(f"{disp}\tU+{ord(ch):04X}\t{unicodedata.name(ch, '?')}\t{cnt}"
                    f"\t{cnt/n:.3e}\t{int(ch.lower() in SAKHA_LOWER)}\n")

    # доля специфически якутских букв — индикатор «настоящего» якутского письма
    rows = []
    for src, c in by_source.items():
        letters = sum(v for k, v in c.items() if k.isalpha())
        special = sum(v for k, v in c.items() if k.lower() in SAKHA_ONLY)
        foreign = sum(v for k, v in c.items() if k.isalpha() and k.lower() not in SAKHA_LOWER)
        rows.append((src, docs[src], sum(c.values()), letters,
                     special / max(letters, 1), foreign / max(letters, 1)))
    rows.sort(key=lambda r: -r[2])
    with (args.out / "source_profile.tsv").open("w", encoding="utf-8") as f:
        f.write("source\tdocs\tchars\tletters\tspecial_share\tforeign_letter_share\n")
        for r in rows:
            f.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]}\t{r[4]:.4f}\t{r[5]:.5f}\n")

    print(f"всего символов {n}, различных {len(total)}, источников {len(by_source)}")


if __name__ == "__main__":
    main()
