"""Чистка эталонных расшифровок ASR и пересчёт WER.

E13 показал, что эталоны в `sakha/lm/pred_*.jsonl` систематически хуже выхода
модели: в 50.9% расхождений не является словом именно эталон, а 46% расшифровок
содержат орфографическую ошибку. Значит WER в проекте ASR завышен — модель
штрафуется за слова, которые написала правильно.

Скрипт применяет спелчекер к эталонам и считает WER до и после. Разница —
это та часть ошибки, которой у модели на самом деле нет.

Правки берутся консервативно: только известные тени искажения и слова, для
которых у чекера есть однозначная подсказка. Гадать в эталоне нельзя — это
испортит метрику в другую сторону.

    /usr/bin/python3 scripts/fix_asr_refs.py --preds /home/jovyan/shared-volume/sakha/lm \
        --lexicon data/lexicon --out data/asr_audit
"""
import argparse, collections, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sakhaspell.checker import SpellChecker
from sakhaspell.lexicon import Lexicon
from sakhaspell.norm import normalize
from sakhaspell.tokenize import tokenize


def words(text: str) -> list[str]:
    return [t.text.lower() for t in tokenize(text) if t.is_word]


def wer(ref: list[str], hyp: list[str]) -> tuple[int, int]:
    """Расстояние редактирования по словам и длина эталона."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return m, 0
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        for j in range(1, m + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return prev[m], n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=pathlib.Path, required=True)
    ap.add_argument("--lexicon", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    lex = Lexicon.load(args.lexicon)
    ch = SpellChecker(lex)

    def clean(text: str) -> tuple[str, int]:
        """Исправляет только уверенные случаи: известные тени искажения."""
        out, prev, n = [], 0, 0
        for iss in ch.check(text):
            w = iss.token.text.lower()
            origin = lex.shadow.get(w)
            if not origin:
                continue
            out.append(text[prev:iss.token.start])
            out.append(origin if w == iss.token.text else origin.capitalize())
            prev = iss.token.end
            n += 1
        out.append(text[prev:])
        return "".join(out), n

    stats = collections.Counter()
    per_file = {}
    fixed_rows = []

    for p in sorted(args.preds.glob("pred_*.jsonl")):
        errs = errs_fixed = length = 0
        with p.open(encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ref, hyp = d.get("ref"), d.get("hyp")
                if not ref or not hyp:
                    continue
                ref, hyp = normalize(ref.strip()), normalize(hyp.strip())
                ref2, n_fix = clean(ref)
                stats["refs"] += 1
                stats["fixes"] += n_fix
                if n_fix:
                    stats["refs_fixed"] += 1
                    if len(fixed_rows) < 3000:
                        fixed_rows.append({"file": p.stem, "ref": ref,
                                           "ref_fixed": ref2, "n": n_fix})
                e1, L = wer(words(ref), words(hyp))
                e2, _ = wer(words(ref2), words(hyp))
                errs += e1
                errs_fixed += e2
                length += L
        if length:
            per_file[p.stem] = {"wer": errs / length, "wer_fixed": errs_fixed / length,
                                "words": length}

    with (args.out / "refs_fixed.jsonl").open("w", encoding="utf-8") as f:
        for r in fixed_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"расшифровок {stats['refs']}, поправлено {stats['refs_fixed']} "
          f"({stats['refs_fixed']/max(stats['refs'],1):.1%}), правок {stats['fixes']}\n")
    print(f"{'набор':34s} {'слов':>8s} {'WER':>8s} {'WER после':>10s} {'разница':>9s}")
    tot_w = tot_e = tot_e2 = 0
    for name, m in sorted(per_file.items(), key=lambda kv: -kv[1]["words"]):
        print(f"{name:34s} {m['words']:8d} {m['wer']:7.2%} {m['wer_fixed']:9.2%} "
              f"{m['wer_fixed']-m['wer']:+8.2%}")
        tot_w += m["words"]
        tot_e += m["wer"] * m["words"]
        tot_e2 += m["wer_fixed"] * m["words"]
    print(f"{'ИТОГО':34s} {tot_w:8d} {tot_e/tot_w:7.2%} {tot_e2/tot_w:9.2%} "
          f"{(tot_e2-tot_e)/tot_w:+8.2%}")

    (args.out / "wer_recount.json").write_text(json.dumps(
        {"per_file": per_file, "stats": dict(stats),
         "total": {"words": tot_w, "wer": tot_e / tot_w, "wer_fixed": tot_e2 / tot_w}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nОтрицательная разница означает, что часть «ошибок распознавания» —")
    print("это орфографические ошибки в самих расшифровках.")


if __name__ == "__main__":
    main()
