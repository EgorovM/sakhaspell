"""Аудит эталонных расшифровок ASR.

Задача постобработки ASR оказалась непригодной как бенчмарк: в парах `ref`/`hyp`
из `sakha/lm/pred_*.jsonl` эталон систематически хуже гипотезы. Проверка
показала, что в половине расхождений именно `ref` не является словом якутского
языка, и каждый пятый эталон — деноминализованное написание.

Это не дефект спелчекера, а находка про данные: WER в проекте ASR завышен,
потому что часть «ошибок распознавания» — это ошибки в самих расшифровках.

Скрипт делает из этого рабочий артефакт: список строк эталона с подозрением на
орфографическую ошибку, отсортированный по уверенности, и предлагаемые правки.

    /usr/bin/python3 scripts/audit_asr_refs.py --preds /home/jovyan/shared-volume/sakha/lm \
        --lexicon data/lexicon --out data/asr_audit
"""
import argparse, collections, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sakhaspell.checker import SpellChecker
from sakhaspell.lexicon import Lexicon
from sakhaspell.norm import normalize
from sakhaspell.tokenize import tokenize


def words(text: str):
    return [t.text for t in tokenize(text) if t.is_word]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=pathlib.Path, required=True)
    ap.add_argument("--lexicon", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    lex = Lexicon.load(args.lexicon)
    ch = SpellChecker(lex)

    seen: set[str] = set()
    stats = collections.Counter()
    bad_words = collections.Counter()
    rows = []

    for p in sorted(args.preds.glob("pred_*.jsonl")):
        with p.open(encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ref = d.get("ref")
                if not ref:
                    continue
                ref = normalize(ref.strip())
                if ref in seen:
                    continue
                seen.add(ref)
                stats["refs"] += 1

                issues = []
                for w in words(ref):
                    low = w.lower()
                    stats["words"] += 1
                    if low in lex.shadow:
                        issues.append((w, lex.shadow[low], "деноминализация"))
                        bad_words[low] += 1
                        stats["shadow"] += 1
                    elif not lex.check_form(w).ok:
                        sug = ch.suggest(w)
                        if sug:
                            issues.append((w, sug[0].form, "не слово"))
                            bad_words[low] += 1
                            stats["unknown"] += 1
                if issues:
                    stats["refs_with_issues"] += 1
                    rows.append({"ref": ref, "n_issues": len(issues),
                                 "issues": [{"before": a, "after": b, "kind": k}
                                            for a, b, k in issues],
                                 "fixed": ch.correct(ref)})

    rows.sort(key=lambda r: -r["n_issues"])
    with (args.out / "ref_issues.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with (args.out / "top_bad_words.tsv").open("w", encoding="utf-8") as f:
        f.write("word\tcount\tsuggestion\n")
        for w, c in bad_words.most_common(500):
            s = lex.shadow.get(w) or (ch.suggest(w)[0].form if ch.suggest(w) else "")
            f.write(f"{w}\t{c}\t{s}\n")

    stats["_summary"] = 0
    n = stats["refs"]
    print(f"эталонных расшифровок: {n}")
    print(f"слов в них: {stats['words']}")
    print(f"с подозрением на ошибку: {stats['refs_with_issues']} "
          f"({stats['refs_with_issues']/max(n,1):.1%} расшифровок)")
    print(f"  деноминализация: {stats['shadow']} слов")
    print(f"  не слово:        {stats['unknown']} слов")
    print(f"  доля ошибочных слов: "
          f"{(stats['shadow']+stats['unknown'])/max(stats['words'],1):.2%}")
    (args.out / "summary.json").write_text(
        json.dumps(dict(stats), ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nчаще всего в эталонах:")
    for w, c in bad_words.most_common(20):
        s = lex.shadow.get(w, "")
        print(f"  {c:5d}  {w:20s} -> {s}")


if __name__ == "__main__":
    main()
