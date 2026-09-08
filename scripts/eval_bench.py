"""Оценка спелчекера на SakhaSpellBench.

Метрики пословные, как в SAGE (`ruspelleval`): сравниваем множества правок,
которые сделала система, с множеством правок эталона.

  precision   доля сделанных правок, которые совпали с эталоном
  recall      доля эталонных правок, которые система сделала
  F1          гармоническое среднее
  detection   доля ошибочных слов, которые система хотя бы пометила (без учёта
              того, верна ли подсказка) — отдельно, потому что «подчеркнуть»
              и «исправить» это разные продуктовые обещания
  FPR         доля правильных слов, которые система тронула зря

FPR на задаче clean — главная метрика. Точка отсчёта: apertium-sah даёт 8.96%
ложных подчёркиваний на газетном тексте (LREC 2022), bashspell помечает 4.89%
токенов на башкирском корпусе, из них ~46% ложно.

    /usr/bin/python3 scripts/eval_bench.py --lexicon data/lexicon --bench data/bench \
        --split dev
"""
import argparse, collections, json, pathlib, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sakhaspell.checker import SpellChecker
from sakhaspell.lexicon import Lexicon
from sakhaspell.norm import normalize
from sakhaspell.tokenize import tokenize

TASKS = ["clean", "denorm_full", "denorm_mixed", "real", "typo", "mixed", "asr"]


def word_list(text: str) -> list[str]:
    return [t.text for t in tokenize(text) if t.is_word]


def evaluate(checker: SpellChecker, path: pathlib.Path, limit: int = 0) -> dict:
    tp = fp = fn = 0            # правки
    det_tp = det_fn = 0         # обнаружение
    fp_tokens = clean_tokens = 0
    n_sent = 0
    t0 = time.perf_counter()
    n_tok = 0

    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            src, tgt = normalize(d["src"]), normalize(d["tgt"])
            a, b = word_list(src), word_list(tgt)
            if len(a) != len(b):
                # порча изменила число слов — на пословных метриках такие
                # предложения не оцениваем, их доля мала и одинакова для всех систем
                continue
            n_sent += 1
            n_tok += len(a)

            gold = {i: b[i] for i in range(len(a)) if a[i].lower() != b[i].lower()}
            clean_tokens += len(a) - len(gold)

            # индексы слов в исходном тексте
            idx = {}
            for i, t in enumerate(t for t in tokenize(src) if t.is_word):
                idx[t.start] = i

            made: dict[int, str] = {}
            flagged: set[int] = set()
            for iss in checker.check(src):
                i = idx.get(iss.token.start)
                if i is None:
                    continue
                flagged.add(i)
                if iss.suggestions:
                    made[i] = iss.suggestions[0].form

            for i, corr in made.items():
                if i in gold and corr.lower() == gold[i].lower():
                    tp += 1
                else:
                    fp += 1
            fn += len(gold) - sum(1 for i in gold if i in made
                                  and made[i].lower() == gold[i].lower())
            det_tp += len(flagged & gold.keys())
            det_fn += len(gold.keys() - flagged)
            fp_tokens += len(flagged - gold.keys())

            if limit and n_sent >= limit:
                break

    dt = time.perf_counter() - t0
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {
        "sentences": n_sent,
        "precision": p,
        "recall": r,
        "f1": 2 * p * r / max(p + r, 1e-9),
        "detection_recall": det_tp / max(det_tp + det_fn, 1),
        "fpr": fp_tokens / max(clean_tokens, 1),
        "tokens_per_sec": n_tok / max(dt, 1e-9),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", type=pathlib.Path, required=True)
    ap.add_argument("--bench", type=pathlib.Path, required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="L1")
    ap.add_argument("--no-grammar", action="store_true",
                    help="отключить правила грамматики целиком")
    ap.add_argument("--tail-grammar", action="store_true",
                    help="отсеивать формы хвоста, нарушающие гармонию")
    args = ap.parse_args()

    lex = Lexicon.load(args.lexicon)
    t0 = time.perf_counter()
    ch = SpellChecker(lex, grammar=not args.no_grammar)
    print(f"лексикон: ядро {len(lex.core)}, дерево построено за "
          f"{time.perf_counter()-t0:.1f} с\n")

    print(f"{'задача':14s} {'P':>7s} {'R':>7s} {'F1':>7s} {'детекция':>9s} "
          f"{'FPR':>7s} {'ток/с':>8s}")
    results = {}
    for task in TASKS:
        p = args.bench / f"{args.split}.{task}.jsonl"
        if not p.exists():
            continue
        m = evaluate(ch, p, args.limit)
        results[task] = m
        print(f"{task:14s} {m['precision']:7.2%} {m['recall']:7.2%} {m['f1']:7.2%} "
              f"{m['detection_recall']:9.2%} {m['fpr']:7.2%} {m['tokens_per_sec']:8.0f}")

    out = args.bench / f"results.{args.tag}.{args.split}.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
