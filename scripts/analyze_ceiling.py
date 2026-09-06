"""Потолок пословной проверки: сколько ошибок в принципе невидимы без контекста.

Спелчекер, который смотрит на слово отдельно, обязан пропустить ошибку, если
испорченное слово само является законным словом языка. Для якутского это не
редкость, а норма: деноминализация ҕ→г, ө→о, ү→у, һ→с схлопывает пары слов,
различающиеся только этими буквами (баҕар/багар, олох/өлөх, тус/түс).

Скрипт считает долю таких случаев на каждой задаче. Полученное число —
жёсткая верхняя граница полноты для L1 и точная мера того, сколько работы
остаётся слою L2 с контекстом.

    /usr/bin/python3 scripts/analyze_ceiling.py --lexicon data/lexicon --bench data/bench
"""
import argparse, collections, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sakhaspell.lexicon import Lexicon
from sakhaspell.norm import normalize
from sakhaspell.tokenize import tokenize

TASKS = ["denorm_full", "denorm_mixed", "real", "typo", "mixed"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", type=pathlib.Path, required=True)
    ap.add_argument("--bench", type=pathlib.Path, required=True)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--limit", type=int, default=1500)
    args = ap.parse_args()

    lex = Lexicon.load(args.lexicon)
    print(f"{'задача':14s} {'ошибок':>8s} {'невидимы':>9s} {'доля':>7s}   примеры")
    summary = {}
    for task in TASKS:
        p = args.bench / f"{args.split}.{task}.jsonl"
        if not p.exists():
            continue
        total = blind = 0
        examples = collections.Counter()
        with p.open(encoding="utf-8") as f:
            for n, line in enumerate(f):
                if args.limit and n >= args.limit:
                    break
                d = json.loads(line)
                a = [t.text for t in tokenize(normalize(d["src"])) if t.is_word]
                b = [t.text for t in tokenize(normalize(d["tgt"])) if t.is_word]
                if len(a) != len(b):
                    continue
                for x, y in zip(a, b):
                    if x.lower() == y.lower():
                        continue
                    total += 1
                    if lex.check_form(x).ok:
                        blind += 1
                        if len(examples) < 4000:
                            examples[f"{x.lower()} (верно {y.lower()})"] += 1
        summary[task] = {"errors": total, "invisible": blind,
                         "share": blind / max(total, 1)}
        ex = ", ".join(w for w, _ in examples.most_common(3))
        print(f"{task:14s} {total:8d} {blind:9d} {blind/max(total,1):7.1%}   {ex}")

    (args.bench / f"ceiling.{args.split}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\nневидимы = испорченное слово само принимается лексиконом;")
    print("это верхняя граница полноты для любой пословной проверки без контекста")


if __name__ == "__main__":
    main()
