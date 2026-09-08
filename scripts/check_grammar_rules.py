"""Проверка книжных правил на корпусе — по каждому правилу отдельно.

Правило полезно, если выполняются два условия, и оба надо мерить порознь:

  1. Согласие с корпусом. Сколько заведомо правильных форм из ядра лексикона
     его нарушают. Много нарушений означает, что правило сформулировано в
     грамматике слишком широко, а не что корпус плох.
  2. Добавка сверх словаря. Сколько ошибок бенчмарка правило ловит из тех,
     которые лексикон и так не принимает. Правило, дублирующее словарь,
     бесполезно, каким бы верным оно ни было.

Второй пункт — главный. В E15 выяснилось, что гармония гласных на
деноминализации не добавляет ничего: из 850 дисгармоничных слов лексикон
отвергает все 850. Здесь тот же вопрос задаётся каждому правилу.

    /usr/bin/python3 scripts/check_grammar_rules.py --lexicon data/lexicon \
        --bench data/bench --errors data/errors
"""
import argparse, collections, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sakhaspell.grammar import RULES, is_loanword
from sakhaspell.lexicon import Lexicon
from sakhaspell.norm import normalize
from sakhaspell.tokenize import tokenize

TASKS = ("denorm_mixed", "typo", "real", "mixed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", type=pathlib.Path, default=None)
    ap.add_argument("--bench", type=pathlib.Path)
    ap.add_argument("--errors", type=pathlib.Path)
    ap.add_argument("--limit", type=int, default=1200)
    args = ap.parse_args()

    lex = Lexicon.load(args.lexicon)
    core = list(lex.core)
    loans = sum(is_loanword(w) for w in core)
    print(f"ядро лексикона: {len(core)} форм, из них с буквами заимствований "
          f"{loans} ({loans/len(core):.1%})\n")

    # --- 1. согласие с корпусом ---------------------------------------------
    print("СОГЛАСИЕ С КОРПУСОМ — доля правильных форм, нарушающих правило")
    print(f"{'правило':22s} {'по типам':>9s} {'по токенам':>11s}   примеры нарушителей")
    agreement = {}
    for name, fn in RULES.items():
        bad_types = bad_tokens = 0
        ex = collections.Counter()
        for w, f in lex.core.items():
            v = fn(w)
            if v:
                bad_types += 1
                bad_tokens += f
                ex[f"{w} ({v[0].detail})"] += f
        r_t = bad_types / len(core)
        r_k = bad_tokens / max(sum(lex.core.values()), 1)
        agreement[name] = {"types": r_t, "tokens": r_k}
        print(f"{name:22s} {r_t:8.2%} {r_k:10.2%}   "
              f"{', '.join(w for w, _ in ex.most_common(3))}")

    # --- 2. добавка сверх словаря -------------------------------------------
    if not args.bench:
        return
    print(f"\nДОБАВКА СВЕРХ СЛОВАРЯ — ошибки, которые ловит правило,")
    print("но которые лексикон принимает как правильные слова")
    header = f"{'правило':22s}" + "".join(f"{t[:12]:>13s}" for t in TASKS)
    print(header)

    gold: dict[str, list[tuple[str, str]]] = {}
    for task in TASKS:
        p = args.bench / f"dev.{task}.jsonl"
        if not p.exists():
            continue
        pairs = []
        with p.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= args.limit:
                    break
                d = json.loads(line)
                a = [t.text for t in tokenize(normalize(d["src"])) if t.is_word]
                b = [t.text for t in tokenize(normalize(d["tgt"])) if t.is_word]
                if len(a) != len(b):
                    continue
                pairs += [(x, y) for x, y in zip(a, b) if x.lower() != y.lower()]
        gold[task] = pairs

    # знаменатель: ошибки, невидимые для словаря
    invisible = {t: [(x, y) for x, y in p if lex.check_form(x).ok]
                 for t, p in gold.items()}
    print(f"{'(всего невидимых)':22s}" +
          "".join(f"{len(invisible[t]):13d}" for t in TASKS if t in invisible))

    added = {}
    for name, fn in RULES.items():
        cells, row = [], {}
        for task in TASKS:
            if task not in invisible:
                continue
            inv = invisible[task]
            caught = sum(1 for x, _ in inv if fn(x))
            # ложные: правило срабатывает на правильной форме из той же пары
            row[task] = {"caught": caught, "n": len(inv),
                         "share": caught / max(len(inv), 1)}
            cells.append(f"{caught:5d} {caught/max(len(inv),1):6.1%}")
        added[name] = row
        print(f"{name:22s}" + "".join(f"{c:>13s}" for c in cells))

    # --- 3. цена: ложные срабатывания на верных формах -----------------------
    print(f"\nЦЕНА — правило сработало на ПРАВИЛЬНОЙ форме из пары")
    print(f"{'правило':22s}" + "".join(f"{t[:12]:>13s}" for t in TASKS))
    cost = {}
    for name, fn in RULES.items():
        cells, row = [], {}
        for task in TASKS:
            if task not in gold:
                continue
            pairs = gold[task]
            bad = sum(1 for _, y in pairs if fn(y))
            row[task] = bad / max(len(pairs), 1)
            cells.append(f"{bad:5d} {row[task]:6.1%}")
        cost[name] = row
        print(f"{name:22s}" + "".join(f"{c:>13s}" for c in cells))

    out = {"agreement": agreement, "added": added, "cost": cost}
    pathlib.Path("data/grammar_check.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n-> data/grammar_check.json")


if __name__ == "__main__":
    main()
