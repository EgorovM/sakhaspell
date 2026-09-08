"""Проверка книжных правил на корпусе.

Правило из грамматики полезно ровно настолько, насколько оно верно на реальном
языке и настолько же отличает правильное слово от ошибочного. Оба свойства
меряются здесь, до того как правило куда-то встраивается.

  1. Согласие с корпусом. Сколько форм из ядра лексикона — то есть заведомо
     правильных слов — нарушают правило. Много нарушений означает, что правило
     сформулировано слишком узко, а не что корпус плох.
  2. Различающая сила. Сколько нарушений у правильных форм против ошибочных,
     добытых в E4. Правило, которое одинаково срабатывает на тех и на других,
     бесполезно, каким бы верным оно ни было.

    python scripts/check_grammar_rules.py --lexicon sakhaspell/data \
        --errors data/errors
"""
import argparse, collections, json, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sakhaspell.grammar import (cluster_violations, harmony_violations, is_loanword,
                                parse_nuclei, positional_violations)
from sakhaspell.lexicon import Lexicon

RULES = {
    "гармония гласных": harmony_violations,
    "начало слова": positional_violations,
    "стечение согласных": cluster_violations,
}


def rate(forms, fn, weights=None) -> tuple[float, collections.Counter]:
    """Доля форм с нарушением. С весами — доля токенов, без — доля типов."""
    bad = tot = 0
    examples = collections.Counter()
    for w in forms:
        n = weights.get(w, 1) if weights else 1
        tot += n
        v = fn(w)
        if v:
            bad += n
            if len(examples) < 5000:
                examples[f"{w} ({v[0].detail})"] += n
    return bad / max(tot, 1), examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", type=pathlib.Path, default=None)
    ap.add_argument("--errors", type=pathlib.Path)
    args = ap.parse_args()

    lex = Lexicon.load(args.lexicon)
    core = list(lex.core)
    print(f"ядро лексикона: {len(core)} форм, "
          f"{sum(lex.core.values())} вхождений\n")

    loans = sum(is_loanword(w) for w in core)
    print(f"с буквами заимствований (в е ё ж з ф ц ш щ ъ ю я): "
          f"{loans} форм, {loans/len(core):.1%} — гармония к ним не применяется\n")

    # 1. согласие с корпусом
    print("СОГЛАСИЕ С КОРПУСОМ — доля правильных форм, нарушающих правило")
    print(f"{'правило':22s} {'по типам':>10s} {'по токенам':>12s}   примеры")
    agreement = {}
    for name, fn in RULES.items():
        r_types, ex = rate(core, fn)
        r_tokens, _ = rate(core, fn, weights=lex.core)
        agreement[name] = {"types": r_types, "tokens": r_tokens}
        top = ", ".join(w for w, _ in ex.most_common(3))
        print(f"{name:22s} {r_types:9.2%} {r_tokens:11.2%}   {top}")

    # 2. различающая сила
    if not args.errors:
        return
    wrong = []
    for tier in ("web", "ocr"):
        p = args.errors / f"pairs_{tier}.tsv"
        if not p.exists():
            continue
        with p.open(encoding="utf-8") as f:
            next(f)
            for line in f:
                c = line.rstrip("\n").split("\t")
                if len(c) >= 2:
                    wrong.append((c[0], c[1]))
    print(f"\nРАЗЛИЧАЮЩАЯ СИЛА — на {len(wrong)} парах «ошибка → верно» из E4")
    print(f"{'правило':22s} {'у ошибок':>10s} {'у верных':>10s} {'разница':>9s}")
    power = {}
    for name, fn in RULES.items():
        bad_wrong = sum(1 for w, _ in wrong if fn(w))
        bad_right = sum(1 for _, r in wrong if fn(r))
        rw, rr = bad_wrong / max(len(wrong), 1), bad_right / max(len(wrong), 1)
        power[name] = {"wrong": rw, "right": rr, "lift": rw - rr}
        print(f"{name:22s} {rw:9.2%} {rr:9.2%} {rw-rr:+8.2%}")

    # где правило меняет вердикт: ошибка нарушает, верная форма — нет
    print("\nПары, где правило видит разницу:")
    shown = 0
    for w, r in wrong:
        vw, vr = harmony_violations(w), harmony_violations(r)
        if vw and not vr:
            print(f"  {w:20s} → {r:20s}  {vw[0].detail}")
            shown += 1
            if shown >= 12:
                break

    out = {"agreement": agreement, "power": power,
           "loanword_share": loans / len(core), "core_forms": len(core)}
    pathlib.Path("data/grammar_check.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
