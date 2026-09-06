"""Задержка и память — то, что решает, годится ли чекер для проверки при наборе.

Точка отсчёта — замеры bashspell на проде: `spell()` 0.17–0.21 мс на слово,
`suggest()` 115–132 мс на слово. Второе число делает подсказки непригодными
для интерактивной работы без кэша.

    /usr/bin/python3 scripts/bench_latency.py --lexicon data/lexicon
"""
import argparse, json, pathlib, random, resource, statistics, sys, time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sakhaspell.lexicon import Lexicon
from sakhaspell.checker import SpellChecker
from sakhaspell.tokenize import tokenize


def rss_mb() -> float:
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / (1024 * 1024) if sys.platform == "darwin" else r / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", type=pathlib.Path, required=True)
    ap.add_argument("--bench", type=pathlib.Path)
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    base = rss_mb()
    t0 = time.perf_counter()
    lex = Lexicon.load(args.lexicon)
    t_lex = time.perf_counter() - t0
    m_lex = rss_mb()

    t0 = time.perf_counter()
    ch = SpellChecker(lex)
    t_trie = time.perf_counter() - t0
    m_trie = rss_mb()

    print(f"загрузка лексикона   {t_lex:6.2f} с   +{m_lex-base:7.1f} МБ")
    print(f"построение дерева    {t_trie:6.2f} с   +{m_trie-m_lex:7.1f} МБ")
    print(f"итого в памяти       {m_trie:7.1f} МБ, форм в ядре {len(lex.core)}")

    rng = random.Random(7)
    good = rng.sample(sorted(lex.core), min(args.n, len(lex.core)))
    bad = [w.translate(str.maketrans("ҕҥөүһ", "гнoус")) for w in good]

    # проверка слова
    t0 = time.perf_counter()
    for w in good:
        lex.check_form(w)
    dt_ok = (time.perf_counter() - t0) / len(good) * 1000

    t0 = time.perf_counter()
    for w in bad:
        lex.check_form(w)
    dt_bad = (time.perf_counter() - t0) / len(bad) * 1000

    # подсказки
    sample = bad[:300]
    times = []
    for w in sample:
        t0 = time.perf_counter()
        ch.suggest(w)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()

    print(f"\nпроверка слова (в словаре)   {dt_ok:8.4f} мс")
    print(f"проверка слова (не в словаре) {dt_bad:8.4f} мс")
    print(f"подсказки: медиана {statistics.median(times):.2f} мс, "
          f"p90 {times[int(len(times)*0.9)]:.2f} мс, макс {times[-1]:.2f} мс")

    if args.bench:
        sents = []
        with (args.bench / "dev.denorm_mixed.jsonl").open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 300:
                    break
                sents.append(json.loads(line)["src"])
        t0 = time.perf_counter()
        n_tok = 0
        for s in sents:
            ch.check(s)
            n_tok += sum(1 for t in tokenize(s) if t.is_word)
        dt = time.perf_counter() - t0
        print(f"\nтекст целиком: {len(sents)/dt:.0f} предл./с, {n_tok/dt:.0f} слов/с")
        print(f"на предложение в среднем {dt/len(sents)*1000:.2f} мс")

    print(f"\nдля сравнения, bashspell на проде: проверка 0.17–0.21 мс, "
          f"подсказки 115–132 мс")


if __name__ == "__main__":
    main()
