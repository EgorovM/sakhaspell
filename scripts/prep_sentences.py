"""Корпус документов -> предложения по источникам.

Почему по источникам, а не в одну кучу: лексикон спелчекера нельзя строить на
OCR-тексте. OCR-ошибка даёт правдоподобное несуществующее слово, которое потом
навсегда поселится в словаре и не будет подчёркиваться никогда. Веб-источники
(oscar/madlad/fineweb2) содержат настоящие орфографические ошибки живых людей —
их тоже нельзя пускать в словарь как эталон.

Поэтому: режем на предложения, помечаем источником, а решение о доверии
принимаем на следующем шаге, при сборке лексикона.

    /usr/bin/python3 scripts/prep_sentences.py \
        --corpus /home/jovyan/shared-volume/sakha-emb/data/corpus/corpus_sah.jsonl \
        --out data/sent --procs 90
"""
import argparse, collections, hashlib, json, pathlib, sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sakhaspell.norm import normalize_for_lexicon
from sakhaspell.tokenize import sentences, tokenize, alphabet_share

# Источники, размеченные по происхождению текста. Уровень доверия здесь не
# зашивается — он выбирается при сборке лексикона, а тут только факт.
TIERS = {
    "kyym": "edited", "eder_saas": "edited", "sakha_sire": "edited",
    "uluus_media": "edited", "svfu": "edited", "sakhapechat": "edited",
    "childrens_lib": "edited", "sakha_texts_v2": "edited",
    "fineweb2": "web", "madlad": "web", "oscar": "web",
    "nii_olonho": "ocr", "ocr_agitator": "ocr", "ocr_cholbon": "ocr",
    "ocr_kyrylgen": "ocr", "ocr_ленин_суолунан": "ocr",
    "ocr_хотугу_сулус": "ocr", "ocr_школаразвитиясайдыыкыьата": "ocr",
}

# Русские служебные слова: пороги из scripts/calibrate_lang.py проекта sakha-emb,
# там они откалиброваны на 800+800 документах. Здесь применяются к предложению,
# потому что якутские новости регулярно вставляют русские цитаты целиком.
RU_STOP = set("""и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по
только ее мне было вот от меня еще нет о из ему когда даже ну вдруг ли если уже или ни быть был
него до вас уж вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их
чем была сам без чего раз тоже себе под будет тогда кто этот того потому этого какой совсем ним
здесь этом один почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два
об другой хоть после над больше тот через эти нас про всего них какая много разве три эту моя
хорошо свою этой перед иногда лучше чуть том нельзя такой им более всегда конечно всю между""".split())

MIN_CHARS, MAX_CHARS, MIN_WORDS = 20, 400, 4


def process_doc(payload: str) -> tuple[str, list[str]] | None:
    try:
        d = json.loads(payload)
    except json.JSONDecodeError:
        return None
    src = d.get("source", "?")
    text = normalize_for_lexicon(d.get("text", ""))
    out = []
    for a, b in sentences(text):
        s = text[a:b].strip()
        if not (MIN_CHARS <= len(s) <= MAX_CHARS):
            continue
        toks = [t for t in tokenize(s) if t.is_word]
        if len(toks) < MIN_WORDS:
            continue
        # чужое письмо внутри предложения
        if alphabet_share(s) < 0.98:
            continue
        # доля не-букв: таблицы, списки, обрывки вёрстки
        if sum(c.isalpha() or c.isspace() for c in s) / len(s) < 0.85:
            continue
        low = [t.text.lower() for t in toks]
        if sum(w in RU_STOP for w in low) / len(low) > 0.18:
            continue
        out.append(s)
    return src, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--procs", type=int, default=90)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    handles = {}
    seen: set[int] = set()
    stats = collections.Counter()

    def sink(src: str):
        if src not in handles:
            handles[src] = (args.out / f"{src}.txt").open("w", encoding="utf-8")
        return handles[src]

    with args.corpus.open(encoding="utf-8") as f, \
            ProcessPoolExecutor(max_workers=args.procs) as ex:
        for i, res in enumerate(ex.map(process_doc, f, chunksize=64), 1):
            if res is None:
                stats["bad_json"] += 1
                continue
            src, sents = res
            fh = sink(src)
            for s in sents:
                stats[f"{src}/seen"] += 1
                h = int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "big")
                if h in seen:
                    stats[f"{src}/dup"] += 1
                    continue
                seen.add(h)
                fh.write(s + "\n")
                stats[f"{src}/kept"] += 1
            if i % 50000 == 0:
                print(f"  {i} док., уникальных предложений {len(seen)}", flush=True)

    for fh in handles.values():
        fh.close()
    stats["_total_unique"] = len(seen)
    (args.out / "stats.json").write_text(
        json.dumps({"tiers": TIERS, "stats": dict(stats)}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"готово: {len(seen)} уникальных предложений в {len(handles)} источниках")


if __name__ == "__main__":
    main()
