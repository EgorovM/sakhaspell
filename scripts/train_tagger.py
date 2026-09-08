"""Обучение посимвольного тэггера восстановления ҕҥөһү.

Обучающие пары делаются на лету: берём чистое предложение из редактируемых
источников, портим его моделью ошибок и учим восстанавливать. Данных поэтому
бесконечно много и каждая эпоха видит новую порчу — переобучаться не на чем.

Порча смешивает три режима в пропорциях, отражающих реальность:
  * полная деноминализация (человек без якутской раскладки);
  * частичная, с ь вместо һ и диграфом нг (как пишут на самом деле —
    это добыто из корпуса, см. scripts/mine_errors.py);
  * нетронутый текст, чтобы модель не считала, что править надо всегда.

Последний режим важнее, чем кажется: без него модель приобретает склонность
заменять буквы даже в правильном тексте, а это ровно та ошибка, которая делает
спелчекер невыносимым.

    $E scripts/train_tagger.py --sent data/sent --out runs/tagger_v1 --steps 20000
"""
import argparse, collections, json, math, pathlib, random, sys, time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sakhaspell.model import CharTagger, TaggerConfig
from sakhaspell.tagger import (KEEP, DELETE, N_TAGS, PAD, CharVocab,
                               apply_tags, make_tags)

EDITED = ["kyym", "eder_saas", "sakha_sire", "uluus_media", "svfu",
          "sakhapechat", "childrens_lib", "sakha_texts_v2"]

# доли режимов порчи
P_CLEAN, P_FULL, P_MIXED = 0.15, 0.35, 0.50


def corrupt(s: str, rng: random.Random) -> str:
    """Порча одного предложения одним из режимов."""
    r = rng.random()
    if r < P_CLEAN:
        return s
    full = r < P_CLEAN + P_FULL
    # в полном режиме заменяем все буквы, в смешанном — часть
    keep_p = 0.0 if full else rng.uniform(0.05, 0.35)
    h_sub = rng.choice(["с", "h", "ь", "ь"])      # ь чаще прочих: так пишут
    ng = rng.random() < 0.25                      # диграф «нг» вместо ҥ
    out = []
    for ch in s:
        low = ch.lower()
        if low not in "ҕҥөүһ" or rng.random() < keep_p:
            out.append(ch)
            continue
        if low == "ҥ":
            rep = "нг" if ng else "н"
        elif low == "һ":
            rep = h_sub
        else:
            rep = {"ҕ": "г", "ө": "о", "ү": "у"}[low]
        out.append(rep.upper() if ch.isupper() else rep)
    return "".join(out)


class Pairs(IterableDataset):
    def __init__(self, files: list[pathlib.Path], vocab: CharVocab,
                 max_len: int, seed: int) -> None:
        self.files, self.vocab, self.max_len, self.seed = files, vocab, max_len, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info else 0
        nw = info.num_workers if info else 1
        rng = random.Random(self.seed + wid * 7919)
        files = self.files[wid::nw] or self.files
        while True:
            rng.shuffle(files)
            for path in files:
                with path.open(encoding="utf-8") as f:
                    for line in f:
                        s = line.rstrip("\n")
                        if not (10 <= len(s) <= self.max_len):
                            continue
                        src = corrupt(s, rng)
                        # Порча ҥ -> «нг» удлиняет строку, поэтому длину проверяем
                        # ПОСЛЕ неё: иначе испорченное предложение вылезает за
                        # позиционные эмбеддинги и обучение падает на ассерте.
                        if len(src) > self.max_len:
                            continue
                        tags = make_tags(src, s)
                        if tags is None or len(tags) != len(src):
                            continue
                        yield (torch.tensor(self.vocab.encode(src)),
                               torch.tensor(tags))


def collate(batch):
    n = max(len(x) for x, _ in batch)
    ids = torch.full((len(batch), n), PAD, dtype=torch.long)
    tgt = torch.full((len(batch), n), -100, dtype=torch.long)
    for i, (x, t) in enumerate(batch):
        ids[i, :len(x)] = x
        tgt[i, :len(t)] = t
    return ids, tgt


@torch.no_grad()
def evaluate(model, vocab, files, device, n_sent=2000, max_len=256, seed=999):
    """Метрика по позициям, которые модель должна была изменить, плюс отдельно
    ложные изменения на позициях, которые надо было оставить."""
    model.eval()
    rng = random.Random(seed)
    pool = []
    for path in files:
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i > 20000:
                    break
                s = line.rstrip("\n")
                if 10 <= len(s) <= max_len:
                    pool.append(s)
    rng.shuffle(pool)
    pool = pool[:n_sent]

    tp = fp = fn = 0
    exact = total = 0
    for i in range(0, len(pool), 128):
        chunk = pool[i:i + 128]
        srcs = [corrupt(s, random.Random(seed + j)) for j, s in enumerate(chunk, i)]
        pairs = [(a, b) for a, b in zip(srcs, chunk)
                 if len(a) <= max_len and make_tags(a, b) is not None]
        if not pairs:
            continue
        srcs = [a for a, _ in pairs]
        golds = [b for _, b in pairs]
        n = max(len(a) for a in srcs)
        ids = torch.full((len(srcs), n), PAD, dtype=torch.long)
        for j, a in enumerate(srcs):
            e = vocab.encode(a)
            ids[j, :len(e)] = torch.tensor(e)
        pred = model(ids.to(device)).argmax(-1).cpu()
        for j, (a, b) in enumerate(zip(srcs, golds)):
            gt = make_tags(a, b)
            pt = pred[j, :len(a)].tolist()
            for g, p in zip(gt, pt):
                if g != KEEP and p == g:
                    tp += 1
                elif g != KEEP and p != g:
                    fn += 1
                elif g == KEEP and p != KEEP:
                    fp += 1
            total += 1
            exact += apply_tags(a, pt) == b
    model.train()
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-9),
            "sentence_exact": exact / max(total, 1), "n": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sent", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=384)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    files = [args.sent / f"{s}.txt" for s in EDITED]
    files = [p for p in files if p.exists() and p.stat().st_size > 0]
    print(f"источников для обучения: {len(files)}", flush=True)

    # словарь символов по выборке
    counts = collections.Counter()
    for path in files:
        with path.open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i > 60000:
                    break
                counts.update(line)
    # символы, возникающие только при порче, должны быть в словаре обязательно
    counts.update({c: 10**6 for c in "гнoоуүсхьцҕҥөһ"})
    vocab = CharVocab.from_counts(counts, min_freq=50)
    vocab.save(args.out / "vocab.json")
    print(f"словарь символов: {len(vocab)}", flush=True)

    cfg = TaggerConfig(vocab_size=len(vocab), d_model=args.d_model,
                       n_layers=args.layers, n_heads=max(args.d_model // 64, 1),
                       d_ff=args.d_model * 4, max_len=args.max_len + 32)
    model = CharTagger(cfg).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"параметров: {n_par/1e6:.1f}M, устройство {device}", flush=True)
    (args.out / "config.json").write_text(
        json.dumps({**cfg.__dict__, "params": n_par}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    ds = Pairs(files, vocab, args.max_len, args.seed)
    dl = DataLoader(ds, batch_size=args.bs, num_workers=args.workers,
                    collate_fn=collate, pin_memory=True, persistent_workers=True,
                    prefetch_factor=4)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01,
                            betas=(0.9, 0.98))
    warmup = min(1000, args.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warmup, 1.0) *
        (0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1.0)))))
    scaler = torch.amp.GradScaler(device)
    lossf = nn.CrossEntropyLoss(ignore_index=-100)

    best, t0, run = 0.0, time.perf_counter(), 0.0
    for step, (ids, tgt) in enumerate(dl, 1):
        ids, tgt = ids.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        with torch.amp.autocast(device, dtype=torch.bfloat16):
            logits = model(ids)
            loss = lossf(logits.reshape(-1, N_TAGS), tgt.reshape(-1))
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        run += loss.item()

        if step % 200 == 0:
            print(f"шаг {step:6d}  loss {run/200:.4f}  lr {sched.get_last_lr()[0]:.2e}"
                  f"  {step*args.bs/(time.perf_counter()-t0):.0f} предл./с", flush=True)
            run = 0.0
        if step % args.eval_every == 0 or step == args.steps:
            m = evaluate(model, vocab, files, device, max_len=args.max_len)
            print(f"  оценка: P {m['precision']:.4f}  R {m['recall']:.4f}  "
                  f"F1 {m['f1']:.4f}  предложений точно {m['sentence_exact']:.4f}",
                  flush=True)
            torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                        "step": step, "metrics": m}, args.out / "last.pt")
            if m["f1"] > best:
                best = m["f1"]
                torch.save({"model": model.state_dict(), "cfg": cfg.__dict__,
                            "step": step, "metrics": m}, args.out / "best.pt")
                print(f"  -> новый лучший F1 {best:.4f}", flush=True)
        if step >= args.steps:
            break

    print(f"готово. лучший F1 {best:.4f}, время {(time.perf_counter()-t0)/60:.1f} мин")


if __name__ == "__main__":
    main()
