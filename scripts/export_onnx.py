"""Экспорт тэггера в ONNX с квантованием — для запуска в браузере.

Страница работает целиком в браузере и потому обходится без контекстной модели:
чекпоинт весит 43 МБ, столько не грузят ради демонстрации. Из-за этого веб-версия
даёт 93.16% на восстановлении ҕҥөһү вместо 96.51%.

Квантование до int8 сокращает модель примерно вчетверо. Тэггер для этого удобен:
он маленький, посимвольный и неавторегрессивный — один проход энкодера, никакого
цикла декодирования, которое в браузере было бы медленным.

Точность после квантования проверяется тут же, на тех же парах, что при
обучении: терять больше половины пункта F1 ради размера не стоит.

    $E scripts/export_onnx.py --run runs/tagger_v2 --out docs/model
"""
import argparse, json, pathlib, random, sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from sakhaspell.model import CharTagger, TaggerConfig
from sakhaspell.tagger import KEEP, PAD, CharVocab, apply_tags, make_tags

MAX_LEN = 256


def load(run: pathlib.Path, device: str = "cpu"):
    ckpt = torch.load(run / "best.pt", map_location=device, weights_only=False)
    cfg = TaggerConfig(**ckpt["cfg"])
    model = CharTagger(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, CharVocab.load(run / "vocab.json"), ckpt.get("metrics", {})


def encode(texts, vocab, n):
    ids = torch.full((len(texts), n), PAD, dtype=torch.long)
    for i, t in enumerate(texts):
        e = vocab.encode(t[:n])
        ids[i, :len(e)] = torch.tensor(e)
    return ids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=pathlib.Path, required=True)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    ap.add_argument("--sent", type=pathlib.Path, default=pathlib.Path("data/sent"))
    ap.add_argument("--no-quantize", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    model, cfg, vocab, metrics = load(args.run)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"модель: {n_par/1e6:.1f}M параметров, словарь {len(vocab)}, "
          f"F1 при обучении {metrics.get('f1', float('nan')):.4f}")

    # Ось длины должна быть динамической: паддинг каждого предложения до 256
    # символов обошёлся бы в браузере дороже самой модели.
    #
    # Классический трассировочный экспортёр здесь не годится: внутри
    # nn.TransformerEncoderLayer форма головы внимания зашивается в Reshape
    # константой из пробного входа, и модель падает на любой другой длине.
    # Экспортёр dynamo строит граф из FX и сохраняет символьные размерности.
    dummy = torch.full((2, 32), 5, dtype=torch.long)
    fp32 = args.out / "tagger.onnx"
    batch = torch.export.Dim("batch", min=1, max=256)
    seq = torch.export.Dim("seq", min=2, max=MAX_LEN)
    torch.onnx.export(
        model, (dummy,), str(fp32),
        input_names=["input_ids"], output_names=["logits"],
        dynamic_shapes={"x": {0: batch, 1: seq}},
        opset_version=17, dynamo=True)
    print(f"fp32: {fp32.stat().st_size/1e6:.1f} МБ")

    out_path = fp32
    if not args.no_quantize:
        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic
        except ImportError:
            print("нет onnxruntime.quantization — оставляю fp32", flush=True)
        else:
            q = args.out / "tagger.int8.onnx"
            # Динамическое квантование: веса в int8, активации считаются на лету.
            # Для трансформера этого достаточно, калибровочный набор не нужен.
            quantize_dynamic(str(fp32), str(q), weight_type=QuantType.QInt8)
            print(f"int8: {q.stat().st_size/1e6:.1f} МБ "
                  f"(в {fp32.stat().st_size/q.stat().st_size:.1f} раза меньше)")
            out_path = q

    vocab.save(args.out / "vocab.json")
    (args.out / "meta.json").write_text(json.dumps(
        {"params": n_par, "max_len": MAX_LEN, "vocab": len(vocab),
         "d_model": cfg.d_model, "layers": cfg.n_layers,
         "train_f1": metrics.get("f1")}, ensure_ascii=False, indent=2),
        encoding="utf-8")

    # --- сверка: квантование не должно заметно портить качество --------------
    files = list(args.sent.glob("*.txt"))
    if not files:
        print("нет данных для сверки, пропускаю")
        return
    rng = random.Random(7)
    pool = []
    with files[0].open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i > 4000:
                break
            s = line.rstrip("\n")
            if 20 <= len(s) <= MAX_LEN:
                pool.append(s)
    rng.shuffle(pool)
    pool = pool[:400]

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from train_tagger import corrupt
    srcs, golds = [], []
    for j, s in enumerate(pool):
        c = corrupt(s, random.Random(1000 + j))
        if len(c) <= MAX_LEN and make_tags(c, s) is not None:
            srcs.append(c)
            golds.append(s)

    def score(pred_fn) -> tuple[float, float]:
        tp = fp = fn = exact = 0
        for i in range(0, len(srcs), 32):
            chunk, gold = srcs[i:i + 32], golds[i:i + 32]
            n = max(len(t) for t in chunk)
            ids = encode(chunk, vocab, n)
            pred = pred_fn(ids)
            for j, (a, b) in enumerate(zip(chunk, gold)):
                gt = make_tags(a, b)
                pt = pred[j][:len(a)]
                for g, p in zip(gt, pt):
                    if g != KEEP and p == g:
                        tp += 1
                    elif g != KEEP and p != g:
                        fn += 1
                    elif g == KEEP and p != KEEP:
                        fp += 1
                exact += apply_tags(a, list(pt)) == b
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        return 2 * p * r / max(p + r, 1e-9), exact / max(len(srcs), 1)

    with torch.no_grad():
        f1_t, ex_t = score(lambda ids: model(ids).argmax(-1).tolist())
    print(f"\nсверка на {len(srcs)} парах")
    print(f"  torch  F1 {f1_t:.4f}  предложений точно {ex_t:.4f}")

    try:
        import onnxruntime as ort
    except ImportError:
        print("  onnxruntime не установлен — ONNX не проверен")
        return
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    f1_o, ex_o = score(lambda ids: sess.run(
        None, {"input_ids": ids.numpy()})[0].argmax(-1).tolist())
    print(f"  onnx   F1 {f1_o:.4f}  предложений точно {ex_o:.4f}")
    print(f"  разница F1 {f1_o - f1_t:+.4f}")


if __name__ == "__main__":
    main()
