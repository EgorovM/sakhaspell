/* Контекстная модель в браузере: посимвольный тэггер через onnxruntime-web.
 *
 * Словарный слой не видит ошибку, если испорченное слово само является законным:
 * «киси», «сана», «ого». Такое решается только контекстом, и на восстановлении
 * ҕҥөһү модель поднимает качество с 93.16% до 96.51%.
 *
 * Модель весит 11 МБ после квантования в int8 (из 43 МБ), поэтому грузится не
 * при открытии страницы, а по требованию: человек, который просто проверил одно
 * предложение, платить за неё не должен.
 *
 * Разметка, а не порождение: модель предсказывает для каждого символа, оставить
 * его или заменить. Изменить символ, которому предсказано «оставить», она
 * физически не может — поэтому текст, которого она не касается, гарантированно
 * не испортится.
 */

const ORT_URL = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js';

// Метки. Порядок обязан совпадать с sakhaspell/tagger.py.
const KEEP = 0, DELETE = 1;
const TAG_CHAR = { 2: 'ҕ', 3: 'ҥ', 4: 'ө', 5: 'ү', 6: 'һ' };
const PAD = 0, UNK = 1;
const MAX_LEN = 256;

export class Tagger {
  constructor(session, stoi, threshold = 0.9) {
    this.session = session;
    this.stoi = stoi;
    // Порог несимметричен и применяется только к правкам: заменить символ
    // модель может лишь при уверенности выше порога, оставить — всегда.
    // Пропущенная ошибка не стоит пользователю ничего, лишняя правка стоит
    // доверия ко всему инструменту.
    this.threshold = threshold;
  }

  static async load(base = 'model', onProgress = null) {
    if (!window.ort) await loadScript(ORT_URL);
    const ort = window.ort;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.simd = true;

    const bytes = await fetchWithProgress(`${base}/tagger.int8.onnx`, onProgress);
    const chars = await (await fetch(`${base}/vocab.json`)).json();
    const itos = ['<pad>', '<unk>', '<bos>', '<eos>', ...chars];
    const stoi = new Map(itos.map((c, i) => [c, i]));

    const session = await ort.InferenceSession.create(bytes, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });
    return new Tagger(session, stoi);
  }

  /** Восстанавливает якутские буквы в строке. */
  async restore(text) {
    if (!text.trim()) return text;
    // Длинный текст режем по строкам: модель обучалась на предложениях, и
    // позиционные эмбеддинги дальше 256 символов не определены.
    const chunks = splitChunks(text, MAX_LEN);
    const out = [];
    for (const chunk of chunks) out.push(await this._one(chunk));
    return out.join('');
  }

  async _one(text) {
    const n = Math.min(text.length, MAX_LEN);
    if (n < 2) return text;
    const ids = new BigInt64Array(n);
    for (let i = 0; i < n; i++) {
      ids[i] = BigInt(this.stoi.get(text[i]) ?? UNK);
    }
    const ort = window.ort;
    const input = new ort.Tensor('int64', ids, [1, n]);
    const { logits } = await this.session.run({ input_ids: input });
    const data = logits.data;
    const nTags = logits.dims[2];

    const parts = [];
    for (let i = 0; i < n; i++) {
      // softmax по метке считаем на месте: тащить его в граф ради семи чисел
      // на позицию незачем
      let max = -Infinity, best = KEEP, sum = 0;
      for (let t = 0; t < nTags; t++) {
        const v = data[i * nTags + t];
        if (v > max) { max = v; best = t; }
      }
      for (let t = 0; t < nTags; t++) sum += Math.exp(data[i * nTags + t] - max);
      const conf = 1 / sum;

      const ch = text[i];
      if (best === KEEP || conf < this.threshold) { parts.push(ch); continue; }
      if (best === DELETE) continue;
      const rep = TAG_CHAR[best];
      parts.push(rep ? (ch === ch.toUpperCase() && ch !== ch.toLowerCase()
        ? rep.toUpperCase() : rep) : ch);
    }
    return parts.join('') + text.slice(n);
  }
}

/** Режет текст на куски не длиннее limit, по границам предложений и пробелов. */
function splitChunks(text, limit) {
  if (text.length <= limit) return [text];
  const out = [];
  let rest = text;
  while (rest.length > limit) {
    let cut = rest.lastIndexOf(' ', limit);
    const dot = rest.lastIndexOf('. ', limit);
    if (dot > limit * 0.5) cut = dot + 1;
    if (cut <= 0) cut = limit;
    out.push(rest.slice(0, cut));
    rest = rest.slice(cut);
  }
  if (rest) out.push(rest);
  return out;
}

function loadScript(src) {
  return new Promise((res, rej) => {
    const s = document.createElement('script');
    s.src = src;
    s.onload = res;
    s.onerror = () => rej(new Error(`не загрузился ${src}`));
    document.head.appendChild(s);
  });
}

async function fetchWithProgress(url, onProgress) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`не загрузилась модель: ${r.status}`);
  const total = +r.headers.get('content-length') || 0;
  if (!onProgress || !total || !r.body) return new Uint8Array(await r.arrayBuffer());

  const reader = r.body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    onProgress(got / total);
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const p of parts) { buf.set(p, off); off += p.length; }
  return buf;
}
