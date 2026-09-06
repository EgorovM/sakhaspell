/* sakhaspell — браузерный движок проверки орфографии якутского языка.
 *
 * Портирует слои L0 (нормализация) и L1 (словарь) из Python-версии. Слой L2,
 * посимвольный тэггер, здесь не работает: он требует torch и весит 43 МБ, а
 * страница должна открываться быстро. Словарь целиком лежит в браузере, так что
 * ни одного запроса на сервер после загрузки не уходит — текст никуда не едет.
 *
 * Отличие от Python-версии одно: вместо префиксного дерева со взвешенным
 * расстоянием здесь порождение кандидатов на расстоянии 1 с проверкой по
 * словарю. На 300 тысячах форм проверка по хэшу дешевле обхода дерева, а
 * расстояние 1 покрывает подавляющее большинство опечаток.
 */

// --- L0: нормализация --------------------------------------------------------

// Чужие кириллические буквы, найденные в корпусе 1.65 млрд символов.
// Число рядом — сколько раз встретилось; это не догадки, а факт.
const CONFUSABLES = {
  'ѳ': 'ө', 'Ѳ': 'Ө',   // фита, 27 513
  'ӧ': 'ө', 'Ӧ': 'Ө',   // о с диерезисом, 24
  'ң': 'ҥ', 'Ң': 'Ҥ',   // казахское эн, 16 158
  'ӊ': 'ҥ', 'Ӊ': 'Ҥ',   // эн с хвостом, 5 591
  'ғ': 'ҕ', 'Ғ': 'Ҕ',   // казахское гha, 142
  'ұ': 'ү', 'Ұ': 'Ү',   // казахское у, 656
  'ҋ': 'й', 'Ҋ': 'Й',   // и краткое с хвостом, 774
};

const INVISIBLE = /[­﻿​‌‍⁠]/g;

const LATIN_TO_CYR = {
  a: 'а', c: 'с', e: 'е', o: 'о', p: 'р', x: 'х', y: 'у',
  A: 'А', B: 'В', C: 'С', E: 'Е', H: 'Н', K: 'К', M: 'М',
  O: 'О', P: 'Р', T: 'Т', X: 'Х', Y: 'У',
};

function isCyrillic(ch) {
  const c = ch.codePointAt(0);
  return (c >= 0x0400 && c <= 0x052f) || (c >= 0xa640 && c <= 0xa69f);
}

/** Латинские гомоглифы внутри кириллического слова. «Cаха» чинится, «CD-ROM» нет. */
function fixMixedScript(word) {
  const letters = [...word].filter((c) => /\p{L}/u.test(c));
  if (!letters.length) return word;
  const lat = letters.filter((c) => !isCyrillic(c));
  if (!lat.length) return word;
  const cyr = letters.length - lat.length;
  if (cyr / letters.length < 0.5) return word;
  if (!lat.every((c) => c in LATIN_TO_CYR)) return word;
  return [...word].map((c) => LATIN_TO_CYR[c] ?? c).join('');
}

/** Полная нормализация. Длина может измениться только за счёт невидимых символов. */
function normalize(text) {
  let t = text.normalize('NFC').replace(INVISIBLE, '');
  t = [...t].map((c) => CONFUSABLES[c] ?? c).join('');
  return t.replace(/[\p{L}'’-]+/gu, (w) => (/[A-Za-z]/.test(w) ? fixMixedScript(w) : w));
}

// --- токенизация с офсетами --------------------------------------------------

const WORD_RE = /([Ѐ-ԯꙀ-ꚟ]+(?:[-'’][Ѐ-ԯꙀ-ꚟ]+)*)|([A-Za-z]+(?:[-'’][A-Za-z]+)*)|(\d+(?:[.,:/-]\d+)*(?:-[Ѐ-ԯꙀ-ꚟ]+)*)/gu;

function tokenize(text) {
  const out = [];
  let m;
  WORD_RE.lastIndex = 0;
  while ((m = WORD_RE.exec(text)) !== null) {
    const kind = m[1] ? 'word' : m[2] ? 'latin' : 'number';
    out.push({ text: m[0], start: m.index, end: m.index + m[0].length, kind });
  }
  return out;
}

// --- модель ошибок -----------------------------------------------------------

/* Замены добыты из корпуса (scripts/mine_errors.py), а не придуманы.
 * Самая частая реальная ошибка в вебе — «ь» вместо «һ»: уьу (997 вхождений),
 * киьини (960), эьиги (512). Мягкий знак стоит на русской раскладке и похож
 * на һ начертанием. В OCR так же ведёт себя «ц» вместо «ҥ». */
const RESTORE = {
  г: ['ҕ'], н: ['ҥ'], о: ['ө'], у: ['ү'],
  с: ['һ'], ь: ['һ'], х: ['һ'], h: ['һ'], x: ['һ'],
  ц: ['ҥ'], g: ['ҕ'], n: ['ҥ'], y: ['ү'],
};
const MAX_VARIANTS = 4096;

/** Все варианты слова с восстановленными якутскими буквами, по числу изменений. */
function restorationVariants(word) {
  const low = word.toLowerCase();
  let slots = [...low].map((ch) => (RESTORE[ch] ? [ch, ...RESTORE[ch]] : [ch]));

  let n = 1;
  for (const s of slots) n *= s.length;
  if (n > MAX_VARIANTS) {
    // слишком много развилок — восстанавливаем только самые частые буквы
    slots = slots.map((s) => (['г', 'о', 'у'].includes(s[0]) ? s : [s[0]]));
  }

  const out = [];
  const seen = new Set();
  const build = (i, acc, changed) => {
    if (out.length >= MAX_VARIANTS) return;
    if (i === slots.length) {
      if (!seen.has(acc)) { seen.add(acc); out.push([changed, acc]); }
      return;
    }
    for (const c of slots[i]) build(i + 1, acc + c, changed + (c !== low[i] ? 1 : 0));
  };
  build(0, '', 0);

  if (low.includes('нг')) {
    const v = low.replaceAll('нг', 'ҥ');
    if (!seen.has(v)) out.push([1, v]);
  }
  out.sort((a, b) => a[0] - b[0]);
  return out;
}

const KEYBOARD = (() => {
  const rows = ['йцукенгшщзхъ', 'фывапролджэ', 'ячсмитьбю'];
  const map = {};
  rows.forEach((row, r) => {
    [...row].forEach((c, i) => {
      const n = new Set();
      if (i) n.add(row[i - 1]);
      if (i + 1 < row.length) n.add(row[i + 1]);
      for (const dr of [-1, 1]) {
        const rr = r + dr;
        if (rr >= 0 && rr < rows.length) {
          for (const j of [i - 1, i, i + 1]) {
            if (j >= 0 && j < rows[rr].length) n.add(rows[rr][j]);
          }
        }
      }
      map[c] = [...n];
    });
  });
  return map;
})();

const ALPHABET = 'абвгҕдеёжзийклмнҥоөпрсһтуүфхцчшщъыьэюя';

// --- движок ------------------------------------------------------------------

const BASE = 100;
const CHEAP = 35;          // замена ҕ↔г и подобные: это не опечатка, а письмо без раскладки
const FREQ_WEIGHT = 12;

export class SakhaSpell {
  constructor(forms, freq, shadows) {
    this.forms = forms;      // Map: форма -> частота
    this.shadows = shadows;  // Map: искажение -> правильная форма
    this.freq = freq;
  }

  static async load(base = 'data') {
    const un = async (p) => {
      const r = await fetch(p);
      if (!r.ok) throw new Error(`не удалось загрузить ${p}`);
      const ds = new DecompressionStream('gzip');
      return new Response(r.body.pipeThrough(ds));
    };

    const meta = await (await fetch(`${base}/meta.json`)).json();
    const text = await (await un(`${base}/forms.txt.gz`)).text();
    const freqBuf = new Uint8Array(await (await un(`${base}/freq.bin.gz`)).arrayBuffer());
    const shadowObj = JSON.parse(await (await un(`${base}/shadows.json.gz`)).text());

    // разворачиваем фронт-кодирование
    const forms = new Map();
    const lines = text.split('\n');
    let prev = '';
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      if (!line) continue;
      const keep = line.charCodeAt(0) - 48;
      const w = prev.slice(0, keep) + line.slice(1);
      prev = w;
      forms.set(w, Math.pow(2, freqBuf[i] / meta.freq_scale));
    }
    const shadows = new Map(Object.entries(shadowObj));
    return new SakhaSpell(forms, freqBuf, shadows);
  }

  get size() { return this.forms.size; }

  /** Принимает ли словарь эту форму. Дефисные сложения проверяются по частям. */
  accepts(word, depth = 0) {
    const w = word.toLowerCase();
    if (this.shadows.has(w)) return false;
    if (this.forms.has(w)) return true;
    if (depth === 0 && /[-'’]/.test(w)) {
      const parts = w.split(/[-'’]/).filter(Boolean);
      if (parts.length > 1 && parts.every((p) => this.accepts(p, 1))) return true;
    }
    return false;
  }

  /** Кандидаты на расстоянии одной правки: удаление, вставка, замена, перестановка. */
  *editsOne(w) {
    const n = w.length;
    for (let i = 0; i < n; i++) yield [w.slice(0, i) + w.slice(i + 1), BASE];
    for (let i = 0; i + 1 < n; i++) {
      yield [w.slice(0, i) + w[i + 1] + w[i] + w.slice(i + 2), BASE];
    }
    for (let i = 0; i < n; i++) {
      const neigh = KEYBOARD[w[i]] ?? [];
      for (const c of ALPHABET) {
        if (c === w[i]) continue;
        yield [w.slice(0, i) + c + w.slice(i + 1), neigh.includes(c) ? 70 : BASE];
      }
    }
    for (let i = 0; i <= n; i++) {
      for (const c of ALPHABET) yield [w.slice(0, i) + c + w.slice(i), BASE];
    }
  }

  suggest(word, limit = 5) {
    const low = word.toLowerCase();
    const found = new Map();      // форма -> цена

    const add = (form, cost) => {
      if (form === low || !this.forms.has(form)) return;
      const prev = found.get(form);
      if (prev === undefined || cost < prev) found.set(form, cost);
    };

    // 0. известная тень искажения — исправление знаем точно
    const origin = this.shadows.get(low);
    if (origin) add(origin, 1);

    // 1. восстановление спецбукв: дёшево и точно, это самый частый случай
    for (const [changed, v] of restorationVariants(low)) {
      if (changed) add(v, CHEAP * changed);
    }

    // 2. Опечатки. Полный перебор нужен, только если восстановление не дало
    // дешёвого кандидата: деноминализация несравнимо частотнее опечаток.
    const best = found.size ? Math.min(...found.values()) : Infinity;
    if (best > 105) {
      for (const [cand, cost] of this.editsOne(low)) add(cand, cost);
    }

    const scored = [...found].map(([form, cost]) => ({
      form,
      cost,
      freq: this.forms.get(form) ?? 1,
      score: cost - FREQ_WEIGHT * Math.log10(Math.max(this.forms.get(form) ?? 1, 1)),
    }));
    scored.sort((a, b) => a.score - b.score);
    return scored.slice(0, limit).map((s) => matchCase(word, s.form));
  }

  /** Проверка текста. Возвращает правки с офсетами в переданной строке. */
  check(text) {
    const issues = [];
    for (const t of tokenize(text)) {
      if (t.kind !== 'word') continue;
      if (this.accepts(t.text)) continue;
      issues.push({
        start: t.start, end: t.end, word: t.text,
        suggestions: this.suggest(t.text),
        reason: this.shadows.has(t.text.toLowerCase()) ? 'искажение' : 'не в словаре',
      });
    }
    return issues;
  }

  correct(text) {
    let out = '';
    let prev = 0;
    for (const iss of this.check(text)) {
      if (!iss.suggestions.length) continue;
      out += text.slice(prev, iss.start) + iss.suggestions[0];
      prev = iss.end;
    }
    return out + text.slice(prev);
  }
}

function matchCase(src, form) {
  if (src === src.toLowerCase()) return form;
  if (src === src.toUpperCase() && src.length > 1) return form.toUpperCase();
  if (src[0] === src[0].toUpperCase()) return form[0].toUpperCase() + form.slice(1);
  return form;
}

export { normalize, tokenize, restorationVariants };
