/* DropScore — prototype frontend.
 *
 * Everything below the URL parsing is a *simulation*: there is no backend yet.
 * The pipeline stages mirror the real plan in docs/PIPELINE.md so that swapping
 * in a live API is mostly a matter of replacing runPipeline() with polling a job.
 */

// ── helpers ──────────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const el = (tag, cls, text) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (text != null) n.textContent = text;
  return n;
};
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Extract an 11-char YouTube video id from the common URL shapes. */
function parseYouTubeId(raw) {
  const s = (raw || '').trim();
  if (!s) return null;
  if (/^[\w-]{11}$/.test(s)) return s;

  let u;
  try {
    u = new URL(s.startsWith('http') ? s : 'https://' + s);
  } catch {
    return null;
  }

  const host = u.hostname.replace(/^www\./, '').replace(/^m\./, '');
  const ok = ['youtube.com', 'youtu.be', 'youtube-nocookie.com', 'music.youtube.com'];
  if (!ok.includes(host)) return null;

  if (host === 'youtu.be') {
    const id = u.pathname.slice(1).split('/')[0];
    return /^[\w-]{11}$/.test(id) ? id : null;
  }

  const v = u.searchParams.get('v');
  if (v && /^[\w-]{11}$/.test(v)) return v;

  const m = u.pathname.match(/\/(?:shorts|embed|v|live)\/([\w-]{11})/);
  return m ? m[1] : null;
}

/** Deterministic PRNG so the same link always "transcribes" the same way. */
function seededRandom(seed) {
  let h = 2166136261 >>> 0;
  for (const ch of String(seed)) {
    h ^= ch.charCodeAt(0);
    h = Math.imul(h, 16777619);
  }
  return () => {
    h += 0x6d2b79f5;
    let t = h;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// ── state ────────────────────────────────────────────────────────────
const state = { videoId: null, job: null, cancelled: false, timer: null, t0: 0 };

const inputStage = $('#input-stage');
const procStage = $('#processing-stage');
const resStage = $('#result-stage');

// ── URL input ────────────────────────────────────────────────────────
const urlInput = $('#url-input');
const field = document.querySelector('.field');
const hint = $('#hint');
const submitBtn = $('#submit-btn');

const DEFAULT_HINT = 'Also accepts youtu.be, /shorts/ and /embed/ links.';

function setHint(msg, isError = false) {
  hint.textContent = msg;
  hint.classList.toggle('error', isError);
  field.classList.toggle('invalid', isError);
}

function refreshPreview() {
  const id = parseYouTubeId(urlInput.value);
  state.videoId = id;
  const preview = $('#preview');

  if (id) {
    $('#preview-img').src = `https://i.ytimg.com/vi/${id}/hqdefault.jpg`;
    $('#preview-id').textContent = id;
    preview.hidden = false;
    setHint('Looks like a valid YouTube link.');
  } else {
    preview.hidden = true;
    setHint(urlInput.value.trim() ? 'Could not find a video id in that link.' : DEFAULT_HINT,
            Boolean(urlInput.value.trim()));
  }
  submitBtn.disabled = !id;
}

urlInput.addEventListener('input', refreshPreview);

$('#paste-btn').addEventListener('click', async () => {
  try {
    urlInput.value = await navigator.clipboard.readText();
    refreshPreview();
    urlInput.focus();
  } catch {
    setHint('Clipboard access was blocked — paste with Ctrl+V instead.', true);
  }
});

$('#clear-btn').addEventListener('click', () => {
  urlInput.value = '';
  refreshPreview();
  urlInput.focus();
});

document.querySelectorAll('.chip').forEach((chip) => {
  chip.addEventListener('click', () => {
    urlInput.value = chip.dataset.url;
    refreshPreview();
  });
});

$('#url-form').addEventListener('submit', (e) => {
  e.preventDefault();
  if (!state.videoId) {
    setHint('Paste a YouTube link first.', true);
    urlInput.focus();
    return;
  }
  startJob(state.videoId);
});

submitBtn.disabled = true;

// ── pipeline simulation ──────────────────────────────────────────────
const STEPS = [
  { key: 'fetch',   label: 'Fetching video stream',        note: '720p · 30 fps', ms: 1400 },
  { key: 'keys',    label: 'Calibrating keyboard geometry', note: '88 keys',      ms: 1200 },
  { key: 'tiles',   label: 'Detecting and tracking tiles',  note: 'per frame',    ms: 2400 },
  { key: 'timing',  label: 'Solving fall speed and onsets', note: 'px → seconds', ms: 1200 },
  { key: 'notes',   label: 'Building note events',          note: 'pitch + hand', ms: 900  },
  { key: 'score',   label: 'Quantizing and engraving',      note: 'MIDI / XML',   ms: 1300 },
];

/** Log lines per step. `n` is the note count so the console agrees with the stats. */
const logLines = (n) => ({
  fetch: ['resolving stream manifest…', 'container mp4 · 1280x720 · 30.00 fps', 'decoding 3421 frames'],
  keys: [
    'keybed strip found at y = 372..478',
    'black-key pattern locked (2-3 grouping, 36 sharps)',
    'grid fit: A0 at x=41.2px, key width 5.83px, rms 0.31px',
  ],
  tiles: [
    'background model: static, 2 tile palettes',
    'palette A #f5d76e → right hand · palette B #7aa2ff → left hand',
    `tracked ${n} tile blobs across 3421 frames`,
    'merged 63 occlusion splits',
  ],
  timing: ['fall speed 214.6 px/s (sigma 1.9)', 'strike line y = 372', 'lead time 1.73 s'],
  notes: [`${n} note-on events`, `sustain inferred for ${Math.round(n * 0.14)} notes from tile tails`],
  score: ['tempo estimate 96.4 BPM', 'meter 4/4 · key F major', 'engraving 2 staves'],
});

function renderSteps() {
  const list = $('#steps');
  list.innerHTML = '';
  STEPS.forEach((s) => {
    const li = el('li');
    li.dataset.key = s.key;
    li.append(el('span', 'dot', '✓'), el('span', 'step-label', s.label), el('span', 'step-note', s.note));
    list.append(li);
  });
}

function log(text, dim = false) {
  const c = $('#console');
  const line = el(dim ? 'i' : 'span', null, text + '\n');
  c.append(line);
  c.scrollTop = c.scrollHeight;
}

function startElapsed() {
  state.t0 = Date.now();
  state.timer = setInterval(() => {
    const s = Math.floor((Date.now() - state.t0) / 1000);
    $('#elapsed').textContent =
      String(Math.floor(s / 60)).padStart(2, '0') + ':' + String(s % 60).padStart(2, '0');
  }, 250);
}

function show(stage) {
  for (const s of [inputStage, procStage, resStage]) s.hidden = s !== stage;
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

async function startJob(videoId) {
  state.cancelled = false;
  $('#console').innerHTML = '';
  $('#progress-bar').style.width = '0%';
  $('#elapsed').textContent = '00:00';
  renderSteps();
  show(procStage);
  startElapsed();

  log(`job queued for youtube:${videoId}`, true);

  // generated up front so the console numbers match the result panel
  const notes = synthesizeNotes(videoId);
  const lookup = logLines(notes.length);

  for (let i = 0; i < STEPS.length; i++) {
    if (state.cancelled) return;
    const step = STEPS[i];
    const li = document.querySelector(`.steps li[data-key="${step.key}"]`);
    li.classList.add('active');

    const lines = lookup[step.key] || [];
    const per = step.ms / Math.max(lines.length, 1);
    for (const line of lines) {
      await sleep(per);
      if (state.cancelled) return;
      log(line);
    }

    li.classList.remove('active');
    li.classList.add('done');
    $('#progress-bar').style.width = `${((i + 1) / STEPS.length) * 100}%`;
  }

  clearInterval(state.timer);
  finish(videoId, notes);
}

$('#cancel-btn').addEventListener('click', () => {
  state.cancelled = true;
  clearInterval(state.timer);
  show(inputStage);
});

$('#again-btn').addEventListener('click', () => show(inputStage));

// ── simulated result ─────────────────────────────────────────────────
function synthesizeNotes(videoId) {
  const rnd = seededRandom(videoId);
  const notes = [];
  const scale = [0, 2, 4, 5, 7, 9, 11];
  const bars = 16;
  const beat = 60 / 96.4;

  for (let bar = 0; bar < bars; bar++) {
    // left hand: broken chord figure
    const root = 41 + scale[Math.floor(rnd() * 4)];
    for (let i = 0; i < 4; i++) {
      notes.push({
        t: (bar * 4 + i) * beat,
        dur: beat * 0.9,
        pitch: root + [0, 7, 12, 7][i],
        hand: 'L',
      });
    }
    // right hand: melody with occasional pairs
    let t = bar * 4 * beat;
    while (t < (bar + 1) * 4 * beat - 1e-6) {
      const d = rnd() < 0.35 ? beat / 2 : beat;
      const pitch = 65 + scale[Math.floor(rnd() * 7)] + (rnd() < 0.25 ? 12 : 0);
      notes.push({ t, dur: d * 0.85, pitch, hand: 'R' });
      if (rnd() < 0.2) notes.push({ t, dur: d * 0.85, pitch: pitch + 4, hand: 'R' });
      t += d;
    }
  }
  return notes.sort((a, b) => a.t - b.t);
}

function drawRoll(notes) {
  const canvas = $('#roll');
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth;
  const h = 240;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const dur = Math.max(...notes.map((n) => n.t + n.dur));
  const lo = 36, hi = 96;
  const x = (t) => (t / dur) * w;
  const y = (p) => h - ((p - lo) / (hi - lo)) * h;
  const rowH = h / (hi - lo);

  // pitch guides on every C
  ctx.strokeStyle = 'rgba(255,255,255,0.05)';
  ctx.lineWidth = 1;
  for (let p = lo; p <= hi; p++) {
    if (p % 12 !== 0) continue;
    ctx.beginPath();
    ctx.moveTo(0, y(p));
    ctx.lineTo(w, y(p));
    ctx.stroke();
  }

  // bar lines
  const beat = 60 / 96.4;
  ctx.strokeStyle = 'rgba(255,255,255,0.07)';
  for (let t = 0; t < dur; t += beat * 4) {
    ctx.beginPath();
    ctx.moveTo(x(t), 0);
    ctx.lineTo(x(t), h);
    ctx.stroke();
  }

  for (const n of notes) {
    const nx = x(n.t);
    const nw = Math.max(2, x(n.t + n.dur) - nx - 1);
    const ny = y(n.pitch) - rowH * 0.5;
    const nh = Math.max(3, rowH * 0.9);
    ctx.fillStyle = n.hand === 'R' ? '#f5d76e' : '#7aa2ff';
    ctx.shadowColor = ctx.fillStyle;
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.roundRect(nx, ny, nw, nh, 2);
    ctx.fill();
  }
  ctx.shadowBlur = 0;
  return dur;
}

function drawKeybed() {
  const bed = $('#keybed');
  bed.innerHTML = '';
  const whitesPerOctave = 7;
  const octaves = 5;
  const total = whitesPerOctave * octaves;
  for (let i = 0; i < total; i++) bed.append(el('span', 'wk'));
  // black keys sit between white keys 0-1,1-2, 3-4,4-5,5-6 of each octave
  const offsets = [1, 2, 4, 5, 6];
  for (let o = 0; o < octaves; o++) {
    for (const off of offsets) {
      const b = el('span', 'bk');
      b.style.left = `${((o * whitesPerOctave + off) / total) * 100}%`;
      bed.append(b);
    }
  }
}

function stat(k, v, unit) {
  const s = el('div', 'stat');
  s.append(el('span', 'k', k));
  const val = el('span', 'v', v);
  if (unit) val.append(el('small', null, ' ' + unit));
  s.append(val);
  return s;
}

function finish(videoId, notes) {
  show(resStage);

  const rnd = seededRandom(videoId + ':conf');
  const conf = (0.86 + rnd() * 0.11).toFixed(2);
  $('#result-conf').textContent = `confidence ${conf}`;

  const stats = $('#stats');
  stats.innerHTML = '';
  const dur = Math.max(...notes.map((n) => n.t + n.dur));
  stats.append(
    stat('Notes', String(notes.length)),
    stat('Tempo', '96.4', 'BPM'),
    stat('Key', 'F major'),
    stat('Meter', '4/4'),
    stat('Length', dur.toFixed(1), 's'),
    stat('Range', 'F2–C6'),
  );

  drawKeybed();
  drawRoll(notes);
  $('#roll-range').textContent = `0.0s – ${dur.toFixed(1)}s · ${notes.length} events`;

  const dl = $('#downloads');
  dl.innerHTML = '';
  const fmts = [...document.querySelectorAll('.checks input:checked')].map((i) => i.dataset.fmt);
  const ext = { MIDI: '.mid', MusicXML: '.musicxml', PDF: '.pdf', JSON: '.json' };
  for (const f of fmts.length ? fmts : ['MIDI']) {
    const b = el('button', 'dl');
    b.disabled = true;
    b.title = 'Backend not implemented yet';
    b.append(el('span', 'ext', ext[f].slice(1).toUpperCase()), el('span', null, `${videoId}${ext[f]}`));
    dl.append(b);
  }

  window.addEventListener('resize', () => drawRoll(notes), { once: true });
}

// roundRect polyfill for older engines
if (!CanvasRenderingContext2D.prototype.roundRect) {
  CanvasRenderingContext2D.prototype.roundRect = function (x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    this.moveTo(x + r, y);
    this.arcTo(x + w, y, x + w, y + h, r);
    this.arcTo(x + w, y + h, x, y + h, r);
    this.arcTo(x, y + h, x, y, r);
    this.arcTo(x, y, x + w, y, r);
    this.closePath();
    return this;
  };
}
