/* DropScore — frontend.
 *
 * Talks to the API when one is reachable, and falls back to a simulation when it
 * is not — opening this file straight from disk still demonstrates the whole
 * flow, which is how it was built and is still the quickest way to look at the
 * UI. Demo mode says so on screen rather than quietly showing invented notes as
 * though they were real.
 *
 * Run the real thing with:  dropscore serve
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

const fmtBytes = (b) => {
  const u = ['B', 'KB', 'MB', 'GB'];
  let i = 0;
  while (b >= 1024 && i < u.length - 1) { b /= 1024; i++; }
  return `${b < 10 && i > 0 ? b.toFixed(1) : Math.round(b)} ${u[i]}`;
};

const fmtClock = (secs) => {
  if (!isFinite(secs)) return '—';
  const s = Math.round(secs);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
};

// ── backend ──────────────────────────────────────────────────────────

/* `ready` is the startup probe. Anything that branches on `available` must
 * await it first: until it settles the answer is "don't know yet", and
 * treating that as "no backend" would run the simulation and present invented
 * notes as a real transcription — the one outcome demo mode exists to prevent. */
const api = { available: false, probing: true, ready: null, jobId: null, poll: null };

/** Probe for a live API once, at startup. */
async function detectApi() {
  try {
    if (location.protocol !== 'file:') {
      try {
        const response = await fetch('/api/health', { cache: 'no-store' });
        api.available = response.ok;
      } catch {
        api.available = false;
      }
    }
    applyMode();
  } finally {
    // Whatever happened, stop blocking submission on the probe.
    api.probing = false;
    syncSubmit();
  }
  return api.available;
}

/** Reflect the detected mode in the parts of the UI that make claims. */
function applyMode() {
  document.body.classList.toggle('demo', !api.available);
  const pill = $('#mode-pill');
  pill.textContent = api.available ? 'connected' : 'demo mode · simulated results';
  pill.classList.toggle('pill-ok', api.available);
  pill.classList.toggle('pill-warn', !api.available);

  // Only now is it known where an uploaded file would actually go, so this is
  // the first point at which either claim can honestly be made. Any error
  // already on screen is left alone rather than overwritten with boilerplate.
  $('#dz-sub').textContent = fileCopy().sub;
  if (!fileHint.classList.contains('error')) setFileHint(defaultFileHint());
}

// ── state ────────────────────────────────────────────────────────────
/** `source` is either {kind:'youtube', id} or {kind:'file', file, meta}. */
const state = { tab: 'link', videoId: null, file: null, cancelled: false, timer: null, t0: 0 };

const inputStage = $('#input-stage');
const procStage = $('#processing-stage');
const resStage = $('#result-stage');

/** The source the active tab currently holds, or null if it is incomplete. */
function currentSource() {
  if (state.tab === 'link') {
    return state.videoId ? { kind: 'youtube', id: state.videoId } : null;
  }
  return state.file ? { kind: 'file', ...state.file } : null;
}

/** Short label used for the job line and download filenames. */
const sourceSlug = (src) =>
  src.kind === 'youtube' ? src.id : src.file.name.replace(/\.[^.]+$/, '').replace(/[^\w-]+/g, '_').slice(0, 48);

// ── tabs ─────────────────────────────────────────────────────────────
function selectTab(which) {
  state.tab = which;
  for (const [name, tab, pane] of [
    ['link', $('#tab-link'), $('#pane-link')],
    ['file', $('#tab-file'), $('#pane-file')],
  ]) {
    const on = name === which;
    tab.classList.toggle('is-active', on);
    tab.setAttribute('aria-selected', String(on));
    pane.hidden = !on;
  }
  syncSubmit();
}

$('#tab-link').addEventListener('click', () => selectTab('link'));
$('#tab-file').addEventListener('click', () => selectTab('file'));

function syncSubmit() {
  const probing = api.probing;
  submitBtn.disabled = probing || !currentSource();
  submitBtn.title = probing ? 'Checking for a transcription server…' : '';
}

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
  syncSubmit();
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

// ── file input ───────────────────────────────────────────────────────
const MAX_BYTES = 2 * 1024 * 1024 * 1024; // 2 GB
const ALLOWED_EXT = ['mp4', 'webm', 'mov', 'mkv', 'm4v'];

/* Where the file actually goes differs between the two modes, and telling
 * someone their video stays local while uploading it would be a lie worth
 * avoiding. Both strings are stated positively so neither mode is ambiguous. */
const FILE_COPY = {
  demo: {
    sub: 'MP4, WebM, MOV or MKV · up to 2 GB · stays on your machine',
    hint: 'Demo mode: the file is read in your browser and never uploaded.',
  },
  server: {
    sub: 'MP4, WebM, MOV or MKV · up to 2 GB · uploaded to the server running this page',
    hint: 'Uploaded to the server running this page, then deleted once transcribed.',
  },
};

const fileCopy = () => (api.available ? FILE_COPY.server : FILE_COPY.demo);
const defaultFileHint = () => fileCopy().hint;

const dropzone = $('#dropzone');
const fileInput = $('#file-input');
const fileHint = $('#file-hint');

function setFileHint(msg, isError = false) {
  fileHint.textContent = msg;
  fileHint.classList.toggle('error', isError);
}

function clearFile() {
  if (state.file) URL.revokeObjectURL(state.file.url);
  state.file = null;
  fileInput.value = '';
  $('#file-preview').hidden = true;
  setFileHint(defaultFileHint());
  syncSubmit();
}

/**
 * Pull metadata and a representative frame straight from the file. Doubles as a
 * decodability check: if the browser cannot open it, neither will the backend.
 */
function probeVideo(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const video = document.createElement('video');
    video.preload = 'metadata';
    video.muted = true;
    video.src = url;

    let settled = false;
    const fail = (why) => {
      if (settled) return;
      settled = true;
      URL.revokeObjectURL(url);
      reject(new Error(why));
    };
    video.addEventListener('error', () => fail('This file could not be decoded in the browser.'));
    // covers both "never loaded" and "loaded but the seek never completed"
    setTimeout(() => fail('Timed out reading that file.'), 12000);

    video.addEventListener('loadedmetadata', () => {
      // a frame a quarter of the way in is far likelier to show tiles than frame 0
      video.currentTime = Math.min(video.duration * 0.25 || 0, 30);
    }, { once: true });

    video.addEventListener('seeked', () => {
      const canvas = $('#file-thumb');
      const ctx = canvas.getContext('2d');
      const scale = Math.min(canvas.width / video.videoWidth, canvas.height / video.videoHeight);
      const w = video.videoWidth * scale;
      const h = video.videoHeight * scale;
      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(video, (canvas.width - w) / 2, (canvas.height - h) / 2, w, h);
      settled = true;
      resolve({
        url,
        duration: video.duration,
        width: video.videoWidth,
        height: video.videoHeight,
      });
    }, { once: true });
  });
}

async function acceptFile(file) {
  if (!file) return;
  const ext = (file.name.split('.').pop() || '').toLowerCase();

  if (!ALLOWED_EXT.includes(ext) && !file.type.startsWith('video/')) {
    setFileHint(`${file.name} is not a video file DropScore can read.`, true);
    return;
  }
  if (file.size > MAX_BYTES) {
    setFileHint(`That file is ${fmtBytes(file.size)} — the limit is ${fmtBytes(MAX_BYTES)}.`, true);
    return;
  }

  setFileHint('Reading video…');
  try {
    const meta = await probeVideo(file);
    if (state.file) URL.revokeObjectURL(state.file.url);
    state.file = { file, ...meta };

    $('#file-name').textContent = file.name;
    $('#file-badge').textContent = ext.toUpperCase();
    $('#file-meta').textContent =
      `${meta.width}×${meta.height} · ${fmtClock(meta.duration)} · ${fmtBytes(file.size)}`;
    $('#file-preview').hidden = false;
    setFileHint(defaultFileHint());
  } catch (err) {
    state.file = null;
    $('#file-preview').hidden = true;
    setFileHint(err.message, true);
  }
  syncSubmit();
}

fileInput.addEventListener('change', () => acceptFile(fileInput.files[0]));
$('#file-clear-btn').addEventListener('click', clearFile);

['dragenter', 'dragover'].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add('dragover');
  }));

['dragleave', 'drop'].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    if (evt === 'dragleave' && dropzone.contains(e.relatedTarget)) return;
    dropzone.classList.remove('dragover');
  }));

dropzone.addEventListener('drop', (e) => acceptFile(e.dataTransfer.files[0]));

// a drop anywhere else on the page should not navigate away from the app
['dragover', 'drop'].forEach((evt) =>
  window.addEventListener(evt, (e) => { if (!dropzone.contains(e.target)) e.preventDefault(); }));

// ── submit ───────────────────────────────────────────────────────────
$('#source-form').addEventListener('submit', (e) => {
  e.preventDefault();
  const src = currentSource();
  if (!src) {
    if (state.tab === 'link') {
      setHint('Paste a YouTube link first.', true);
      urlInput.focus();
    } else {
      setFileHint('Choose a video file first.', true);
    }
    return;
  }
  startJob(src);
});

// Disabled until both a source is chosen and the backend probe has settled.
syncSubmit();

// ── pipeline simulation ──────────────────────────────────────────────
const STEPS = [
  { key: 'fetch',   label: 'Fetching video stream',        note: '720p · 30 fps', ms: 1400,
    fileLabel: 'Reading video file',                       fileNote: 'local' },
  { key: 'keys',    label: 'Calibrating keyboard geometry', note: '88 keys',      ms: 1200 },
  { key: 'tiles',   label: 'Detecting and tracking tiles',  note: 'per frame',    ms: 2400 },
  { key: 'timing',  label: 'Solving fall speed and onsets', note: 'px → seconds', ms: 1200 },
  { key: 'notes',   label: 'Building note events',          note: 'pitch + hand', ms: 900  },
  { key: 'score',   label: 'Quantizing and engraving',      note: 'MIDI / XML',   ms: 1300 },
];

/** Log lines per step. `n` is the note count so the console agrees with the stats. */
const logLines = (n, src) => ({
  fetch: src.kind === 'youtube'
    ? ['resolving stream manifest…', 'container mp4 · 1280x720 · 30.00 fps', 'decoding 3421 frames']
    : [
        `opening ${src.file.name} (${fmtBytes(src.file.size)})`,
        `container ${(src.file.name.split('.').pop() || '').toLowerCase()} · ${src.width}x${src.height} · ${fmtClock(src.duration)}`,
        'decoding 3421 frames',
      ],
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

function renderSteps(src) {
  const list = $('#steps');
  const local = src.kind === 'file';
  list.innerHTML = '';
  STEPS.forEach((s) => {
    const li = el('li');
    li.dataset.key = s.key;
    li.append(
      el('span', 'dot', '✓'),
      el('span', 'step-label', (local && s.fileLabel) || s.label),
      el('span', 'step-note', (local && s.fileNote) || s.note),
    );
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

async function startJob(src) {
  // The probe may still be in flight if the user was quick, or the network
  // slow. Deciding now would silently pick the simulation.
  await api.ready;

  state.cancelled = false;
  api.jobId = null;
  $('#console').innerHTML = '';
  $('#progress-bar').style.width = '0%';
  $('#elapsed').textContent = '00:00';
  renderSteps(src);
  show(procStage);
  startElapsed();

  if (api.available) {
    return runRemoteJob(src);
  }

  const slug = sourceSlug(src);
  log(`job queued for ${src.kind === 'youtube' ? 'youtube:' + src.id : 'file:' + src.file.name}`, true);

  // generated up front so the console numbers match the result panel
  const notes = synthesizeNotes(slug);
  const lookup = logLines(notes.length, src);

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
  finish(slug, notes);
}

// ── the real job ─────────────────────────────────────────────────────

/** Submit to the API and poll until it settles. */
async function runRemoteJob(src) {
  let submitted;
  try {
    if (src.kind === 'file') {
      const body = new FormData();
      body.append('file', src.file);
      submitted = await postJson('/api/jobs/upload', { method: 'POST', body });
    } else {
      submitted = await postJson('/api/jobs/url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: `https://www.youtube.com/watch?v=${src.id}` }),
      });
    }
  } catch (err) {
    return failJob(err.message);
  }

  api.jobId = submitted.id;
  log(`job ${submitted.id.slice(0, 8)} queued`, true);
  pollJob();
}

async function postJson(url, options) {
  const response = await fetch(url, options);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `Request failed (${response.status})`);
  return payload;
}

function pollJob() {
  clearInterval(api.poll);
  let seen = 0;

  api.poll = setInterval(async () => {
    if (state.cancelled) return;

    let job;
    try {
      job = await postJson(`/api/jobs/${api.jobId}`, { cache: 'no-store' });
    } catch (err) {
      clearInterval(api.poll);
      return failJob(err.message);
    }

    // The server owns the log; append only what has not been shown yet.
    for (const line of job.log.slice(seen)) log(line);
    seen = job.log.length;

    for (const stage of job.stages) {
      const li = document.querySelector(`.steps li[data-key="${stage.key}"]`);
      if (!li) continue;
      li.classList.toggle('active', stage.state === 'active');
      li.classList.toggle('done', stage.state === 'done');
    }
    $('#progress-bar').style.width = `${job.progress * 100}%`;

    if (job.status === 'done') {
      clearInterval(api.poll);
      clearInterval(state.timer);
      finishRemote(job);
    } else if (job.status === 'error' || job.status === 'cancelled') {
      clearInterval(api.poll);
      failJob(job.error || 'Transcription was cancelled');
    }
  }, 700);
}

function failJob(message) {
  clearInterval(state.timer);
  log(`error: ${message}`);
  const hintEl = state.tab === 'link' ? setHint : setFileHint;
  hintEl(message, true);
  show(inputStage);
}

$('#cancel-btn').addEventListener('click', () => {
  state.cancelled = true;
  clearInterval(state.timer);
  clearInterval(api.poll);
  if (api.jobId) {
    fetch(`/api/jobs/${api.jobId}`, { method: 'DELETE' }).catch(() => {});
  }
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

  // The canvas measures zero until the stage it lives in has been laid out.
  // Drawing then would silently produce an empty roll, so bail and let the
  // caller retry on the next frame.
  if (!w) return 0;
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

const PITCH_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];
const pitchName = (p) => `${PITCH_NAMES[p % 12]}${Math.floor(p / 12) - 1}`;

/** Render a completed API job. */
function finishRemote(job) {
  const r = job.result || {};
  const notes = (r.notes || []).map((n) => ({ ...n, hand: n.hand || 'R' }));

  showResult({
    notes,
    confidence: r.confidence,
    stats: [
      ['Notes', String(r.count ?? notes.length)],
      ['Tempo', r.tempo ? String(r.tempo) : '—', r.tempo ? 'BPM' : ''],
      ['Key', r.key || '—'],
      ['Meter', r.meter || '—'],
      ['Length', (r.duration ?? 0).toFixed(1), 's'],
      [
        'Range',
        r.lowest != null ? `${pitchName(r.lowest)}–${pitchName(r.highest)}` : '—',
      ],
    ],
    downloads: (job.formats || []).map((format) => ({
      label: format.toUpperCase(),
      name: `${job.label}.${format === 'midi' ? 'mid' : format}`,
      href: `/api/jobs/${job.id}/download/${format}`,
    })),
  });
}

/** Render a simulated result, for demo mode. */
function finish(slug, notes) {
  const rnd = seededRandom(slug + ':conf');
  const dur = Math.max(...notes.map((n) => n.t + n.dur));
  const fmts = [...document.querySelectorAll('.checks input:checked')].map((i) => i.dataset.fmt);
  const ext = { MIDI: '.mid', MusicXML: '.musicxml', PDF: '.pdf', JSON: '.json' };

  showResult({
    notes,
    confidence: 0.86 + rnd() * 0.11,
    stats: [
      ['Notes', String(notes.length)],
      ['Tempo', '96.4', 'BPM'],
      ['Key', 'F major'],
      ['Meter', '4/4'],
      ['Length', dur.toFixed(1), 's'],
      ['Range', 'F2–C6'],
    ],
    downloads: (fmts.length ? fmts : ['MIDI']).map((f) => ({
      label: ext[f].slice(1).toUpperCase(),
      name: `${slug}${ext[f]}`,
      href: null,
    })),
  });
}

function showResult({ notes, confidence, stats: rows, downloads }) {
  show(resStage);

  $('#result-conf').textContent = `confidence ${(confidence ?? 0).toFixed(2)}`;

  const stats = $('#stats');
  stats.innerHTML = '';
  for (const [key, value, unit] of rows) stats.append(stat(key, value, unit));

  drawKeybed();

  const dur = notes.length ? Math.max(...notes.map((n) => n.t + n.dur)) : 0;
  $('#roll-range').textContent = `0.0s – ${dur.toFixed(1)}s · ${notes.length} events`;

  // The canvas is still zero-width in the task that unhides its section, and
  // requestAnimationFrame does not run at all while the page is hidden. A
  // ResizeObserver covers both: it fires once the element has a real size, and
  // again whenever that size changes, so it doubles as the resize handler.
  const paint = () => notes.length && drawRoll(notes);
  paint();
  state.rollObserver?.disconnect();
  state.rollObserver = new ResizeObserver(paint);
  state.rollObserver.observe($('#roll'));

  const dl = $('#downloads');
  dl.innerHTML = '';
  for (const item of downloads) {
    const node = item.href ? el('a', 'dl') : el('button', 'dl');
    if (item.href) {
      node.href = item.href;
      node.setAttribute('download', item.name);
    } else {
      node.disabled = true;
      node.title = 'Demo mode — run `dropscore serve` for real output';
    }
    node.append(el('span', 'ext', item.label), el('span', null, item.name));
    dl.append(node);
  }

  $('#disclaimer').hidden = api.available;
}

api.ready = detectApi();

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
