# DropScore

Turn falling-tile piano videos (Synthesia-style) into sheet music.

**Status: all 10 stages built, none executed.** The pipeline runs video → MIDI and
MusicXML, with a synthetic corpus and a scoring harness, and the frontend is wired
to a real API. See [docs/ROADMAP.md](docs/ROADMAP.md) for what each stage does.

> **Nothing here has ever been run.** The machine this was written on has no
> Python interpreter — only the Windows Store stub — so every module is
> hand-reviewed and unverified. Install Python 3.11+, `pip install -e ".[dev]"`,
> and run `pytest` before trusting any of it. `dropscore eval` is what will show
> whether the vision stages actually work; until it runs, their accuracy is
> design intent rather than measurement.

## The backend

```bash
pip install -e ".[dev]"
dropscore probe path/to/video.mp4
```

`probe` reports what the decoder sees — resolution, frame rate, frame count, and
the downscale factor the pipeline will work at. `--dump-frames DIR` writes evenly
spaced sample frames, which is how you check a video is actually the kind
DropScore expects before blaming the pipeline.

```bash
dropscore synth --theme all --key-range 88
```

`synth` renders falling-tile videos from procedurally generated music, each with a
`.truth.json` sidecar giving the notes *and* the geometry that drew them — strike
line, scroll speed, key grid. This is the evaluation corpus: it exists before any
vision code so stages 3–5 can be scored against an exact answer instead of being
tuned by eye. Six themes and four keyboard ranges, so a change that only works on
one look shows up as a regression on the rest.

```bash
dropscore calibrate out/synth/classic_88key_seed0.mp4
```

`calibrate` fits the keyboard: it locates the keybed from temporal activity,
measures the white-key grid from the dominant spatial frequency of the separator
edges, and anchors that grid to pitch via the black-key 2-3 grouping. Given a
`.truth.json` sidecar — found automatically next to synthetic clips — it also
scores the fit and exits non-zero on a mismatch, so it doubles as the stage-3
regression check.

```bash
dropscore tiles out/synth/synthesia_88key_seed0.mp4
```

`tiles` discovers the video's own tile colours, then reports the tiles found in
sampled frames and which key each sits over. Detection classifies pixels by
*closeness to a tile colour* rather than difference from the background, which
excludes bloom halos without eroding real tiles; merged tiles are split on the key
grid horizontally and at thin rows vertically.

```bash
dropscore transcribe out/synth/synthesia_88key_seed0.mp4
```

`transcribe` runs the whole vision path and writes note JSON. Timing comes from
tile geometry rather than frame indices, so it is **sub-frame accurate**: a tile's
bottom edge crossing the strike line is the note-on, its top edge crossing is the
note-off, and both are extrapolated from frames where that edge is unclipped. At
30fps a frame is 33ms while sixteenths at 140 BPM are 107ms apart, so rounding to
frames would quantize badly before any musical quantization happened.

By default it then runs stage 7: tempo, downbeat, key, hand assignment and
quantization. `--raw` skips that. Add `-f midi -f musicxml` to write those
directly.

```bash
dropscore export video.notes.json -f midi -f musicxml
```

**Trust the MIDI over the notation.** MIDI says exactly what the tiles said, with
no editorial decisions. MusicXML is mechanically correct but musically
approximate: one voice per staff, notes tied across barlines, no beaming or
dynamics. Turning a note stream into *readable* notation is a partly aesthetic
problem, and this does the mechanical part only. PDF hands the MusicXML to
MuseScore, which must be installed separately.

```bash
dropscore synth --theme all -o out/synth
dropscore eval --save-baseline baseline.json
dropscore eval --baseline baseline.json
```

`eval` transcribes every clip that has a `.truth.json` sidecar and reports
note-level precision, recall and F1, plus how far the fitted geometry landed from
the geometry that drew each clip. With a baseline it exits non-zero when any clip
gets worse, so it works as a CI gate.

Corpus F1 pools notes rather than averaging per-clip scores — averaging would let
a two-note clip outvote a thousand-note one. A clip that disappears from the
corpus counts as a regression to zero, since quietly dropping an awkward case is
how a corpus stops catching anything.

```bash
dropscore debug out/synth/classic_88key_seed0.mp4 --video
```

`debug` draws the fitted grid, strike line and detected tiles onto the video.
**Check this first when a transcription looks wrong.** If the bright C lines do
not land on the real C keys, calibration is off and everything downstream is
transposed — a failure that is obvious here and invisible in the note output.
`--grid-only` skips detection; without `--video` it writes stills instead.

Reading from a YouTube URL needs the optional extra (`pip install -e ".[youtube]"`)
and violates YouTube's Terms of Service — it exists for local experimentation.
Pass a local file for anything else.

Run the tests with `pytest`. They render their own synthetic clip, so there are no
fixture files to check out.

## The web app

```bash
pip install -e ".[service]"
dropscore serve
```

Serves the API and the frontend from one origin at `http://127.0.0.1:8000`. Upload
a video, watch the real pipeline run stage by stage, and download the MIDI,
MusicXML and note JSON it produces.

The uploaded video is **deleted as soon as the job settles**, whatever the
outcome — a source is up to 2 GB and useless once read, while the outputs are
kilobytes and are kept until retention evicts the job. Pass `--keep-sources` to
hold onto them while debugging a clip that transcribes badly.

Jobs run in a thread pool in-process, and the frontend polls — no Redis, no
separate worker, no websockets. Transcription takes seconds, so polling is
sufficient, and shipping infrastructure before there is evidence it is needed
would be the wrong trade. The stage keys the API reports are exactly the ones the
UI already rendered, so progress needed no new vocabulary.

Opening `web/index.html` straight from disk still works and falls back to a
simulation, which is the quickest way to look at the UI. **Demo mode says so on
screen** — a badge in the header and a note under the results — rather than
quietly presenting invented notes as though they were transcribed.

**What the frontend does:**

- Two source paths behind a tab switcher:
  - **YouTube link** — parses `watch?v=`, `youtu.be/`, `/shorts/`, `/embed/`,
    `/live/` and bare 11-char ids, with live validation and a thumbnail preview
  - **Upload a file** — drag-and-drop or browse, with type/size checks, and a
    local probe that reads duration and resolution and grabs a real frame for the
    preview. Prefer this path: downloading from YouTube violates their ToS.
- Staged progress view with a live log from the server, elapsed timer, and cancel
- Result view: stats, a rendered piano roll on a keybed, downloads

The **transcription settings** panel is not yet wired to the API — the server
currently runs with defaults and ignores those choices. They are honoured by the
CLI equivalents (`--raw`, and the knobs in `dropscore/config.py`); plumbing them
through the job submission is the obvious next piece of work.

Against a live API the piano roll and stats come from the real transcription and
the download buttons are real links. In demo mode the notes come from a seeded
PRNG keyed on the video id or filename, and the download buttons are disabled.

## Layout

```
dropscore/    the Python pipeline
  config.py     every tunable in one place
  video.py      decoding, frame iteration, calibration sampling
  sources.py    path or URL -> local file
  notes.py      Note / NoteSequence, the currency every stage trades in
  keyboard.py   pitch <-> pixel-column geometry
  calibrate.py  fits that geometry to a real video
  tiles.py      palette discovery, blob extraction, blob -> key
  tracking.py   scroll speed, tile tracks, geometry -> timed notes
  overlay.py    annotated frames for debugging the vision stages
  score.py      tempo, downbeat, key, hands, quantization
  export/       MIDI, MusicXML, PDF writers
  evaluate.py   note-level scoring, corpus runs, regression detection
  service/      FastAPI app and the in-process job queue
  synth/        synthetic clip generation with ground truth
tests/        pytest suite; generates its own video fixtures
web/          static frontend (index.html, styles.css, app.js)
docs/         ROADMAP.md — the staged build plan
              PIPELINE.md — how the video processing is meant to work
```

## The approach in one paragraph

These videos are *rendered*, not filmed, so this is closer to reverse-engineering
a renderer than to general video understanding. Classical CV is the spine —
calibrate the keyboard grid from the black-key pattern, segment tiles by colour,
and recover timing from scroll geometry rather than frame indices — with learned
models only where that measurably fails. Details and open questions in
[docs/PIPELINE.md](docs/PIPELINE.md).
