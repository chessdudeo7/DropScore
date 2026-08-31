# DropScore

Turn falling-tile piano videos (Synthesia-style) into sheet music.

**Status: frontend prototype, backend stage 2 of 10.** The UI is complete and
interactive but still simulates its results. The Python pipeline has its
foundation — video I/O, note and keyboard models, and a synthetic clip generator
that produces ground truth to score against — and none of the vision work yet.
See [docs/ROADMAP.md](docs/ROADMAP.md) for the staged plan.

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

`transcribe` is registered but raises until stages 3–8 land.

Reading from a YouTube URL needs the optional extra (`pip install -e ".[youtube]"`)
and violates YouTube's Terms of Service — it exists for local experimentation.
Pass a local file for anything else.

Run the tests with `pytest`. They render their own synthetic clip, so there are no
fixture files to check out.

## The frontend

No build step, no dependencies — open the file:

```bash
start web/index.html
```

(Or serve `web/` with any static server if you prefer a real origin.)

**What works:**

- Two source paths behind a tab switcher:
  - **YouTube link** — parses `watch?v=`, `youtu.be/`, `/shorts/`, `/embed/`,
    `/live/` and bare 11-char ids, with live validation and a thumbnail preview
  - **Upload a file** — drag-and-drop or browse, with type/size checks, and a
    local probe that reads duration and resolution and grabs a real frame for the
    preview. Prefer this path: downloading from YouTube violates their ToS.
- Transcription settings: hand assignment, quantization, tempo, key, output formats
- Staged progress view with a log console, elapsed timer, and cancel
- Result view: stats, a rendered piano roll on a keybed, download slots

**What is simulated:** everything after the source is chosen. Note data comes from
a seeded PRNG keyed on the video id or filename, so the same input always produces
the same fake result. Download buttons are deliberately disabled. The frontend
gets wired to the real pipeline in stage 10.

## Layout

```
dropscore/    the Python pipeline
  config.py     every tunable in one place
  video.py      decoding, frame iteration, calibration sampling
  sources.py    path or URL -> local file
  notes.py      Note / NoteSequence, the currency every stage trades in
  keyboard.py   pitch <-> pixel-column geometry
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
