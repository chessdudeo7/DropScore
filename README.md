# DropScore

Turn falling-tile piano videos (Synthesia-style) into sheet music.

**Status: frontend prototype, backend stage 8 of 10.** The UI is complete and
interactive but still simulates its results. The Python pipeline has its
foundation — video I/O, note and keyboard models, a synthetic clip generator that
produces ground truth to score against, and a working path from video all the way
to MIDI and MusicXML. What is missing is measured accuracy and the web service.
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
  calibrate.py  fits that geometry to a real video
  tiles.py      palette discovery, blob extraction, blob -> key
  tracking.py   scroll speed, tile tracks, geometry -> timed notes
  overlay.py    annotated frames for debugging the vision stages
  score.py      tempo, downbeat, key, hands, quantization
  export/       MIDI, MusicXML, PDF writers
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
