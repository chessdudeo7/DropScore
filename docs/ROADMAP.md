# Build roadmap

Ten stages. Each one lands as its own commit, is useful on its own, and is
testable before the next begins. Order is chosen so that every stage can be
*checked* by something already built — the synthetic renderer (stage 2) exists
early precisely so stages 3–5 have ground truth to be measured against instead of
being eyeballed.

| # | Stage | Delivers | Status |
|---|-------|----------|--------|
| 1 | **Skeleton, config, video I/O** | Package layout, CLI, frame iteration, source resolution, `probe` command | ✅ done |
| 2 | **Synthetic renderer** | Generate Synthesia-style videos from known note data, in six themes and four key ranges — free perfect ground truth | ✅ done |
| 3 | **Keyboard calibration** | Keybed detection, white-key grid fit, black-key anchoring → `x → MIDI pitch` | ✅ done |
| 4 | **Tile detection** | Palette clustering, masking, blob extraction, blob → key mapping | ⬜ |
| 5 | **Tracking and timing** | Scroll-speed estimation, onset/duration from geometry → raw note events | ⬜ |
| 6 | **Debug overlays** | Annotated frame dumps: keybed, key grid, blob boxes with assigned pitch | ⬜ |
| 7 | **Symbolic post-processing** | Hand assignment, tempo/downbeat, quantization, key detection | ⬜ |
| 8 | **Export** | MIDI, then MusicXML, then PDF | ⬜ |
| 9 | **Evaluation harness** | Note-level P/R/F1 against ground truth, over a corpus, with a regression report | ⬜ |
| 10 | **Service + frontend wiring** | FastAPI, job queue, progress events matching the UI's existing stage list | ⬜ |

## Ground truth (stage 2)

Every synthetic clip ships a `.truth.json` sidecar carrying both the notes and
**the geometry that drew them** — strike line, scroll speed, white-key width,
first pitch. That second half is what makes stages 3–5 scoreable independently:
calibration can be checked against the exact key grid, and timing against the
exact pixels-per-second, instead of only being judged at the end of the pipeline
by whether the notes came out right.

The generator deliberately produces the cases that break tile readers rather than
music that sounds good: repeated notes with hairline gaps (the main cause of
under-counting, since the tiles merge vertically), block chords on adjacent keys
(merge horizontally), long notes held under moving ones, and both hands in the
same register so hand assignment cannot fall back to a pitch split.

The six themes vary what the vision code could accidentally depend on: palette,
tile shape (flat / rounded / outlined / gradient), bloom, keybed proportions, lane
separators, and whether a strike line is drawn at all. `paper` is intentionally
hostile — dark outlined tiles on a light background — so any "bright rectangle on
dark background" assumption fails loudly rather than silently.

## Why this order

**Stage 2 before 3–5.** Building the renderer first is the single highest-leverage
move: it converts "does the calibration look right?" into a number. Without it,
stages 3–5 are tuned by eye and silently regress.

**Stage 6 is deliberately not last.** Debug overlays are worth more than tests for
CV work — nearly every bug here is obvious at a glance in an annotated frame and
invisible in a stack trace. It sits after 3–5 only because it needs something to
draw; in practice it will get built alongside them.

**Stage 8 after 7.** Emitting MIDI is trivial and could happen at stage 5, but
emitting *good* MusicXML depends on the tempo grid and hand split from stage 7.
MIDI is the honest output; notation is the lossy, partly aesthetic one.

**Stage 10 last.** The frontend already renders a stage list matching 3–8, so
wiring is mechanical once the pipeline produces real events.

## Scope boundaries

Out of scope for now, listed so they are decisions rather than omissions:

- **Sustain pedal.** Essentially never rendered in these videos, so it is not
  recoverable from the picture. Would need the audio track.
- **Audio cross-checking.** Combining tile events with an audio transcriber would
  be a real accuracy win and roughly doubles the project. Revisit after stage 9
  gives a baseline number to beat.
- **3D / perspective keyboards.** Stage 3 targets the flat, axis-aligned common
  case. A learned keybed segmenter is the fallback, and should only be built once
  stage 9 shows how often the classical path actually fails.
