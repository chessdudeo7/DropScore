# DropScore — how to actually read a falling-tile piano video

Notes on the processing plan. Nothing here is implemented yet; the frontend in
`web/` simulates these stages so the UI and the eventual API line up.

---

## 0. The key insight

These videos are **rendered, not filmed**. Synthesia (and the dozens of clones)
draws axis-aligned rectangles on a static background at a constant scroll speed.
That means the problem is much closer to *reverse-engineering a renderer* than to
general video understanding.

So the pipeline should be **classical CV first, learned models only where CV breaks
down**. A well-tuned geometric solver will beat a from-scratch neural net here, and
it produces exact integers (MIDI pitch numbers) rather than probabilities.

The realistic failure modes are all *variation between channels*: different
palettes, glow/bloom, particle effects, camera-ish zooms, keyboard overlays on real
hands, tiles with rounded corners or gradients, 3D-perspective keyboards.

---

## 1. Ingest

- `yt-dlp` for the stream, capped at 720p — more resolution buys nothing since a
  white key is ~6 px wide at 720p and that is already enough to disambiguate.
- Decode with PyAV / OpenCV at native fps. Most of these are 30 or 60 fps.
- Sample a few hundred frames spread across the video for the calibration pass,
  then stream sequentially for tracking.

**Legal note:** downloading YouTube video is against their ToS. For a prototype
this is fine locally, but if this ever becomes a hosted service that is the first
thing that will bite. The frontend therefore offers an **upload-your-own-file**
path alongside the URL path, and the URL pane carries a notice pointing at it —
the file path is the one to treat as primary, with the link path as a local-only
convenience. Both converge on the same job: the backend should take a decoded
video, not a URL, so ingestion stays a thin swappable layer.

## 2. Keyboard calibration (the load-bearing step)

Everything downstream is a function of *"which x-range is which MIDI note"*. Get
this wrong by one key and the entire transcription is transposed garbage.

1. **Find the keybed strip.** Take a temporal median of sampled frames to kill the
   moving tiles. The keyboard region is the horizontal band that stays constant
   and has high vertical edge density. Row-wise variance profile gives you the top
   and bottom edges.
2. **Find the white-key boundaries.** Column-wise gradient of the keybed band gives
   a near-periodic spike train. FFT / autocorrelation over that gives white-key
   pitch (px per white key), and phase gives the offset.
3. **Anchor pitch using the black-key pattern.** Black keys come in groups of 2 and
   3. Detect dark blobs in the upper half of the keybed, look at the gap sequence,
   and find the unique alignment — this pins C. That converts "white key #17" into
   "MIDI 60" with no guessing about how many keys the keyboard has.
4. **Fit and store a model:** `x → midi`, plus the strike line `y`. Refit if the
   layout changes mid-video (some videos zoom or switch camera angles) — detect via
   a running check that the keybed band is still where it was.

Fallback when this fails (3D perspective keyboards, heavily stylized ones): a small
segmentation net (U-Net / SAM-style) trained on a few hundred hand-labeled keybeds
to output the key polygons directly. Worth building only after the classical path
has a measured failure rate.

## 3. Tile detection

Per frame, in the region *above* the strike line:

1. **Background/palette model.** Sample frames, cluster pixel colours (k-means in
   Lab space). The background is the dominant dark cluster; tiles are the 1–4
   saturated clusters. This auto-discovers each channel's palette instead of
   hardcoding colours, and the cluster identity is directly the hand/track label.
2. **Mask + connected components.** Threshold against the palette, morphological
   open to kill particles and glow, then connected components. Filter to
   axis-aligned rectangles with plausible width (≈ one key width from step 2).
3. **Blob → key.** Take the blob's x-centre, map through the calibration to a MIDI
   pitch. Snap: a blob whose width is ~1 key is a note; a wide blob is *two adjacent
   notes merged* and must be split on the key grid.

Known nastiness:
- **Glow/bloom** inflates blobs. Erode by a fixed fraction of key width, or
  threshold on the saturated core rather than the halo.
- **Vertically adjacent repeated notes** merge into one tall blob. Detect the
  1–2 px seam, or use the fact that repeated notes have a gap of ≥ 1 px at typical
  scroll speeds.
- **Particle effects at the strike line** — just ignore everything below the strike
  line + a small margin; the tile has already been measured by then.

## 4. Tracking and timing

- Tiles move purely vertically at a constant speed. Track by matching blobs between
  frames on `(pitch, x)` and expected `Δy`.
- **Estimate scroll speed** globally: phase-correlate consecutive frames restricted
  to the tile region, or take the median `Δy` of all tracked blobs. This should be
  a tight distribution; its spread is a great confidence signal.
- Once speed `v` (px/s) is known, geometry gives timing directly:
  - `onset = t_frame + (y_bottom(tile) - y_strike) / v`
  - `duration = height(tile) / v`
- This is **sub-frame accurate**, which matters: at 30 fps a frame is 33 ms, but
  16th notes at 140 BPM are 107 ms apart. Reading onsets off frame indices alone
  would quantize badly; reading them off pixel geometry does not.
- Only tiles fully visible at some point can be measured precisely. Tiles longer
  than the screen need duration accumulated across frames.

## 5. Notes → score

Now it is a symbolic-music problem, which is well-trodden:

- **Hand assignment** — free when there are two tile colours (the common case).
  Fall back to pitch clustering / a voice-separation heuristic when there is only
  one colour.
- **Tempo + downbeat.** Onset intervals cluster around multiples of the beat. Fit
  a tempo grid (autocorrelation of the onset histogram, or a simple DP over
  candidate BPMs) and allow a slow tempo curve for rubato.
- **Quantize** to 1/16 or 1/32, with a penalty for moving a note far — never snap
  blindly, snapping is what makes machine transcriptions unreadable.
- **Key signature** via Krumhansl-style pitch-class profile correlation, then spell
  accidentals accordingly (F# vs Gb).
- **Emit** MIDI first (trivial, and the honest ground truth of what was played),
  then MusicXML via `music21`, then PDF via MuseScore/LilyPond CLI.

Important expectation-setting: **the MIDI will be excellent, the engraved sheet
music will be mediocre.** Turning a note stream into readable notation (voicing,
beaming, rests, ties, enharmonics) is a genuinely hard, partly aesthetic problem.
That gap should be visible in the UI, not hidden.

---

## Suggested build order

1. **Offline CLI on a local video file** — `dropscore video.mp4 -o out.mid`.
   No web, no YouTube. Get the calibration + tile reading right on 5–10 videos.
2. **Visual debugger** — dump annotated frames: detected keybed, key grid overlay,
   blob boxes with their assigned pitch. This is worth more than any test suite
   here; almost every bug is visible in one glance at an overlay frame.
3. **Evaluation set.** Find videos where the original MIDI is available (many
   Synthesia channels link it, and MIDI→Synthesia renderers exist). Render your own
   videos from known MIDI in several styles — that gives *free, perfect ground
   truth* and is the single highest-leverage thing to build early. Score with
   `mir_eval`-style note-level precision/recall/F1 with onset+pitch tolerance.
4. **Wrap in an API** — FastAPI, job queue (Redis/RQ), the frontend polls
   `GET /jobs/{id}` and gets exactly the stage list it already renders.
5. **Learned components only where measured failures justify them** — keybed
   segmentation, and possibly a small tile detector for exotic renderers.

## Where a "vision model" genuinely helps

- **Renderer classification.** A tiny classifier over a sample frame ("Synthesia /
  PianoFromAbove / custom-3D / real-video-with-overlay") that picks a preset. Cheap
  and removes a lot of per-video tuning.
- **Keybed segmentation** for perspective/3D layouts.
- **A VLM as a supervisor, not a transcriber.** Feed it a debug overlay frame and
  ask "is the key grid aligned with the keyboard?" It is good at catching gross
  misalignment and bad at reading 400 notes. Use it for QA and confidence, never
  in the inner loop — it would be thousands of times more expensive and less exact.

## Open questions worth deciding

- Ceiling for videos with a *real* keyboard + overlay (hands occlude keys). Probably
  fine, since the tiles above the keyboard are what is being read, not the keys.
- Videos that scroll upward, or horizontally (rare, but exist).
- Sustain pedal — almost never rendered, so it is unrecoverable from video. The
  audio track could supply it, but that is a separate project.
- Whether to cross-check against the **audio** at all. Combining the video tile
  stream with an audio transcriber for verification would be a meaningful accuracy
  win, but doubles the scope — worth listing as v2.
