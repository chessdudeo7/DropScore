# DropScore — how to actually read a falling-tile piano video

The processing plan, with what was built written under each stage as "as
built". Every stage is implemented; the plan is kept because it says why.

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
   pitch (px per white key), and phase gives the offset. A board filmed off a
   screen with a phone is wider at one end than the other, so a grid whose key
   width grows steadily across the board is tried too, and kept only when it
   explains clearly more edge per gridline than the evenly spaced one.
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
   A board filmed at an angle has lanes that lean, so a tall tile's box reaches
   over the key beside it: the lean is measured from the sample frames' tall
   tiles (a line in x, since one end leans left and the other right) and the
   fall area is pulled straight before masking, once the drift at the top of
   the picture is a quarter of a key. A screen recording measures nothing and
   is left untouched.
3. **Blob → key.** Take the blob's x-centre, map through the calibration to a MIDI
   pitch. Snap: a blob whose width is ~1 key is a note; a wide blob is *two adjacent
   notes merged* and must be split on the key grid. A black key is only claimed
   from a merged blob when the pixels say its tile is there: rows the black lane
   fills that no white neighbour does, or, when it was struck together with one
   white neighbour, the blob's far edge running straight down the black key's
   own outer edge, a third of a key into the unclaimed white's lane, on most of
   the lane's rows. Two merged white tiles end on a key boundary; a spark cloud
   has no steady edge at all. A one-tile blob takes the key with the nearest
   centre, except that a blob too wide for a black key's tile (1.2 black keys)
   and off its centre goes to the nearest white key: where the fitted board
   sits a fifth of a key off the tiles (one end of a keystoned copy), a white
   tile's centre falls nearer the black key beside it. Glow spreads a black
   tile to a white tile's width, but evenly, so a halo stays on its key's centre.

Known nastiness:
- **Glow/bloom** inflates blobs. Erode by a fixed fraction of key width, or
  threshold on the saturated core rather than the halo. As built, there is no
  erosion: each colour accepts pixels within a radius set by how tightly its
  own pixels cluster. That spread is read from the colour's densest band when
  most pixels sit there, so a particle cloud around flat tiles cannot widen
  the radius (`tiles._member_spread`). On a smoke-effect render it had widened
  from 22 to 37, and every cloud read as a tile.
- **A tile shaded through hue** clusters into two colours either side of its
  ramp when the shading runs past the hue tolerance that merges a gradient's
  shades: one recording's right hand runs from green to yellow-green, 51
  degrees apart, and split with the default sample while holding together
  with a smaller one. Its tiles were cut across two track masks and the
  notes between the shades came back as fragments. Two colours whose own
  pixels run unbroken along the segment between them are shades of one ramp
  and feed one track (`tiles._merge_ramps`, `Palette.tracks`); two hands'
  colours leave a gap. They stay two colours to match against. Merged into
  one colour, the ramp's spread widened its radius in every direction, from
  the 22 the two shades held each other to, to 39, and the tile's halo on
  the white keys either side read as ten short notes. Along the ramp a pixel
  within the tolerance of the line between the shades is the tile's: that
  recording shades by height, and halfway down the fall its tiles sat just
  past where the two shades meet and vanished for most of a second.
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
- As built (`tracking.transcribe_video`), the video is decoded once: every frame
  is detected, the detections are kept (a few rectangles a frame), the speed is
  measured from them at eight windows across the clip, and only then are tracks
  built. Measuring the speed from its own decode first used to cost two fifths
  of the runtime. Frames are read for tiles on a small thread pool, in order
  (`detect_threads`, 4): each frame is read on its own, so the notes are the
  same at any count, and a 27s render takes 43s instead of 94s. The eval scores
  its corpus clips two at a time in worker processes (`corpus_workers`), each
  with the caller's logging; the report is the same as one at a time.
  The calibration sample (60 frames, 170MB at 720p) is dropped before that
  pass begins (`tracking.prepare_video`); kept in a local, it was a quarter
  of the peak.
  The sample's temporal medians, and the deviation map that finds a video
  inside a screen recording, are taken 32 rows at a time
  (`calibrate.median_images`): stacked whole, each copy of the sample was
  as large as the sample itself, and `np.median` makes one more to
  partition. The result is the same to the pixel at a twentieth of the
  memory.
- A keyboard filmed off a screen is rarely level. `VideoReader.sample()`
  measures the tilt of the keybed's top edge and, from 0.15° up, rotates every
  frame level before anything else sees it.
- Not every video scrolls steadily. A screen recording can pause, and a
  re-upload can change playback speed part-way. After a first pass,
  `tracking._scroll_clock` measures the speed a second at a time from the
  top edges of the tracks that became notes, and finds runs of frames in
  which no tile with its top in view moves at all. Top edges only: a held
  note's bottom stops short of the strike line, and a held chord looks the
  same in every frame whether the video plays or not. Where the speed holds two different values for
  at least two seconds each, or the picture stands still, every frame is
  re-timed by the distance fallen, and the notes come out in the music's own
  time. A video that scrolls steadily, which is nearly all of them, is left
  exactly as it was.
- A tile is known by its top as well as its bottom. Smoke or glow joining a
  tile from below moves its bottom edge ahead of the scroll in one frame,
  and matched by bottom alone the tile began a fresh track, timed early by
  the depth of the smoke. A track that found nothing by its bottom takes a
  tile whose top is on its course, keeping its own bottom; only a track seen
  at least twice has a course.
- A phone's vertical video, or a letterboxed one, holds the picture under a
  blank band. The calibration records the first row in which anything moves
  (`picture_top`), and a tile top at that row is clipped, not a top: read as
  one, it timed a 2.4s note as two. A band shallower than 5% of the frame is
  a player's own bar, and is left alone. A top clipped under that bar, or
  under anything else, is found the way a clipped bottom is: by standing
  still while the other edge moves, for more than one frame (a falling top
  is at its highest once, on its first frame, and that sample is as real as
  any other). Read as tops, the
  pinned samples each report the note's end a frame apart, and on one page
  45 of them against 16 real ones timed a 1.65s note at 1.04 -- and one
  frame more or less moved it. A top that climbs marks a later tile reaching
  this one, and the samples after it are the later tile's -- unless the
  climb ends at the clipped top, which is this tile coming out from under
  something at the picture's top (a header that dimmed one recording's
  tiles for their first hundred rows; the top read 98, 58, then 4). Taken
  for an arrival that threw away every real top sample after it: a
  six-beat note timed by the one sample before, at 0.4s, and 54 of Fur
  Elise's notes at a tenth of their length. Any other climb is an arrival
  even before the tile lands: on a glowing render a tile's halo runs ahead
  as a blob of its own and the tile reaches it mid-fall, and read on as one
  the halo took the tile's length and was merged with the real note behind.
- A board cut off at one side of the picture is placed in the one octave
  where that many white keys fit on a piano.
- A screen recording can show the video in a window beside other windows.
  When the whole picture will not calibrate, `calibrate.video_columns` finds
  the columns whose fall-area rows are the moving region's background colour,
  and the keyboard is fitted inside them and shifted back into the frame.

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
  As built, a note that misses the steady grid is tried again against its
  neighbours' own lateness over two beats (`score._locally_snapped`), so a
  passage played consistently late still lands on the grid.
  Two strikes of one key that would snap to the same step stay two notes,
  the second at its measured onset (`keep_repeats`): snapped together they
  were one note to the MIDI writer, and a waltz lost six repeats played
  faster than a sixteenth.
  A page in declared sections (`score.sections`) is read in beat positions
  throughout, and each section snaps to its own beats: the count runs over
  the join by a fraction of a beat, and against whole numbers that fraction
  put 96 of 266 notes past the shift allowed once a note twenty seconds
  before the page changed the first section's reading.
- **Key signature** via Krumhansl-style pitch-class profile correlation, then spell
  accidentals accordingly (F# vs Gb). As built, each note is spelled by its scale
  degree in the key (`musicxml.spell_in_key`), so F# major gets its E# and C
  major its B♭, not A#.
- **Emit** MIDI first (trivial, and the honest ground truth of what was played),
  then MusicXML via `music21`, then PDF via MuseScore/LilyPond CLI. As built, a
  MIDI note is released no later than the next strike of its key: rounding can
  leave a note held past it, and a note-off names only the key. MusicXML is
  written directly, with eighths and shorter beamed a beat at a time (a dotted
  quarter in compound time, the bar in 3/8), because Verovio, which engraves
  the PDF first, does not beam a file that leaves it out. A compound metre's
  tempo mark names the dotted quarter. In the PDF the tempo mark is written
  as words on one line ("Dotted quarter = 48"), since the PDF step has no
  music font and draws each piece of a line at the same spot.

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
