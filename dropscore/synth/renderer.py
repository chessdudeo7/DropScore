"""Render a note sequence as a falling-tile video, with exact ground truth.

This is the highest-leverage thing in the project: it turns "does the calibration
look right?" into a number. Every clip it writes comes with a JSON sidecar giving
not just the notes but the geometry that produced them — key grid, strike line,
scroll speed — so stages 3-5 can be scored against the exact answer instead of
being eyeballed.

Geometry convention, which stage 5 has to invert:

* ``speed = strike_y / lead_time`` pixels per second, constant.
* A tile's **bottom edge** crosses the strike line exactly at the note's onset,
  so ``y_bottom(t) = strike_y + (t - onset) * speed``.
* A tile's height is ``duration * speed``.
* Tiles are clipped at the strike line; nothing is drawn over the keybed.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from ..keyboard import COMMON_RANGES, KeyboardLayout, is_black
from ..notes import Note, NoteSequence
from ..video import VideoError, open_writer
from .themes import DEFAULT_THEME, RGB, Theme, get_theme

log = logging.getLogger(__name__)



class RenderError(RuntimeError):
    """Raised when no usable video encoder is available."""


def _bgr(color: RGB) -> tuple[int, int, int]:
    r, g, b = color
    return (b, g, r)


def _blend(base: RGB, tint: RGB, strength: float) -> tuple[int, int, int]:
    return _bgr(
        (
            int(base[0] + (tint[0] - base[0]) * strength),
            int(base[1] + (tint[1] - base[1]) * strength),
            int(base[2] + (tint[2] - base[2]) * strength),
        )
    )


@dataclass(frozen=True)
class RenderSpec:
    """Everything needed to render, and everything worth recording as truth."""

    width: int = 1280
    height: int = 720
    fps: float = 30.0
    theme: Theme | None = None
    key_range: str = "88"
    lead_in: float = 1.0  # silence before the first note sounds
    lead_out: float = 1.5  # tail after the last note ends

    @property
    def resolved_theme(self) -> Theme:
        return self.theme or get_theme(DEFAULT_THEME)


class SynthRenderer:
    """Draws frames for a note sequence. Stateless between calls to ``frame``."""

    def __init__(self, sequence: NoteSequence, spec: RenderSpec | None = None) -> None:
        self.sequence = sequence
        self.spec = spec or RenderSpec()
        self.theme = self.spec.resolved_theme

        self.keybed_height = int(self.spec.height * self.theme.keybed_ratio)
        self.bottom_margin = int(self.spec.height * self.theme.bottom_margin)
        self.strike_y = self.spec.height - self.keybed_height - self.bottom_margin
        self.keybed_bottom = self.strike_y + self.keybed_height
        self.speed = self.strike_y / self.theme.lead_time  # px/s

        first, last = COMMON_RANGES[self.spec.key_range]
        self.layout = KeyboardLayout(
            first_pitch=first, last_pitch=last, x0=0.0, width=float(self.spec.width)
        )

        # Notes are shifted by lead_in so nothing is already mid-fall at t=0.
        self.notes = [
            Note(
                onset=n.onset + self.spec.lead_in,
                pitch=n.pitch,
                duration=n.duration,
                hand=n.hand,
                velocity=n.velocity,
            )
            for n in sequence
            if self.layout.contains(n.pitch)
        ]
        self.notes.sort()

        dropped = len(sequence) - len(self.notes)
        if dropped:
            log.warning(
                "%d note(s) fall outside the %s-key range and were not rendered",
                dropped,
                self.spec.key_range,
            )

        self._keybed = self._draw_keybed_base()
        self._hand_x = self._hand_tracks() if self.theme.hands else {}

    def _hand_tracks(self) -> dict[str, np.ndarray]:
        """A damped x-position per hand, per frame.

        Hands are drawn where the hand actually is, which is not where the next
        note is: a player's hand travels, it does not teleport. Snapping it to
        each note's key made the temporal median smear both hands across the
        whole board, wiping out the key edges that stage 3 fits its grid to --
        an artefact of the renderer, not something a real capture does.
        """
        frames = self.frame_count
        tracks: dict[str, np.ndarray] = {}
        for hand in ("L", "R"):
            target = np.full(frames, np.nan, dtype=np.float32)
            for note in self.notes:
                if note.hand != hand:
                    continue
                lo = max(0, int(note.onset * self.spec.fps))
                hi = min(frames, int(note.offset * self.spec.fps) + 1)
                target[lo:hi] = self.layout.key_center(note.pitch)

            # Hold the last known position through the rests, then smooth over
            # about a second so the hand glides between positions.
            last = np.nan
            for i in range(frames):
                if np.isnan(target[i]):
                    target[i] = last
                else:
                    last = target[i]
            fallback = self.spec.width / 2.0
            target[np.isnan(target)] = fallback
            window = max(3, int(self.spec.fps))
            kernel = np.ones(window, dtype=np.float32) / window
            padded = np.pad(target, window, mode="edge")
            tracks[hand] = np.convolve(padded, kernel, mode="same")[window:-window]
        return tracks

    # ── timing ───────────────────────────────────────────────────────

    @property
    def duration(self) -> float:
        """Clip length: the last note's tail plus the configured lead-out."""
        last = max((n.offset for n in self.notes), default=self.spec.lead_in)
        return last + self.spec.lead_out

    @property
    def frame_count(self) -> int:
        return max(1, int(round(self.duration * self.spec.fps)))

    # ── drawing ──────────────────────────────────────────────────────

    def _draw_keybed_base(self) -> np.ndarray:
        """The unlit keyboard, drawn once and copied per frame."""
        theme = self.theme
        bed = np.zeros((self.keybed_height, self.spec.width, 3), dtype=np.uint8)
        bed[:] = _bgr(theme.white_key_color)

        edge = _bgr(theme.key_edge_color)
        for pitch in self.layout.pitches:
            if is_black(pitch):
                continue
            left, right = self.layout.white_span(pitch)
            cv2.line(bed, (int(right), 0), (int(right), self.keybed_height), edge, 1)

        black_h = int(self.keybed_height * self.layout.black_height_ratio)
        for pitch in self.layout.pitches:
            if not is_black(pitch):
                continue
            left, right = self.layout.key_span(pitch)
            cv2.rectangle(
                bed,
                (int(round(left)), 0),
                (int(round(right)), black_h),
                _bgr(theme.black_key_color),
                -1,
            )

        return bed

    def _draw_keybed(self, canvas: np.ndarray, active: list[Note]) -> None:
        bed = self._keybed.copy()
        theme = self.theme
        black_h = int(self.keybed_height * self.layout.black_height_ratio)

        for note in active:
            left, right = self.layout.key_span(note.pitch)
            black = is_black(note.pitch)
            base = theme.black_key_color if black else theme.white_key_color
            color = _blend(base, theme.color_for(note.hand), theme.highlight_strength)
            cv2.rectangle(
                bed,
                (int(round(left)) + 1, 0),
                (int(round(right)) - 1, black_h if black else self.keybed_height),
                color,
                -1,
            )

        canvas[self.strike_y : self.keybed_bottom, :] = bed

        if theme.keybed_shadow:
            band = max(2, self.keybed_height // 12)
            strip = canvas[self.strike_y : self.strike_y + band].astype(np.float32)
            fade = np.linspace(0.35, 1.0, band, dtype=np.float32)[:, None, None]
            canvas[self.strike_y : self.strike_y + band] = (strip * fade).astype(np.uint8)

        if theme.strike_line:
            cv2.line(
                canvas,
                (0, self.strike_y),
                (self.spec.width, self.strike_y),
                _bgr(theme.strike_color),
                2,
            )

    def _tile_rect(self, note: Note, t: float) -> tuple[int, int, int, int] | None:
        """Pixel rect for a tile at time ``t``, or None when off screen."""
        y_bottom = self.strike_y + (t - note.onset) * self.speed
        y_top = y_bottom - note.duration * self.speed

        # Clipped at the strike line, and above the top of the frame.
        bottom = min(y_bottom, self.strike_y)
        top = max(y_top, 0.0)
        if bottom <= 0 or top >= self.strike_y or bottom - top < 1:
            return None

        left, right = self.layout.key_span(note.pitch)
        inset = self.layout.key_width(note.pitch) * self.theme.tile_gap / 2
        left += inset
        right -= inset
        if right - left < 1:
            return None

        return int(round(left)), int(round(top)), int(round(right)), int(round(bottom))

    def _draw_tile(self, canvas: np.ndarray, rect: tuple[int, int, int, int], color: RGB) -> None:
        x0, y0, x1, y1 = rect
        if x1 <= x0 or y1 <= y0:  # rounding can collapse a thin tile
            return
        theme = self.theme
        bgr = _bgr(color)

        if theme.tile_style == "outline":
            thickness = max(1, (x1 - x0) // 8)
            cv2.rectangle(canvas, (x0, y0), (x1, y1), bgr, thickness)
            return

        if theme.tile_style == "gradient":
            height, width = y1 - y0, x1 - x0
            # Bright at the bottom (nearest the strike line), fading upward.
            ramp = np.linspace(0.45, 1.0, height, dtype=np.float32)[:, None, None]
            patch = np.empty((height, width, 3), dtype=np.float32)
            patch[:] = np.array(bgr, dtype=np.float32)
            canvas[y0:y1, x0:x1] = (patch * ramp).astype(np.uint8)
            return

        if theme.tile_style == "rounded" and theme.corner_radius > 0:
            radius = int(min((x1 - x0) * theme.corner_radius, (y1 - y0) / 2))
            if radius >= 1:
                _rounded_rect(canvas, x0, y0, x1, y1, radius, bgr)
                return

        cv2.rectangle(canvas, (x0, y0), (x1, y1), bgr, -1)

    def frame(self, index: int) -> np.ndarray:
        """Render frame ``index``."""
        theme = self.theme
        t = index / self.spec.fps

        canvas = np.empty((self.spec.height, self.spec.width, 3), dtype=np.uint8)
        canvas[:] = _bgr(theme.background)

        if theme.lane_separators:
            lane = _bgr(theme.lane_color)
            for pitch in self.layout.pitches:
                if is_black(pitch):
                    continue
                _, right = self.layout.white_span(pitch)
                cv2.line(canvas, (int(right), 0), (int(right), self.strike_y), lane, 1)

        rects: list[tuple[tuple[int, int, int, int], RGB]] = []
        active: list[Note] = []
        for note in self.notes:
            if note.onset - theme.lead_time > t:
                break  # sorted by onset: nothing later is visible yet
            rect = self._tile_rect(note, t)
            if rect is not None:
                rects.append((rect, theme.color_for(note.hand)))
            if note.onset <= t < note.offset:
                active.append(note)

        if theme.glow > 0 and rects:
            canvas = _apply_glow(canvas, rects, theme.glow)

        for rect, color in rects:
            self._draw_tile(canvas, rect, color)

        self._draw_keybed(canvas, active)
        if theme.highlight_bloom > 0 and active:
            self._draw_key_bloom(canvas, active)
        if theme.hands:
            self._draw_hands(canvas, index)
        if theme.caption and self.bottom_margin > 8:
            self._draw_caption(canvas)
        return canvas

    def _draw_key_bloom(self, canvas: np.ndarray, active: list[Note]) -> None:
        """Blow struck keys out past white, the way a real renderer does."""
        halo = np.zeros(canvas.shape[:2], dtype=np.float32)
        for note in active:
            left, right = self.layout.key_span(note.pitch)
            cv2.rectangle(
                halo,
                (int(round(left)), self.strike_y - 8),
                (int(round(right)), self.keybed_bottom),
                1.0,
                -1,
            )
        halo = cv2.GaussianBlur(halo, (0, 0), self.layout.white_width * 0.6)
        strength = self.theme.highlight_bloom * halo[:, :, None]
        lit = canvas.astype(np.float32) + 255.0 * strength
        np.clip(lit, 0, 255, out=lit)
        canvas[:] = lit.astype(np.uint8)

    def _draw_hands(self, canvas: np.ndarray, index: int) -> None:
        """Two blobs over the lower keybed, following what is being played.

        Not anatomy — the point is that something opaque and skin-coloured
        covers the bottom of the keys and adds structure to those rows.
        """
        skin = (105, 125, 150)  # BGR; close to the keys, as a real hand is
        top = self.strike_y + int(self.keybed_height * 0.55)
        for hand in ("L", "R"):
            track = self._hand_x.get(hand)
            if track is None or not len(track):
                continue
            centre = float(track[min(index, len(track) - 1)])
            span = max(self.layout.white_width * 2.5, 40.0)
            cv2.ellipse(
                canvas,
                (int(centre), self.keybed_bottom),
                (int(span), int((self.keybed_bottom - top) * 1.1)),
                0,
                180,
                360,
                skin,
                -1,
            )

    def _draw_caption(self, canvas: np.ndarray) -> None:
        """Credit text burned into the dark space under the keyboard."""
        text = self.theme.caption
        scale = 0.6
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 1)
        x = (self.spec.width - tw) // 2
        y = self.keybed_bottom + (self.bottom_margin + th) // 2
        cv2.putText(
            canvas, text, (x, y), cv2.FONT_HERSHEY_DUPLEX, scale, (238, 238, 238), 1,
            cv2.LINE_AA,
        )

    # ── output ───────────────────────────────────────────────────────

    def truth(self) -> dict:
        """Ground truth: the notes as rendered, plus the geometry behind them."""
        return {
            "version": 1,
            "video": {
                "width": self.spec.width,
                "height": self.spec.height,
                "fps": self.spec.fps,
                "frames": self.frame_count,
                "duration": self.duration,
            },
            "theme": self.theme.name,
            "geometry": {
                "strike_y": self.strike_y,
                "keybed_height": self.keybed_height,
                "speed_px_per_s": self.speed,
                "lead_time": self.theme.lead_time,
                "first_pitch": self.layout.first_pitch,
                "last_pitch": self.layout.last_pitch,
                "white_key_width": self.layout.white_width,
                "x0": self.layout.x0,
                "key_range": self.spec.key_range,
            },
            "sequence": NoteSequence.of(
                self.notes,
                tempo=self.sequence.tempo,
                key=self.sequence.key,
                source=self.sequence.source,
            ).to_dict(),
        }


def _rounded_rect(
    canvas: np.ndarray, x0: int, y0: int, x1: int, y1: int, radius: int, color: tuple[int, int, int]
) -> None:
    cv2.rectangle(canvas, (x0 + radius, y0), (x1 - radius, y1), color, -1)
    cv2.rectangle(canvas, (x0, y0 + radius), (x1, y1 - radius), color, -1)
    for cx, cy in ((x0 + radius, y0 + radius), (x1 - radius, y0 + radius),
                   (x0 + radius, y1 - radius), (x1 - radius, y1 - radius)):
        cv2.circle(canvas, (cx, cy), radius, color, -1)


def _apply_glow(
    canvas: np.ndarray, rects: list[tuple[tuple[int, int, int, int], RGB]], strength: float
) -> np.ndarray:
    """Bloom, drawn as one blurred layer rather than per tile.

    Real renderers bloom heavily, and the halo inflates blob sizes — which is
    precisely the thing stage 4 has to erode back off, so it has to be here.
    """
    layer = np.zeros_like(canvas)
    for (x0, y0, x1, y1), color in rects:
        cv2.rectangle(layer, (x0, y0), (x1, y1), _bgr(color), -1)

    blur = max(3, int(canvas.shape[1] * 0.012) | 1)  # odd kernel
    layer = cv2.GaussianBlur(layer, (blur, blur), 0)
    return cv2.addWeighted(canvas, 1.0, layer, strength, 0)


def render(
    sequence: NoteSequence,
    path: str | Path,
    spec: RenderSpec | None = None,
    write_truth: bool = True,
) -> tuple[Path, Path | None]:
    """Render ``sequence`` to a video file, returning (video, truth) paths.

    The extension may change if the preferred encoder is unavailable, so use the
    returned path rather than assuming the one passed in.
    """
    renderer = SynthRenderer(sequence, spec)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        writer, target = open_writer(
            path, renderer.spec.fps, (renderer.spec.width, renderer.spec.height)
        )
    except VideoError as exc:
        raise RenderError(str(exc)) from exc

    try:
        for index in range(renderer.frame_count):
            writer.write(renderer.frame(index))
    finally:
        writer.release()

    truth_path = None
    if write_truth:
        truth_path = target.with_suffix(".truth.json")
        truth_path.write_text(json.dumps(renderer.truth(), indent=2), encoding="utf-8")

    log.info("rendered %d frames to %s", renderer.frame_count, target)
    return target, truth_path
