from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any

import numpy as np

from .config import PrivacyBlurConfig
from .errors import PrivacyBlurNotReadyError, UnsupportedFrameTypeError
from .runtime import SharedGpuRuntime, get_shared_runtime


def box_iou_xywh(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax, ay, aw, ah = box_a[:4].astype(float)
    bx, by, bw, bh = box_b[:4].astype(float)
    intersection_x1 = max(ax, bx)
    intersection_y1 = max(ay, by)
    intersection_x2 = min(ax + aw, bx + bw)
    intersection_y2 = min(ay + ah, by + bh)
    intersection_width = max(0.0, intersection_x2 - intersection_x1)
    intersection_height = max(0.0, intersection_y2 - intersection_y1)
    intersection = intersection_width * intersection_height
    union = aw * ah + bw * bh - intersection
    return float(intersection / union) if union > 1e-6 else 0.0


def track_match_score(
    face_xywh: np.ndarray,
    previous_xywh: np.ndarray,
    config: PrivacyBlurConfig,
) -> float:
    iou = box_iou_xywh(face_xywh, previous_xywh)
    fx, fy, fw, fh = face_xywh[:4].astype(float)
    px, py, pw, ph = previous_xywh[:4].astype(float)
    distance = float(
        np.linalg.norm(
            np.array([fx + fw * 0.5, fy + fh * 0.5])
            - np.array([px + pw * 0.5, py + ph * 0.5])
        )
    )
    allowed_distance = max(fw, fh, pw, ph) * config.track_center_distance_scale
    center_score = (
        max(0.0, 1.0 - distance / allowed_distance) if allowed_distance > 1e-6 else 0.0
    )
    if iou < config.track_iou_threshold and center_score <= 0.0:
        return 0.0
    return max(iou, center_score * 0.75)


@dataclass
class FaceTrack:
    track_id: int
    xywh: np.ndarray
    confidence: float
    is_me: bool
    similarity: float
    last_seen_frame: int
    last_checked_frame: int


class FacePrivacyEngine:
    """Per-stream tracking state backed by the process-wide GPU runtime."""

    def __init__(
        self,
        config: PrivacyBlurConfig | None = None,
        runtime: SharedGpuRuntime | None = None,
    ):
        self.config = config or PrivacyBlurConfig()
        self.runtime = runtime or get_shared_runtime(self.config)
        self.frame_index = 0
        self.next_track_id = 1
        self.tracks: list[FaceTrack] = []
        self.last_detection_count = 0
        self._lock = threading.RLock()

    def _active_tracks(self) -> list[FaceTrack]:
        return [
            track
            for track in self.tracks
            if self.frame_index - track.last_seen_frame <= self.config.mask_hold_frames
        ]

    def active_tracks(self) -> list[FaceTrack]:
        with self._lock:
            return list(self._active_tracks())

    def needs_detection(self) -> bool:
        with self._lock:
            return (
                self.frame_index == 0
                or self.config.detect_every_n_frames <= 1
                or self.frame_index % self.config.detect_every_n_frames == 0
            )

    def _find_matching_track(
        self,
        face_xywh: np.ndarray,
        tracks: list[FaceTrack],
        matched_track_ids: set[int],
    ) -> FaceTrack | None:
        best_track: FaceTrack | None = None
        best_score = 0.0
        for track in tracks:
            if track.track_id in matched_track_ids:
                continue
            score = track_match_score(face_xywh, track.xywh, self.config)
            if score > best_score:
                best_score = score
                best_track = track
        return best_track if best_score > 0.0 else None

    def begin_frame(
        self,
        detections: list[tuple[np.ndarray, float]] | None,
    ) -> list[FaceTrack]:
        """Update boxes and return tracks that need a batched identity check."""
        if detections is None:
            return []

        with self._lock:
            self.last_detection_count = len(detections)
            active_tracks = self._active_tracks()
            matched_track_ids: set[int] = set()
            updated_tracks: list[FaceTrack] = []
            identity_checks: list[FaceTrack] = []

            for xywh, confidence in detections:
                track = self._find_matching_track(
                    xywh, active_tracks, matched_track_ids
                )
                needs_identity_check = False
                if track is None:
                    track = FaceTrack(
                        track_id=self.next_track_id,
                        xywh=xywh,
                        confidence=confidence,
                        is_me=False,
                        similarity=-1.0,
                        last_seen_frame=self.frame_index,
                        last_checked_frame=self.frame_index,
                    )
                    self.next_track_id += 1
                    needs_identity_check = self.config.enable_identity_exclusion
                else:
                    track.xywh = xywh
                    track.confidence = confidence
                    track.last_seen_frame = self.frame_index
                    needs_identity_check = (
                        self.config.enable_identity_exclusion
                        and self.frame_index - track.last_checked_frame
                        >= self.config.identity_recheck_interval
                    )
                    if needs_identity_check:
                        track.last_checked_frame = self.frame_index

                if needs_identity_check:
                    if (
                        min(float(xywh[2]), float(xywh[3]))
                        >= self.config.identity_min_face_size
                    ):
                        identity_checks.append(track)
                    else:
                        track.similarity = -1.0
                        track.is_me = False

                matched_track_ids.add(track.track_id)
                updated_tracks.append(track)

            updated_tracks.extend(
                track
                for track in active_tracks
                if track.track_id not in matched_track_ids
            )
            self.tracks = updated_tracks
            return identity_checks

    def apply_identity_matches(self, matches: list[tuple[FaceTrack, float]]) -> None:
        with self._lock:
            for track, similarity in matches:
                previous_is_me = track.is_me
                track.similarity = similarity
                track.is_me = similarity >= self.config.face_match_cosine_threshold or (
                    previous_is_me
                    and similarity >= self.config.face_match_lost_cosine_threshold
                )

    def finish_frame(self) -> list[np.ndarray]:
        with self._lock:
            privacy_boxes = [
                track.xywh.copy() for track in self._active_tracks() if not track.is_me
            ]
            self.frame_index += 1
            return privacy_boxes

    def process(self, frame: np.ndarray) -> np.ndarray:
        return self.runtime.process(self, frame)

    async def process_async(self, frame: np.ndarray) -> np.ndarray:
        return await self.runtime.process_async(self, frame)

    def reset(self) -> None:
        with self._lock:
            self.frame_index = 0
            self.next_track_id = 1
            self.tracks = []
            self.last_detection_count = 0

    def stats(self) -> dict[str, Any]:
        with self._lock:
            active_tracks = self._active_tracks()
            return {
                "frame_index": self.frame_index,
                "last_detection_count": self.last_detection_count,
                "active_tracks": len(active_tracks),
                "tracks": [
                    {
                        "track_id": track.track_id,
                        "confidence": track.confidence,
                        "is_me": track.is_me,
                        "similarity": track.similarity,
                        "last_seen_frame": track.last_seen_frame,
                        "last_checked_frame": track.last_checked_frame,
                    }
                    for track in active_tracks
                ],
                "runtime": self.runtime.stats(),
            }


class FacePrivacyFilter:
    def __init__(
        self,
        config: PrivacyBlurConfig | None = None,
        runtime: SharedGpuRuntime | None = None,
    ):
        self.config = config or PrivacyBlurConfig()
        self.engine = FacePrivacyEngine(self.config, runtime=runtime)
        self._closed = False

    def process_ndarray(self, frame: np.ndarray) -> np.ndarray:
        if self._closed:
            raise RuntimeError("The privacy filter is closed")
        return self.engine.process(frame)

    def apply(self, frame: Any) -> Any:
        if isinstance(frame, np.ndarray):
            return self.process_ndarray(frame)
        video_frame_type = _video_frame_type()
        if video_frame_type is not None and isinstance(frame, video_frame_type):
            return self._apply_video_frame(frame)
        raise UnsupportedFrameTypeError(
            f"Unsupported frame type: {type(frame).__name__}"
        )

    async def apply_async(self, frame: Any) -> Any:
        if self._closed:
            raise RuntimeError("The privacy filter is closed")
        if isinstance(frame, np.ndarray):
            return await self.engine.process_async(frame)
        video_frame_type = _video_frame_type()
        if video_frame_type is not None and isinstance(frame, video_frame_type):
            image = frame.to_ndarray(format="bgr24")
            processed = await self.engine.process_async(image)
            protected = video_frame_type.from_ndarray(processed, format="bgr24")
            protected.pts = frame.pts
            protected.time_base = frame.time_base
            return protected
        raise UnsupportedFrameTypeError(
            f"Unsupported frame type: {type(frame).__name__}"
        )

    def reset(self) -> None:
        self.engine.reset()

    def stats(self) -> dict[str, Any]:
        return self.engine.stats()

    def close(self) -> None:
        self._closed = True

    def _apply_video_frame(self, frame: Any) -> Any:
        video_frame_type = _video_frame_type()
        if video_frame_type is None:
            raise UnsupportedFrameTypeError("av.VideoFrame is not available")
        image = frame.to_ndarray(format="bgr24")
        processed = self.process_ndarray(image)
        protected = video_frame_type.from_ndarray(processed, format="bgr24")
        protected.pts = frame.pts
        protected.time_base = frame.time_base
        return protected


def _video_frame_type() -> Any | None:
    try:
        from av import VideoFrame
    except ImportError:
        return None
    return VideoFrame


def create_filter(config: PrivacyBlurConfig | None = None) -> FacePrivacyFilter:
    try:
        return FacePrivacyFilter(config=config)
    except FileNotFoundError as exc:
        raise PrivacyBlurNotReadyError(str(exc)) from exc
