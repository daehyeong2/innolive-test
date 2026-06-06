from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from .config import PrivacyBlurConfig
from .errors import PrivacyBlurNotReadyError, UnsupportedFrameTypeError


def make_odd(value: int) -> int:
    value = int(value)
    return value if value % 2 == 1 else value + 1


def require_file(path: Path, hint: str = "") -> None:
    if not path.exists():
        message = f"Required file not found: {path}"
        if hint:
            message += f"\n{hint}"
        raise FileNotFoundError(message)


def clamp_box(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_w: int,
    frame_h: int,
) -> tuple[int, int, int, int]:
    x1_i = max(0, min(frame_w - 1, int(round(x1))))
    y1_i = max(0, min(frame_h - 1, int(round(y1))))
    x2_i = max(0, min(frame_w, int(round(x2))))
    y2_i = max(0, min(frame_h, int(round(y2))))
    return x1_i, y1_i, x2_i, y2_i


def xyxy_to_xywh(box: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = box[:4].astype(float)
    return np.array([x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)], dtype=np.float32)


def box_iou_xywh(box_a: np.ndarray, box_b: np.ndarray) -> float:
    ax, ay, aw, ah = box_a[:4].astype(float)
    bx, by, bw, bh = box_b[:4].astype(float)

    a_x2 = ax + aw
    a_y2 = ay + ah
    b_x2 = bx + bw
    b_y2 = by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = (aw * ah) + (bw * bh) - intersection

    if union <= 1e-6:
        return 0.0
    return float(intersection / union)


def track_match_score(
    face_xywh: np.ndarray,
    previous_xywh: np.ndarray,
    config: PrivacyBlurConfig,
) -> float:
    iou = box_iou_xywh(face_xywh, previous_xywh)

    fx, fy, fw, fh = face_xywh[:4].astype(float)
    px, py, pw, ph = previous_xywh[:4].astype(float)
    face_center = np.array([fx + fw * 0.5, fy + fh * 0.5])
    previous_center = np.array([px + pw * 0.5, py + ph * 0.5])

    distance = float(np.linalg.norm(face_center - previous_center))
    allowed_distance = max(fw, fh, pw, ph) * config.track_center_distance_scale
    center_score = 0.0

    if allowed_distance > 1e-6:
        center_score = max(0.0, 1.0 - distance / allowed_distance)

    if iou < config.track_iou_threshold and center_score <= 0.0:
        return 0.0

    return max(iou, center_score * 0.75)


def is_basic_valid_face_box(xywh: np.ndarray, config: PrivacyBlurConfig) -> bool:
    _, _, w, h = xywh[:4].astype(float)

    if w < config.min_face_size or h < config.min_face_size:
        return False

    ratio = max(w / max(h, 1e-6), h / max(w, 1e-6))
    if ratio > config.max_face_aspect_ratio:
        return False

    return True


def create_expanded_ellipse_mask(
    frame_shape: tuple[int, int, int],
    face_xywh: np.ndarray,
    config: PrivacyBlurConfig,
) -> np.ndarray | None:
    frame_h, frame_w = frame_shape[:2]
    x, y, w, h = face_xywh[:4].astype(float)

    if w <= 1 or h <= 1:
        return None

    small = max(w, h) < config.small_face_threshold
    expand_w = config.small_box_expand_w if small else config.box_expand_w
    expand_h = config.small_box_expand_h if small else config.box_expand_h

    center_x = x + w * 0.5
    center_y = y + h * (0.5 + config.center_y_shift)
    axis_w = max(2, int(round(w * expand_w * 0.5)))
    axis_h = max(2, int(round(h * expand_h * 0.5)))

    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (int(round(center_x)), int(round(center_y))),
        (axis_w, axis_h),
        0,
        0,
        360,
        255,
        thickness=-1,
    )
    return mask.astype(bool)


def apply_blur(
    frame: np.ndarray,
    mask: np.ndarray | None,
    config: PrivacyBlurConfig,
) -> np.ndarray:
    if mask is None or not mask.any():
        return frame

    blur_kernel = make_odd(config.blur_kernel)
    feather_kernel = make_odd(config.feather * 2 + 1)
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return frame

    output = frame.copy()
    frame_h, frame_w = frame.shape[:2]
    pad = max(blur_kernel // 2, feather_kernel // 2)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame_w, x + w + pad)
        y2 = min(frame_h, y + h + pad)

        roi = frame[y1:y2, x1:x2]
        roi_mask = mask_u8[y1:y2, x1:x2].astype(np.float32) / 255.0

        if roi.size == 0:
            continue

        blurred = cv2.GaussianBlur(roi, (blur_kernel, blur_kernel), 0)
        alpha = cv2.GaussianBlur(roi_mask, (feather_kernel, feather_kernel), 0)
        alpha = np.clip(alpha, 0.0, 1.0)[..., None]

        blended = roi.astype(np.float32) * (1.0 - alpha) + blurred.astype(np.float32) * alpha
        output[y1:y2, x1:x2] = blended.astype(np.uint8)

    return output


def select_reference_face(faces: np.ndarray | None) -> np.ndarray | None:
    if faces is None or len(faces) == 0:
        return None
    areas = faces[:, 2] * faces[:, 3]
    scores = faces[:, -1]
    rank = areas * scores
    return faces[int(np.argmax(rank))]


class IdentityMatcher:
    def __init__(self, config: PrivacyBlurConfig):
        self.config = config
        require_file(
            config.me_image_path,
            "Pass enable_identity_exclusion=False if reference-face exclusion is not needed.",
        )
        require_file(config.face_detection_model_path)
        require_file(config.face_recognition_model_path)

        self.detector = cv2.FaceDetectorYN.create(
            str(config.face_detection_model_path),
            "",
            (320, 320),
            0.55,
            0.3,
            5000,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(config.face_recognition_model_path), "")
        self.reference_feature = self._load_reference_feature()

    def _detect_best_yunet_face(self, image: np.ndarray) -> np.ndarray | None:
        h, w = image.shape[:2]
        if w <= 0 or h <= 0:
            return None
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)
        if faces is None:
            return None
        return select_reference_face(faces.astype(np.float32))

    def _load_reference_feature(self) -> np.ndarray:
        reference = cv2.imread(str(self.config.me_image_path))
        if reference is None:
            raise RuntimeError(f"Failed to read reference image: {self.config.me_image_path}")

        face = self._detect_best_yunet_face(reference)
        if face is None:
            raise RuntimeError(f"No face found in reference image: {self.config.me_image_path}")

        aligned = self.recognizer.alignCrop(reference, face)
        return self.recognizer.feature(aligned)

    def similarity_from_yolo_box(self, frame: np.ndarray, face_xywh: np.ndarray) -> float:
        frame_h, frame_w = frame.shape[:2]
        x, y, w, h = face_xywh[:4].astype(float)

        if min(w, h) < self.config.identity_min_face_size:
            return -1.0

        pad_x = w * 0.35
        pad_y = h * 0.35
        x1, y1, x2, y2 = clamp_box(x - pad_x, y - pad_y, x + w + pad_x, y + h + pad_y, frame_w, frame_h)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return -1.0

        try:
            face_in_crop = self._detect_best_yunet_face(crop)
            if face_in_crop is None:
                return -1.0
            aligned = self.recognizer.alignCrop(crop, face_in_crop)
            feature = self.recognizer.feature(aligned)
            return float(
                self.recognizer.match(
                    self.reference_feature,
                    feature,
                    cv2.FaceRecognizerSF_FR_COSINE,
                )
            )
        except cv2.error:
            return -1.0

    def should_treat_as_me(self, previous_is_me: bool, similarity: float) -> bool:
        if similarity >= self.config.face_match_cosine_threshold:
            return True
        if previous_is_me and similarity >= self.config.face_match_lost_cosine_threshold:
            return True
        return False


@dataclass
class Detection:
    xywh: np.ndarray
    confidence: float


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
    def __init__(self, config: PrivacyBlurConfig | None = None):
        cv2.setUseOptimized(True)
        self.config = config or PrivacyBlurConfig()
        require_file(
            self.config.face_model_path,
            "face_model_path must point to face-specific YOLO weights such as models/yolov8n-face.pt.",
        )
        self.detector = YOLO(str(self.config.face_model_path))
        self.identity_matcher = IdentityMatcher(self.config) if self.config.enable_identity_exclusion else None

        self.frame_index = 0
        self.next_track_id = 1
        self.tracks: list[FaceTrack] = []
        self.last_detection_count = 0

    def detect_faces(self, frame: np.ndarray) -> list[Detection]:
        result = self.detector.predict(
            frame,
            imgsz=self.config.face_imgsz,
            conf=self.config.face_conf_threshold,
            iou=self.config.face_iou_threshold,
            max_det=self.config.face_max_detections,
            verbose=False,
            device=self.config.device,
        )[0]

        detections: list[Detection] = []
        if result.boxes is None or len(result.boxes) == 0:
            self.last_detection_count = 0
            return detections

        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()

        for box, conf in zip(xyxy, confs):
            xywh = xyxy_to_xywh(box)
            if not is_basic_valid_face_box(xywh, self.config):
                continue
            detections.append(Detection(xywh=xywh, confidence=float(conf)))

        self.last_detection_count = len(detections)
        return detections

    def active_tracks(self) -> list[FaceTrack]:
        return [
            track
            for track in self.tracks
            if self.frame_index - track.last_seen_frame <= self.config.mask_hold_frames
        ]

    def find_matching_track(
        self,
        face_xywh: np.ndarray,
        tracks: list[FaceTrack],
        matched_track_ids: set[int],
    ) -> FaceTrack | None:
        best_track = None
        best_score = 0.0

        for track in tracks:
            if track.track_id in matched_track_ids:
                continue
            score = track_match_score(face_xywh, track.xywh, self.config)
            if score > best_score:
                best_score = score
                best_track = track

        if best_score <= 0.0:
            return None
        return best_track

    def check_identity(
        self,
        frame: np.ndarray,
        xywh: np.ndarray,
        previous_is_me: bool = False,
    ) -> tuple[bool, float]:
        if self.identity_matcher is None:
            return False, -1.0

        similarity = self.identity_matcher.similarity_from_yolo_box(frame, xywh)
        is_me = self.identity_matcher.should_treat_as_me(previous_is_me, similarity)
        return is_me, similarity

    def update_tracks(self, frame: np.ndarray) -> None:
        detections = self.detect_faces(frame)
        active_tracks = self.active_tracks()
        matched_track_ids: set[int] = set()
        updated_tracks: list[FaceTrack] = []

        for det in detections:
            track = self.find_matching_track(det.xywh, active_tracks, matched_track_ids)

            if track is None:
                is_me, similarity = self.check_identity(frame, det.xywh, previous_is_me=False)
                track = FaceTrack(
                    track_id=self.next_track_id,
                    xywh=det.xywh,
                    confidence=det.confidence,
                    is_me=is_me,
                    similarity=similarity,
                    last_seen_frame=self.frame_index,
                    last_checked_frame=self.frame_index,
                )
                self.next_track_id += 1
            else:
                track.xywh = det.xywh
                track.confidence = det.confidence
                track.last_seen_frame = self.frame_index

                if self.frame_index - track.last_checked_frame >= self.config.identity_recheck_interval:
                    is_me, similarity = self.check_identity(frame, det.xywh, previous_is_me=track.is_me)
                    track.is_me = is_me
                    track.similarity = similarity
                    track.last_checked_frame = self.frame_index

            matched_track_ids.add(track.track_id)
            updated_tracks.append(track)

        for track in active_tracks:
            if track.track_id not in matched_track_ids:
                updated_tracks.append(track)

        self.tracks = updated_tracks

    def build_privacy_mask(self, frame: np.ndarray) -> np.ndarray | None:
        should_detect = (
            self.frame_index == 0
            or self.config.detect_every_n_frames <= 1
            or self.frame_index % self.config.detect_every_n_frames == 0
        )

        if should_detect:
            self.update_tracks(frame)

        active_tracks = self.active_tracks()
        if not active_tracks:
            self.frame_index += 1
            return None

        combined = np.zeros(frame.shape[:2], dtype=bool)

        for track in active_tracks:
            if track.is_me:
                continue

            face_mask = create_expanded_ellipse_mask(frame.shape, track.xywh, self.config)
            if face_mask is not None:
                combined |= face_mask

        self.frame_index += 1
        return combined if combined.any() else None

    def process(self, frame: np.ndarray) -> np.ndarray:
        mask = self.build_privacy_mask(frame)
        return apply_blur(frame, mask, self.config)

    def reset(self) -> None:
        self.frame_index = 0
        self.next_track_id = 1
        self.tracks = []
        self.last_detection_count = 0

    def stats(self) -> dict[str, Any]:
        return {
            "frame_index": self.frame_index,
            "last_detection_count": self.last_detection_count,
            "active_tracks": len(self.active_tracks()),
            "tracks": [
                {
                    "track_id": track.track_id,
                    "confidence": track.confidence,
                    "is_me": track.is_me,
                    "similarity": track.similarity,
                    "last_seen_frame": track.last_seen_frame,
                    "last_checked_frame": track.last_checked_frame,
                }
                for track in self.active_tracks()
            ],
        }


class FacePrivacyFilter:
    def __init__(self, config: PrivacyBlurConfig | None = None):
        self.config = config or PrivacyBlurConfig()
        self.engine = FacePrivacyEngine(self.config)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.worker_count,
            thread_name_prefix="privacy-blur",
        )

    def process_ndarray(self, frame: np.ndarray) -> np.ndarray:
        with self._lock:
            return self.engine.process(frame)

    def apply(self, frame: Any) -> Any:
        if isinstance(frame, np.ndarray):
            return self.process_ndarray(frame)

        video_frame_type = _video_frame_type()
        if video_frame_type is not None and isinstance(frame, video_frame_type):
            return self._apply_video_frame(frame)

        raise UnsupportedFrameTypeError(f"Unsupported frame type: {type(frame).__name__}")

    async def apply_async(self, frame: Any) -> Any:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self.apply, frame)

    def reset(self) -> None:
        with self._lock:
            self.engine.reset()

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return self.engine.stats()

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

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

