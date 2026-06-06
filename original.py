from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import queue
import shutil
import subprocess
import threading
import time
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# Config
# ============================================================

# Input / output
VIDEO_PATH = Path("input.mp4")
OUTPUT_VIDEO_PATH = Path("output_blurred_raw.mp4")
OUTPUT_BROWSER_VIDEO_PATH = Path("output_blurred_browser.mp4")
SAVE_OUTPUT = True
MAKE_BROWSER_COMPATIBLE_MP4 = True

# 1st stage face detector
# 중요: 일반 YOLO(person/car...)가 아니라 face 전용 weight를 넣어야 합니다.
# 예: models/yolov8n-face.pt, models/yolov11n-face.pt 등
FACE_MODEL_PATH = Path("models/yolov8n-face.pt")
FACE_IMGSZ = 1280               # 작은 얼굴 노출 방지 우선. 느리면 960으로 낮추세요.
FACE_CONF_THRESHOLD = 0.22      # 작은 얼굴 recall 우선. 배경 오탐 많으면 0.28~0.35.
FACE_IOU_THRESHOLD = 0.45
FACE_MAX_DETECTIONS = 300
DEVICE = None                   # 예: "cuda:0", "cpu", None이면 ultralytics 기본값

# Basic bbox filtering only. 복잡한 검사 금지.
MIN_FACE_SIZE = 8               # 작은 얼굴까지 잡기 위해 낮게 둠.
MAX_FACE_ASPECT_RATIO = 2.2     # w/h 또는 h/w가 이 이상이면 얼굴 후보에서 제외.

# Identity exclusion: 본인 얼굴은 블러하지 않기.
# 얼굴 노출 방지가 더 중요하면 False로 두고 모든 얼굴을 블러하세요.
ENABLE_IDENTITY_EXCLUSION = True
ME_IMAGE_PATH = Path("me.jpeg")
FACE_DETECTION_MODEL_PATH = Path("models/face_detection_yunet_2023mar.onnx")
FACE_RECOGNITION_MODEL_PATH = Path("models/face_recognition_sface_2021dec.onnx")
FACE_MATCH_COSINE_THRESHOLD = 0.40
FACE_MATCH_LOST_COSINE_THRESHOLD = 0.30
IDENTITY_RECHECK_INTERVAL = 12
IDENTITY_MIN_FACE_SIZE = 36     # 이보다 작은 얼굴은 식별 실패 가능성이 커서 그냥 블러.

# Tracking / mask hold
DETECT_EVERY_N_FRAMES = 1       # 절대 놓치면 안 되면 1 유지.
MASK_HOLD_FRAMES = 18           # 순간 검출 실패 시 이전 마스크 유지.
TRACK_IOU_THRESHOLD = 0.08
TRACK_CENTER_DISTANCE_SCALE = 1.10

# Mask shape
BOX_EXPAND_W = 1.35
BOX_EXPAND_H = 1.60
SMALL_FACE_THRESHOLD = 40
SMALL_BOX_EXPAND_W = 1.75
SMALL_BOX_EXPAND_H = 2.10
CENTER_Y_SHIFT = -0.06          # 얼굴 bbox 기준 살짝 위로 올려 머리/이마 커버.

# Blur
BLUR_KERNEL = 101               # 강하게 하려면 151, 더 빠르게 하려면 71.
FEATHER = 21

# Console logging
LOG_EVERY_N_FRAMES = 30
PRINT_TRACK_DETAIL = False

# Writer worker
# 0이면 무제한 큐라 처리 FPS에 거의 영향이 없습니다. 긴 영상에서는 메모리 사용량 주의.
WRITER_QUEUE_MAXSIZE = 0
WRITER_FOURCC = "mp4v"          # raw 저장용. 마지막에 ffmpeg로 H.264 변환.


# ============================================================
# Utils
# ============================================================

def make_odd(value: int) -> int:
    value = int(value)
    return value if value % 2 == 1 else value + 1


def require_file(path: Path, hint: str = "") -> None:
    if not path.exists():
        message = f"Required file not found: {path}"
        if hint:
            message += f"\n{hint}"
        raise FileNotFoundError(message)


def clamp_box(x1: float, y1: float, x2: float, y2: float, frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
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


def track_match_score(face_xywh: np.ndarray, previous_xywh: np.ndarray) -> float:
    iou = box_iou_xywh(face_xywh, previous_xywh)

    fx, fy, fw, fh = face_xywh[:4].astype(float)
    px, py, pw, ph = previous_xywh[:4].astype(float)
    face_center = np.array([fx + fw * 0.5, fy + fh * 0.5])
    previous_center = np.array([px + pw * 0.5, py + ph * 0.5])

    distance = float(np.linalg.norm(face_center - previous_center))
    allowed_distance = max(fw, fh, pw, ph) * TRACK_CENTER_DISTANCE_SCALE
    center_score = 0.0

    if allowed_distance > 1e-6:
        center_score = max(0.0, 1.0 - distance / allowed_distance)

    if iou < TRACK_IOU_THRESHOLD and center_score <= 0.0:
        return 0.0

    return max(iou, center_score * 0.75)


def is_basic_valid_face_box(xywh: np.ndarray) -> bool:
    _, _, w, h = xywh[:4].astype(float)

    if w < MIN_FACE_SIZE or h < MIN_FACE_SIZE:
        return False

    ratio = max(w / max(h, 1e-6), h / max(w, 1e-6))
    if ratio > MAX_FACE_ASPECT_RATIO:
        return False

    return True


def create_expanded_ellipse_mask(frame_shape: tuple[int, int, int], face_xywh: np.ndarray) -> Optional[np.ndarray]:
    frame_h, frame_w = frame_shape[:2]
    x, y, w, h = face_xywh[:4].astype(float)

    if w <= 1 or h <= 1:
        return None

    small = max(w, h) < SMALL_FACE_THRESHOLD
    expand_w = SMALL_BOX_EXPAND_W if small else BOX_EXPAND_W
    expand_h = SMALL_BOX_EXPAND_H if small else BOX_EXPAND_H

    center_x = x + w * 0.5
    center_y = y + h * (0.5 + CENTER_Y_SHIFT)
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


def apply_blur(frame: np.ndarray, mask: Optional[np.ndarray]) -> np.ndarray:
    if mask is None or not mask.any():
        return frame

    blur_kernel = make_odd(BLUR_KERNEL)
    feather_kernel = make_odd(FEATHER * 2 + 1)
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


def select_reference_face(faces: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if faces is None or len(faces) == 0:
        return None
    areas = faces[:, 2] * faces[:, 3]
    scores = faces[:, -1]
    rank = areas * scores
    return faces[int(np.argmax(rank))]


# ============================================================
# Async writer
# ============================================================

class AsyncVideoWriter:
    def __init__(self, path: Path, fps: float, frame_size: tuple[int, int]):
        self.path = path
        self.fps = fps if fps and fps > 1e-6 else 30.0
        self.frame_size = frame_size
        self.queue: queue.Queue[Optional[np.ndarray]] = queue.Queue(maxsize=WRITER_QUEUE_MAXSIZE)
        self.error: Optional[BaseException] = None
        self.frames_written = 0
        self.started_at = time.perf_counter()

        fourcc = cv2.VideoWriter_fourcc(*WRITER_FOURCC)
        self.writer = cv2.VideoWriter(str(path), fourcc, self.fps, frame_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter: {path}")

        self.thread = threading.Thread(target=self._worker, name="AsyncVideoWriter", daemon=True)
        self.thread.start()

    def _worker(self) -> None:
        try:
            while True:
                item = self.queue.get()
                try:
                    if item is None:
                        return
                    self.writer.write(item)
                    self.frames_written += 1
                finally:
                    self.queue.task_done()
        except BaseException as exc:
            self.error = exc
        finally:
            self.writer.release()

    def write(self, frame: np.ndarray) -> None:
        if self.error is not None:
            raise RuntimeError("Video writer worker failed") from self.error
        self.queue.put(frame.copy())

    def qsize(self) -> int:
        return self.queue.qsize()

    def close(self) -> None:
        self.queue.put(None)
        self.queue.join()
        self.thread.join()
        if self.error is not None:
            raise RuntimeError("Video writer worker failed") from self.error


# ============================================================
# Identity matcher: SFace uses YuNet only inside detected face crop
# ============================================================

class IdentityMatcher:
    def __init__(self):
        require_file(ME_IMAGE_PATH, "본인 얼굴 제외가 필요 없으면 ENABLE_IDENTITY_EXCLUSION = False 로 바꾸세요.")
        require_file(FACE_DETECTION_MODEL_PATH)
        require_file(FACE_RECOGNITION_MODEL_PATH)

        self.detector = cv2.FaceDetectorYN.create(
            str(FACE_DETECTION_MODEL_PATH),
            "",
            (320, 320),
            0.55,
            0.3,
            5000,
        )
        self.recognizer = cv2.FaceRecognizerSF.create(str(FACE_RECOGNITION_MODEL_PATH), "")
        self.reference_feature = self._load_reference_feature()

    def _detect_best_yunet_face(self, image: np.ndarray) -> Optional[np.ndarray]:
        h, w = image.shape[:2]
        if w <= 0 or h <= 0:
            return None
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(image)
        if faces is None:
            return None
        return select_reference_face(faces.astype(np.float32))

    def _load_reference_feature(self) -> np.ndarray:
        reference = cv2.imread(str(ME_IMAGE_PATH))
        if reference is None:
            raise RuntimeError(f"Failed to read reference image: {ME_IMAGE_PATH}")

        face = self._detect_best_yunet_face(reference)
        if face is None:
            raise RuntimeError(f"No face found in reference image: {ME_IMAGE_PATH}")

        aligned = self.recognizer.alignCrop(reference, face)
        feature = self.recognizer.feature(aligned)
        print(f"Loaded reference face from {ME_IMAGE_PATH}")
        return feature

    def similarity_from_yolo_box(self, frame: np.ndarray, face_xywh: np.ndarray) -> float:
        frame_h, frame_w = frame.shape[:2]
        x, y, w, h = face_xywh[:4].astype(float)

        if min(w, h) < IDENTITY_MIN_FACE_SIZE:
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

    @staticmethod
    def should_treat_as_me(previous_is_me: bool, similarity: float) -> bool:
        if similarity >= FACE_MATCH_COSINE_THRESHOLD:
            return True
        if previous_is_me and similarity >= FACE_MATCH_LOST_COSINE_THRESHOLD:
            return True
        return False


# ============================================================
# Face privacy engine
# ============================================================

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
    def __init__(self):
        require_file(
            FACE_MODEL_PATH,
            "FACE_MODEL_PATH에는 일반 YOLO가 아니라 face 전용 weight를 넣어야 합니다. 예: models/yolov8n-face.pt",
        )
        self.detector = YOLO(str(FACE_MODEL_PATH))
        self.identity_matcher = IdentityMatcher() if ENABLE_IDENTITY_EXCLUSION else None

        self.frame_index = 0
        self.next_track_id = 1
        self.tracks: list[FaceTrack] = []
        self.last_detection_count = 0

    def detect_faces(self, frame: np.ndarray) -> list[Detection]:
        result = self.detector.predict(
            frame,
            imgsz=FACE_IMGSZ,
            conf=FACE_CONF_THRESHOLD,
            iou=FACE_IOU_THRESHOLD,
            max_det=FACE_MAX_DETECTIONS,
            verbose=False,
            device=DEVICE,
        )[0]

        detections: list[Detection] = []
        if result.boxes is None or len(result.boxes) == 0:
            self.last_detection_count = 0
            return detections

        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        confs = result.boxes.conf.detach().cpu().numpy()

        for box, conf in zip(xyxy, confs):
            xywh = xyxy_to_xywh(box)
            if not is_basic_valid_face_box(xywh):
                continue
            detections.append(Detection(xywh=xywh, confidence=float(conf)))

        self.last_detection_count = len(detections)
        return detections

    def active_tracks(self) -> list[FaceTrack]:
        return [
            track
            for track in self.tracks
            if self.frame_index - track.last_seen_frame <= MASK_HOLD_FRAMES
        ]

    def find_matching_track(
        self,
        face_xywh: np.ndarray,
        tracks: list[FaceTrack],
        matched_track_ids: set[int],
    ) -> Optional[FaceTrack]:
        best_track = None
        best_score = 0.0

        for track in tracks:
            if track.track_id in matched_track_ids:
                continue
            score = track_match_score(face_xywh, track.xywh)
            if score > best_score:
                best_score = score
                best_track = track

        if best_score <= 0.0:
            return None
        return best_track

    def check_identity(self, frame: np.ndarray, xywh: np.ndarray, previous_is_me: bool = False) -> tuple[bool, float]:
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

                if self.frame_index - track.last_checked_frame >= IDENTITY_RECHECK_INTERVAL:
                    is_me, similarity = self.check_identity(frame, det.xywh, previous_is_me=track.is_me)
                    track.is_me = is_me
                    track.similarity = similarity
                    track.last_checked_frame = self.frame_index

            matched_track_ids.add(track.track_id)
            updated_tracks.append(track)

        # 감지 실패한 얼굴은 잠깐 유지해서 순간 노출 방지.
        for track in active_tracks:
            if track.track_id not in matched_track_ids:
                updated_tracks.append(track)

        self.tracks = updated_tracks

    def build_privacy_mask(self, frame: np.ndarray) -> Optional[np.ndarray]:
        should_detect = (
            self.frame_index == 0
            or DETECT_EVERY_N_FRAMES <= 1
            or self.frame_index % DETECT_EVERY_N_FRAMES == 0
        )

        if should_detect:
            self.update_tracks(frame)

        active_tracks = self.active_tracks()
        if not active_tracks:
            self.frame_index += 1
            return None

        combined = np.zeros(frame.shape[:2], dtype=bool)

        for track in active_tracks:
            # 본인으로 확실히 판별된 경우만 제외.
            # 작거나 애매한 얼굴은 privacy 우선으로 블러합니다.
            if track.is_me:
                continue

            face_mask = create_expanded_ellipse_mask(frame.shape, track.xywh)
            if face_mask is not None:
                combined |= face_mask

        self.frame_index += 1
        return combined if combined.any() else None


# ============================================================
# Browser-compatible MP4 conversion
# ============================================================

def make_browser_compatible_mp4(raw_video_path: Path, source_video_path: Path, output_path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        print("[WARN] ffmpeg not found. Install it with: sudo apt install -y ffmpeg")
        return False

    command = [
        ffmpeg,
        "-y",
        "-i", str(raw_video_path),
        "-i", str(source_video_path),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "128k",
        "-shortest",
        "-movflags", "+faststart",
        str(output_path),
    ]

    print("\nConverting to browser-compatible H.264 MP4...")
    try:
        subprocess.run(command, check=True)
        print(f"Saved browser-compatible video: {output_path}")
        return True
    except subprocess.CalledProcessError as exc:
        print(f"[WARN] ffmpeg conversion failed: {exc}")
        return False


# ============================================================
# Main
# ============================================================

def main() -> None:
    cv2.setUseOptimized(True)

    require_file(VIDEO_PATH)
    engine = FacePrivacyEngine()

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {VIDEO_PATH}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if src_fps <= 1e-6:
        src_fps = 30.0

    print(f"Input: {VIDEO_PATH}")
    print(f"Resolution: {src_w}x{src_h}, source FPS: {src_fps:.3f}, frames: {total_frames or 'unknown'}")
    print(f"Detector: {FACE_MODEL_PATH}, imgsz={FACE_IMGSZ}, conf={FACE_CONF_THRESHOLD}")
    print(f"Identity exclusion: {ENABLE_IDENTITY_EXCLUSION}")
    print("GUI disabled. Processing in console only.\n")

    writer: Optional[AsyncVideoWriter] = None
    if SAVE_OUTPUT:
        writer = AsyncVideoWriter(OUTPUT_VIDEO_PATH, src_fps, (src_w, src_h))
        print(f"Async writer started: {OUTPUT_VIDEO_PATH}")

    processed = 0
    start_time = time.perf_counter()
    last_log_time = start_time
    last_log_frame = 0
    last_process_ms = 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_start = time.perf_counter()

            mask = engine.build_privacy_mask(frame)
            output = apply_blur(frame, mask)

            # Writer worker로 넘기기만 함. 실제 인코딩/파일 쓰기는 별도 스레드에서 진행.
            if writer is not None:
                writer.write(output)

            processed += 1
            last_process_ms = (time.perf_counter() - frame_start) * 1000.0

            should_log = processed % LOG_EVERY_N_FRAMES == 0 or processed == 1
            if should_log:
                now = time.perf_counter()
                total_elapsed = max(now - start_time, 1e-6)
                interval_elapsed = max(now - last_log_time, 1e-6)
                avg_fps = processed / total_elapsed
                interval_fps = (processed - last_log_frame) / interval_elapsed
                progress = (processed / total_frames * 100.0) if total_frames > 0 else 0.0
                qsize = writer.qsize() if writer is not None else 0
                active = len(engine.active_tracks())

                print(
                    f"[{processed}/{total_frames or '?'} | {progress:6.2f}%] "
                    f"interval FPS={interval_fps:6.2f}, avg FPS={avg_fps:6.2f}, "
                    f"last={last_process_ms:7.2f} ms, "
                    f"detected={engine.last_detection_count:3d}, active_tracks={active:3d}, writer_q={qsize}"
                )

                if PRINT_TRACK_DETAIL:
                    for track in engine.active_tracks():
                        x, y, w, h = track.xywh.astype(int)
                        print(
                            f"  track={track.track_id}, box=({x},{y},{w},{h}), "
                            f"conf={track.confidence:.2f}, is_me={track.is_me}, sim={track.similarity:.3f}"
                        )

                last_log_time = now
                last_log_frame = processed

    finally:
        cap.release()

    processing_done_time = time.perf_counter()
    processing_elapsed = max(processing_done_time - start_time, 1e-6)
    print("\nProcessing loop finished.")
    print(f"Processed frames: {processed}")
    print(f"Processing-loop FPS: {processed / processing_elapsed:.2f}")

    if writer is not None:
        print("Waiting for writer worker to flush remaining frames...")
        writer.close()
        writer_elapsed = max(time.perf_counter() - processing_done_time, 1e-6)
        print(f"Raw video saved: {OUTPUT_VIDEO_PATH}")
        print(f"Writer flush time: {writer_elapsed:.2f}s, frames written: {writer.frames_written}")

    if SAVE_OUTPUT and MAKE_BROWSER_COMPATIBLE_MP4:
        make_browser_compatible_mp4(OUTPUT_VIDEO_PATH, VIDEO_PATH, OUTPUT_BROWSER_VIDEO_PATH)

    print("\nDone.")
    if MAKE_BROWSER_COMPATIBLE_MP4:
        print("JupyterLab preview file:", OUTPUT_BROWSER_VIDEO_PATH)
        print("Notebook display example:")
        print("from IPython.display import Video")
        print(f"Video('{OUTPUT_BROWSER_VIDEO_PATH}', embed=True, width=960)")


if __name__ == "__main__":
    main()
