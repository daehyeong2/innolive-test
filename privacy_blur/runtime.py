from __future__ import annotations

import asyncio
import atexit
from collections import defaultdict
from concurrent.futures import Future
from dataclasses import dataclass
import logging
from pathlib import Path
from queue import Empty, Full, Queue
import threading
import time
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from .config import PrivacyBlurConfig
from .errors import PrivacyBlurNotReadyError

if TYPE_CHECKING:
    from .core import FacePrivacyEngine, FaceTrack


LOGGER = logging.getLogger(__name__)
_STOP = object()


@dataclass(slots=True)
class _FrameRequest:
    engine: FacePrivacyEngine
    frame: np.ndarray
    future: Future[np.ndarray]


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def _make_odd(value: int) -> int:
    value = int(value)
    return value if value % 2 else value + 1


def _resolve_torch_device(config: PrivacyBlurConfig) -> torch.device:
    requested = config.device
    if requested is None:
        if torch.cuda.is_available():
            requested = "cuda:0"
        elif torch.backends.mps.is_available():
            requested = "mps"
        else:
            requested = "cpu"
    elif str(requested).isdigit():
        requested = f"cuda:{requested}"

    device = torch.device(str(requested))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise PrivacyBlurNotReadyError(
            f"CUDA device {device} was requested, but torch.cuda.is_available() is false. "
            "Install a CUDA-enabled PyTorch build and verify the NVIDIA driver."
        )
    if config.require_gpu and device.type != "cuda":
        raise PrivacyBlurNotReadyError(
            f"A CUDA GPU is required, but the configured device is {device}."
        )
    return device


def _remove_initializer_inputs(model: onnx.ModelProto) -> None:
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    real_inputs = [
        value for value in model.graph.input if value.name not in initializer_names
    ]
    del model.graph.input[:]
    model.graph.input.extend(real_inputs)


def _dynamic_onnx_model(path: Path, kind: str, identity_size: int) -> bytes:
    model = onnx.load(str(path))
    _remove_initializer_inputs(model)

    if kind == "sface":
        values = (
            list(model.graph.input)
            + list(model.graph.output)
            + list(model.graph.value_info)
        )
        for value in values:
            dimensions = value.type.tensor_type.shape.dim
            if dimensions and dimensions[0].dim_value == 1:
                dimensions[0].ClearField("dim_value")
                dimensions[0].dim_param = "batch"
    elif kind == "yunet":
        dimensions = model.graph.input[0].type.tensor_type.shape.dim
        dimensions[0].ClearField("dim_value")
        dimensions[0].dim_param = "batch"
        dimensions[2].dim_value = identity_size
        dimensions[3].dim_value = identity_size
        for value in list(model.graph.output) + list(model.graph.value_info):
            output_dimensions = value.type.tensor_type.shape.dim
            if len(output_dimensions) > 1 and output_dimensions[1].dim_value in {
                400,
                1600,
                6400,
            }:
                output_dimensions[1].ClearField("dim_value")
                output_dimensions[1].dim_param = "batched_anchors"
    else:  # pragma: no cover - internal programming error
        raise ValueError(f"Unsupported ONNX model kind: {kind}")

    return model.SerializeToString()


class SharedGpuRuntime:
    """One model set and one dynamic batching worker shared by all video streams."""

    def __init__(self, config: PrivacyBlurConfig):
        self.config = config
        self.device = _resolve_torch_device(config)
        self.cuda_enabled = self.device.type == "cuda"
        self.half_enabled = self.cuda_enabled and config.use_half
        self._device_index = self.device.index or 0

        if self.cuda_enabled:
            torch.cuda.set_device(self.device)
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self._cuda_stream: torch.cuda.Stream | None = torch.cuda.Stream(
                device=self.device
            )
        else:
            self._cuda_stream = None

        _require_file(config.face_model_path)
        self.detector = YOLO(str(config.face_model_path))

        self.yunet_session: ort.InferenceSession | None = None
        self.sface_session: ort.InferenceSession | None = None
        self.reference_feature: torch.Tensor | None = None
        if config.enable_identity_exclusion:
            _require_file(config.me_image_path)
            _require_file(config.face_detection_model_path)
            _require_file(config.face_recognition_model_path)
            self.yunet_session = self._create_ort_session(
                config.face_detection_model_path,
                kind="yunet",
            )
            self.sface_session = self._create_ort_session(
                config.face_recognition_model_path,
                kind="sface",
            )
            self.reference_feature = self._load_reference_feature()

        self._queue: Queue[_FrameRequest | object] = Queue(
            maxsize=config.max_pending_frames
        )
        self._closed = False
        self._stats_lock = threading.Lock()
        self._batches = 0
        self._frames = 0
        self._last_batch_size = 0
        self._total_batch_latency_ms = 0.0
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="privacy-blur-gpu-batcher",
            daemon=True,
        )
        self._worker.start()

    def _create_ort_session(self, path: Path, kind: str) -> ort.InferenceSession:
        available = ort.get_available_providers()
        if self.cuda_enabled and "CUDAExecutionProvider" not in available:
            if self.config.require_gpu:
                raise PrivacyBlurNotReadyError(
                    "onnxruntime does not expose CUDAExecutionProvider. "
                    "Install onnxruntime-gpu with a CUDA/cuDNN version compatible with PyTorch."
                )
            LOGGER.warning(
                "CUDAExecutionProvider is unavailable; %s will run on CPU", path.name
            )

        providers: list[Any]
        if self.cuda_enabled and "CUDAExecutionProvider" in available:
            provider_options = {
                "device_id": str(self._device_index),
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": "1",
            }
            if self._cuda_stream is not None:
                provider_options["user_compute_stream"] = str(
                    self._cuda_stream.cuda_stream
                )
            providers = [("CUDAExecutionProvider", provider_options)]
        else:
            providers = ["CPUExecutionProvider"]

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.intra_op_num_threads = 1
        options.log_severity_level = 3
        model_bytes = _dynamic_onnx_model(
            path, kind, self.config.identity_detector_size
        )
        session = ort.InferenceSession(
            model_bytes, sess_options=options, providers=providers
        )

        if (
            self.config.require_gpu
            and session.get_providers()[0] != "CUDAExecutionProvider"
        ):
            raise PrivacyBlurNotReadyError(
                f"{path.name} did not initialize on CUDAExecutionProvider"
            )
        return session

    def _load_reference_feature(self) -> torch.Tensor:
        reference = cv2.imread(str(self.config.me_image_path))
        if reference is None:
            raise PrivacyBlurNotReadyError(
                f"Failed to read reference image: {self.config.me_image_path}"
            )
        detections = self._predict_yolo([reference])[0]
        if not detections:
            raise PrivacyBlurNotReadyError(
                f"No face found in reference image: {self.config.me_image_path}"
            )
        box, _ = max(detections, key=lambda item: item[0][2] * item[0][3] * item[1])
        embeddings, valid = self._face_embeddings([(reference, box)])
        if not bool(valid[0]):
            raise PrivacyBlurNotReadyError(
                f"YuNet could not align the face in reference image: {self.config.me_image_path}"
            )
        return embeddings[0].detach()

    def submit(
        self, engine: FacePrivacyEngine, frame: np.ndarray
    ) -> Future[np.ndarray]:
        if self._closed:
            raise RuntimeError("The shared inference runtime is closed")
        future: Future[np.ndarray] = Future()
        request = _FrameRequest(engine=engine, frame=frame, future=future)
        try:
            self._queue.put_nowait(request)
        except Full as exc:
            raise RuntimeError(
                "GPU inference queue is full; reduce incoming FPS or increase max_pending_frames"
            ) from exc
        return future

    def process(self, engine: FacePrivacyEngine, frame: np.ndarray) -> np.ndarray:
        return self.submit(engine, frame).result()

    async def process_async(
        self, engine: FacePrivacyEngine, frame: np.ndarray
    ) -> np.ndarray:
        return await asyncio.wrap_future(self.submit(engine, frame))

    def _worker_loop(self) -> None:
        if self.cuda_enabled:
            torch.cuda.set_device(self.device)

        while True:
            first = self._queue.get()
            if first is _STOP:
                return
            assert isinstance(first, _FrameRequest)

            requests = [first]
            deadline = time.perf_counter() + self.config.batch_wait_ms / 1000.0
            while len(requests) < self.config.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    item = self._queue.get(timeout=remaining)
                except Empty:
                    break
                if item is _STOP:
                    self._queue.put_nowait(_STOP)
                    break
                assert isinstance(item, _FrameRequest)
                requests.append(item)

            requests = [
                request for request in requests if not request.future.cancelled()
            ]
            if not requests:
                continue

            started = time.perf_counter()
            try:
                outputs = self._run_batch(requests)
            except BaseException as exc:
                LOGGER.exception("GPU inference batch failed")
                for request in requests:
                    if not request.future.done():
                        request.future.set_exception(exc)
                continue

            latency_ms = (time.perf_counter() - started) * 1000.0
            with self._stats_lock:
                self._batches += 1
                self._frames += len(requests)
                self._last_batch_size = len(requests)
                self._total_batch_latency_ms += latency_ms

            for request, output in zip(requests, outputs):
                if not request.future.done():
                    request.future.set_result(output)

    @torch.inference_mode()
    def _run_batch(self, requests: list[_FrameRequest]) -> list[np.ndarray]:
        detect_indices = [
            index
            for index, request in enumerate(requests)
            if request.engine.needs_detection()
        ]
        detections: dict[int, list[tuple[np.ndarray, float]]] = {}
        if detect_indices:
            detected = self._predict_yolo(
                [requests[index].frame for index in detect_indices]
            )
            detections = dict(zip(detect_indices, detected))

        identity_work: list[tuple[FacePrivacyEngine, FaceTrack, np.ndarray]] = []
        for index, request in enumerate(requests):
            tracks = request.engine.begin_frame(detections.get(index))
            identity_work.extend(
                (request.engine, track, request.frame) for track in tracks
            )

        if identity_work:
            similarities = self._identity_similarities(
                [(frame, track.xywh) for _, track, frame in identity_work]
            )
            grouped: dict[FacePrivacyEngine, list[tuple[FaceTrack, float]]] = (
                defaultdict(list)
            )
            for (engine, track, _), similarity in zip(identity_work, similarities):
                grouped[engine].append((track, similarity))
            for engine, matches in grouped.items():
                engine.apply_identity_matches(matches)

        privacy_boxes = [request.engine.finish_frame() for request in requests]
        return self._blur_frames([request.frame for request in requests], privacy_boxes)

    def _predict_yolo(
        self, frames: list[np.ndarray]
    ) -> list[list[tuple[np.ndarray, float]]]:
        results = self.detector.predict(
            source=frames,
            imgsz=self.config.face_imgsz,
            conf=self.config.face_conf_threshold,
            iou=self.config.face_iou_threshold,
            max_det=self.config.face_max_detections,
            verbose=False,
            device=str(self.device),
            half=self.half_enabled,
            batch=len(frames),
        )

        parsed: list[list[tuple[np.ndarray, float]]] = []
        for result in results:
            frame_detections: list[tuple[np.ndarray, float]] = []
            if result.boxes is not None and len(result.boxes):
                boxes = (
                    torch.as_tensor(result.boxes.xyxy).detach().float().cpu().numpy()
                )
                confidences = (
                    torch.as_tensor(result.boxes.conf).detach().float().cpu().numpy()
                )
                for box, confidence in zip(boxes, confidences):
                    x1, y1, x2, y2 = box[:4]
                    xywh = np.array(
                        [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)],
                        dtype=np.float32,
                    )
                    width, height = float(xywh[2]), float(xywh[3])
                    if (
                        width < self.config.min_face_size
                        or height < self.config.min_face_size
                    ):
                        continue
                    ratio = max(width / max(height, 1e-6), height / max(width, 1e-6))
                    if ratio <= self.config.max_face_aspect_ratio:
                        frame_detections.append((xywh, float(confidence)))
            parsed.append(frame_detections)
        return parsed

    def _identity_similarities(
        self,
        faces: list[tuple[np.ndarray, np.ndarray]],
    ) -> list[float]:
        if self.reference_feature is None:
            return [-1.0] * len(faces)
        embeddings, valid = self._face_embeddings(faces)
        scores = embeddings @ self.reference_feature
        scores = torch.where(valid, scores, torch.full_like(scores, -1.0))
        return [float(value) for value in scores.detach().float().cpu().tolist()]

    def _face_embeddings(
        self,
        faces: list[tuple[np.ndarray, np.ndarray]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        size = self.config.identity_detector_size
        crops: list[torch.Tensor] = []
        source_indices: list[int] = []
        frame_tensors: dict[int, torch.Tensor] = {}

        for index, (frame, box) in enumerate(faces):
            frame_h, frame_w = frame.shape[:2]
            x, y, width, height = box[:4].astype(float)
            pad_x, pad_y = width * 0.35, height * 0.35
            x1 = max(0, min(frame_w - 1, int(round(x - pad_x))))
            y1 = max(0, min(frame_h - 1, int(round(y - pad_y))))
            x2 = max(0, min(frame_w, int(round(x + width + pad_x))))
            y2 = max(0, min(frame_h, int(round(y + height + pad_y))))
            if x2 <= x1 or y2 <= y1:
                continue

            frame_key = id(frame)
            frame_tensor = frame_tensors.get(frame_key)
            if frame_tensor is None:
                frame_tensor = torch.from_numpy(np.ascontiguousarray(frame)).to(
                    self.device
                )
                frame_tensor = frame_tensor.permute(2, 0, 1).float()
                frame_tensors[frame_key] = frame_tensor
            crop = frame_tensor[:, y1:y2, x1:x2].unsqueeze(0)
            crop = F.interpolate(
                crop, size=(size, size), mode="bilinear", align_corners=False
            )
            crops.append(crop[0])
            source_indices.append(index)

        embeddings = torch.zeros(
            (len(faces), 128), device=self.device, dtype=torch.float32
        )
        valid = torch.zeros(len(faces), device=self.device, dtype=torch.bool)
        if not crops:
            return embeddings, valid

        crop_batch = torch.stack(crops)
        stream_context = (
            torch.cuda.stream(self._cuda_stream)
            if self._cuda_stream is not None
            else torch.no_grad()
        )
        with stream_context:
            landmarks, landmark_valid = self._yunet_landmarks(crop_batch)
            if landmark_valid.any():
                valid_positions = torch.nonzero(
                    landmark_valid, as_tuple=False
                ).flatten()
                aligned = self._align_faces(
                    crop_batch[valid_positions], landmarks[valid_positions]
                )
                aligned_rgb = aligned[:, [2, 1, 0], :, :].contiguous()
                features = self._run_sface(aligned_rgb)
                features = F.normalize(features.float(), dim=1)
                mapped_indices = torch.as_tensor(source_indices, device=self.device)[
                    valid_positions
                ]
                embeddings[mapped_indices] = features
                valid[mapped_indices] = True
        return embeddings, valid

    def _yunet_landmarks(
        self, crops: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.yunet_session is None:  # pragma: no cover - guarded by configuration
            raise RuntimeError("YuNet session is not initialized")
        batch_size = crops.shape[0]
        outputs = self._run_yunet(crops)
        all_landmarks: list[torch.Tensor] = []
        all_ranks: list[torch.Tensor] = []
        size = self.config.identity_detector_size

        for level, stride in enumerate((8, 16, 32)):
            rows = size // stride
            anchors = rows * rows
            cls = outputs[level].reshape(batch_size, anchors, 1).clamp(0, 1)
            obj = outputs[level + 3].reshape(batch_size, anchors, 1).clamp(0, 1)
            bbox = outputs[level + 6].reshape(batch_size, anchors, 4)
            keypoints = outputs[level + 9].reshape(batch_size, anchors, 5, 2)

            score = torch.sqrt(cls * obj).squeeze(-1)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(rows, device=self.device, dtype=torch.float32),
                torch.arange(rows, device=self.device, dtype=torch.float32),
                indexing="ij",
            )
            grid = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=-1)
            width = torch.exp(bbox[..., 2].clamp(max=10)) * stride
            height = torch.exp(bbox[..., 3].clamp(max=10)) * stride
            rank = width * height * score
            rank = torch.where(
                score >= self.config.identity_detector_confidence,
                rank,
                torch.full_like(rank, -1.0),
            )
            decoded = (keypoints + grid.view(1, anchors, 1, 2)) * stride
            all_landmarks.append(decoded)
            all_ranks.append(rank)

        landmarks = torch.cat(all_landmarks, dim=1)
        ranks = torch.cat(all_ranks, dim=1)
        best_rank, best_index = ranks.max(dim=1)
        batch_indices = torch.arange(batch_size, device=self.device)
        return landmarks[batch_indices, best_index], best_rank >= 0

    def _run_yunet(self, inputs: torch.Tensor) -> list[torch.Tensor]:
        assert self.yunet_session is not None
        batch_size = inputs.shape[0]
        size = self.config.identity_detector_size
        output_shapes: list[tuple[int, ...]] = []
        for channels in (1, 1, 4, 10):
            for stride in (8, 16, 32):
                anchors = (size // stride) ** 2
                output_shapes.append((1, batch_size * anchors, channels))
        return self._run_ort(
            self.yunet_session, inputs.float().contiguous(), output_shapes
        )

    def _run_sface(self, inputs: torch.Tensor) -> torch.Tensor:
        assert self.sface_session is not None
        return self._run_ort(
            self.sface_session,
            inputs.float().contiguous(),
            [(inputs.shape[0], 128)],
        )[0]

    def _run_ort(
        self,
        session: ort.InferenceSession,
        inputs: torch.Tensor,
        output_shapes: list[tuple[int, ...]],
    ) -> list[torch.Tensor]:
        input_name = session.get_inputs()[0].name
        output_names = [output.name for output in session.get_outputs()]
        if session.get_providers()[0] != "CUDAExecutionProvider":
            numpy_outputs = session.run(
                output_names, {input_name: inputs.detach().cpu().numpy()}
            )
            return [
                torch.from_numpy(output).to(self.device) for output in numpy_outputs
            ]

        binding = session.io_binding()
        binding.bind_input(
            input_name,
            device_type="cuda",
            device_id=self._device_index,
            element_type=np.float32,
            shape=tuple(inputs.shape),
            buffer_ptr=inputs.data_ptr(),
        )
        outputs = [
            torch.empty(shape, device=self.device, dtype=torch.float32)
            for shape in output_shapes
        ]
        for name, output in zip(output_names, outputs):
            binding.bind_output(
                name,
                device_type="cuda",
                device_id=self._device_index,
                element_type=np.float32,
                shape=tuple(output.shape),
                buffer_ptr=output.data_ptr(),
            )
        session.run_with_iobinding(binding)
        return outputs

    def _align_faces(
        self, crops: torch.Tensor, landmarks: torch.Tensor
    ) -> torch.Tensor:
        destination = torch.tensor(
            [
                [38.2946, 51.6963],
                [73.5318, 51.5014],
                [56.0252, 71.7366],
                [41.5493, 92.3655],
                [70.7299, 92.2041],
            ],
            device=self.device,
            dtype=torch.float32,
        ).expand(landmarks.shape[0], -1, -1)

        source_mean = landmarks.mean(dim=1, keepdim=True)
        destination_mean = destination.mean(dim=1, keepdim=True)
        source_centered = landmarks - source_mean
        destination_centered = destination - destination_mean
        covariance = destination_centered.transpose(1, 2) @ source_centered / 5.0
        u, singular_values, vh = torch.linalg.svd(covariance)
        signs = torch.ones_like(singular_values)
        signs[:, 1] = torch.where(
            torch.linalg.det(covariance) < 0,
            -torch.ones_like(signs[:, 1]),
            torch.ones_like(signs[:, 1]),
        )
        rotation = u @ torch.diag_embed(signs) @ vh
        variance = source_centered.square().sum(dim=(1, 2)) / 5.0
        scale = (singular_values * signs).sum(dim=1) / variance.clamp_min(1e-8)
        translation = destination_mean[:, 0] - (
            scale[:, None] * (rotation @ source_mean.transpose(1, 2))[:, :, 0]
        )

        transform = torch.zeros((landmarks.shape[0], 3, 3), device=self.device)
        transform[:, :2, :2] = scale[:, None, None] * rotation
        transform[:, :2, 2] = translation
        transform[:, 2, 2] = 1.0
        inverse = torch.linalg.inv(transform)

        output_size = 112
        y, x = torch.meshgrid(
            torch.arange(output_size, device=self.device, dtype=torch.float32),
            torch.arange(output_size, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        homogeneous = torch.stack((x, y, torch.ones_like(x)), dim=-1).reshape(-1, 3)
        source = torch.einsum("bij,pj->bpi", inverse, homogeneous)
        crop_size = crops.shape[-1]
        grid_x = source[..., 0] * (2.0 / (crop_size - 1)) - 1.0
        grid_y = source[..., 1] * (2.0 / (crop_size - 1)) - 1.0
        grid = torch.stack((grid_x, grid_y), dim=-1).reshape(
            -1, output_size, output_size, 2
        )
        return F.grid_sample(
            crops,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    def _blur_frames(
        self,
        frames: list[np.ndarray],
        boxes_per_frame: list[list[np.ndarray]],
    ) -> list[np.ndarray]:
        outputs = list(frames)
        grouped: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, (frame, boxes) in enumerate(zip(frames, boxes_per_frame)):
            if boxes:
                grouped[frame.shape[:2]].append(index)

        for (height, width), indices in grouped.items():
            dtype = torch.float16 if self.half_enabled else torch.float32
            batch = np.stack([frames[index] for index in indices])
            images = torch.from_numpy(np.ascontiguousarray(batch)).to(self.device)
            images = images.permute(0, 3, 1, 2).to(dtype)
            masks = torch.zeros(
                (len(indices), 1, height, width), device=self.device, dtype=dtype
            )
            y = torch.arange(height, device=self.device, dtype=dtype).view(height, 1)
            x = torch.arange(width, device=self.device, dtype=dtype).view(1, width)

            for batch_index, frame_index in enumerate(indices):
                for box in boxes_per_frame[frame_index]:
                    box_x, box_y, box_w, box_h = box[:4].astype(float)
                    small = max(box_w, box_h) < self.config.small_face_threshold
                    expand_w = (
                        self.config.small_box_expand_w
                        if small
                        else self.config.box_expand_w
                    )
                    expand_h = (
                        self.config.small_box_expand_h
                        if small
                        else self.config.box_expand_h
                    )
                    center_x = box_x + box_w * 0.5
                    center_y = box_y + box_h * (0.5 + self.config.center_y_shift)
                    axis_x = max(2.0, box_w * expand_w * 0.5)
                    axis_y = max(2.0, box_h * expand_h * 0.5)
                    ellipse = ((x - center_x) / axis_x).square() + (
                        (y - center_y) / axis_y
                    ).square() <= 1.0
                    masks[batch_index, 0] = torch.maximum(
                        masks[batch_index, 0], ellipse.to(dtype)
                    )

            blurred = self._separable_gaussian(
                images, _make_odd(self.config.blur_kernel)
            )
            feathered = self._separable_gaussian(
                masks,
                _make_odd(self.config.feather * 2 + 1),
            ).clamp(0, 1)
            protected = images * (1.0 - feathered) + blurred * feathered
            protected = protected.clamp(0, 255).round().to(torch.uint8)
            protected_numpy = protected.permute(0, 2, 3, 1).contiguous().cpu().numpy()
            for batch_index, frame_index in enumerate(indices):
                outputs[frame_index] = protected_numpy[batch_index]
        return outputs

    def _separable_gaussian(
        self, tensor: torch.Tensor, kernel_size: int
    ) -> torch.Tensor:
        if kernel_size <= 1:
            return tensor
        sigma = 0.3 * ((kernel_size - 1) * 0.5 - 1) + 0.8
        coordinates = torch.arange(kernel_size, device=self.device, dtype=tensor.dtype)
        coordinates -= (kernel_size - 1) / 2.0
        kernel = torch.exp(-(coordinates.square()) / (2.0 * sigma * sigma))
        kernel /= kernel.sum()
        channels = tensor.shape[1]
        horizontal = kernel.view(1, 1, 1, kernel_size).expand(
            channels, 1, 1, kernel_size
        )
        vertical = kernel.view(1, 1, kernel_size, 1).expand(channels, 1, kernel_size, 1)
        padding = kernel_size // 2
        padding_mode = "reflect" if min(tensor.shape[-2:]) > padding else "replicate"
        tensor = F.pad(tensor, (padding, padding, 0, 0), mode=padding_mode)
        tensor = F.conv2d(tensor, horizontal, groups=channels)
        tensor = F.pad(tensor, (0, 0, padding, padding), mode=padding_mode)
        return F.conv2d(tensor, vertical, groups=channels)

    def stats(self) -> dict[str, Any]:
        with self._stats_lock:
            average_batch = self._frames / self._batches if self._batches else 0.0
            average_latency = (
                self._total_batch_latency_ms / self._batches if self._batches else 0.0
            )
            return {
                "device": str(self.device),
                "cuda": self.cuda_enabled,
                "half": self.half_enabled,
                "onnx_providers": (
                    self.sface_session.get_providers()
                    if self.sface_session is not None
                    else []
                ),
                "queue_depth": self._queue.qsize(),
                "batches": self._batches,
                "frames": self._frames,
                "last_batch_size": self._last_batch_size,
                "average_batch_size": round(average_batch, 2),
                "average_batch_latency_ms": round(average_latency, 2),
            }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(_STOP)
        except Full:
            self._queue.put(_STOP)
        self._worker.join(timeout=10)


_RUNTIMES: dict[PrivacyBlurConfig, SharedGpuRuntime] = {}
_RUNTIMES_LOCK = threading.Lock()


def get_shared_runtime(config: PrivacyBlurConfig) -> SharedGpuRuntime:
    with _RUNTIMES_LOCK:
        runtime = _RUNTIMES.get(config)
        if runtime is None:
            runtime = SharedGpuRuntime(config)
            _RUNTIMES[config] = runtime
        return runtime


def shared_runtime_stats() -> list[dict[str, Any]]:
    with _RUNTIMES_LOCK:
        runtimes = list(_RUNTIMES.values())
    return [runtime.stats() for runtime in runtimes]


def shutdown_shared_runtimes() -> None:
    with _RUNTIMES_LOCK:
        runtimes = list(_RUNTIMES.values())
        _RUNTIMES.clear()
    for runtime in runtimes:
        runtime.close()


atexit.register(shutdown_shared_runtimes)
