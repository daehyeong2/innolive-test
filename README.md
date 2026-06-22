# innolive privacy blur

GPU-batched WebRTC face privacy filtering for multi-user services. A process-wide runtime shares
one YOLO, YuNet, and SFace model set across every stream; only tracking state is kept per stream.

## NVIDIA server setup

The production server requires a CUDA-enabled PyTorch build and ONNX Runtime CUDA Execution
Provider. The following example targets a Linux host with an RTX 3090 and a compatible NVIDIA
driver. Select the PyTorch CUDA wheel index that matches the deployed driver.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/python -m pip install -r requirements.txt
```

Verify both GPU runtimes before serving traffic:

```bash
.venv/bin/python - <<'PY'
import onnxruntime as ort
import torch

print("torch CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0))
print("ONNX providers:", ort.get_available_providers())
PY
```

`torch CUDA` must be `True` and `ONNX providers` must contain `CUDAExecutionProvider`.

## Run the WebRTC server

```bash
.venv/bin/python examples/webrtc/server.py \
  --host 0.0.0.0 \
  --port 8080 \
  --device cuda:0 \
  --batch-size 8 \
  --batch-wait-ms 4
```

The production command fails during startup if CUDA or ONNX Runtime CUDA is unavailable. For a
local CPU smoke test only, pass `--device cpu --allow-cpu --no-half`.

The `/health` endpoint reports peer count, device/provider selection, queue depth, processed
batches, average batch size, and average batch latency.

Cloud browsers require HTTPS for camera access. Configure TLS at the reverse proxy and provide
STUN/TURN servers when clients and the server are separated by NAT or restrictive firewalls.

## Processing architecture

1. `aiortc` receives one frame from each active WebRTC video track.
2. A shared worker gathers up to `max_batch_size` frames for `batch_wait_ms`.
3. YOLOv8n-Face runs one FP16 CUDA batch for all gathered streams.
4. Per-stream tracking selects new or periodic identity checks.
5. YuNet performs batched CUDA landmark detection on the selected face crops.
6. PyTorch CUDA aligns the faces; SFace performs batched CUDA embedding inference.
7. PyTorch CUDA creates expanded ellipse masks and applies separable Gaussian blur.
8. Processed frames are returned to their original WebRTC tracks.

Backpressure is bounded by `max_pending_frames`; model objects and GPU memory are not duplicated
per connection.

Run one application process per GPU. Starting multiple web workers on the same RTX 3090 creates
one model set per process and defeats process-wide batching; scale to another process only when it
is assigned a separate GPU.

## Python API

Warm the shared runtime before accepting traffic, then create any number of independent filters:

```python
from privacy_blur import configure, initialize_runtime, new_filter, register_reference_face

register_reference_face("me.jpeg")
config = configure(
    device="cuda:0",
    require_gpu=True,
    max_batch_size=8,
    batch_wait_ms=4,
)
initialize_runtime(config)

first_stream_filter = new_filter()
second_stream_filter = new_filter()
```

Wrap an incoming WebRTC video track with the same shared runtime:

```python
from privacy_blur import protect_video_track

protected_track = protect_video_track(source_track)
```

Call `shutdown_runtimes()` during application shutdown. The example server already does this.

## Models

- `models/yolov8n-face.pt`: primary face detector, PyTorch CUDA FP16
- `models/face_detection_yunet_2023mar.onnx`: facial landmarks, ONNX Runtime CUDA
- `models/face_recognition_sface_2021dec.onnx`: 128-dimensional embeddings, ONNX Runtime CUDA

YuNet and SFace are patched to dynamic batch shapes in memory at startup; model files are not
modified on disk.
