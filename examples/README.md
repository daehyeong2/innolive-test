# Examples

These examples use only the public API.

## ndarray Image

```bash
python examples/ndarray_image.py \
  --input input.jpg \
  --output protected.jpg \
  --profile privacy
```

## av.VideoFrame

```bash
python examples/video_frame.py
```

## Async Frame Processing

```bash
python examples/async_frame.py
```

## Django aiortc

```python
from aiortc import RTCPeerConnection
from privacy_blur import protect_video_track

pc = RTCPeerConnection()

@pc.on("track")
def on_track(track):
    if track.kind == "video":
        pc.addTrack(protect_video_track(track))
```

## Local WebRTC Camera Test

```bash
python examples/webrtc/server.py \
  --profile fast \
  --asset models/yolov8n-face.pt
```

Then open:

```text
http://127.0.0.1:8080
```
