# privacy_blur

This package accepts a video frame and returns a privacy-protected frame.
The caller does not need to know how the protection is performed internally.

이 패키지는 영상 프레임을 받아 개인정보 보호 처리가 된 프레임을 반환합니다.
호출하는 쪽에서는 내부 처리 방식을 알 필요가 없습니다.

## Django/WebRTC Usage

```python
from privacy_blur import protect_video_track

protected_track = protect_video_track(source_track)
```

For direct frame processing:

```python
from privacy_blur import get_privacy_filter

privacy_filter = get_privacy_filter()

processed = privacy_filter.apply(frame)
```

### Django aiortc Example

```python
from aiortc import RTCPeerConnection
from privacy_blur import protect_video_track

pc = RTCPeerConnection()

@pc.on("track")
def on_track(track):
    if track.kind == "video":
        protected_track = protect_video_track(track)
        pc.addTrack(protected_track)
```

Or with an explicitly shared filter:

```python
from privacy_blur import get_privacy_filter, protect_video_track

privacy_filter = get_privacy_filter()

@pc.on("track")
def on_track(track):
    if track.kind == "video":
        pc.addTrack(protect_video_track(track, privacy_filter))
```

## Frame Usage

### ndarray

```python
from privacy_blur import get_privacy_filter

privacy_filter = get_privacy_filter()

protected_image = privacy_filter.apply(image)
```

### av.VideoFrame

```python
from privacy_blur import get_privacy_filter

privacy_filter = get_privacy_filter()

protected_frame = privacy_filter.apply(video_frame)
```

## Performance

Most callers only need to choose a profile.

```python
from privacy_blur import PrivacyFilterSettings, get_privacy_filter

privacy_filter = get_privacy_filter(
    PrivacyFilterSettings(profile="balanced")
)
```

Profiles:

```text
privacy  : small-face protection first
balanced : default real-time usage
fast     : speed first
```

## Deployment Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Install the private asset used by the filter:

```bash
mkdir -p models
wget -O models/yolov8n-face.pt <다운로드 URL>
```

You can also provide the asset path through Django settings:

```python
PRIVACY_FILTER = {
    "ASSET_PATH": "/secure/path/privacy-filter-asset.pt",
}
```

Or through an environment variable:

```bash
export PRIVACY_FILTER_ASSET_PATH=/secure/path/privacy-filter-asset.pt
```

## Offline Scripts

Process a video:

```bash
python scripts/process_video.py \
  --input input.mp4 \
  --output protected.mp4 \
  --profile balanced \
  --asset models/yolov8n-face.pt \
  --accelerator auto
```

Benchmark a video:

```bash
python scripts/benchmark_video.py \
  --input input.mp4 \
  --profile fast \
  --asset models/yolov8n-face.pt
```
