# innolive privacy blur

Install and activate the local environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Wrap an incoming WebRTC video track:

```python
from privacy_blur import protect_video_track, register_reference_face

register_reference_face("me.jpeg")
protected_track = protect_video_track(source_track)
```

`protected_track` is an `aiortc.VideoStreamTrack`. Internally it keeps the original script's
YOLO face detection, YuNet/SFace reference-face exclusion, track hold, expanded ellipse mask,
and Gaussian blur flow.

For direct frame processing:

```python
from privacy_blur import new_filter, register_reference_face

register_reference_face("me.jpeg")
privacy_filter = new_filter()
processed_frame = privacy_filter.apply(frame)
```

Run the local WebRTC test site:

```bash
.venv/bin/python examples/webrtc/server.py
```

Then open `http://127.0.0.1:8080` and press `Start Camera`.
You can also choose a local video file in the page and press `Start File`; the browser streams
that file through the same WebRTC processing path.
