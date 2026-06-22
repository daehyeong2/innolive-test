from __future__ import annotations

import asyncio
from pathlib import Path
import unittest

import cv2
import numpy as np

from privacy_blur import PrivacyBlurConfig, new_filter, shutdown_runtimes
from privacy_blur.errors import PrivacyBlurNotReadyError
from privacy_blur.runtime import _resolve_torch_device


ROOT = Path(__file__).resolve().parent.parent


def test_config(**overrides) -> PrivacyBlurConfig:
    defaults = {
        "face_model_path": ROOT / "models/yolov8n-face.pt",
        "me_image_path": ROOT / "me.jpeg",
        "face_detection_model_path": ROOT / "models/face_detection_yunet_2023mar.onnx",
        "face_recognition_model_path": ROOT
        / "models/face_recognition_sface_2021dec.onnx",
        "device": "cpu",
        "require_gpu": False,
        "face_imgsz": 320,
        "identity_detector_size": 320,
        "max_batch_size": 4,
        "batch_wait_ms": 10,
        "blur_kernel": 21,
        "feather": 5,
    }
    defaults.update(overrides)
    return PrivacyBlurConfig(**defaults)


class ConfigTests(unittest.TestCase):
    def test_production_mode_rejects_cpu(self) -> None:
        with self.assertRaises(PrivacyBlurNotReadyError):
            _resolve_torch_device(test_config(require_gpu=True))

    def test_batch_limits_are_validated(self) -> None:
        with self.assertRaises(ValueError):
            test_config(max_batch_size=8, max_pending_frames=4)


class RuntimeTests(unittest.IsolatedAsyncioTestCase):
    def tearDown(self) -> None:
        shutdown_runtimes()

    async def test_filters_share_models_and_frames_are_dynamically_batched(
        self,
    ) -> None:
        config = test_config()
        filters = [new_filter(config) for _ in range(4)]
        self.assertTrue(
            all(item.engine.runtime is filters[0].engine.runtime for item in filters)
        )

        image = cv2.imread(str(ROOT / "me.jpeg"))
        self.assertIsNotNone(image)
        frame = cv2.resize(image, (320, 240))
        outputs = await asyncio.gather(
            *(item.apply_async(frame.copy()) for item in filters)
        )

        self.assertEqual([output.shape for output in outputs], [(240, 320, 3)] * 4)
        stats = filters[0].engine.runtime.stats()
        self.assertEqual(stats["frames"], 4)
        self.assertEqual(stats["batches"], 1)
        self.assertEqual(stats["last_batch_size"], 4)
        self.assertTrue(all(item.stats()["tracks"][0]["is_me"] for item in filters))

    async def test_non_reference_faces_receive_a_masked_gpu_pipeline_result(
        self,
    ) -> None:
        config = test_config(enable_identity_exclusion=False)
        privacy_filter = new_filter(config)
        image = cv2.imread(str(ROOT / "me.jpeg"))
        frame = cv2.resize(image, (320, 240))

        output = await privacy_filter.apply_async(frame)

        self.assertEqual(output.shape, frame.shape)
        self.assertEqual(output.dtype, np.uint8)
        self.assertFalse(np.array_equal(output, frame))
        self.assertGreater(privacy_filter.stats()["last_detection_count"], 0)


if __name__ == "__main__":
    unittest.main()
