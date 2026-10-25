import unittest

import numpy as np

from csdemo_mapextractor.cs2render import bc6h


def _rgba(rgb: np.ndarray) -> np.ndarray:
    return np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,), np.float32)], axis=-1).astype(np.float32)


def _modes(data: bytes) -> set[int]:
    return set((np.frombuffer(data, "<u8")[::2] & 31).tolist())


class Bc6hEncoderTests(unittest.TestCase):
    def test_smooth_hdr_falloff_matches_texconv_quality(self):
        y, x = np.mgrid[0:64, 0:64].astype(np.float32)
        intensity = 0.05 + 8.0 * np.exp(-((x - 20) ** 2 + (y - 40) ** 2) / 400.0)
        rgb = intensity[..., None] * np.array([1.0, 0.8, 0.55], np.float32)
        encoded = bc6h.encode_layer(_rgba(rgb))
        decoded = bc6h.decode_layer(encoded, 64, 64)
        self.assertEqual(len(encoded), 16 * 16 * 16)
        # texconv reaches max 5.9% and mean 0.93% on this image
        relative = np.abs(decoded - rgb) / rgb
        self.assertLess(float(relative.max()), 0.06)
        self.assertLess(float(relative.mean()), 0.01)

    def test_chromatic_gradient_stays_within_single_region_limits(self):
        # red and green change independently, which a single-region block can only approximate
        y, x = np.mgrid[0:64, 0:64].astype(np.float32)
        rgb = np.stack([0.5 + x * 0.2, 0.3 + y * 0.1, np.full_like(x, 2.0)], axis=-1)
        decoded = bc6h.decode_layer(bc6h.encode_layer(_rgba(rgb)), 64, 64)
        relative = np.abs(decoded - rgb) / rgb
        self.assertLess(float(relative.mean()), 0.02)
        self.assertLess(float(np.percentile(relative, 99)), 0.1)

    def test_uses_every_one_region_mode_and_decodes_them(self):
        rng = np.random.default_rng(1)
        smooth = np.full((32, 32, 3), 1.5, np.float32) + rng.normal(0, 0.002, (32, 32, 3)).astype(np.float32)
        edges = np.exp2(rng.uniform(-8, 8, (32, 32, 3))).astype(np.float32)
        ramps = np.repeat(np.linspace(0.01, 40, 32, dtype=np.float32)[None, :, None], 32, axis=0).repeat(3, axis=2)
        rgb = np.concatenate([smooth, edges, ramps * np.array([1.0, 0.6, 0.3], np.float32)], axis=1)
        encoded = bc6h.encode_layer(_rgba(rgb))
        self.assertTrue({0x03, 0x07, 0x0F} <= _modes(encoded))
        decoded = bc6h.decode_layer(encoded, rgb.shape[1], rgb.shape[0])
        log_error = np.abs(np.log2(decoded[:, :32] + 1e-3) - np.log2(smooth + 1e-3))
        self.assertLess(float(log_error.max()), 0.01)

    def test_odd_sizes_pad_to_whole_blocks(self):
        rgb = np.full((5, 6, 3), 0.25, np.float32)
        encoded = bc6h.encode_layer(_rgba(rgb))
        self.assertEqual(len(encoded), 2 * 2 * 16)
        decoded = bc6h.decode_layer(encoded, 6, 5)
        self.assertEqual(decoded.shape, (5, 6, 3))
        np.testing.assert_allclose(decoded, rgb, rtol=2e-3)

    def test_negative_and_non_finite_values_are_clamped(self):
        rgb = np.array([[[-1.0, np.nan, np.inf]] * 4] * 4, np.float32)
        decoded = bc6h.decode_layer(bc6h.encode_layer(_rgba(rgb)), 4, 4)
        self.assertTrue(np.all(np.isfinite(decoded)))
        np.testing.assert_allclose(decoded[..., :2], 0.0, atol=1e-6)
        self.assertGreater(float(decoded[0, 0, 2]), 60000.0)


if __name__ == "__main__":
    unittest.main()
