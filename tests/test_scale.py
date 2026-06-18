"""Unit tests for pendant_swap.scale - pure arithmetic, no images needed."""

import pytest
from pendant_swap.scale import pixels_per_mm, scale_for_image, target_pixels


# --- pixels_per_mm -----------------------------------------------------------

class TestPixelsPerMm:
    def test_reference_calibration(self):
        """130 px / 13 mm → exactly 10.0 px/mm (canonical reference values)."""
        assert pixels_per_mm(130, 13) == pytest.approx(10.0)

    def test_fractional_result(self):
        """100 px / 15 mm → ~6.667 px/mm."""
        assert pixels_per_mm(100, 15) == pytest.approx(100 / 15)

    def test_one_to_one(self):
        assert pixels_per_mm(1, 1) == pytest.approx(1.0)

    def test_float_inputs(self):
        assert pixels_per_mm(130.0, 13.0) == pytest.approx(10.0)

    def test_zero_height_raises(self):
        with pytest.raises(ValueError, match="ref_pixel_height"):
            pixels_per_mm(0, 13)

    def test_negative_height_raises(self):
        with pytest.raises(ValueError, match="ref_pixel_height"):
            pixels_per_mm(-10, 13)

    def test_zero_mm_raises(self):
        with pytest.raises(ValueError, match="ref_mm"):
            pixels_per_mm(130, 0)

    def test_negative_mm_raises(self):
        with pytest.raises(ValueError, match="ref_mm"):
            pixels_per_mm(130, -5)


# --- target_pixels -----------------------------------------------------------

class TestTargetPixels:
    def test_reference_target(self):
        """21 mm × 10 px/mm → exactly 210 px (canonical reference values)."""
        ppm = pixels_per_mm(130, 13)
        assert target_pixels(21, ppm) == pytest.approx(210.0)

    def test_direct_multiplication(self):
        assert target_pixels(15, 8.0) == pytest.approx(120.0)

    def test_float_precision(self):
        """Result should be a float, not truncated."""
        result = target_pixels(7, 3)
        assert result == pytest.approx(21.0)
        assert isinstance(result, float)

    def test_zero_target_mm_raises(self):
        with pytest.raises(ValueError, match="target_mm"):
            target_pixels(0, 10)

    def test_negative_target_mm_raises(self):
        with pytest.raises(ValueError, match="target_mm"):
            target_pixels(-1, 10)

    def test_zero_ppm_raises(self):
        with pytest.raises(ValueError, match="ppm"):
            target_pixels(21, 0)

    def test_negative_ppm_raises(self):
        with pytest.raises(ValueError, match="ppm"):
            target_pixels(21, -2)


# --- scale_for_image ---------------------------------------------------------

class TestScaleForImage:
    def test_canonical_values(self):
        """Convenience wrapper returns the same result as calling each function."""
        ppm, target_px = scale_for_image(130, 13, 21)
        assert ppm == pytest.approx(10.0)
        assert target_px == pytest.approx(210.0)

    def test_matches_individual_calls(self):
        ppm_direct = pixels_per_mm(200, 20)
        target_direct = target_pixels(30, ppm_direct)
        ppm_wrap, target_wrap = scale_for_image(200, 20, 30)
        assert ppm_wrap == pytest.approx(ppm_direct)
        assert target_wrap == pytest.approx(target_direct)

    def test_propagates_value_error(self):
        with pytest.raises(ValueError):
            scale_for_image(0, 13, 21)

    def test_small_pendant(self):
        """A tiny 5 mm pendant at 10 px/mm → 50 px."""
        _, target_px = scale_for_image(130, 13, 5)
        assert target_px == pytest.approx(50.0)

    def test_large_pendant(self):
        """A large 50 mm pendant at 10 px/mm → 500 px."""
        _, target_px = scale_for_image(130, 13, 50)
        assert target_px == pytest.approx(500.0)
