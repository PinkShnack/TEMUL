import numpy as np
import pytest

from temul.dummy_data import get_simple_cubic_signal
import temul.topotem.fft_mapping as fft_mapping


def test_get_masked_ifft_returns_signal_with_same_shape():
    image = get_simple_cubic_signal()
    mask_coords = [[170.2, 170.8], [129.8, 130.0]]

    image_ifft = fft_mapping.get_masked_ifft(image, mask_coords, mask_radius=8)

    assert image_ifft.data.shape == image.data.shape
    assert np.isrealobj(image_ifft.data)
    assert np.isfinite(image_ifft.data).all()


def test_get_masked_ifft_fourier_input_matches_real_path():
    image = get_simple_cubic_signal()
    mask_coords = [[170.2, 170.8], [129.8, 130.0]]

    real_space = fft_mapping.get_masked_ifft(
        image, mask_coords, mask_radius=8, image_space="real")
    fourier_space = fft_mapping.get_masked_ifft(
        image.fft(shift=True), mask_coords, mask_radius=8,
        image_space="fourier")

    assert real_space.data.shape == fourier_space.data.shape
    assert np.allclose(real_space.data, fourier_space.data)


def test_get_masked_ifft_keep_masked_area_changes_output():
    image = get_simple_cubic_signal()
    mask_coords = [[170.2, 170.8], [129.8, 130.0]]

    kept = fft_mapping.get_masked_ifft(
        image, mask_coords, keep_masked_area=True, mask_radius=8)
    removed = fft_mapping.get_masked_ifft(
        image, mask_coords, keep_masked_area=False, mask_radius=8)

    assert kept.data.shape == removed.data.shape
    assert not np.allclose(kept.data, removed.data)


def test_get_masked_ifft_rejects_invalid_image_space():
    image = get_simple_cubic_signal()

    with pytest.raises(ValueError, match="image_space must be either"):
        fft_mapping.get_masked_ifft(
            image, [[170.2, 170.8]], image_space="invalid")


def test_get_masked_ifft_rejects_empty_mask_coords():
    image = get_simple_cubic_signal()

    with pytest.raises(ValueError, match="mask_coords"):
        fft_mapping.get_masked_ifft(image, [])
