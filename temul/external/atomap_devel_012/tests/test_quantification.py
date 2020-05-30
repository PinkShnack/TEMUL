import numpy as np
import math
import atomap.quantification as quant
from atomap.example_data import get_detector_image_signal
from atomap.dummy_data import get_simple_cubic_signal


class TestDetectorNormalisation:

    def test_centered_distance_matrix(self):
        s = quant.centered_distance_matrix((32, 32), np.zeros((64, 64)))
        assert s[32, 32] == 1
        assert s[63, 63] == np.sqrt((63-31)**2 + (63-32)**2)

    def test_detector_threshold(self):
        det_image = get_detector_image_signal()
        threshold_image = quant._detector_threshold(det_image.data)
        assert not (np.sum(threshold_image) == 0)
        assert det_image.data.shape == threshold_image.shape

    def test_radial_profile(self):
        det_image = get_detector_image_signal()
        profile = quant._radial_profile(det_image.data, (256, 256))
        assert len(np.shape(profile)) == 1
        assert np.shape(profile)[0] == math.ceil(math.sqrt(2) * 256)

    def test_detector_normalisation(self):
        det_image = get_detector_image_signal()
        img = get_simple_cubic_signal(image_noise=True)
        img = (img) * 300000 + 4000
        image_normalised = quant.detector_normalisation(img, det_image, 60)
        assert image_normalised.data.max() < 1
        assert image_normalised.data.shape == img.data.shape

    def test_func(self):
        result = quant._func(4, 2, 0.5, 5)
        assert result == 6

    def test_find_flux_limits_running(self):
        flux1 = quant.centered_distance_matrix((63, 63), np.zeros((128, 128)))
        (profiler, flux_profile) = quant.find_flux_limits(100 - flux1, 25)
        assert len(flux_profile) == math.ceil((64**2 + 64**2)**0.5)
