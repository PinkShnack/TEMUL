import pytest
import numpy as np
from numpy.testing import assert_allclose
from atomap.atom_position import Atom_Position
from atomap.sublattice import Sublattice
from atomap.testing_tools import MakeTestData
import atomap.dummy_data as dd
import atomap.atom_finding_refining as afr


class TestMakeMaskFromPositions:

    def test_radius_1(self):
        x, y, r = 10, 20, 1
        pos = [[x, y]]
        rad = [r]
        mask = afr._make_mask_from_positions(pos, rad, (40, 40))
        assert mask.sum() == 5.
        mask[x, y] = False
        mask[x+r, y] = False
        mask[x-r, y] = False
        mask[x, y+1] = False
        mask[x, y-1] = False
        assert not mask.any()

    def test_2_positions_radius_1(self):
        x0, y0, x1, y1, r = 10, 20, 20, 30, 1
        pos = [[x0, y0], [x1, y1]]
        rad = [r, r]
        mask = afr._make_mask_from_positions(pos, rad, (40, 40))
        assert mask.sum() == 10.
        mask[x0, y0] = False
        mask[x0+r, y0] = False
        mask[x0-r, y0] = False
        mask[x0, y0+1] = False
        mask[x0, y0-1] = False
        mask[x1, y1] = False
        mask[x1+r, y1] = False
        mask[x1-r, y1] = False
        mask[x1, y1+1] = False
        mask[x1, y1-1] = False
        assert not mask.any()

    def test_radius_2(self):
        x, y, r = 10, 5, 2
        pos = [[x, y]]
        rad = [r]
        mask = afr._make_mask_from_positions(pos, rad, (40, 40))
        assert mask.sum() == 13.

    def test_2_positions_radius_2(self):
        x0, y0, x1, y1, r = 5, 7, 17, 25, 2
        pos = [[x0, y0], [x1, y1]]
        rad = [r, r]
        mask = afr._make_mask_from_positions(pos, rad, (40, 40))
        assert mask.sum() == 26.

    def test_wrong_input(self):
        x, y, r = 10, 5, 2
        pos = [[x, y]]
        rad = [r, r]
        with pytest.raises(ValueError):
            afr._make_mask_from_positions(
                    position_list=pos, radius_list=rad, data_shape=(40, 40))


class TestCropMask:

    def test_radius_1(self):
        x, y, r = 10, 20, 1
        pos = [[x, y]]
        rad = [r]
        mask = afr._make_mask_from_positions(pos, rad, (40, 40))
        x0, x1, y0, y1 = afr._crop_mask_slice_indices(mask)
        assert x0 == x-r
        assert x1 == x+r+1
        assert y0 == y-r
        assert y1 == y+r+1
        mask_crop = mask[x0:x1, y0:y1]
        assert mask_crop.shape == (2*r+1, 2*r+1)

    def test_radius_2(self):
        x, y, r = 15, 10, 2
        pos = [[x, y]]
        rad = [r]
        mask = afr._make_mask_from_positions(pos, rad, (40, 40))
        x0, x1, y0, y1 = afr._crop_mask_slice_indices(mask)
        mask_crop = mask[x0:x1, y0:y1]
        assert mask_crop.shape == (2*r+1, 2*r+1)

    def test_radius_5(self):
        x, y, r = 15, 10, 5
        pos = [[x, y]]
        rad = [r]
        mask = afr._make_mask_from_positions(pos, rad, (40, 40))
        x0, x1, y0, y1 = afr._crop_mask_slice_indices(mask)
        mask_crop = mask[x0:x1, y0:y1]
        assert mask_crop.shape == (2*r+1, 2*r+1)


class TestFindBackgroundValue:

    def test_percentile(self):
        data = np.arange(100)
        value = afr._find_background_value(data, lowest_percentile=0.01)
        assert value == 0.
        value = afr._find_background_value(data, lowest_percentile=0.1)
        assert value == 4.5
        value = afr._find_background_value(data, lowest_percentile=0.5)
        assert value == 24.5


class TestFindMedianUpperPercentile:

    def test_percentile(self):
        data = np.arange(100)
        value = afr._find_median_upper_percentile(data, upper_percentile=0.01)
        assert value == 99.
        value = afr._find_median_upper_percentile(data, upper_percentile=0.1)
        assert value == 94.5
        value = afr._find_median_upper_percentile(data, upper_percentile=0.5)
        assert value == 74.5


class TestMakeModelFromAtomList:

    def setup_method(self):
        image_data = np.random.random(size=(100, 100))
        position_list = []
        for x in range(10, 100, 5):
            for y in range(10, 100, 5):
                position_list.append([x, y])
        sublattice = Sublattice(np.array(position_list), image_data)
        sublattice.find_nearest_neighbors()
        self.sublattice = sublattice

    def test_1_atom(self):
        sublattice = self.sublattice
        model, mask = afr._make_model_from_atom_list(
                [sublattice.atom_list[10]],
                sublattice.image)
        assert len(model) == 1

    def test_2_atom(self):
        sublattice = self.sublattice
        model, mask = afr._make_model_from_atom_list(
                sublattice.atom_list[10:12],
                sublattice.image)
        assert len(model) == 2

    def test_5_atom(self):
        sublattice = self.sublattice
        model, mask = afr._make_model_from_atom_list(
                sublattice.atom_list[10:15],
                sublattice.image)
        assert len(model) == 5

    def test_set_mask_radius_atom(self):
        atom_list = [Atom_Position(2, 2), Atom_Position(4, 4)]
        image = np.random.random((20, 20))
        model, mask = afr._make_model_from_atom_list(
                atom_list=atom_list,
                image_data=image,
                mask_radius=3)
        assert len(model) == 2


class TestFitAtomPositionsWithGaussianModel:

    def setup_method(self):
        test_data = MakeTestData(100, 100)
        x, y = np.mgrid[10:90:10j, 10:90:10j]
        x, y = x.flatten(), y.flatten()
        sigma, A = 1, 50
        test_data.add_atom_list(
                x, y, sigma_x=sigma, sigma_y=sigma, amplitude=A)
        self.sublattice = test_data.sublattice
        self.sublattice.find_nearest_neighbors()

    def test_1_atom(self):
        sublattice = self.sublattice
        g_list = afr._fit_atom_positions_with_gaussian_model(
                [sublattice.atom_list[5]],
                sublattice.image)
        assert len(g_list) == 1

    def test_2_atom(self):
        sublattice = self.sublattice
        g_list = afr._fit_atom_positions_with_gaussian_model(
                sublattice.atom_list[5:7],
                sublattice.image)
        assert len(g_list) == 2

    def test_5_atom(self):
        sublattice = self.sublattice
        g_list = afr._fit_atom_positions_with_gaussian_model(
                sublattice.atom_list[5:10],
                sublattice.image)
        assert len(g_list) == 5

    def test_wrong_input_0(self):
        sublattice = self.sublattice
        with pytest.raises(TypeError):
            afr._fit_atom_positions_with_gaussian_model(
                    sublattice.atom_list[5],
                    sublattice.image)

    def test_wrong_input_1(self):
        sublattice = self.sublattice
        with pytest.raises(TypeError):
            afr._fit_atom_positions_with_gaussian_model(
                    [sublattice.atom_list[5:7]],
                    sublattice.image)


class TestAtomToGaussianComponent:

    def test_simple(self):
        x, y, sX, sY, r = 7.1, 2.8, 2.1, 3.3, 1.9
        atom_position = Atom_Position(
                x=x, y=y,
                sigma_x=sX, sigma_y=sY,
                rotation=r)
        gaussian = afr._atom_to_gaussian_component(atom_position)
        assert x == gaussian.centre_x.value
        assert y == gaussian.centre_y.value
        assert sX == gaussian.sigma_x.value
        assert sY == gaussian.sigma_y.value
        assert r == gaussian.rotation.value


class TestMakeCircularMask:

    def test_small_radius_1(self):
        imX, imY = 3, 3
        mask = afr._make_circular_mask(1, 1, imX, imY, 1)
        assert mask.size == imX*imY
        assert mask.sum() == 5
        true_index = [[1, 0], [0, 1], [1, 1],  [2, 1], [1, 2]]
        false_index = [[0, 0], [0, 2], [2, 0],  [2, 2]]
        for index in true_index:
            assert mask[index[0], index[1]]
        for index in false_index:
            assert not mask[index[0], index[1]]

    def test_all_true_mask(self):
        imX, imY = 5, 5
        mask = afr._make_circular_mask(1, 1, imX, imY, 5)
        assert mask.all()
        assert mask.size == imX*imY
        assert mask.sum() == imX*imY

    def test_all_false_mask(self):
        mask = afr._make_circular_mask(10, 10, 5, 5, 3)
        assert not mask.any()


class TestFitAtomPositionsGaussian:

    def setup_method(self):
        test_data = MakeTestData(100, 100)
        x, y = np.mgrid[5:95:10j, 5:95:10j]
        x, y = x.flatten(), y.flatten()
        test_data.add_atom_list(x, y)
        sublattice = test_data.sublattice
        sublattice.construct_zone_axes()
        self.sublattice = sublattice
        self.x, self.y = x, y

    def test_one_atoms(self):
        sublattice = self.sublattice
        atom_index = 55
        atom_list = [sublattice.atom_list[atom_index]]
        image_data = sublattice.image
        afr.fit_atom_positions_gaussian(atom_list, image_data)
        assert_allclose(
                sublattice.atom_list[atom_index].pixel_x,
                self.x[atom_index], rtol=1e-7)
        assert_allclose(
                sublattice.atom_list[atom_index].pixel_y,
                self.y[atom_index], rtol=1e-7)

    def test_two_atoms(self):
        sublattice = self.sublattice
        atom_indices = [44, 45]
        atom_list = []
        for index in atom_indices:
            atom_list.append(sublattice.atom_list[index])
        image_data = sublattice.image
        afr.fit_atom_positions_gaussian(atom_list, image_data)
        for atom_index in atom_indices:
            assert_allclose(
                    sublattice.atom_list[atom_index].pixel_x,
                    self.x[atom_index], rtol=1e-7)
            assert_allclose(
                    sublattice.atom_list[atom_index].pixel_y,
                    self.y[atom_index], rtol=1e-7)

    def test_four_atoms(self):
        sublattice = self.sublattice
        atom_indices = [35, 36, 45, 46]
        atom_list = []
        for index in atom_indices:
            atom_list.append(sublattice.atom_list[index])
        image_data = sublattice.image
        afr.fit_atom_positions_gaussian(atom_list, image_data)
        for atom_index in atom_indices:
            assert_allclose(
                    sublattice.atom_list[atom_index].pixel_x,
                    self.x[atom_index], rtol=1e-7)
            assert_allclose(
                    sublattice.atom_list[atom_index].pixel_y,
                    self.y[atom_index], rtol=1e-7)

    def test_nine_atoms(self):
        sublattice = self.sublattice
        atom_indices = [34, 35, 36, 44, 45, 46, 54, 55, 56]
        atom_list = []
        for index in atom_indices:
            atom_list.append(sublattice.atom_list[index])
        image_data = sublattice.image
        afr.fit_atom_positions_gaussian(atom_list, image_data)
        for atom_index in atom_indices:
            assert_allclose(
                    sublattice.atom_list[atom_index].pixel_x,
                    self.x[atom_index], rtol=1e-7)
            assert_allclose(
                    sublattice.atom_list[atom_index].pixel_y,
                    self.y[atom_index], rtol=1e-7)


class TestGetAtomPositions:

    def test_find_number_of_columns(self):
        test_data = MakeTestData(50, 50)
        x, y = np.mgrid[5:48:5, 5:48:5]
        x, y = x.flatten(), y.flatten()
        test_data.add_atom_list(x, y)
        peaks = afr.get_atom_positions(test_data.signal, separation=3)
        assert len(peaks) == len(x)

    @pytest.mark.parametrize("separation", [-1000, -1, 0, 0.0, 0.2, 0.9999])
    def test_too_low_separation(self, separation):
        s = dd.get_simple_cubic_signal()
        with pytest.raises(ValueError):
            afr.get_atom_positions(s, separation)


class TestBadFitCondition:

    def setup_method(self):
        t = MakeTestData(40, 40)
        x, y = np.mgrid[5:40:10, 5:40:10]
        x, y = x.flatten(), y.flatten()
        t.add_atom_list(x, y)
        self.sublattice = t.sublattice
        self.sublattice.find_nearest_neighbors()

    def test_initial_position_inside_mask_x(self):
        sublattice = self.sublattice
        atom = [sublattice.atom_list[6]]
        x0 = atom[0].pixel_x
        atom[0].pixel_x += 2
        g = afr._fit_atom_positions_with_gaussian_model(
                atom, sublattice.image, mask_radius=4)
        assert_allclose(g[0].centre_x.value, x0, rtol=1e-2)

    def test_initial_position_outside_mask_x(self):
        sublattice = self.sublattice
        atom = [sublattice.atom_list[6]]
        atom[0].pixel_x += 3
        g = afr._fit_atom_positions_with_gaussian_model(
                atom, sublattice.image, mask_radius=2)
        assert not g

    def test_initial_position_outside_mask_y(self):
        sublattice = self.sublattice
        atom = [sublattice.atom_list[6]]
        atom[0].pixel_y -= 4
        g = afr._fit_atom_positions_with_gaussian_model(
                atom, sublattice.image, mask_radius=2)
        assert not g

    def test_initial_position_outside_mask_xy(self):
        sublattice = self.sublattice
        atom = [sublattice.atom_list[6]]
        atom[0].pixel_y += 3
        atom[0].pixel_x += 3
        g = afr._fit_atom_positions_with_gaussian_model(
                atom, sublattice.image, mask_radius=2)
        assert not g


class TestGetFeatureSeparation:

    def test_simple(self):
        s = dd.get_simple_cubic_signal()
        s_fs = afr.get_feature_separation(s)
        s_fs.plot()

    def test_separation_range(self):
        sr0, sr1 = 10, 15
        s = dd.get_simple_cubic_signal()
        s_fs = afr.get_feature_separation(s, separation_range=(sr0, sr1))
        assert s_fs.axes_manager.navigation_size == (sr1 - sr0)
        assert s_fs.axes_manager.navigation_extent == (sr0, sr1 - 1)

    def test_separation_step(self):
        sr0, sr1, s_step = 10, 16, 2
        s = dd.get_simple_cubic_signal()
        s_fs = afr.get_feature_separation(
                s, separation_range=(sr0, sr1), separation_step=s_step)
        assert s_fs.axes_manager.navigation_size == ((sr1 - sr0) / s_step)

    def test_pca_subtract_background_normalize_intensity(self):
        s = dd.get_simple_cubic_signal()
        s_fs = afr.get_feature_separation(
                s, pca=True, subtract_background=True,
                normalize_intensity=True)
        assert hasattr(s_fs, 'data')

    def test_dtypes(self):
        dtype_list = [
                'float64', 'float32', 'uint64', 'uint32', 'uint16', 'uint8',
                'int64', 'int32', 'int16', 'int8']
        s = dd.get_simple_cubic_signal()
        s *= 10**9
        for dtype in dtype_list:
            print(dtype)
            s.change_dtype(dtype)
            afr.get_feature_separation(s, separation_range=(10, 15))
        s.change_dtype('float16')
        with pytest.raises(ValueError):
            afr.get_feature_separation(s, separation_range=(10, 15))

    @pytest.mark.parametrize("separation_low", [-1000, -1, 0, 0.0, 0.2, 0.999])
    def test_too_low_separation_low(self, separation_low):
        separation_range = (separation_low, 3)
        s = dd.get_simple_cubic_signal()
        with pytest.raises(ValueError):
            afr.get_feature_separation(s, separation_range)

    @pytest.mark.parametrize("separation_range", [(10, 2), (1000, 2), (2, 1)])
    def test_separation_range_bad(self, separation_range):
        s = dd.get_simple_cubic_signal()
        with pytest.raises(ValueError):
            afr.get_feature_separation(s, separation_range)
