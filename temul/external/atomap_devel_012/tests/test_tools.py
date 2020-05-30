import pytest
from pytest import approx
import numpy as np
import atomap.api as am
import atomap.tools as to
from hyperspy.signals import Signal2D
from atomap.tools import array2signal1d, array2signal2d, Fingerprinter
from atomap.tools import remove_atoms_from_image_using_2d_gaussian
import atomap.dummy_data as dd
from atomap.tools import integrate
import atomap.testing_tools as tt
import hyperspy as hs


class TestArray2Signal:

    def test_array2signal2d(self):
        array = np.arange(100).reshape(10, 10)
        scale = 0.2
        s = array2signal2d(array, scale=scale)
        assert s.axes_manager[0].scale == scale
        assert s.axes_manager[1].scale == scale
        assert s.axes_manager.shape == array.shape

    def test_array2signal1d(self):
        array = np.arange(100)
        scale = 0.2
        s = array2signal1d(array, scale=scale)
        assert s.axes_manager[0].scale == scale
        assert s.axes_manager.shape == array.shape


class TestFingerprinter:

    def test_running_simple(self):
        xN, yN, n = 3, 3, 60
        point_list = tt.make_nn_test_dataset(xN=xN, yN=yN, n=n)
        fp = Fingerprinter()
        fp.fit(point_list)
        clusters = (xN*2+1)*(yN*2+1)-1
        assert fp.cluster_centers_.shape[0] == clusters

    def test_running_advanced(self):
        xN, yN, xS, yS, n = 2, 2, 9, 12, 70
        tolerance = 0.5
        point_list = tt.make_nn_test_dataset(xN=xN, yN=yN, xS=xS, yS=yS, n=n)
        fp = Fingerprinter()
        fp.fit(point_list)
        bool_list = []
        for ix in list(range(-xN, xN + 1)):
            for iy in list(range(-yN, yN + 1)):
                if (ix == 0) and (iy == 0):
                    pass
                else:
                    for point in point_list:
                        x, y = ix * xS, iy * yS
                        xD, yD = abs(x - point[0]), abs(y - point[1])
                        if (xD < tolerance) and (yD < tolerance):
                            bool_list.append(True)
                            break
        clusters = (xN*2+1)*(yN*2+1)-1
        assert len(bool_list) == clusters


class TestRemoveAtomsFromImageUsing2dGaussian:

    def setup_method(self):
        test_data = tt.MakeTestData(520, 520)
        x, y = np.mgrid[10:510:8j, 10:510:8j]
        x, y = x.flatten(), y.flatten()
        test_data.add_atom_list(x, y)
        sublattice = test_data.sublattice
        sublattice.find_nearest_neighbors()
        self.sublattice = sublattice

    def test_running(self):
        sublattice = self.sublattice
        remove_atoms_from_image_using_2d_gaussian(
            sublattice.image,
            sublattice,
            percent_to_nn=0.40)


@pytest.mark.parametrize(
    "x,y,sx,sy,ox,oy", [
        (20, 20, 1, 1, 0, 0), (50, 20, 1, 1, 0, 0),
        (20, 20, 0.5, 2, 0, 0), (20, 20, 0.5, 2, 10, 5),
        (50, 20, 0.5, 4, -10, 20), (15, 50, 0.7, 2.3, -10, -5)])
def test_get_signal_centre(x, y, sx, sy, ox, oy):
    s = Signal2D(np.zeros((y, x)))
    print(s)
    am = s.axes_manager
    am[0].scale, am[0].offset, am[1].scale, am[1].offset = sx, ox, sy, oy
    xC, yC = to._get_signal_centre(s)
    print(xC, yC)
    assert xC == ((0.5 * (x - 1) * sx) + ox)
    assert yC == ((0.5 * (y - 1) * sy) + oy)


class TestRotatePointsAroundSignalCentre:

    def setup_method(self):
        sublattice = dd.get_simple_cubic_sublattice()
        self.x, self.y = sublattice.x_position, sublattice.y_position
        self.s = sublattice.get_atom_list_on_image()

    @pytest.mark.parametrize("rot", [10, 30, 60, 90, 180, 250])
    def test_simple_rotation(self, rot):
        x_rot, y_rot = to.rotate_points_around_signal_centre(
            self.s, self.x, self.y, rot)
        assert len(self.x) == len(x_rot)
        assert len(self.y) == len(y_rot)

    @pytest.mark.parametrize("rot", [0, 360])
    def test_zero_rotation(self, rot):
        x_rot, y_rot = to.rotate_points_around_signal_centre(
            self.s, self.x, self.y, rot)
        np.testing.assert_allclose(self.x, x_rot)
        np.testing.assert_allclose(self.y, y_rot)

    def test_180_double_rotation(self):
        x_rot, y_rot = to.rotate_points_around_signal_centre(
            self.s, self.x, self.y, 180)
        x_rot, y_rot = to.rotate_points_around_signal_centre(
            self.s, x_rot, y_rot, 180)
        np.testing.assert_allclose(self.x, x_rot)
        np.testing.assert_allclose(self.y, y_rot)

    def test_90_double_rotation(self):
        x_rot, y_rot = to.rotate_points_around_signal_centre(
            self.s, self.x, self.y, 90)
        x_rot, y_rot = to.rotate_points_around_signal_centre(
            self.s, x_rot, y_rot, -90)
        np.testing.assert_allclose(self.x, x_rot)
        np.testing.assert_allclose(self.y, y_rot)


class TestRotatePointsAndSignal:

    def setup_method(self):
        sublattice = dd.get_simple_cubic_sublattice()
        self.x, self.y = sublattice.x_position, sublattice.y_position
        self.s = sublattice.get_atom_list_on_image()

    def test_simple_rotation(self):
        s_rot, x_rot, y_rot = to.rotate_points_and_signal(
            self.s, self.x, self.y, 180)
        assert self.s.axes_manager.shape == s_rot.axes_manager.shape
        assert len(self.x) == len(x_rot)
        assert len(self.y) == len(y_rot)

    def test_double_180_rotation(self):
        s_rot, x_rot, y_rot = to.rotate_points_and_signal(
            self.s, self.x, self.y, 180)
        s_rot, x_rot, y_rot = to.rotate_points_and_signal(
            s_rot, x_rot, y_rot, 180)
        np.testing.assert_allclose(self.x, x_rot)
        np.testing.assert_allclose(self.y, y_rot)


class TestFliplrPointsAroundSignalCentre:

    def setup_method(self):
        sublattice = dd.get_simple_cubic_sublattice()
        self.x, self.y = sublattice.x_position, sublattice.y_position
        self.s = sublattice.get_atom_list_on_image()

    def test_simple_flip(self):
        x_flip, y_flip = to.fliplr_points_around_signal_centre(
            self.s, self.x, self.y)
        assert len(self.x) == len(x_flip)
        assert len(self.y) == len(y_flip)

    def test_flip_double(self):
        x_flip, y_flip = to.fliplr_points_around_signal_centre(
            self.s, self.x, self.y)
        x_flip, y_flip = to.fliplr_points_around_signal_centre(
            self.s, x_flip, y_flip)
        np.testing.assert_allclose(self.x, x_flip)
        np.testing.assert_allclose(self.y, y_flip)


class TestFliplrPointsAndSignal:

    def setup_method(self):
        sublattice = dd.get_simple_cubic_sublattice()
        self.x, self.y = sublattice.x_position, sublattice.y_position
        self.s = sublattice.get_atom_list_on_image()

    def test_simple_flip(self):
        s_flip, x_flip, y_flip = to.fliplr_points_and_signal(
            self.s, self.x, self.y)
        s_flip, x_flip, y_flip = to.fliplr_points_and_signal(
            s_flip, x_flip, y_flip)
        np.testing.assert_allclose(self.s.data, s_flip.data)
        np.testing.assert_allclose(self.x, x_flip)
        np.testing.assert_allclose(self.y, y_flip)


class TestIntegrate:

    def test_two_atoms(self):
        test_data = tt.MakeTestData(50, 100)
        x, y, A = [25, 25], [25, 75], [5, 10]
        test_data.add_atom_list(x=x, y=y, amplitude=A)
        s = test_data.signal
        i_points, i_record, p_record = integrate(s, x, y, max_radius=500)

        assert approx(i_points) == A
        assert i_record.axes_manager.signal_shape == (50, 100)
        assert (i_record.isig[:, :51].data == i_points[0]).all()
        assert (i_record.isig[:, 51:].data == i_points[1]).all()
        assert (p_record[:51] == 0).all()
        assert (p_record[51:] == 1).all()

    def test_four_atoms(self):
        test_data = tt.MakeTestData(60, 100)
        x, y, A = [20, 20, 40, 40], [25, 75, 25, 75], [5, 10, 15, 20]
        test_data.add_atom_list(x=x, y=y, amplitude=A)
        s = test_data.signal
        i_points, i_record, p_record = integrate(s, x, y, max_radius=500)

        assert approx(i_points) == A
        assert (i_record.isig[:31, :51].data == i_points[0]).all()
        assert (i_record.isig[:31, 51:].data == i_points[1]).all()
        assert (i_record.isig[31:, :51].data == i_points[2]).all()
        assert (i_record.isig[31:, 51:].data == i_points[3]).all()
        assert (p_record[:51, :31] == 0).all()
        assert (p_record[51:, :31] == 1).all()
        assert (p_record[:51, 31:] == 2).all()
        assert (p_record[51:, 31:] == 3).all()

    def test_max_radius_bad_value(self):
        s = hs.signals.Signal2D(np.zeros((10, 10)))
        with pytest.raises(ValueError):
            integrate(s, [5, ], [5, ], max_radius=-1)

    def test_max_radius_1(self):
        test_data = tt.MakeTestData(60, 100)
        x, y, A = [30, 30], [25, 75], [5, 10]
        test_data.add_atom_list(
                x=x, y=y, amplitude=A, sigma_x=0.1, sigma_y=0.1)
        s = test_data.signal
        i_points, i_record, p_record = integrate(s, x, y, max_radius=1)

        assert (i_points[1] / i_points[0]) == 2.
        assert i_record.data[y[0], x[0]] == i_points[0]
        assert i_record.data[y[1], x[1]] == i_points[1]
        i_record.data[y[0], x[0]] = 0
        i_record.data[y[1], x[1]] = 0
        assert not i_record.data.any()

    def test_too_few_dimensions(self):
        s = hs.signals.Signal1D(np.random.rand(110))
        y, x = np.mgrid[5:96:10, 5:96:10]
        x, y = x.flatten(), y.flatten()
        with pytest.raises(ValueError):
            integrate(s, x, y)

    def test_sum_2d_random_data(self):
        s = hs.signals.Signal2D(np.random.rand(100, 110))
        y, x = np.mgrid[5:96:10, 5:96:10]
        x, y = x.flatten(), y.flatten()
        result = integrate(s, x, y)
        assert approx(np.sum(result[0])) == np.sum(s.data)
        assert result[2].shape == s.data.shape

    def test_3d_data_running(self):
        s = dd.get_eels_spectrum_survey_image()
        s_eels = dd.get_eels_spectrum_map()
        peaks = am.get_atom_positions(s, separation=4)
        i_points, i_record, p_record = integrate(
                s_eels, peaks[:, 0], peaks[:, 1], max_radius=3)
        assert p_record.shape == (100, 100)
        assert s_eels.data.shape == i_record.data.shape

    def test_watershed_method_running(self):
        test_data = tt.MakeTestData(60, 100)
        x, y, A = [20, 20, 40, 40], [25, 75, 25, 75], [5, 10, 15, 20]
        test_data.add_atom_list(x=x, y=y, amplitude=A)
        s = test_data.signal
        i_points, i_record, p_record = integrate(s, x, y, method='Watershed')

    def test_wrong_method(self):
        s = hs.signals.Signal2D(np.zeros((10, 10)))
        with pytest.raises(NotImplementedError):
            integrate(s, [5, ], [5, ], method='bad_method')

    def test_array_input(self):
        sublattice = am.dummy_data.get_simple_cubic_sublattice()
        x, y = sublattice.x_position, sublattice.y_position
        i_points0, i_record0, p_record0 = integrate(sublattice.image, x, y)

        signal = am.dummy_data.get_simple_cubic_signal()
        i_points1, i_record1, p_record1 = integrate(signal, x, y)
        assert (i_points0 == i_points1).all()
        assert (i_record0.data == i_record1.data).all()
        assert (p_record0 == p_record1).all()
