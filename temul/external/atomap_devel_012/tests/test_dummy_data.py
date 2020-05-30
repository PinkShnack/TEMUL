import atomap.dummy_data as dd


class TestDummyData:

    def test_make_simple_cubic_testdata(self):
        dd._make_simple_cubic_testdata()
        dd._make_simple_cubic_testdata(image_noise=False)
        dd._make_simple_cubic_testdata(image_noise=True)

    def test_get_simple_cubic_signal(self):
        s0 = dd.get_simple_cubic_signal()
        s0.plot()
        s1 = dd.get_simple_cubic_signal(image_noise=False)
        s1.plot()
        s2 = dd.get_simple_cubic_signal(image_noise=True)
        s2.plot()

    def test_get_simple_cubic_sublattice(self):
        s0 = dd.get_simple_cubic_sublattice()
        s0.plot()
        s1 = dd.get_simple_cubic_sublattice(image_noise=False)
        s1.plot()
        s2 = dd.get_simple_cubic_sublattice(image_noise=True)
        s2.plot()

    def test_get_two_sublattice_signal(self):
        s = dd.get_two_sublattice_signal()
        s.plot()

    def test_get_simple_heterostructure_signal(self):
        s0 = dd.get_simple_heterostructure_signal()
        s0.plot()
        s1 = dd.get_simple_heterostructure_signal(image_noise=False)
        s1.plot()
        s2 = dd.get_simple_heterostructure_signal(image_noise=True)
        s2.plot()

    def test_get_dumbbell_signal(self):
        s = dd.get_dumbbell_signal()
        s.plot()

    def test_get_perovskite110_ABF_signal(self):
        s0 = dd.get_perovskite110_ABF_signal()
        s0.plot()
        s1 = dd.get_perovskite110_ABF_signal(image_noise=False)
        s1.plot()
        s2 = dd.get_perovskite110_ABF_signal(image_noise=True)
        s2.plot()

    def test_get_simple_atom_lattice_two_sublattices(self):
        s0 = dd.get_simple_atom_lattice_two_sublattices()
        s0.plot()
        s1 = dd.get_simple_atom_lattice_two_sublattices(image_noise=True)
        s1.plot()
        s2 = dd.get_simple_atom_lattice_two_sublattices(image_noise=False)
        s2.plot()


class TestDummyDataFantasite:

    def test_signal(self):
        s = dd.get_fantasite()
        s.plot()
        s1 = dd.get_fantasite()
        assert (s.data == s1.data).all()

    def test_sublattice(self):
        sublattice = dd.get_fantasite_sublattice()
        assert len(sublattice.x_position) == len(sublattice.y_position)

    def test_atom_lattice(self):
        atom_lattice = dd.get_fantasite_atom_lattice()
        assert len(atom_lattice.sublattice_list) == 2


class TestDummyDataDistortedCubic:

    def test_signal(self):
        s0 = dd.get_distorted_cubic_signal(image_noise=False)
        s1 = dd.get_distorted_cubic_signal(image_noise=True)
        s0.plot()
        s1.plot()
        assert not (s0.data == s1.data).all()

    def test_sublattice(self):
        sublattice = dd.get_distorted_cubic_sublattice()
        assert len(sublattice.x_position) == len(sublattice.y_position)


class TestDummyDataEELSMap:

    def test_make_eels_map_spatial_image_la(self):
        x, y = 100, 100
        s = dd._make_eels_map_spatial_image_la(x_size=x, y_size=y)
        assert s.axes_manager.signal_shape == (100, 100)
        assert hasattr(s, 'plot')

    def test_make_eels_map_spatial_image_mn(self):
        x, y = 100, 100
        s = dd._make_eels_map_spatial_image_mn(x_size=x, y_size=y)
        assert s.axes_manager.signal_shape == (100, 100)
        assert hasattr(s, 'plot')

    def test_make_mn_eels_spectrum(self):
        data = dd._make_mn_eels_spectrum(energy_range=(400, 700))
        assert len(data) == 300

    def test_make_la_eels_spectrum(self):
        data = dd._make_la_eels_spectrum(energy_range=(400, 700))
        assert len(data) == 300

    def test_get_eels_spectrum_survey_image(self):
        s = dd.get_eels_spectrum_survey_image()
        assert hasattr(s, 'plot')

    def test_get_eels_spectrum_map(self):
        s = dd.get_eels_spectrum_map()
        assert len(s.axes_manager.shape) == 3
        assert hasattr(s, 'plot')

        s_no_noise = dd.get_eels_spectrum_map(add_noise=False)
        assert len(s.axes_manager.shape) == 3
        assert hasattr(s, 'plot')
        assert not (s_no_noise.data == s.data).all()
