from pytest import approx
from numpy import pi
import math
from atomap.atom_position import Atom_Position


class TestCreateAtomPositionObject:

    def test_position(self):
        atom_x, atom_y = 10, 20
        atom_position = Atom_Position(atom_x, atom_y)
        assert atom_position.pixel_x == 10
        assert atom_position.pixel_y == 20

    def test_sigma(self):
        sigma_x, sigma_y = 2, 3
        atom_position = Atom_Position(1, 2, sigma_x=sigma_x, sigma_y=sigma_y)
        assert atom_position.sigma_x == sigma_x
        assert atom_position.sigma_y == sigma_y

        sigma_x, sigma_y = -5, -6
        atom_position = Atom_Position(1, 2, sigma_x=sigma_x, sigma_y=sigma_y)
        assert atom_position.sigma_x == abs(sigma_x)
        assert atom_position.sigma_y == abs(sigma_y)

    def test_amplitude(self):
        amplitude = 30
        atom_position = Atom_Position(1, 2, amplitude=amplitude)
        assert atom_position.amplitude_gaussian == amplitude

    def test_rotation(self):
        rotation0 = 0.0
        atom_position = Atom_Position(1, 2, rotation=rotation0)
        assert atom_position.rotation == rotation0

        rotation1 = math.pi/2
        atom_position = Atom_Position(1, 2, rotation=rotation1)
        assert atom_position.rotation == rotation1

        rotation2 = math.pi
        atom_position = Atom_Position(1, 2, rotation=rotation2)
        assert atom_position.rotation == 0

        rotation3 = math.pi*3/2
        atom_position = Atom_Position(1, 2, rotation=rotation3)
        assert atom_position.rotation == math.pi/2

        rotation4 = math.pi*2
        atom_position = Atom_Position(1, 2, rotation=rotation4)
        assert atom_position.rotation == 0


class TestAtomPositionObjectTools:

    def setup_method(self):
        self.atom_position = Atom_Position(1, 1)

    def test_get_atom_angle(self):
        atom_position0 = Atom_Position(1, 2)
        atom_position1 = Atom_Position(3, 1)
        atom_position2 = Atom_Position(1, 0)
        atom_position3 = Atom_Position(5, 1)
        atom_position4 = Atom_Position(2, 2)

        angle90 = self.atom_position.get_angle_between_atoms(
                atom_position0, atom_position1)
        angle180 = self.atom_position.get_angle_between_atoms(
                atom_position0, atom_position2)
        angle0 = self.atom_position.get_angle_between_atoms(
                atom_position1, atom_position3)
        angle45 = self.atom_position.get_angle_between_atoms(
                atom_position1, atom_position4)

        assert approx(angle90) == pi/2
        assert approx(angle180) == pi
        assert approx(angle0) == 0
        assert approx(angle45) == pi/4

    def test_as_gaussian(self):
        x, y, sx, sy, A, r = 10., 5., 2., 3.5, 9.9, 1.5
        atom_position = Atom_Position(
                x=x, y=y, sigma_x=sx, sigma_y=sy, amplitude=A, rotation=r)
        g = atom_position.as_gaussian()
        assert g.centre_x.value == x
        assert g.centre_y.value == y
        assert g.sigma_x.value == sx
        assert g.sigma_y.value == sy
        assert g.A.value == A
        assert g.rotation.value == r
