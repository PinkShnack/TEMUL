"""This module contains the Atom_Position class.

The Atom_Position is the "base unit" in Atomap, since it contains
the information about the individual atomic columns.
"""
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import math
from temul.external.atomap_devel_012.atom_finding_refining import _make_circular_mask
from temul.external.atomap_devel_012.atom_finding_refining import fit_atom_positions_gaussian
from temul.external.atomap_devel_012.atom_finding_refining import _atom_to_gaussian_component


class Atom_Position:

    def __init__(
            self, x, y, sigma_x=1., sigma_y=1., rotation=0.01,
            amplitude=1.):
        """
        The Atom_Position class contain information about a single atom column.

        Parameters
        ----------
        x : float
        y : float
        sigma_x : float, optional
        sigma_y : float, optional
        rotation : float, optional
            In radians. The rotation of the axes of the 2D-Gaussian relative
            to the image axes. In other words: the rotation of the sigma_x
            relative to the horizontal axis (x-axis). This is different
            from the rotation_ellipticity, which is the rotation of the
            largest sigma in relation to the horizontal axis.
            For the rotation of the ellipticity, see rotation_ellipticity.
        amplitude : float, optional
            Amplitude of Gaussian. Stored as amplitude_gaussian attribute.

        Attributes
        ----------
        ellipticity : float
        rotation : float
            The rotation of sigma_x axis, relative to the x-axis in radians.
            This value will always be between 0 and pi, as the elliptical
            2D-Gaussian used here is always symmetric in the rotation
            direction, and the perpendicular rotation direction.
        rotation_ellipticity : float
            The rotation of the longest sigma, relative to the x-axis in
            radians.
        refine_position : bool
            If True (default), the atom position will be fitted to the image
            data when calling the sublattice.refine_... methods. Note, the
            atom will still be fitted when directly calling the refine_...
            methods in the atom position class itself. Setting it to False
            can be useful when dealing with vacanies, or other features where
            the automatic fitting doesn't work.

        POSSIBLE ADDITIONS:
        structure_element : str
            The element of the atom in each atom position, given by atomic 
            symbol e.g., S for sulphur, Mo for Molybdenum. Possible to give
            it all attributes from periodic table? or just link the two 
            Default is None
            Multiple elements available to single atom position.
        structure_count_of_element : int
            the count of each element in attribute "element".
            Default = 0
        measured_element : str
            The element of the atom in each atom position, given by atomic 
            symbol e.g., S for sulphur, Mo for Molybdenum. Possible to give
            it all attributes from periodic table? or just link the two 
            Default is None
            Multiple elements available to single atom position.
        measured_count_of_element : int
            the count of each element in attribute "element".
            Default = 0    
        energy : float
            the energy above zero for the atom position. Calculated by having
            some prerequisite knowledge of the structure.
            Example: If sublattice.atom_list[0] should be a single Mo atom
            and it is not (vacancy), take the minimum DFT calculated energy
            (to be tabulated) required to create an Mo vacancy and assign it 
            to "energy". Units of eV.


        Examples
        --------
        >>> from temul.external.atomap_devel_012.atom_position import Atom_Position
        >>> atom_position = Atom_Position(10, 5)

        POSSIBLE:
        >>> sublattice.atom_list[2].structure_element = S
        >>> sublattice.atom_list[2].structure_count_of_element = 2
        >>> sublattice.atom_list[2].measured_element = S
        >>> sublattice.atom_list[2].measured_count_of_element = 1
        >>> sublattice.atom_list[2].energy = 5.90 eV

        More parameters

        >>> atom_pos = Atom_Position(10, 5, sigma_x=2, sigma_y=4, rotation=2)

        """
        self.pixel_x, self.pixel_y = x, y
        self.sigma_x, self.sigma_y = sigma_x, sigma_y
        self.rotation = rotation
        self.nearest_neighbor_list = None
        self.in_atomic_plane = []
        self._start_atom = []
        self._end_atom = []
        self.atom_planes = []
        self._tag = ''
        self.old_pixel_x_list = []
        self.old_pixel_y_list = []
        self.amplitude_gaussian = amplitude
        self._gaussian_fitted = False
        self.amplitude_max_intensity = 1.0
        self.intensity_mask = 0.
        self.refine_position = True

    def __repr__(self):
        return '<%s, %s (x:%s,y:%s,sx:%s,sy:%s,r:%s,e:%s)>' % (
            self.__class__.__name__,
            self._tag,
            round(self.pixel_x, 1), round(self.pixel_y, 1),
            round(self.sigma_x, 1), round(self.sigma_y, 1),
            round(self.rotation, 1), round(self.ellipticity, 1),
        )

    @property
    def sigma_x(self):
        return(self.__sigma_x)

    @sigma_x.setter
    def sigma_x(self, new_sigma_x):
        self.__sigma_x = abs(new_sigma_x)

    @property
    def sigma_y(self):
        return(self.__sigma_y)

    @sigma_y.setter
    def sigma_y(self, new_sigma_y):
        self.__sigma_y = abs(new_sigma_y)

    @property
    def sigma_average(self):
        sigma = (abs(self.sigma_x) + abs(self.sigma_y)) * 0.5
        return(sigma)

    @property
    def rotation(self):
        """The rotation of the atom relative to the horizontal axis.

        Given in radians.
        For the rotation of the ellipticity, see rotation_ellipticity.
        """
        return(self.__rotation)

    @rotation.setter
    def rotation(self, new_rotation):
        self.__rotation = new_rotation % math.pi

    @property
    def rotation_ellipticity(self):
        """Rotation between the "x-axis" and the major axis of the ellipse.

        Rotation between the horizontal axis, and the longest part of the
        atom position, given by the longest sigma.
        Basically giving the direction of the ellipticity.
        """
        if self.sigma_x > self.sigma_y:
            temp_rotation = self.__rotation % math.pi
        else:
            temp_rotation = (self.__rotation + (math.pi / 2)) % math.pi
        return(temp_rotation)

    @property
    def ellipticity(self):
        """Largest sigma divided by the shortest"""
        if self.sigma_x > self.sigma_y:
            return(self.sigma_x / self.sigma_y)
        else:
            return(self.sigma_y / self.sigma_x)

    def as_gaussian(self):
        g = _atom_to_gaussian_component(self)
        g.A.value = self.amplitude_gaussian
        return(g)

    def get_pixel_position(self):
        return((self.pixel_x, self.pixel_y))

    def get_pixel_difference(self, atom):
        """Vector between self and given atom"""
        x_distance = self.pixel_x - atom.pixel_x
        y_distance = self.pixel_y - atom.pixel_y
        return((x_distance, y_distance))

    def get_angle_between_atoms(self, atom0, atom1=None):
        """
        Return the angle between atoms in radians.

        Can either find the angle between self and two other atoms,
        or between another atom and the horizontal axis.

        Parameters
        ----------
        atom0 : Atom Position object
            The first atom.
        atom1 : Atom Position object, optional
            If atom1 is not specified, the angle between
            itself, atom0 and the horizontal axis will be
            returned.

        Returns
        -------
        Angle : float
            Angle in radians

        Examples
        --------
        >>> from temul.external.atomap_devel_012.atom_position import Atom_Position
        >>> atom0 = Atom_Position(0, 0)
        >>> atom1 = Atom_Position(1, 1)
        >>> atom2 = Atom_Position(-1, 1)
        >>> angle0 = atom0.get_angle_between_atoms(atom1, atom2)
        >>> angle1 = atom0.get_angle_between_atoms(atom1)

        """
        vector0 = np.array([
            atom0.pixel_x - self.pixel_x,
            atom0.pixel_y - self.pixel_y])
        if atom1 is None:
            vector1 = np.array([
                self.pixel_x + 1000,
                0])
        else:
            vector1 = np.array([
                atom1.pixel_x - self.pixel_x,
                atom1.pixel_y - self.pixel_y])
        cosang = np.dot(vector0, vector1)
        sinang = np.linalg.norm(np.cross(vector0, vector1))
        return(np.arctan2(sinang, cosang))

    def get_angle_between_zone_vectors(
            self,
            zone_vector0,
            zone_vector1):
        """
        Return the angle between itself and the next atoms in
        the atom planes belonging to zone_vector0 and zone_vector1
        """
        atom0 = self.get_next_atom_in_zone_vector(zone_vector0)
        atom1 = self.get_next_atom_in_zone_vector(zone_vector1)
        if atom0 is False:
            return(False)
        if atom1 is False:
            return(False)
        angle = self.get_angle_between_atoms(atom0, atom1)
        return(angle)

    def _get_image_slice_around_atom(
            self,
            image_data,
            slice_size):
        """
        Return a square slice of the image data.

        The atom is in the center of this slice.

        Parameters
        ----------
        image_data : Numpy 2D array
        slice_size : int
            Width and height of the square slice

        Returns
        -------
        2D numpy array

        """
        x0 = self.pixel_x - slice_size / 2
        x1 = self.pixel_x + slice_size / 2
        y0 = self.pixel_y - slice_size / 2
        y1 = self.pixel_y + slice_size / 2

        if x0 < 0.0:
            x0 = 0
        if y0 < 0.0:
            y0 = 0
        if x1 > image_data.shape[1]:
            x1 = image_data.shape[1]
        if y1 > image_data.shape[0]:
            y1 = image_data.shape[0]
        x0, x1, y0, y1 = int(x0), int(x1), int(y0), int(y1)
        data_slice = copy.deepcopy(image_data[y0:y1, x0:x1])
        return data_slice, x0, y0

    def _plot_gaussian2d_debug(
            self,
            slice_radius,
            gaussian,
            data_slice):

        X, Y = np.meshgrid(
            np.arange(-slice_radius, slice_radius, 1),
            np.arange(-slice_radius, slice_radius, 1))
        s_m = gaussian.function(X, Y)

        fig, axarr = plt.subplots(2, 2)
        ax0 = axarr[0][0]
        ax1 = axarr[0][1]
        ax2 = axarr[1][0]
        ax3 = axarr[1][1]

        ax0.imshow(data_slice, interpolation="nearest")
        ax1.imshow(s_m, interpolation="nearest")
        ax2.plot(data_slice.sum(0))
        ax2.plot(s_m.sum(0))
        ax3.plot(data_slice.sum(1))
        ax3.plot(s_m.sum(1))

        fig.tight_layout()
        fig.savefig(
            "debug_plot_2d_gaussian_" +
            str(np.random.randint(1000, 10000)) + ".jpg", dpi=400)
        plt.close('all')

    def get_closest_neighbor(self):
        """
        Find the closest neighbor to an atom in the same sub lattice.

        Returns
        -------
        Atomap atom_position object

        """
        closest_neighbor = 100000000000000000
        for neighbor_atom in self.nearest_neighbor_list:
            distance = self.get_pixel_distance_from_another_atom(
                neighbor_atom)
            if distance < closest_neighbor:
                closest_neighbor = distance
        return(closest_neighbor)

    def calculate_max_intensity(
            self,
            image_data,
            percent_to_nn=0.40):
        """Find the maximum intensity of the atom.

        The maximum is found within the the distance to the nearest
        neighbor times percent_to_nn.
        """
        closest_neighbor = self.get_closest_neighbor()

        slice_size = closest_neighbor * percent_to_nn * 2
        data_slice, x0, y0 = self._get_image_slice_around_atom(
            image_data, slice_size)

        data_slice_max = data_slice.max()
        self.amplitude_max_intensity = data_slice_max

        return(data_slice_max)

    def refine_position_using_2d_gaussian(
            self,
            image_data,
            rotation_enabled=True,
            percent_to_nn=0.40,
            mask_radius=None,
            centre_free=True):
        """
        Use 2D Gaussian to refine the parameters of the atom position.

        Parameters
        ----------
        image_data : Numpy 2D array
        rotation_enabled : bool, optional
            If True, the Gaussian will be able to rotate.
            Note, this can increase the chance of fitting failure.
            Default True.
        percent_to_nn : float, optional
            The percent of the distance to the nearest neighbor atom
            in the same sub lattice. The distance times this percentage
            defines the mask around the atom where the Gaussian will be
            fitted. A smaller value can reduce the effect from
            neighboring atoms, but might also decrease the accuracy of
            the fitting due to less data to fit to.
            Default 0.4 (40%).
        mask_radius : float, optional
            Radius of the mask around each atom. If this is not set,
            the radius will be the distance to the nearest atom in the
            same sublattice times the `percent_to_nn` value.
            Note: if `mask_radius` is not specified, the Atom_Position objects
            must have a populated nearest_neighbor_list.
        centre_free : bool, default True
            If True, the centre parameter will be free, meaning that
            the Gaussian can move.

        """
        fit_atom_positions_gaussian(
            [self],
            image_data=image_data,
            rotation_enabled=rotation_enabled,
            percent_to_nn=percent_to_nn,
            mask_radius=mask_radius,
            centre_free=centre_free)

    def get_center_position_com(
            self,
            image_data,
            percent_to_nn=0.40,
            mask_radius=None):
        if mask_radius is None:
            closest_neighbor = 100000000000000000
            for neighbor_atom in self.nearest_neighbor_list:
                distance = self.get_pixel_distance_from_another_atom(
                    neighbor_atom)
                if distance < closest_neighbor:
                    closest_neighbor = distance
            mask_radius = closest_neighbor * percent_to_nn
        mask = _make_circular_mask(
            self.pixel_y,
            self.pixel_x,
            image_data.shape[0],
            image_data.shape[1],
            mask_radius)
        data = copy.deepcopy(image_data)
        mask = np.invert(mask)
        data[mask] = 0

        center_of_mass = self._calculate_center_of_mass(data)

        new_x, new_y = center_of_mass[1], center_of_mass[0]
        return(new_x, new_y)

    def refine_position_using_center_of_mass(
            self,
            image_data,
            percent_to_nn=0.40,
            mask_radius=None):
        new_x, new_y = self.get_center_position_com(
            image_data,
            percent_to_nn=percent_to_nn,
            mask_radius=mask_radius)
        self.old_pixel_x_list.append(self.pixel_x)
        self.old_pixel_y_list.append(self.pixel_y)
        self.pixel_x = new_x
        self.pixel_y = new_y

    def _calculate_center_of_mass(self, data):
        center_of_mass = ndimage.measurements.center_of_mass(
            data.astype('float32'))
        return(center_of_mass)

    def get_atomic_plane_from_zone_vector(self, zone_vector):
        for atomic_plane in self.in_atomic_plane:
            if atomic_plane.zone_vector[0] == zone_vector[0]:
                if atomic_plane.zone_vector[1] == zone_vector[1]:
                    return(atomic_plane)
        return(False)

    def get_neighbor_atoms_in_atomic_plane_from_zone_vector(
            self, zone_vector):
        atom_plane = self.get_atomic_plane_from_zone_vector(zone_vector)
        atom_plane_atom_neighbor_list = []
        for atom in self.nearest_neighbor_list:
            if atom in atom_plane.atom_list:
                atom_plane_atom_neighbor_list.append(atom)
        return(atom_plane_atom_neighbor_list)

    def is_in_atomic_plane(self, zone_direction):
        for atomic_plane in self.in_atomic_plane:
            if atomic_plane.zone_vector[0] == zone_direction[0]:
                if atomic_plane.zone_vector[1] == zone_direction[1]:
                    return(True)
        return(False)

    def get_ellipticity_vector(self):
        elli = self.ellipticity - 1
        rot = self.get_ellipticity_rotation_vector()
        vector = (elli * rot[0], elli * rot[1])
        return(vector)

    def get_rotation_vector(self):
        rot = self.rotation
        vector = (
            math.cos(rot),
            math.sin(rot))
        return(vector)

    def get_ellipticity_rotation_vector(self):
        rot = self.rotation_ellipticity
        vector = (math.cos(rot), math.sin(rot))
        return(vector)

    def get_pixel_distance_from_another_atom(self, atom):
        x_distance = self.pixel_x - atom.pixel_x
        y_distance = self.pixel_y - atom.pixel_y
        total_distance = math.hypot(x_distance, y_distance)
        return(total_distance)

    def pixel_distance_from_point(self, point=(0, 0)):
        dist = math.hypot(
            self.pixel_x - point[0], self.pixel_y - point[1])
        return(dist)

    def get_index_in_atom_plane(self, atom_plane):
        for atom_index, atom in enumerate(atom_plane.atom_list):
            if atom == self:
                return(atom_index)

    def get_next_atom_in_atom_plane(self, atom_plane):
        current_index = self.get_index_in_atom_plane(atom_plane)
        if self == atom_plane.end_atom:
            return(False)
        else:
            next_atom = atom_plane.atom_list[current_index + 1]
            return(next_atom)

    def get_previous_atom_in_atom_plane(self, atom_plane):
        current_index = self.get_index_in_atom_plane(atom_plane)
        if self == atom_plane.start_atom:
            return(False)
        else:
            previous_atom = atom_plane.atom_list[current_index - 1]
            return(previous_atom)

    def get_next_atom_in_zone_vector(self, zone_vector):
        """Get the next atom in the atom plane belonging to zone vector."""
        atom_plane = self.get_atomic_plane_from_zone_vector(zone_vector)
        if atom_plane is False:
            return(False)
        next_atom = self.get_next_atom_in_atom_plane(atom_plane)
        return(next_atom)

    def get_previous_atom_in_zone_vector(self, zone_vector):
        atom_plane = self.get_atomic_plane_from_zone_vector(zone_vector)
        if atom_plane is False:
            return(False)
        previous_atom = self.get_previous_atom_in_atom_plane(atom_plane)
        return(previous_atom)

    def can_atom_plane_be_reached_through_zone_vector(
            self, atom_plane, zone_vector):
        for test_atom_plane in self.atom_planes:
            if test_atom_plane.zone_vector == zone_vector:
                for temp_atom in test_atom_plane.atom_list:
                    for temp_atom_plane in temp_atom.atom_planes:
                        if temp_atom_plane == atom_plane:
                            return(test_atom_plane)
        return(False)

    def get_position_convergence(
            self, distance_to_first_position=False):
        x_list = self.old_pixel_x_list
        y_list = self.old_pixel_y_list
        distance_list = []
        for index, (x, y) in enumerate(zip(x_list[1:], y_list[1:])):
            if distance_to_first_position:
                previous_x = x_list[0]
                previous_y = y_list[0]
            else:
                previous_x = x_list[index]
                previous_y = y_list[index]
            dist = math.hypot(x - previous_x, y - previous_y)
            distance_list.append(dist)
        return(distance_list)

    def find_atom_intensity_inside_mask(self, image_data, radius):
        """Find the average intensity inside a circle.

        The circle is defined by the atom position, and the given
        radius (in pixels).
        The outside this area is covered by a mask. The average
        intensity is saved to self.intensity_mask.
        """
        if radius is None:
            radius = 1
        centerX, centerY = self.pixel_x, self.pixel_y
        mask = _make_circular_mask(
            centerY, centerX,
            image_data.shape[0], image_data.shape[1], radius)
        data_mask = image_data * mask
        self.intensity_mask = np.mean(data_mask[np.nonzero(mask)])
