import copy
import math
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from temul.external.atomap_devel_012.atom_finding_refining import get_atom_positions
from temul.external.atomap_devel_012.tools import _get_n_nearest_neighbors, Fingerprinter
from temul.external.atomap_devel_012.atom_finding_refining import _make_circular_mask, do_pca_on_signal
from temul.external.atomap_devel_012.sublattice import Sublattice
from temul.external.atomap_devel_012.atom_lattice import Dumbbell_Lattice
from operator import itemgetter
from matplotlib.colors import LogNorm


def find_dumbbell_vector(s, separation):
    """
    Find the vector between the atoms in structure with dumbbells.
    For example GaAs-(110).

    Parameters
    ----------
    s : HyperSpy 2D signal
    separation : int
        Pixel separation between the atoms in the dumbbells.

    Returns
    -------
    dumbbell_vector : tuple

    Examples
    --------
    >>> import temul.external.atomap_devel_012.api as am
    >>> from temul.external.atomap_devel_012.initial_position_finding import find_dumbbell_vector
    >>> s = am.dummy_data.get_dumbbell_signal()
    >>> dumbbell_vector = find_dumbbell_vector(s, 4)
    """
    position_list = get_atom_positions(s, separation=separation)
    test = _get_n_nearest_neighbors(position_list, 10, leafsize=100)
    fp = Fingerprinter()
    fp.fit(test)
    clusters = fp.cluster_centers_
    clusters_dist = (clusters[:, 0]**2 + clusters[:, 1]**2)**0.5
    sort_zip = zip(list(clusters_dist), clusters[:, 0], clusters[:, 1])
    cluster_dist, cluster_x, cluster_y = zip(
        *sorted(sort_zip, key=itemgetter(0)))
    vec0 = cluster_x[0], cluster_y[0]
    vec1 = cluster_x[1], cluster_y[1]

    if (abs(vec0[0] + vec1[0]) > 0.1) or (abs(vec0[1] + vec1[1]) > 0.1):
        raise ValueError(
            "Dumbbell vectors should be antiparallel, but are %r and %r"
            % (vec0, vec1))
    return(vec0)


def _get_dumbbell_arrays(
        s, position_list, dumbbell_vector, show_progressbar=True):
    """
    Parameters
    ----------
    s : HyperSpy 2D signal
    position_list : list of atomic positions
        In the form [[x0, y0], [x1, y1], [x2, y2], ...]
    dumbbell_vector : tuple
    show_progressbar : bool, default True

    Returns
    -------
    Dumbbell lists : tuple of lists

    Examples
    --------
    >>> import temul.external.atomap_devel_012.api as am
    >>> from temul.external.atomap_devel_012.initial_position_finding import find_dumbbell_vector
    >>> from temul.external.atomap_devel_012.initial_position_finding import _get_dumbbell_arrays
    >>> from temul.external.atomap_devel_012.atom_finding_refining import get_atom_positions
    >>> s = am.dummy_data.get_dumbbell_signal()
    >>> position_list = get_atom_positions(s, separation=16)
    >>> dumbbell_vector = find_dumbbell_vector(s, 4)
    >>> d0, d1 = _get_dumbbell_arrays(s, position_list, dumbbell_vector)
    """
    next_pos_list0 = []
    next_pos_list1 = []
    for x, y in zip(position_list[:, 0], position_list[:, 1]):
        next_pos_list0.append([dumbbell_vector[0] + x, dumbbell_vector[1] + y])
        next_pos_list1.append(
            [-dumbbell_vector[0] + x, -dumbbell_vector[1] + y])
    next_pos_list0 = np.array(next_pos_list0)
    next_pos_list1 = np.array(next_pos_list1)

    mask_radius = 0.5 * (dumbbell_vector[0]**2 + dumbbell_vector[1]**2)**0.5

    iterator = zip(
        position_list[:, 0], position_list[:, 1],
        next_pos_list0, next_pos_list1)
    total_num = len(next_pos_list0)
    dumbbell_list0, dumbbell_list1 = [], []
    for x, y, next_pos0, next_pos1 in tqdm(
            iterator, total=total_num, desc="Finding dumbbells",
            disable=not show_progressbar):
        mask1 = _make_circular_mask(
            next_pos0[1], next_pos0[0],
            s.data.shape[0], s.data.shape[1],
            mask_radius)
        mask2 = _make_circular_mask(
            next_pos1[1], next_pos1[0],
            s.data.shape[0], s.data.shape[1],
            mask_radius)
        pos1_sum = (s.data * mask1).sum()
        pos2_sum = (s.data * mask2).sum()
        if pos1_sum > pos2_sum:
            dumbbell_list0.append([x, y])
            dumbbell_list1.append(next_pos0)
        else:
            dumbbell_list0.append(next_pos1)
            dumbbell_list1.append([x, y])
    dumbbell_list0 = np.array(dumbbell_list0)
    dumbbell_list1 = np.array(dumbbell_list1)
    return(dumbbell_list0, dumbbell_list1)


def make_atom_lattice_dumbbell_structure(
        s, position_list, dumbbell_vector, show_progressbar=True):
    """
    Make Atom_Lattice object from image of dumbbell structure.

    Parameters
    ----------
    s : HyperSpy 2D signal
    position_list : list of atomic positions
        In the form [[x0, y0], [x1, y1], [x2, y2], ...]
    dumbbell_vector : tuple
    show_progressbar : bool, default True

    Returns
    -------
    dumbbell_lattice: Atomap Dumbbell_Lattice object

    Examples
    --------
    >>> import temul.external.atomap_devel_012.api as am
    >>> import temul.external.atomap_devel_012.initial_position_finding as ipf
    >>> from temul.external.atomap_devel_012.atom_finding_refining import get_atom_positions
    >>> s = am.dummy_data.get_dumbbell_signal()
    >>> position_list = get_atom_positions(s, separation=16)
    >>> dumbbell_vector = ipf.find_dumbbell_vector(s, 4)
    >>> dumbbell_lattice = ipf.make_atom_lattice_dumbbell_structure(
    ...     s, position_list, dumbbell_vector)
    """
    dumbbell_list0, dumbbell_list1 = _get_dumbbell_arrays(
        s, position_list, dumbbell_vector,
        show_progressbar=show_progressbar)
    s_modified = do_pca_on_signal(s)
    sublattice0 = Sublattice(
        atom_position_list=dumbbell_list0,
        original_image=s.data,
        image=s_modified.data,
        color='blue')
    sublattice1 = Sublattice(
        atom_position_list=dumbbell_list1,
        original_image=s.data,
        image=s_modified.data,
        color='red')
    sublattice0.find_nearest_neighbors()
    sublattice1.find_nearest_neighbors()
    atom_lattice = Dumbbell_Lattice(
        image=sublattice0.image,
        name="Dumbbell structure",
        sublattice_list=[sublattice0, sublattice1])
    return(atom_lattice)


'''
The log and norm functionality has been added to the atomap 
upstream master
'''


class AtomAdderRemover:

    def __init__(self, image, atom_list=None, distance_threshold=4,
                 norm='linear', vmin=None, vmax=None):
        self.image = image
        self.distance_threshold = distance_threshold
        self.fig, self.ax = plt.subplots()
        if norm == 'linear':
            self.cax = self.ax.imshow(self.image, vmin=vmin, vmax=vmax)
        elif norm == 'log':
            self.cax = self.ax.imshow(self.image,
                                      norm=LogNorm(vmin=np.min(image),
                                                   vmax=np.max(image)),
                                      vmin=vmin, vmax=vmax)
        if atom_list is None:
            self.atom_list = []
        else:
            if hasattr(atom_list, 'tolist'):
                atom_list = atom_list.tolist()
            self.atom_list = copy.deepcopy(atom_list)
        x_pos, y_pos = self.get_xy_pos_lists()
        self.line, = self.ax.plot(x_pos, y_pos, 'o', color='red')
        self.cid = self.fig.canvas.mpl_connect(
            'button_press_event', self.onclick)
        self.fig.tight_layout()

    def onclick(self, event):
        if event.inaxes != self.ax.axes:
            return
        if event.button == 1:  # Left mouse button
            x = np.float(event.xdata)
            y = np.float(event.ydata)
            atom_nearby = self.is_atom_nearby(x, y)
            if atom_nearby is False:
                self.atom_list.append([x, y])
            else:
                self.atom_list.pop(atom_nearby)
            self.replot()

    def is_atom_nearby(self, x_press, y_press):
        dt = self.distance_threshold
        index = False
        closest_dist = 9999999999999999
        x_pos, y_pos = self.get_xy_pos_lists()
        for i, (x, y) in enumerate(zip(x_pos, y_pos)):
            if x - dt < x_press < x + dt:
                if y - dt < y_press < y + dt:
                    dist = math.hypot(x_press - x, y_press - y)
                    if dist < closest_dist:
                        index = i
        return index

    def get_xy_pos_lists(self):
        if self.atom_list:
            x_pos_list = np.array(self.atom_list)[:, 0]
            y_pos_list = np.array(self.atom_list)[:, 1]
        else:
            x_pos_list = []
            y_pos_list = []
        return(x_pos_list, y_pos_list)

    def replot(self):
        x_pos, y_pos = self.get_xy_pos_lists()
        self.line.set_xdata(x_pos)
        self.line.set_ydata(y_pos)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


def add_atoms_with_gui(image, atom_list=None, distance_threshold=4,
                       norm='linear', vmin=None, vmax=None):
    """Add or remove atoms from a list of atom positions.
    Will open a matplotlib figure, where atoms can be added or
    removed by pressing them.
    Parameters
    ----------
    image : array-like
        Signal or NumPy array
    atom_list : list of lists, optional
        In the form [[x0, y0], [x1, y1], ...]
    distance_threshold : int
        Default 4
    Returns
    -------
    atom_positions : list of lists
        In the form [[x0, y0], [x1, y1], ...].
        The list can be updated until the figure is closed.
    Examples
    --------
    >>> import temul.external.atomap_devel_012.api as am
    >>> s = am.dummy_data.get_simple_cubic_signal()
    >>> peaks = am.get_atom_positions(s, separation=9)
    >>> peaks_new = am.add_atoms_with_gui(peaks, s)
    """
    global atom_adder_remover
    atom_adder_remover = AtomAdderRemover(
        image, atom_list, distance_threshold=distance_threshold, norm=norm, vmin=vmin, vmax=vmax)
    new_atom_list = atom_adder_remover.atom_list
    return new_atom_list
