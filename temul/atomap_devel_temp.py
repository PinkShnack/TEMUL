'''
Should make a PR on atomap for these changes.
They are in EOC's personal atomap version.

from matplotlib.colors import LogNorm
import numpy as np
import copy
import matplotlib.pyplot as plt
import math


####
# added to line 179 in atomap.initial_position_finding in 0.1.2:
# allows imshow here to have lognorm
####
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
    import atomap.api as am
    s = am.dummy_data.get_simple_cubic_signal()
    peaks = am.get_atom_positions(s, separation=9)
    peaks_new = am.add_atoms_with_gui(peaks, s)
    """
    global atom_adder_remover
    atom_adder_remover = AtomAdderRemover(
        image, atom_list, distance_threshold=distance_threshold, norm=norm,
        vmin=vmin, vmax=vmax)
    new_atom_list = atom_adder_remover.atom_list
    return new_atom_list
'''
