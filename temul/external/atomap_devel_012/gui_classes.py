import math
import numpy as np
import matplotlib.pyplot as plt


class AtomToggleRefine:

    def __init__(self, image, sublattice, distance_threshold=4):
        self.image = image
        self.distance_threshold = distance_threshold
        self.sublattice = sublattice
        self.fig, self.ax = plt.subplots()
        self.cax = self.ax.imshow(self.image)
        x_pos = self.sublattice.x_position
        y_pos = self.sublattice.y_position
        refine_list = self._get_refine_position_list(
                self.sublattice.atom_list)
        color_list = self._refine_position_list_to_color_list(
                refine_list)
        self.path = self.ax.scatter(x_pos, y_pos, c=color_list)
        self.cid = self.fig.canvas.mpl_connect(
                'button_press_event', self.onclick)
        self.fig.tight_layout()

    def _get_refine_position_list(self, atom_list):
        refine_position_list = []
        for atom in atom_list:
            refine_position_list.append(atom.refine_position)
        return refine_position_list

    def _refine_position_list_to_color_list(
            self, refine_position_list,
            color_true='green', color_false='red'):
        color_list = []
        for refine_position in refine_position_list:
            if refine_position:
                color_list.append(color_true)
            else:
                color_list.append(color_false)
        return color_list

    def onclick(self, event):
        if event.inaxes != self.ax.axes:
            return
        if event.button == 1:  # Left mouse button
            x = np.float(event.xdata)
            y = np.float(event.ydata)
            atom_nearby = self.is_atom_nearby(x, y)
            if atom_nearby is not None:
                ref_pos_current = self.sublattice.atom_list[
                        atom_nearby].refine_position
                self.sublattice.atom_list[
                        atom_nearby].refine_position = not ref_pos_current
                self.replot()

    def is_atom_nearby(self, x_press, y_press):
        dt = self.distance_threshold
        index = None
        closest_dist = 9999999999999999
        x_pos = self.sublattice.x_position
        y_pos = self.sublattice.y_position
        for i, (x, y) in enumerate(zip(x_pos, y_pos)):
            if x - dt < x_press < x + dt:
                if y - dt < y_press < y + dt:
                    dist = math.hypot(x_press - x, y_press - y)
                    if dist < closest_dist:
                        index = i
        return index

    def replot(self):
        refine_list = self._get_refine_position_list(
                self.sublattice.atom_list)
        color_list = self._refine_position_list_to_color_list(
                refine_list)
        self.path.set_color(color_list)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
