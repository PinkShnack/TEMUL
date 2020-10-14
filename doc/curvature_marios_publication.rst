.. _curvature_marios_publication:

***********************************
Calculation of Atom Plane Curvature
***********************************

The :python:`calculate_atom_plane_curvature` function in the
`temul.lattice_structure_tools` module can be used to find the curvature of the
displacement of atoms along an atom plane in a sublattice. Using the default
parameter :python:`func="strain_grad"`, the function will approximate the
curvature as the strain gradient, as in cases where the first derivative is
negligible. See "Landau and Lifshitz, Theory of Elasticity, Vol 7, pp 47-49, 1981"
for more details. One can use any :python:`func` that can be used by
`scipy.optimize.curve_fit`.

This function has been adapted from the MATLAB script written by Dr. Marios
Hadjimichael for the publication "M. Hadjimichael *et al*, Metal-ferroelectric
supercrystals with periodically curved metallic layers, Nature Materials 2020".
This MATLAB script can be found along with this and other publication examples
in the TEMUL repository in the folder "TEMUL/publication_examples/PTO_marios_hadj".


This tutorial follows the python scripts and jupyter notebooks found in the
"TEMUL/publication_examples/PTO_marios_hadj" folder in the
`TEMUL repository <https://github.com/PinkShnack/TEMUL>`_

