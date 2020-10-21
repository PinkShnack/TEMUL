    >>> import temul.polarisation as tmlp
    >>> from temul.dummy_data import get_polarisation_dummy_dataset
    >>> atom_lattice = get_polarisation_dummy_dataset(image_noise=True)
    >>> sublatticeA = atom_lattice.sublattice_list[0]
    >>> sublatticeB = atom_lattice.sublattice_list[1]
    >>> sublatticeA.construct_zone_axes()
    >>> sublatticeB.construct_zone_axes()
    >>> sampling = 0.1  # example of 0.1 nm/pix
    >>> units = 'nm'
    >>> sublatticeB.plot()