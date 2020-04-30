import os
from hyperspy.io import load

my_path = os.path.dirname(__file__)

example_MoS2_vesta_xyz = None


def path_to_example_data_MoS2_vesta_xyz():
    """
    Get the path of the vesta xyz file for bilayer MoS2

    Example
    -------
    >>> import temul.example_data as example_data
    >>> path_vesta_file = example_data.path_to_example_data_MoS2_vesta_xyz()
    """
    if example_MoS2_vesta_xyz is None:
        path = os.path.join(
            my_path, 'example_data', 'prismatic', 'example_MoS2_vesta_xyz.xyz')
    return path


example_Au_nanoparticle = None


def load_example_Au_nanoparticle():
    """
    Get the emd STEM image of an example Au nanoparticle

    Example
    -------
    >>> import temul.example_data as example_data
    >>> s = example_data.load_example_Au_nanoparticle()

    """
    global example_Au_nanoparticle
    if example_Au_nanoparticle is None:
        path = os.path.join(
            my_path, 'example_data',
            'experimental', 'example_Au_nanoparticle.emd')
        example_Au_nanoparticle = load(path)
    s = example_Au_nanoparticle.deepcopy()
    return s


example_Cu_nanoparticle_sim = None


def load_example_Cu_nanoparticle():
    """
    Get the hspy simulated image of an example Cu nanoparticle

    Example
    -------
    >>> import temul.example_data as example_data
    >>> s = example_data.load_example_Cu_nanoparticle()

    """
    global example_Cu_nanoparticle_sim
    if example_Cu_nanoparticle_sim is None:
        path = os.path.join(
            my_path, 'example_data',
            'structures', 'example_Cu_nanoparticle_sim.hspy')
        example_Cu_nanoparticle_sim = load(path)
    s = example_Cu_nanoparticle_sim.deepcopy()
    return s


example_Se_implanted_MoS2 = None


def load_Se_implanted_MoS2_data():
    """
    Get the hspy simulated image of an example Cu nanoparticle

    Example
    -------
    >>> import temul.example_data as example_data
    >>> s = example_data.load_example_Cu_nanoparticle()

    """
    global example_Se_implanted_MoS2
    if example_Se_implanted_MoS2 is None:
        path = os.path.join(
            my_path, 'example_data',
            'experimental', 'example_Se_implanted_MoS2.dm3')
        example_Se_implanted_MoS2 = load(path)
    s = example_Se_implanted_MoS2.deepcopy()
    return s


example_Se_implanted_MoS2_simulation = None


def load_Se_implanted_MoS2_simulation():
    """
    Get the hspy simulated image of an example Cu nanoparticle

    Example
    -------
    >>> import temul.example_data as example_data
    >>> s = example_data.load_example_Cu_nanoparticle()

    """
    global example_Se_implanted_MoS2_simulation
    if example_Se_implanted_MoS2_simulation is None:
        path = os.path.join(
            my_path, 'example_data',
            'prismatic', 'calibrated_data_prismatic_simulation.hspy')
        example_Se_implanted_MoS2_simulation = load(path)
    s = example_Se_implanted_MoS2_simulation.deepcopy()
    return s
