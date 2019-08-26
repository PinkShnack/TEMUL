import os
from hyperspy.io import load

my_path = os.path.dirname(__file__)

example_MoS2_vesta_xyz = None


def path_to_example_data_vesta_MoS2_vesta_xyz():
    """
    Get the path of the vesta xyz file for bilayer MoS2

    Example
    -------

    """
#     global example_MoS2_vesta_xyz
    if example_MoS2_vesta_xyz is None:
        path = os.path.join(
            my_path, 'example_data', 'prismatic', 'example_MoS2_vesta_xyz.xyz')
        # example_MoS2_vesta_xyz = load(path)
#     s = example_MoS2_vesta_xyz.deepcopy()
    return path


example_Au_nanoparticle = None


def load_example_Au_nanoparticle():
    """
    Get an example signal of a STEM detector
    Get the emd STEM image of an example Au nanoparticle

    Example
    -------
    >>> import atomap.api as am
    >>> s = am.example_data.path_to_example_data_prismatic()

    """
    global example_Au_nanoparticle
    if example_Au_nanoparticle is None:
        path = os.path.join(
            my_path, 'example_data', 'experimental', 'example_Au_nanoparticle.emd')
        example_Au_nanoparticle = load(path)
    s = example_Au_nanoparticle.deepcopy()
    return s
