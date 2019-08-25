import os
# from hyperspy.io import load

my_path = os.path.dirname(__file__)

example_MoS2_vesta_xyz = None


def path_to_example_data_vesta_MoS2_vesta_xyz():
    """Get an example signal of a STEM detector

    Example
    -------
    >>> import atomap.api as am
    >>> s = am.example_data.path_to_example_data_prismatic()

    """
#     global example_MoS2_vesta_xyz
    if example_MoS2_vesta_xyz is None:
        path = os.path.join(
            my_path, 'example_data', 'prismatic', 'example_MoS2_vesta_xyz.xyz')
        # example_MoS2_vesta_xyz = load(path)
#     s = example_MoS2_vesta_xyz.deepcopy()
    return path
