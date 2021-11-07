
import numpy as np
from hyperspy._signals.complex_signal2d import ComplexSignal2D

from temul.external.atomap_devel_012.initial_position_finding import (
    add_atoms_with_gui)
from temul.external.atomap_devel_012.atom_finding_refining import (
    _make_circular_mask)


def choose_points_on_image(image, norm='linear', distance_threshold=4):

    coords = add_atoms_with_gui(image=image, norm=norm,
                                distance_threshold=distance_threshold)
    return coords


def choose_mask_coordinates(image, norm="log"):
    """
    Pick the mask locations for an FFT. See get_masked_ifft() for examples and
    commit 5ba307b5af0b598bedc0284aa989d44e23fdde4d on Atomap for more details.

    Parameters
    ----------
    image : Hyperspy 2D Signal
    norm : str, default "log"
        How to scale the intensity value for the displayed image.
        Options are "linear" and "log".

    Returns
    -------
    list of pixel coordinates

    """

    fft = image.fft(shift=True)
    fft_amp = fft.amplitude

    mask_coords = choose_points_on_image(
        fft_amp.data, norm=norm)

    return mask_coords


def get_masked_ifft(image, mask_coords, mask_radius=10, image_space="real",
                    keep_masked_area=True, plot_masked_fft=False):
    """
    Creates an inverse fast Fourier transform (iFFT) from an image and mask
    coordinates. Use ``choose_mask_coordinates`` to manually choose mask
    coordinates in the FFT.

    Parameters
    ----------
    image : Hyperspy 2D Signal
    mask_coords : list of pixel coordinates
        Pixel coordinates of the masking locations. See the example below for
        two simple coordinates found using ``choose_mask_coordinates``.
    mask_radius : int, default 10
        Radius in pixels of the mask.
    image_space : str, default "real"
        If the input image is in Fourier/diffraction/reciprocal space already,
        set space="fourier".
    keep_masked_area : bool, default True
        If True, this will set the mask at the mask_coords.
        If False, this will set the mask as everything other than the
        mask_coords. Can be thought of as inversing the mask.
    plot_masked_fft : bool, default False
        If True, the mask used to filter the FFT will be plotted. Good for
        checking that the mask is doing what you want. Can fail sometimes due
        to matplotlib plotting issues.

    Returns
    -------
    Inverse FFT as a Hyperspy 2D Signal

    Examples
    --------
    >>> from temul.dummy_data import get_simple_cubic_signal
    >>> import temul.api as tml
    >>> image = get_simple_cubic_signal()
    >>> mask_coords = tml.choose_mask_coordinates(image) #  manually
    >>> mask_coords = [[170.2, 170.8],[129.8, 130]]
    >>> image_ifft = tml.get_masked_ifft(image, mask_coords)
    >>> image_ifft.plot()

    Plot the masked fft:

    >>> image_ifft = tml.get_masked_ifft(image, mask_coords,
    ...                                  plot_masked_fft=True)

    Use unmasked fft area and plot the masked fft:

    >>> image_ifft = tml.get_masked_ifft(
    ...     image, mask_coords, plot_masked_fft=True, keep_masked_area=False)
    >>> image_ifft.plot()

    If the input image is already a Fourier transform:

    >>> fft_image = image.fft(shift=True)
    >>> image_ifft = tml.get_masked_ifft(fft_image, mask_coords,
    ...     image_space='fourier')
    >>> image_ifft.plot()

    """
    if image_space.lower() == 'real':
        fft = image.fft(shift=True)
    elif image_space.lower() == 'fourier':
        fft = image
    else:
        raise ValueError("image_space must be either ``real`` or ``fourier``.")

    if len(mask_coords) == 0:
        raise ValueError("`mask_coords`` has not been set.")
    for mask_coord in mask_coords:

        x_pix = mask_coord[0]
        y_pix = mask_coord[1]

        # Create a mask over that location and transpose the axes
        mask = _make_circular_mask(centerX=x_pix, centerY=y_pix,
                                   imageSizeX=image.data.shape[1],
                                   imageSizeY=image.data.shape[0],
                                   radius=mask_radius).T

        # check if the combined masked_fft exists
        try:
            mask_combined
        except NameError:
            mask_combined = mask
        else:
            mask_combined += mask

    # the tilda ~ inverses the mask
    # fill in nothings with 0
    if keep_masked_area:
        masked_fft = np.ma.array(fft.data, mask=~mask_combined).filled(0)
    elif not keep_masked_area:
        masked_fft = np.ma.array(fft.data, mask=mask_combined).filled(0)

    masked_fft_image = ComplexSignal2D(masked_fft)
    if plot_masked_fft:
        masked_fft_image.amplitude.plot(norm="log")
    # sort out units here
    image_ifft = masked_fft_image.ifft()
    image_ifft = np.absolute(image_ifft)
    image_ifft.axes_manager = image.axes_manager

    return image_ifft
