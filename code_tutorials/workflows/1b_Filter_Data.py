
import temul.api as tml


''' You may need to filter your image so that the sublattices can be better
    detected. Only use this if the atom finding in the next step isn't going
    well.
'''


# single Boracite images may need to be FFT masked:
mask_radius = 8  # pixels
mask_coords = tml.choose_mask_coordinates(image)

image_ifft = tml.get_masked_ifft(image, mask_coords, mask_radius=mask_radius)
image_ifft.plot()  # check if the filtering worked

# save the mask coordinates for later use (how to reload them is below)
np.save('mask_coordinates', np.array(mask_coords))
# mask_coords = np.load('mask_coordinates.npy')
