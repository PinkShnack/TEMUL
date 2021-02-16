
import os
import hyperspy.api as hs

''' Set the directory (what folder your data is in!)
    Notice the slashes must be forward slashes. '''

directory = 'C:/Users/Example/notice/the/forward/slashes'
os.chdir(directory)

''' Load the file, it must be in the directory!
    Hyperspy can load many different filetypes: dm3, dm4, emd, tif.
    Check their website! '''

image = hs.load('example_awesome_image.tif')

# plot the image to see how it looks!
image.plot()

# set the pixel size (called sampling in this toolkit) and the units
sampling = image.axes_manager[-1].scale
units = image.axes_manager[-1].units

print(sampling)  # if sampling is 1, then the image is probably not calibrated.
