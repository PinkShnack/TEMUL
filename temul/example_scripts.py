
import os
import my_code_functions_all as temul
import atomap.api as am
import hyperspy.api as hs
# choose directory

directory = 'G:/Titan Images/08-10-19_MHEOC_SampleImaging stem/Cross Grating for STEM alignment/Au NP'
os.chdir(directory)


'''
Au NP example
'''
# open file
s_raw, sampling = temul.load_data_and_sampling(
    'STEM 20190813 HAADF 1732.emd', save_image=False)

cropping_area = am.add_atoms_with_gui(s_raw.data)

cropping_area = [[1, 2], [1, 2]]

s_crop = temul.crop_image_hs(s_raw, cropping_area)

roi = hs.roi.RectangularROI(left=1, right=5, top=1, bottom=5)
s_raw.plot()
s_crop = roi.interactive(s_raw)
s_crop.plot()
