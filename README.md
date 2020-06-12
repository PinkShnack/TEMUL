# TEMUL
Functions for image and signal processing, based on the packages Scikit-image, Hyperspy, Atomap, PyPrismatic, ASE, periodictable etc. Written by members of the TEMUL group from the University of Limerick, Ireland (though we welcome all help, hints and contributions!)

## How to Install
The easiest way to use this package is to place the TEMUL/**temul** folder into your Python packages directory.

1.  Click on the green "Clone or Download" button. Use ` git clone https://github.com/PinkShnack/TEMUL.git ` with git bash, or Download ZIP to download the TEMUL package.
2.  Navigate into the TEMUL folder, copy the "temul" folder.
3.  Navigate to your python packages directory and paste the "temul" folder there. In an Anaconda installation, this can be found at "anaconda3/Lib/site-packages"
4.  Open Anaconda prompt and run `python`. Run `import temul`. If no error occurs, run `temul`, which should show the directory in which you just pasted the temul folder.
5.  If you're using any of the functions or classes that require element quantification, navigate to the "temul/external" directory, copy the "atomap_devel_012" folder and paste that in your "site-packages" directory. Then, when using atomap to create sublattices and quantify elements call atomap like this: `import atomap_devel_012.api as am`. This development version is slowly being folded into the master branch here: https://gitlab.com/PinkShnack/atomap/-/tree/chemical_info. The original issue can be found here: https://gitlab.com/atomap/atomap/-/issues/93 and any help or tips on implementation are welcome!

## Cite

To cite the TEMUL toolkit, use the following DOI:

[![DOI](https://www.zenodo.org/badge/203785298.svg)](https://www.zenodo.org/badge/latestdoi/203785298)
