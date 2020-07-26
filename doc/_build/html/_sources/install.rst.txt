.. _install:

Installation
------------

TLDR: Place the temul folder into your site packages folder. 
Now that PyPrismatic can be installed via conda, we hope to have a pip or conda
installation available for the TEMUL Toolkit soon. 

The easiest way to use the TEMUL Toolkit is to place the temul folder (lowercase) into your Python packages directory as follows:

#. Go to the `TEMUL Toolkit source code <https://github.com/PinkShnack/TEMUL>`_.
#. Click on the green "Code" button. Git clone or Download the ZIP to download the TEMUL package.
#. Navigate into the TEMUL folder, copy the "temul" folder.
#. Navigate to your python packages directory and paste the "temul" folder there. In an Anaconda installation, this can be found at "anaconda3/Lib/site-packages"
#. Open Anaconda prompt and run :bash:`python`. Run :bash:`import temul`. If no error occurs, run :bash:`temul`, which should show the directory in which you just pasted the temul folder.
#. If you're using any of the functions or classes that require element quantification:

   * navigate to the "temul/external" directory, copy the "atomap_devel_012" folder and paste that in your "site-packages" directory. 
   * Then, when using atomap to create sublattices and quantify elements call atomap like this: :python:`import atomap_devel_012.api as am`.
   * This development version is slowly being folded into the master branch here: https://gitlab.com/PinkShnack/atomap/-/tree/chemical_info.
   * The original issue can be found here: https://gitlab.com/atomap/atomap/-/issues/93 and any help or tips on implementation are welcome!
