.. _install:

.. include:: define_roles.rst


Installation
------------

The TEMUL Toolkit can be installed easily with PIP (those using Windows may need to download VS C++ Build Tools, see below).  

:bash:`$ pip install temul-toolkit`

Then, it can be imported with the name "temul". For example, to import the 
:python:`polarisation.py` module, use:

:python:`import temul.polarisation as tmlp`

Matplotlib 3.3 currently has compatability issues with iPython, see below for the fix.
 

-----------------------------
Installation Problems & Notes
-----------------------------

* If installing on Windows, you will need Visual Studio C++ Build Tools. Download it `here <https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019>`_. After downloading, choose the "C++ Build Tools" Workload and click install.

* Matplotlib seems to have issues with iPython at the moment. Install matplotlib==3.2 until this issue is resolved by Matplotlib. See `here <https://stackoverflow.com/questions/64291087/matplotlib-module-sip-has-no-attribute-setapi>`_ for more details.

* If you want to use the :python:`io.write_cif_from_dataframe` function, you will need to install pyCifRW version 4.3. This requires Visual Studio.

* If you wish to use the :python:`simulations.py` or :python:`model_refiner.py` modules, you will need to install PyPrismatic. This requires Visual Studio and other dependencies.

* If you're using any of the functions or classes that require element quantification:

   * navigate to the "temul/external" directory, copy the "atomap_devel_012" folder and paste that in your "site-packages" directory. 
   * Then, when using atomap to create sublattices and quantify elements call atomap like this: :python:`import atomap_devel_012.api as am`.
   * This development version is slowly being folded into the master branch here: https://gitlab.com/atomap/atomap/-/issues/93 and any help or tips on implementation are welcome!
