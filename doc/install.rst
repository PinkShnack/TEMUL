.. _install:

.. include:: define_roles.rst


Installation
------------

The TEMUL Toolkit can be installed easily with PIP.

:bash:`$ pip install temul-toolkit`

Then, it can be imported with the name "temul". For example, to import the 
:python:`polarisation.py` module, use:

:python:`import temul.polarisation as tmlpol`

Note 1: If you wish to use the :python:`simulations.py` or :python:`model_refiner.py` 
modules, you will need to install PyPrismatic. 

Note 2: If you're using any of the functions or classes that require element quantification:

   * navigate to the "temul/external" directory, copy the "atomap_devel_012" folder and paste that in your "site-packages" directory. 
   * Then, when using atomap to create sublattices and quantify elements call atomap like this: :python:`import atomap_devel_012.api as am`.
   * This development version is slowly being folded into the master branch here: https://gitlab.com/atomap/atomap/-/issues/93 and any help or tips on implementation are welcome!
