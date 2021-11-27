.. _install:

.. include:: define_roles.rst


Installation
------------

The TEMUL Toolkit can be installed easily with PIP (those using Windows may
need to download VS C++ Build Tools, see below).

:bash:`$ pip install temul-toolkit`

Then, it can be imported with the name "temul". For example, to import most
of the temul functionality use:

:python:`import temul.api as tml`


-----------------------------
Installation Problems & Notes
-----------------------------

* If installing on Windows, you will need Visual Studio C++ Build Tools.
Download it `here <https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2019>`_.
After downloading, choose the "C++ Build Tools" Workload and click install.

* If you want to use the :py:func:`temul.io.write_cif_from_dataframe`
function, you will need to install pyCifRW version 4.3. This requires
Visual Studio.

* If you wish to use the :py:mod:`temul.simulations` or
:py:mod:`temul.model_refiner` modules, you will need to install PyPrismatic.
This requires Visual Studio and other dependencies.
**It is unfortunately not guaranteed to work**. If you want to help develop
the :py:class:`temul.model_refiner.Model_Refiner`, please create an issue
and/or a pull request on the
`TEMUL github <https://github.com/PinkShnack/TEMUL>`_.

* If you're using any of the functions or classes that require element
quantification:

   * navigate to the "temul/external" directory, copy the "atomap_devel_012"
     folder and paste that in your "site-packages" directory.
   * Then, when using atomap to create sublattices and quantify elements call
     atomap like this: :python:`import atomap_devel_012.api as am`.
   * This development version is slowly being folded into the master branch
     here: https://gitlab.com/atomap/atomap/-/issues/93 and any help or
     tips on implementation are welcome!
