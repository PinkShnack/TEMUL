.. TEMUL Toolkit documentation master file, created by
   sphinx-quickstart on Fri Jul 10 17:39:02 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TEMUL Toolkit's documentation!
=========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

.. role:: bash(code)
   :language: bash

.. role:: python(code)
   :language: python


The TEMUL Toolkit is a suit of functions and classes for analysis and visualisation of
atomic resolution images. It is mostly built upon the data structure of 
`HyperSpy <https://hyperspy.org/>`_ and `Atomap <https://atomap.org/>`_.

.. Look how easy it is to use:
    import project
    # Get your stuff done
    project.do_stuff()

.. Features
   --------
   - Be awesome


Installation
------------

The easiest way to use the TEMUL Toolkit is to place the TEMUL/temul folder into your Python packages directory as follows:

#. Click on the green "Code" button. Git clone or Download the ZIP to download the TEMUL package.
#. Navigate into the TEMUL folder, copy the "temul" folder.
#. Navigate to your python packages directory and paste the "temul" folder there. In an Anaconda installation, this can be found at "anaconda3/Lib/site-packages"
#. Open Anaconda prompt and run :bash:`python`. Run :bash:`import temul`. If no error occurs, run :bash:`temul`, which should show the directory in which you just pasted the temul folder.
#. If you're using any of the functions or classes that require element quantification:

   * navigate to the "temul/external" directory, copy the "atomap_devel_012" folder and paste that in your "site-packages" directory. 
   * Then, when using atomap to create sublattices and quantify elements call atomap like this: :python:`import atomap_devel_012.api as am`.
   * This development version is slowly being folded into the master branch here: https://gitlab.com/PinkShnack/atomap/-/tree/chemical_info.
   * The original issue can be found here: https://gitlab.com/atomap/atomap/-/issues/93 and any help or tips on implementation are welcome!

Contribute
----------

- `Issue Tracker <github.com/PinkShnack/TEMUL/issues>`_
- `Source Code <github.com/PinkShnack/TEMUL>`_

Support
-------

If you are having issues, please let us know in the issue tracker on 
`GitHub <github.com/PinkShnack/TEMUL/issues>`_.


License
-------

The project is licensed under the `GPL-3.0 License <https://github.com/PinkShnack/TEMUL/blob/master/LICENSE>`_.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
