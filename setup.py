from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
        name = 'temul-toolkit',
        packages = [
            'temul',
            'temul.tests',
            'temul.external.atomap_devel_012',
            'temul.external.atomap_devel_012.external',
            'temul.external.skimage_devel_0162',
            ],
        version = '1.0.20',
        description = 'Functions for analysis of high resolution electron microscopy and spectroscopy data.',
        long_description=long_description,
        long_description_content_type='text/markdown',
        author = "Eoghan N O'Connell",
        author_email = 'eoclives@hotmail.com',
        license = 'GPL v3',
        url = 'https://temul-toolkit.readthedocs.io/',
        keywords = [
            'STEM',
            'data analysis',
            'microscopy',
            'data science',
            'image simulation',
            'atomic models'
            ],
        install_requires = [
            'atomap',
            'colorcet',
            'glob2',
            'hyperspy',
            'numpy',
            'matplotlib',
            'matplotlib-scalebar',
            'pandas',
            'periodictable',
            'PyCifRW==4.3',
            'pyprismatic',
            'scikit-image==0.16.2',
            'scipy',
            'tifffile',
            ],
        classifiers = [
            'Development Status :: 3 - Alpha',
            'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python :: 3',
            ],
        include_package_data=True,
)
