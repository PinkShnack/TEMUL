from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
        name = 'temul-toolkit',
        packages = [
            'temul',
            'temul.tests',
            'temul.external',
            ],
        version = '1.0.1',
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
            'colorcet<=2.0.2',
            'glob2<=0.6',
            'hyperspy>=1.5.2',
            'numpy',
            'matplotlib>=3.1.0',
            'matplotlib-scalebar',
            'pandas',
            'periodictable<=1.5.0',
            'PyCifRW==4.3',
            'pyprismatic',
            'scikit-image>=0.13',
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
