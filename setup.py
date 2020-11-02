from setuptools import setup

with open('README.md') as f:
    long_description = f.read()

setup(
    name='temul-toolkit',
    packages=[
        'temul',
        'temul.tests',
        'temul.example_data.experimental',
        'temul.example_data.prismatic',
        'temul.example_data.structures',
        'temul.external.atomap_devel_012',
        'temul.external.atomap_devel_012.external',
        'temul.external.skimage_devel_0162',
    ],
    version='0.1.1',
    description='Functions for analysis of high resolution electron microscopy and spectroscopy data.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author="Eoghan N O'Connell",
    author_email='eoclives@hotmail.com',
    license='GPL v3',
    url='https://temul-toolkit.readthedocs.io/',
    keywords=[
        'STEM',
            'data analysis',
            'microscopy',
            'data science',
            'image simulation',
            'atomic models'
    ],
    install_requires=[
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
        'scikit-image>=0.16.2',
        'scipy',
        'tifffile',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
    ],
    package_data={
        'temul.example_data.experimental': [
            'example_Au_nanoparticle.emd',
            'example_Se_implanted_MoS2.dm3'],
        'temul.example_data.prismatic': [
            'example_MoS2_vesta_xyz.xyz',
            'MoS2_hex_prismatic.xyz',
            'calibrated_data_probeStep0.01_interpolationFactor4_crop0.5.hspy'],
        'temul.example_data.structures': [
            'example_Cu_nanoparticle_sim.hspy'],
    },
    include_package_data=True,
)
