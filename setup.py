from setuptools import setup

setup(
    name='NRSS',
    version='0.1.0',
    description='NIST RSoXS Simulation Suite',
    url='https://github.com/pdudenas/NRSS',
    author='Peter Dudenas',
    author_email='peter.dudenas@nist.gov',
    license='NIST',
    packages=['NRSS'],
    install_requires=['PyHyperScattering>=0.1.7',
                        'numpy',
                        'scipy',
                        'scikit-image',
                        'pandas',
                        'matplotlib',
                        'h5py'],
    include_package_data=True,
    package_data={'': ['cmap/*']},
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: NIST',
        'Operating System :: POSIX :: Linux',
        'Topic :; Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics'
        'Programming Language :: Python :: 3'
    ],
)
