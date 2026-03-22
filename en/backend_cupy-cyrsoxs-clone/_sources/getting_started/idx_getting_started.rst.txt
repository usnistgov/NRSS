.. _Getting_Started:

===============
Getting Started
===============

Hardware Requirements
_____________________
.. warning::
    CyRSoXS requires an NVIDIA GPU.

Installation on Linux
_____________________

.. note:: 
    This installation guide uses the Anaconda Python distribution. Any Python 
    installation ``version >=3.6`` should work, you will just need to point CMake to your 
    specific installation. git, conda, and pip are also used, and it is assumed you have 
    working installations of all three.

**CyRSoXS v1.1.5.0** is now Conda installable and no longer requires building from 
source. If you need to compile from source (to enable double-precision, for example) 
the instructions are provided below.

Conda Installation
^^^^^^^^^^^^^^^^^^^^^^^^^

Clone NRSS from the github repository:

.. code-block:: bash

    git clone https://github.com/usnistgov/NRSS.git


Use the ``environment.yml`` file to create a new virtual environment, and activate it:

.. code-block:: bash

    conda env create -f environment.yml
    conda activate nrss

CyRSoXS is listed as a dependency in ``environment.yml``, and will automatically be installed.

Now we can pip install NRSS, which will also install PyHyperScattering as a dependency:

.. code-block:: bash

    pip install .

The conda-forge distribution of CyRSoXS includes the executable and Python Shared Library File. 
You can use the CyRSoXS executable from the shell:

.. code-block:: bash
    
    CyRSoXS <Morphology HDF5>

or import CyRSoXS to a python script or jupyter notebook:

.. code-block:: python

    import CyRSoXS

After importing, you should see the following output:

.. code-block:: console

    CyRSoXS
    ============================================================================
    Size of Real               : 4
    Maximum Number Of Material : 32
     __________________________________________________________________________________________________
    |                                 Thanks for using Cy-RSoXS                                        |
    |--------------------------------------------------------------------------------------------------|
    |  Copyright          : Iowa State University                                                      |
    |  License            : NIST                                                                       |
    |  Acknowledgement    : ONR MURI                                                                   |
    |                                                                                                  |
    |  Developed at Iowa State University in collaboration with NIST                                   |
    |                                                                                                  |
    |  Please cite the following publication :                                                         |
    |  Comments/Questions :                                                                            |
    |          1. Dr. Baskar GanapathySubramanian (baskarg@iastate.edu)                                |
    |          2. Dr. Adarsh Krishnamurthy        (adarsh@iastate.edu)                                 |
    |          3. Dr. Dean DeLongchamp            (dean.delongchamp@nist.gov)                          |
    -------------------------------------------------------------------------------------------------- 
    Version   :  <version_number>
    Git patch :  <git_patch_number>

Building CyRSoXS from source
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**NOTE:** These installation instructions use Conda to install the required dependencies. 
If you prefer to manually install and manage these dependencies, please see the 
installation instructions at https://github.com/usnistgov/cyrsoxs/blob/main/docs/INSTALL.md

**Dependencies**

*Required Dependencies*

* A C++ compiler with C++14 support is required.
* gcc >= 7 (CUDA specific versions might have GCC requirements )
* Cuda Toolkit (>=9)
* HDF5
* OpenMP
* libconfig
* Python >= 3.6 (only for building with Pybind)

*Optional Dependencies*

* Doxygen
* Docker

Clone CyRSoXS from the github repository:

.. code-block:: bash

    git clone https://github.com/usnistgov/cyrsoxs.git

Use the ``environment-build.yml`` file to create a new virtual environment, and activate it:

.. code-block:: bash

    conda env create -f environment-build.yml
    conda activate cyrsoxs-build

**Building CyRSoXS without Pybind**

.. code-block:: bash

    cd $CyRSoXS_DIR
    mkdir build;
    cd build;
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make


Create a ``bin`` directory and move the CyRSoXS executable inside:

.. code-block:: bash

    mkdir bin
    mv CyRSoXS bin/

Add CyRSoXS to your PATH:

.. code-block:: bash

    cd bin
    echo "export PATH=$PATH:`pwd`" >> ~/.bashrc
    source ~/.bashrc

At this point you should have a working CyRSoXS installation. If you also want to import CyRSoXS as a Python library, you need to compile with Pybind.

**Building CyRSoXS with Pybind**

.. code-block:: bash

    cd $CyRSoXS_DIR
    mkdir build_pybind;
    cd build_pybind;
    cmake .. -DCMAKE_BUILD_TYPE=Release -DPYBIND=Yes -DUSE_SUBMODULE_PYBIND=No

Depending on where your python installation is, you may need to point CMake to it by including the following compile flags:

.. code-block:: bash

    -DPYTHON_EXECUTABLE=[path_to_anaconda]/anaconda/bin/python
    -DPYTHON_LIBRARY=[path_to_anaconda]/anaconda/lib/libpython3.9.so
    -DPYTHON_INCLUDE=[path_to_anaconda]/anaconda/include/python3.9/

If this still doesn't work, you can edit the ``CMakeLists.txt`` file on line 82 to include the three ``set`` commands:

.. code-block:: cmake

    if (PYBIND)
        set(Python_EXECUTABLE [path_to_anaconda]/anaconda/bin/python)
        set(Python_INCLUDE_DIR [path_to_anaconda]/anaconda/include/python3.9)
        set(Python_LIBRARIES [path_to_anaconda]/anaconda/lib/libpython3.9.so)
        find_package(Python COMPONENTS Interpreter Development REQUIRED)

Once the CMake files have been generated run the following command:

.. code-block:: bash

    make

This will generate a shared library ``CyRSoXS.so`` file. Create a ``lib`` directory and move ``CyRSoXS.so`` inside:

.. code-block:: bash

    mkdir lib
    mv CyRSoXS.so lib/

Add to your PYTHONPATH:

.. code-block:: bash

    cd lib
    echo "export PYTHONPATH=$PYTHONPATH:`pwd`" >> ~/.bashrc
    source ~/.bashrc

Now you can import CyRSoXS in a python script or jupyter notebook:

.. code-block:: python

    import CyRSoXS

Again, you should see the following output:

.. code-block:: console

    CyRSoXS
    ============================================================================
    Size of Real               : 4
    Maximum Number Of Material : 32
     __________________________________________________________________________________________________
    |                                 Thanks for using Cy-RSoXS                                        |
    |--------------------------------------------------------------------------------------------------|
    |  Copyright          : Iowa State University                                                      |
    |  License            : NIST                                                                       |
    |  Acknowledgement    : ONR MURI                                                                   |
    |                                                                                                  |
    |  Developed at Iowa State University in collaboration with NIST                                   |
    |                                                                                                  |
    |  Please cite the following publication :                                                         |
    |  Comments/Questions :                                                                            |
    |          1. Dr. Baskar GanapathySubramanian (baskarg@iastate.edu)                                |
    |          2. Dr. Adarsh Krishnamurthy        (adarsh@iastate.edu)                                 |
    |          3. Dr. Dean DeLongchamp            (dean.delongchamp@nist.gov)                          |
    -------------------------------------------------------------------------------------------------- 
    Version   :  <version_number>
    Git patch :  <git_patch_number>


*Optional CMake Flags*

.. code-block:: console
    
    -DPYBIND=Yes            # Compiling with Pybind: 
    -DMAX_NUM_MATERIAL=64   # To change the maximum number of materials (default is 32) 
    -DDOUBLE_PRECISION=Yes  # Double precision mode
    -DPROFILING=Yes         # Profiling
    -DBUILD_DOCS=Yes        # To build documentation
    -DCMAKE_CXX_COMPILER=icpc -DCMAKE_C_COMPILER=icc # Compiling with the Intel compiler (does not work with Pybind)



NRSS Tutorials
__________

Several tutorials are available in the `tutorials <https://github.com/usnistgov/NRSS/tree/main/tutorials>`_ 
folder. Most of these tutorials include Jupyter notebooks with explanatory prose, example code, and code use strategies.

For all tutorials, it is recommended to copy the entire tutorial folder out of the installed repository location before 
running, because some tutorials will generate file changes and new serialized objects within repository directories.

kkcalc for NRSS
^^^^^^^^^^^^^^^
This Jupyter notebook series describes how to convert Near Edge X-ray Absorption Fine Structure 
(NEXAFS) data into a complex index of refraction for use in NRSS computations. Conditioning of the NEXAFS and the use of the 
`kkcalc library <https://github.com/benajamin/kkcalc>`_ to develop this complex index are demonstrated.
Both notebooks additionally include calculation code to predict binary contrast from pairs of complex 
indices in real materials.

Tutorial Series Overview
^^^^^^^^^^^^^^^^^^^^^

NRSS provides several tutorial series to help you get started with different aspects of the software:

3D Polymer Adsorption
--------------------
A series of tutorials demonstrating NRSS modeling and simulation capabilities:

1. `Basic Model Setup <https://github.com/usnistgov/NRSS/blob/main/src/NRSS_tutorials/3D_polymer_adsorption/01_polymer_adsorption_model.ipynb>`_: Introduction to setting up NRSS simulations, including environment configuration, defining model parameters, and working with optical constants for materials like polystyrene (PS) and silicon dioxide (SiO2).

2. `Function Framework <https://github.com/usnistgov/NRSS/blob/main/src/NRSS_tutorials/3D_polymer_adsorption/02_polymer_adsorption_functions.ipynb>`_: Creates a code framework around the basic model for rapid exploration, organizing model construction into functions with arguments.

3. `Parameter Sweeps <https://github.com/usnistgov/NRSS/blob/main/src/NRSS_tutorials/3D_polymer_adsorption/03_polymer_adsorption_sweeps.ipynb>`_: Demonstrates automated parameter sweeps using Python loop structures, with results saved using pickle.

4. `Streamlined Sweeps <https://github.com/usnistgov/NRSS/blob/main/src/NRSS_tutorials/3D_polymer_adsorption/04_polymer_adsorption_sweep_streamlined.ipynb>`_: A more efficient implementation of parameter sweeps using imported library functions, demonstrated with PMMA as the example material.

5. `Basic Visualization <https://github.com/usnistgov/NRSS/blob/main/src/NRSS_tutorials/3D_polymer_adsorption/05_visualizing_simulation_results.ipynb>`_: Examining and plotting NRSS simulation results using matplotlib.

6. `Advanced Visualization <https://github.com/usnistgov/NRSS/blob/main/src/NRSS_tutorials/3D_polymer_adsorption/06_visualizing_sweeps_of_sweeps.ipynb>`_: Creating animations from parameter sweep results using ffmpeg.


Multi-Walled Carbon Nanotubes (MWCNTs)
------------------------------------
This series focuses on applying NRSS to study the structure and properties of multi-walled carbon nanotubes using various experimental techniques and modeling approaches.

1. `RSoXS Analysis <https://github.com/usnistgov/NRSS/blob/main/src/NRSS_tutorials/MWCNTs/nb1_rsoxs.ipynb>`_: A detailed walkthrough of resonant soft X-ray scattering data processing for MWCNT samples. Learn how to import, process, and analyze RSoXS data, including background subtraction, normalization, and initial interpretation.

2. `NEXAFS Analysis <https://github.com/usnistgov/NRSS/blob/main/src/NRSS_tutorials/MWCNTs/nb2_nexafs.ipynb>`_: Covers the analysis of Near Edge X-ray Absorption Fine Structure spectroscopy data from MWCNTs. Understand how to extract chemical and structural information from NEXAFS spectra and prepare this information for use in NRSS modeling.

3. `NRSS Modeling <https://github.com/usnistgov/NRSS/blob/main/src/NRSS_tutorials/MWCNTs/nb3_nrss.ipynb>`_: Demonstrates how to build and apply NRSS models to MWCNT systems. Learn to combine RSoXS and NEXAFS data with NRSS modeling to extract detailed structural information about MWCNT organization and properties.

KKcalc Integration
----------------
These tutorials show how to determine optical constants from experimental data using the Kramers-Kronig calculator (KKcalc) and integrate them into NRSS simulations.

1. `PS-PMMA Analysis <https://github.com/usnistgov/NRSS/blob/main/src/NRSS_tutorials/kkcalc_for_NRSS/kkcalc_ps_pmma_xr.ipynb>`_: A step-by-step guide to determining scalar complex refractive indices from NEXAFS data of PS and PMMA. Learn the complete workflow from raw data processing to final optical constant determination, including data validation and error analysis.

2. `Y6-PM6 Orientation <https://github.com/usnistgov/NRSS/blob/main/src/NRSS_tutorials/kkcalc_for_NRSS/kkcalc_orientation_Y6_PM6.ipynb>`_: Advanced tutorial on calculating tensor (uniaxial) optical constants for oriented organic semiconductor materials. Understand how to handle orientation effects in NEXAFS spectra and determine direction-dependent optical properties.


Core-Shell Radial Disk in the NRSS
^^^^^^^^^^^^^^^^^^^^^^^^^
This Jupyter notebook is a basic introduction to the NRSS. It has two different versions. 
The pybind version is recommended, as it demonstrates the preferred NRSS pybind workflow.

Both tutorials describe a radial disk scattering object, a simple 2D structure with a radial
orientation of the extraordinary component of a tensor uniaxial index of refraction.

* `Commandline / slurm queue version. <https://github.com/usnistgov/NRSS/blob/main/tutorials/coreshell_disk/CoreShell.ipynb>`_ 
This tutorial describes how to create the radial disk morphology and serialize the model to hard drive,
submit a CyRSoXS job via slurm, and examine the simulation result.

* `Pybind version. <https://github.com/usnistgov/NRSS/blob/main/tutorials/pybind/MorphologyClass.ipynb>`_ 
This tutorial describes how to create the radial disk morphology, submit the simulation job within Python, and examine the simulation result.
This tutorial relies on index of refraction information stored in the commandline / slurm queue version subdirectory,
so it is recommended to copy both at the same time into the same parent directory.

Polymer Grafted Nanoparticles (PGNs)
^^^^^^^^^^^^^^^^^^^^^^^^^
* `Polymer Grafted Nanoparticles <https://github.com/usnistgov/NRSS/blob/main/src/NRSS_tutorials/polymer_grafted_nanoparticles/parametric_pgn_model.ipynb>`_
This tutorial demonstrates how to simulate resonant soft X-ray scattering from polymer-grafted nanoparticles using a parametric model. 
The model is based on published work examining polystyrene-grafted gold nanoparticles, and allows exploration of orientation decay 
away from particle surfaces. The tutorial includes both morphology generation and NRSS simulation, with a focus on the carbon K-edge 
energies relevant to polystyrene orientation.

