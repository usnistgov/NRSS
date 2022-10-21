===============
Getting Started
===============

Hardware Requirements
_____________________

CyRSoXS requires an NVIDIA GPU.

Installation on Linux
_____________________

**NOTE:** This installation guide uses the Anaconda Python distribution. Any Python 
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





