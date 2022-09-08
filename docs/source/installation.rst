============
Installation
============

Hardware Requirements
_____________________

* The CyRSoXS submodule requires an NVIDIA GPU.

Installation on Linux
_____________________

**NOTE:** This installation guide uses git for cloning github repositories and Anaconda/conda as the Python distribution/package and virtual environment manager, and assumes you have working installations of both.

NRSS & PyHyperScattering
^^^^^^^^^^^^^^^^^^^^^^^^^

Clone NRSS from the github repository:

.. code-block:: bash

    git clone https://github.com/usnistgov/NRSS.git

Navigate to the cloned repository and initialize the CyRSoXS submodule and its submodules:

.. code-block:: bash

    git submodule update --init --recursive 

Use the environment.yml file to create a new virtual environment, and activate it:

.. code-block:: bash

    conda env create -f environment.yml
    conda activate nrss

Now we can pip install NRSS, which will also install PyHyperScattering as a dependency:

.. code-block:: bash

    pip install .

CyRSoXS
^^^^^^^

The build instructions here mirror what is provided on the usnistgov/cyrsoxs github repo, with slight modifications for the repo as a submodule.

**NOTE:** Before installing the dependencies below, make sure to navigate out of the NRSS directory.

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

**Compiling Libconfig**

Libconfig's build system is `Autotools <https://www.gnu.org/software/automake/manual/html_node/Autotools-Introduction.html>`_, which means you'll need to run ``./configure`` and then ``make`` to build.

This guide recommends passing ``--prefix=`pwd`/install`` to ``./configure``, which will cause ``make install`` to copy the output files to ``[your_libconfig_dir]/install`` instead of ``/usr``. This way your libconfig install lives completely inside your libconfig folder. This is necessary if you are working on a system where you don't have admin privileges (i.e. an HPC cluster).


Download and extract:

.. code-block:: bash

    cd $LIBCONFIG_INSTALL_DIRECTORY
    wget http://hyperrealm.github.io/libconfig/dist/libconfig-1.7.2.tar.gz
    tar xvf libconfig-1.7.2.tar.gz
    rm libconfig-1.7.2.tar.gz

Compile and copy output files to libconfig-1.7.2/install:

.. code-block:: bash

    cd libconfig-1.7.2
    ./configure --prefix=`pwd`/install
    make -j8  # compile with 8 threads
    make install

**NOTE:** On some HPC clusters (when using the Intel compiler), the ``make`` step gives a linker error. This is the libconfig example program failing to link with the Intel runtime. This is okay - the libconfig library itself compiles just fine. Just run ``make install`` and double check that ``install/lib`` contains some ``*.a`` files.


Permanently set ``$LIBCONFIG_DIR`` environment variable and set ``LD_LIBRARY_PATH`` to include the libconfig lib directory to prevent dynamic linking errors with libconfig++.so:

.. code-block:: bash

    echo "export LIBCONFIG_DIR=`pwd`/install" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:\$LIBCONFIG_DIR/lib" >> ~/.bashrc
    source ~/.bashrc


**Installing HDF5**

Cy-RSoXS uses the `HDF5 <https://en.wikipedia.org/wiki/Hierarchical_Data_Format>`_ library to store morphology models and simulated scattering patterns. To install:

.. code-block:: bash

    cd $HDF5_INSTALL_DIRECTORY
    wget https://support.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-1.10.5/src/CMake-hdf5-1.10.5.tar.gz
    tar -xzvf CMake-hdf5-1.10.5.tar.gz
    rm CMake-hdf5-1.10.5.tar.gz
    cd CMake-hdf5-1.10.5
    ./build-unix.sh

This step might take some time. Do not cancel until all the tests have passed.
This step will create cmake files at ``$HFD5_DIR/build/_CPack_Packages/Linux/TGZ/HDF5-1.10.5-Linux/HDF_Group/HDF5/1.10.5/share/cmake/hdf5``

Export the path for HDF5:

.. code-block:: bash

    cd build/_CPack_Packages/Linux/TGZ/HDF5-1.10.5-Linux/HDF_Group/HDF5/1.10.5/share/cmake/hdf5;
    echo "export HDF5_DIR=`pwd`" >> ~/.bashrc
    source ~/.bashrc


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
    cmake .. -DCMAKE_BUILD_TYPE=Release -DPYBIND=Yes

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

Add to your PATH:

.. code-block:: bash

    cd lib
    echo "export PATH=$PATH:`pwd`" >> ~/.bashrc
    source ~/.bashrc

Now you can import CyRSoXS in a python script or jupyter notebook:

.. code-block:: python

    import CyRSoXS

You should see the following output:

.. code-block:: console

    CyRSoXS
    ============================================================================
    Size of Real               : 4
    Maximum Number Of Material : 32
    __________________________________________________________________________________________________
    |                                 Thanks for using Cy-RSoXS                                        |
    |--------------------------------------------------------------------------------------------------|
    |  Copyright          : Iowa State University                                                      |
    |  License            : MIT                                                                        |
    |  Acknowledgement    : ONR MURI                                                                   |
    |  Developed at Iowa State University in collaboration with NIST                                   |
    |  Please cite the following publication :                                                         |
    |  Comments/Questions :                                                                            |
    |          1. Dr. Baskar GanapathySubramanian (baskarg@iastate.edu)                                |
    |          2. Dr. Adrash Krishnamurthy        (adarsh@iastate.edu)                                 |
    -------------------------------------------------------------------------------------------------- 
    Version   :  1 . 1 . 4 . 0
    Git patch :  d88e168


*Optional CMake Flags*

.. code-block:: console
    
    -DPYBIND=Yes            # Compiling with Pybind: 
    -DMAX_NUM_MATERIAL=64   # To change the maximum number of materials (default is 32) 
    -DDOUBLE_PRECISION=Yes  # Double precision mode
    -DPROFILING=Yes         # Profiling
    -DBUILD_DOCS=Yes        # To build documentation
    -DCMAKE_CXX_COMPILER=icpc -DCMAKE_C_COMPILER=icc # Compiling with the Intel compiler (does not work with Pybind)





