=============================
CyRSoXS Simulation Components
=============================

`Cy-RSoXS <https://github.com/usnistgov/cyrsoxs>`_ is a voxel-based forward-scattering simulator calculated in the Born Approximation. This page describes the required files and file structure for use with CyRSoXS.

- **Morphology file**
    - .hdf5 format
- **Material optical constant files**
    - Material1.txt, Material2.txt, Material3.txt, etc.
- **Configuration file**
    - config.txt



Morphology File Structure
_________________________

Cy-RSoXS accepts two different types of morphologies: Vector and Euler. A vector morphology will use a vector to assign the direction and amount of alignment in each voxel for each material. A Euler morphology will use a set of Euler angles to assign the direction of alignment. A separate S parameter is used to denote the amount of alignment in each voxel for each material.

The two structures below assume a morphology of size [1, 256, 256] where the dimensions are [Z, Y, X], and two materials in the morphology.

Vector Morphology HDF5 Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: console

    Vector_Morphology/  
        Mat_1_alignment/   
            data = alignment vector array of shape [1, 256, 256, 3] and order [Z, Y, X, (XYZ)]
                dims = ['Z', 'Y', 'X']
        Mat_1_unaligned/
            data = volume fraction array of shape [1, 256, 256] and order [Z, Y, X]
                dims = ['Z', 'Y', 'X']
        Mat_2_alignment/   
            data = alignment vector array of shape [1, 256, 256, 3] and order [Z, Y, X, (XYZ)]
                dims = ['Z', 'Y', 'X']
        Mat_2_unaligned/
            data = volume fraction array of shape [1, 256, 256] and order [Z, Y, X]
                dims = ['Z', 'Y', 'X']

    Morphology_Parameters/
    *required*
        PhysSize/
            data = size of each voxel edge in nanometers
        NumMaterial/
            data = number of materials in morphology (integer)
    *optional*
        creation_date/
            data = date and time
        film_normal/
            data = [Z, Y, X] vector denoting the film normal direction
        morphology_creator/
            data = author of the morphology
        name/
            data = name of morphology
    


Euler Morphology HDF5 Structure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The Euler Morphology uses a ZYZ convention. Currently, Cy-RSoXS only supports uniaxial materials and the first Euler rotation (Phi) is unused. Theta is the rotation around the Y axis. Psi is the last rotation around the Z axis.

.. image:: Euler_ZYZ-v4.gif

.. code-block:: console

    Euler_Angles/
        Mat_1_Vfrac/
            data = Rotation angle array of shape [1, 256, 256] and order [Z, Y, X]
                dims = ['Z', 'Y', 'X']
        Mat_1_S/   
            data = alignment magnitude array of shape [1, 256, 256] and order [Z, Y, X]
                dims = ['Z', 'Y', 'X']
        Mat_1_Theta/
            data = Rotation angle array of shape [1, 256, 256] and order [Z, Y, X]
                dims = ['Z', 'Y', 'X']
        Mat_1_Psi/
            data = Rotation angle array of shape [1, 256, 256] and order [Z, Y, X]
                dims = ['Z', 'Y', 'X']

        Mat_2_Vfrac/
            data = Rotation angle array of shape [1, 256, 256] and order [Z, Y, X]
                dims = ['Z', 'Y', 'X']
        Mat_2_S/   
            data = alignment magnitude array of shape [1, 256, 256] and order [Z, Y, X]
                dims = ['Z', 'Y', 'X']
        Mat_2_Theta/
            data = Rotation angle array of shape [1, 256, 256] and order [Z, Y, X]
                dims = ['Z', 'Y', 'X']
        Mat_2_Psi/
            data = Rotation angle array of shape [1, 256, 256] and order [Z, Y, X]
                dims = ['Z', 'Y', 'X']

    Morphology Parameters/
    *required*
        PhysSize/
            data = size of each voxel edge in nanometers
        NumMaterial/
            data = number of materials in morphology (integer)
    *optional*
        creation_date/
            data = date and time
        film_normal/
            data = [Z, Y, X] vector denoting the film normal direction
        morphology_creator/
            data = author of the morphology
        name/
            data = name of morphology

Material Optical Constant File Structure
________________________________________

For each material in the simulation, we need a corresponding MaterialX.txt file. This file contains the optical constants at each energy for the extraordinary (Para) and ordinary (Perp) axes of the uniaxial dielectric function.

.. code-block:: console

    EnergyData0:
    {
    Energy = 275.0;
    BetaPara = 6.388392448251455e-05;
    BetaPerp = 6.303899730113871e-05;
    DeltaPara = 0.0010635346640931634;
    DeltaPerp = 0.0011221433414215483;
    }

    EnergyData1:
    {
    Energy = 275.1;
    BetaPara = 6.309144102259152e-05;
    BetaPerp = 6.304376809350212e-05;
    DeltaPara = 0.0010567115883113286;
    DeltaPerp = 0.0011157664852560843;
    }

    .
    .
    .

    EnergyData249:
    {
    Energy = 299.9;
    BetaPara = 0.0024365306249853557;
    BetaPerp = 0.0025455166691934236;
    DeltaPara = 0.0017547293997892883;
    DeltaPerp = 0.001774225207859871;
    }
    
Config.txt File Structure
_________________________

.. code-block:: console

    Energies = [Energy0, Energer morphology, 1 for Vector morphology