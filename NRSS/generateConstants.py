#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# """
# Created on Mon Dec 16 14:27:24 2019

# @author: maksbh
# """


import numpy as np



def find_nearest(array, value):
    """
    Function to find the nearest index 

    Parameters
    ----------

    array : Numpy array
    value : value of energy

    Returns
    -------
    idx : Integer
        index location corresponding to the closest location
    """
    idx = (np.abs(array - value)).argmin()
    return idx


def get_interpolated_value(array, value, nearest_id, energy_id):
    """
    Function to get the interpolated value

    Parameters
    ----------

    array : Numpy array
    value : value of energy
    nearest_id : id corresponding to the nearest value

    Returns
    -------
    valArray : Numpy array
            array of the interpolated values
    """
    valArray = np.zeros(array.shape[1])
    if (array[nearest_id][energy_id] > value):
        xp = [array[nearest_id - 1][energy_id], array[nearest_id][energy_id]]
        for i in range(0, array.shape[1]):
            yp = [array[nearest_id - 1][i], array[nearest_id][i]]
            valArray[i] = np.interp(value, xp, yp)

    elif (array[nearest_id][energy_id] < value):
        xp = [array[nearest_id][energy_id], array[nearest_id + 1][energy_id]]
        for i in range(0, array.shape[1]):
            yp = [array[nearest_id][i], array[nearest_id + 1][i]]
            valArray[i] = np.interp(value, xp, yp)

    else:
        for i in range(0, len(valArray)):
            valArray[i] = array[nearest_id][i]

    return valArray


def removeDuplicates(Data, energy_id):
    """
    Function to remove duplicate energies
    
    Parameters
    ----------
    
    Data : Numpy array
    energy_id : int

    Returns
    -------
    
    listOut : Numpy array

    """
    listIn = Data.tolist()
    listOut = []
    listOut.append(listIn[0])
    currEnergy = listIn[0][energy_id]
    duplicateFound = False
    for i in range(1, len(listIn)):
        if (listIn[i][energy_id] == currEnergy):
            duplicateFound = True
            continue
        else:
            listOut.append(listIn[i])
            currEnergy = listIn[i][energy_id]

    if (duplicateFound):
        print('Duplicates in Energy found. Removing it')
    return (np.array(listOut))


def dump_dataVacuum(index, energy, f):
    """
    Function to write vacuum optical contants (n = 1 + i0)

    Parameters
    ----------

    index : int
    energy : float

    """
    Header = "EnergyData" + str(index) + ":\n{\n"
    f.write(Header)
    Energy = "Energy = " + str(energy) + ";\n"
    f.write(Energy)
    BetaPara = "BetaPara = " + str(0.0) + ";\n"
    f.write(BetaPara)
    BetaPerp = "BetaPerp = " + str(0.0) + ";\n"
    f.write(BetaPerp)
    DeltaPara = "DeltaPara = " + str(0.0) + ";\n"
    f.write(DeltaPara)
    DeltaPerp = "DeltaPerp = " + str(0.0) + ";\n"
    f.write(DeltaPerp)
    f.write("}\n")


def dump_data(valArray, index, labelEnergy, f):
    """
    Function to write material optical constants to file

    Parameters
    ----------

    valArray : Numpy array
    index : int
    labelEnergy : dict
    f : file object


    """
    Header = "EnergyData" + str(index) + ":\n{\n";
    f.write(Header)
    Energy = "Energy = " + str(valArray[labelEnergy["Energy"]]) + ";\n"
    f.write(Energy)
    BetaPara = "BetaPara = " + str(valArray[labelEnergy["BetaPara"]]) + ";\n"
    f.write(BetaPara)
    BetaPerp = "BetaPerp = " + str(valArray[labelEnergy["BetaPerp"]]) + ";\n"
    f.write(BetaPerp)
    DeltaPara = "DeltaPara = " + str(valArray[labelEnergy["DeltaPara"]]) + ";\n"
    f.write(DeltaPara)
    DeltaPerp = "DeltaPerp = " + str(valArray[labelEnergy["DeltaPerp"]]) + ";\n"
    f.write(DeltaPerp)
    f.write("}\n")


def writeList(name: str, value: list, file):
    """
    Function to write list to file

    Parameters
    ----------

    name : str
    value : list
    file : file object

    """
    valStr: str = name + "["
    for i in range(len(value) - 1):
        valStr = valStr + str(value[i]) + ","
    valStr = valStr + str(value[len(value) - 1])
    file.write(valStr + "];\n")


def write_materials(energies, dict, labelEnergy, numMaterial):
    """
    Function to write optical constants for all energies supplied

    Parameters
    ----------

    energies : Numpy array
    dict : dict
    labelEnergy : dict
    numMaterial : int


    """
    NumEnergy = len(energies)

    for numMat in range(1, numMaterial+1):
        f = open("Material" + str(numMat) + ".txt", "w")
        fname = dict["Material" + str(numMat)]
        if (fname != 'vacuum'):
            Data = np.loadtxt(fname, skiprows=1)
            Data = Data[Data[:, labelEnergy["Energy"]].argsort()]
            Data = removeDuplicates(Data, labelEnergy["Energy"])
            for i in range(0, NumEnergy):
                currentEnergy = energies[i]
                nearest_id = find_nearest(Data[:, labelEnergy["Energy"]], currentEnergy)
                ValArray = get_interpolated_value(Data, currentEnergy, nearest_id, labelEnergy["Energy"])
                dump_data(ValArray, i, labelEnergy, f)

        else:
            for i in range(0, NumEnergy):
                currentEnergy = energies[i]
                dump_dataVacuum(i, currentEnergy, f)
        f.close()