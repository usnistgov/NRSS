def read_material(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    energy = []
    betapara = []
    betaperp = []
    deltapara = []
    deltaperp = []
    for line in lines:
        if line.startswith('Energy '):
            split_line = line.split(' = ')
            energy.append(float(split_line[1].strip(';\n')))
        elif line.startswith('BetaPara'):
            split_line = line.split(' = ')
            betapara.append(float(split_line[1].strip(';\n')))
        elif line.startswith('BetaPerp'):
            split_line = line.split(' = ')
            betaperp.append(float(split_line[1].strip(';\n')))
        elif line.startswith('DeltaPara'):
            split_line = line.split(' = ')
            deltapara.append(float(split_line[1].strip(';\n')))
        elif line.startswith('DeltaPerp'):
            split_line = line.split(' = ')
            deltaperp.append(float(split_line[1].strip(';\n')))

    deltabeta = [[val1, val2, val3, val4] for val1, val2, val3, val4 in zip(deltapara, betapara, deltaperp, betaperp)]
    deltabeta = dict(zip(energy, deltabeta))
    return energy, deltabeta


def read_config(config_file):
    config_dict = dict()
    with open(config_file, 'r') as f:
        d = f.readlines()

    # clean and split each line, adding to config dictionary
    for line in d:
        clean_line = line.strip(';\n')
        split_line = clean_line.split(' = ')
        # test if it's a list
        if '[' in split_line[1]:
            # if it is, parse string back to real list
            config_dict[split_line[0]] = [float(val) for val in split_line[1].strip('[]').split(',')]
        else:
            # not a list, assume bool or int for now
            if 'False' in split_line[1]:
                config_dict[split_line[0]] = False
            elif 'True' in split_line[1]:
                config_dict[split_line[0]] = True
            else:
                config_dict[split_line[0]] = int(split_line[1])

    return config_dict
