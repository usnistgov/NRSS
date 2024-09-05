import numpy as np


def Rz(angle):
    return np.array([[np.cos(angle), -np.sin(angle), 0],
                     [np.sin(angle), np.cos(angle), 0],
                     [0, 0, 1]])

def Ry(angle):
    return np.array([[np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])

def create_R(theta, psi):
    rz = Rz(psi)
    ry = Ry(theta)
    return rz@ry # ignores first euler angle

def convert_nrss_opts(nrss_opts):
    uni_tensor = np.zeros((len(nrss_opts), 3, 3), dtype=complex)
    for i, key in enumerate(nrss_opts.keys()):
        uni_tensor[i, 0, 0] = complex(1-nrss_opts[key][2], nrss_opts[key][3])
        uni_tensor[i, 1, 1] = complex(1-nrss_opts[key][2], nrss_opts[key][3])
        uni_tensor[i, 2, 2] = complex(1-nrss_opts[key][0], nrss_opts[key][1])
    return uni_tensor

def calculate_abs(R, dielectric, E, tfilm, wvl):
    nrot = R@dielectric@R.T
    p = 1/4/np.pi*(nrot@nrot - np.identity(3))@E
    beta_val = 2*np.pi*np.abs(p.imag)@E
    abs_val = np.exp(-4*np.pi/wvl*beta_val*tfilm)
    return abs_val


def calculate_abs_dist(dist, dielectric, E, tfilm, wvl):
    H, x_edges, y_edges = dist
    x_centers = np.diff(x_edges) + x_edges[:-1]
    y_centers = np.diff(y_edges) + y_edges[:-1]
    abs_vals = np.zeros(len(wvl))
    for ii, theta in enumerate(x_centers):
        for jj, psi in enumerate(y_centers):
            R = create_R(theta, psi)
            abs_vals += calculate_abs(R, dielectric, E, tfilm, wvl)*H[ii,jj]
    return abs_vals/np.sum(dist[0])
