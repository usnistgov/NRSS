import numpy as np
from numba import njit, prange
from numba import boolean, float64, float32, int32
from numba.experimental import jitclass
from numba.typed import List
from cupyx.scipy.ndimage import gaussian_filter
import cupy as cp


@njit(float64(float64, float64, float64, float64, float64, float64, int32, int32))
def calc_distance3D(x1, y1, z1, x2, y2, z2, boxsize, boxsize_z):
    # periodic in x
    dx = abs(x1-x2)
    if dx > boxsize/2:
        min_x = min(x1,x2) + boxsize
        dx = min_x - max(x1,x2)
        
    # periodic in y
    dy = abs(y1-y2)
    if dy > boxsize/2:
        min_y = min(y1,y2) + boxsize
        dy = min_y - max(y1,y2)
        
    # periodic in z
    dz = abs(z1-z2)
    if dz > boxsize_z/2:
        min_z = min(z1,z2) + boxsize_z
        dz = min_z - max(z1,z2)
    return np.sqrt(dx**2 + dy**2 + dz**2)

@njit(boolean(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], int32, int32, int32))
def _intersect(x1, y1, z1, x2, y2, z2, min_dist, boxsize, boxsize_z):
    for i in range(len(x1)):
        for j in range(len(x2)):
            test_dist = calc_distance3D(x1[i], y1[i], z1[i], x2[j], y2[j], z2[j], boxsize, boxsize_z)
            
            if test_dist < min_dist:
                return True
    return False



spec = [('x_center',float32),
        ('y_center',float32),
        ('z_center',float32),
        ('x',float64[:]),
        ('y',float64[:]),
        ('z',float64[:]),
        ('psi',float32),
        ('theta',float32),
        ('length',float32),
        ('radius',float32),
        ('internal_radius',float32)]

@jitclass(spec)
class Fiber:
    
    def __init__(self, x_center, y_center, z_center, 
                 x, y, z, psi, theta, length, radius):
        self.x_center = x_center
        self.y_center = y_center
        self.z_center = z_center
        self.x = x
        self.y = y
        self.z = z
        self.psi = psi
        self.theta = theta
        self.length = length
        self.radius = radius
    
    # def __repr__(self):
    #     return f"Center: [{self.x[0]}, {self.y[0]}, {self.z[0]}], Length: {self.length}, Radius: {self.radius}"

    

def create_random_fiber(BoxXY, BoxZ, theta, psi, length, radius):
    r = np.linspace(-length/2, length/2, int(abs(length/2/radius+1)))
    x_center = np.random.random()*BoxXY
    y_center = np.random.random()*BoxXY
    z_center = np.random.random()*BoxZ
    x_test = (x_center + r*np.sin(theta)*np.cos(psi))%BoxXY
    y_test = (y_center + r*np.sin(theta)*np.sin(psi))%BoxXY
    z_test = (z_center + r*np.cos(theta))%BoxZ
    return Fiber(x_center, y_center, z_center, 
                 x_test, y_test, z_test, psi, 
                 theta, length, radius)


@njit()
def test_intersect(test_fiber, all_fibers, boxsize, boxsize_z):
    intersect = False
    # if np.any(test_fiber.z < test_fiber.radius):
    #     intersect = True
    #     return intersect
    radius1 = test_fiber.radius
    for k in range(len(all_fibers)):
        min_dist = all_fibers[k].radius + radius1
        if _intersect(test_fiber.x, test_fiber.y, test_fiber.z, 
                      all_fibers[k].x, all_fibers[k].y, all_fibers[k].z, 
                      min_dist, boxsize, boxsize_z):
            intersect = True
            return intersect
    return intersect

from numba.typed import List

@njit()
def create_ball_coords(radius):
    """Generates a flat, disk-shaped footprint.
    A pixel is within the neighborhood if the Euclidean distance between
    it and the origin is no greater than radius.
    Parameters
    ----------
    radius : int
        The radius of the disk-shaped footprint.
    Other Parameters
    ----------------
    dtype : data-type, optional
        The data type of the footprint.
    Returns
    -------
    footprint : ndarray
        The footprint where elements of the neighborhood are 1 and 0 otherwise.
    """
    L = np.arange(-radius, radius + 1)
    # disk_out = np.zeros(shape=(L.size,L.size),dtype=dtype)
    ball_coords = List()
    for k in range(L.size):
        for j in range(L.size):
            for i in range(L.size):
                if (L[k]**2 + L[j]**2 + L[i]**2 <= radius**2):
                    ball_coords.append((L[k], L[j], L[i]))
                
    return ball_coords



@njit()
def select_dilate_nb(xlist, ylist, zlist, coords, output_array, sizeXY, sizeZ, val):
    for i in range(xlist.shape[0]):
        for val_trio in coords:
            zcoord = int((val_trio[0]+round(zlist[i]))%sizeZ)
            ycoord = int((val_trio[1]+round(ylist[i]))%sizeXY)
            xcoord = int((val_trio[2]+round(xlist[i]))%sizeXY)
            output_array[zcoord, ycoord, xcoord] = val


            
def create_all_fiber(numfiber, radius_mu, radius_sigma, theta_mu, theta_sigma, length_lower, length_upper, BoxXY, BoxZ, lower_cutoff):
    all_fibers = List()
    psi = np.random.random()*np.pi*2
    theta = np.random.normal(theta_mu,theta_sigma)
    radius = np.random.lognormal(radius_mu, sigma=radius_sigma)
    # length= np.random.normal(length_mu, length_sigma)
    length = np.random.uniform(length_lower, length_upper)
    all_fibers.append(create_random_fiber(BoxXY, BoxZ, theta, psi, length, radius))
    for i in range(numfiber):
        psi = np.random.random()*np.pi*2
        theta = np.random.normal(theta_mu,theta_sigma)
        radius = np.random.lognormal(radius_mu, sigma=radius_sigma)
        length = np.random.uniform(length_lower, length_upper)
        test_fiber = create_random_fiber(BoxXY, BoxZ, theta, psi, length, radius)
        if not test_intersect(test_fiber, all_fibers, BoxXY, BoxZ):
            all_fibers.append(test_fiber)
    return all_fibers


def upscale_fiber(fiber,BoxXY, BoxZ):
    r = np.linspace(-fiber.length/2, fiber.length/2, int(fiber.length+1))
    fiber.x = (fiber.x_center + r*np.sin(fiber.theta)*np.cos(fiber.psi))%BoxXY
    fiber.y = (fiber.y_center + r*np.sin(fiber.theta)*np.sin(fiber.psi))%BoxXY
    fiber.z = (fiber.z_center + r*np.cos(fiber.theta))%BoxZ

@njit(parallel=True)
def dilate_all_par(fibers, all_fibers_dilated, BoxXY, BoxZ, val):
    for i in prange(len(fibers)):
        coords  = create_ball_coords(np.round(fibers[i].radius))
        select_dilate_nb(fibers[i].x, fibers[i].y, fibers[i].z, coords, all_fibers_dilated, BoxXY, BoxZ, val)

@njit(parallel=True)
def dilate_all_thetapsi(fibers, theta_dilated, psi_dilated, BoxXY, BoxZ):
    for i in prange(len(fibers)):
        coords  = create_ball_coords(np.round(fibers[i].radius))
        select_dilate_nb(fibers[i].x, fibers[i].y, fibers[i].z, coords, theta_dilated, BoxXY, BoxZ, fibers[i].theta)
        select_dilate_nb(fibers[i].x, fibers[i].y, fibers[i].z, coords, psi_dilated, BoxXY, BoxZ, fibers[i].psi)
        

@njit(parallel=True)
def dilate_all_par_hollow(fibers, all_fibers_dilated, BoxXY, BoxZ, fraction_hollow, inside_value=0):
    for i in prange(len(fibers)):
        radius_val = np.round(fibers[i].radius)
        coords  = create_ball_coords(radius_val)
        select_dilate_nb(fibers[i].x[radius_val:-radius_val], fibers[i].y[radius_val:-radius_val], fibers[i].z[radius_val:-radius_val], coords, all_fibers_dilated, BoxXY, BoxZ, 1)
        coords = create_ball_coords(np.round(fibers[i].radius*fraction_hollow))
        select_dilate_nb(fibers[i].x, fibers[i].y, fibers[i].z, coords, all_fibers_dilated, BoxXY, BoxZ, inside_value)

@njit(parallel=True)
def dilate_all_par_const(fibers, all_fibers_dilated, BoxXY, BoxZ,radius):
    for i in prange(len(fibers)):
        coords  = create_ball_coords(np.round(radius))
        select_dilate_nb(fibers[i].x, fibers[i].y, fibers[i].z, coords, all_fibers_dilated, BoxXY, BoxZ)

@njit(parallel=True)
def upscale_all_fibers(fibers, BoxXY, BoxZ):
    for i in prange(len(fibers)):
        r = np.linspace(-fibers[i].length/2, fibers[i].length/2, int(fibers[i].length+1))
        fibers[i].x = (fibers[i].x_center + r*np.sin(fibers[i].theta)*np.cos(fibers[i].psi))%BoxXY
        fibers[i].y = (fibers[i].y_center + r*np.sin(fibers[i].theta)*np.sin(fibers[i].psi))%BoxXY
        fibers[i].z = (fibers[i].z_center + r*np.cos(fibers[i].theta))%BoxZ



def find_angles(fibers, sigma=3):
    try:
        blurred = gaussian_filter(fibers, sigma=sigma)
    except TypeError:
        blurred = gaussian_filter(cp.asarray(fibers), sigma=sigma)
    grad_z, grad_y, grad_x = cp.gradient(blurred)
    orient_mag = cp.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
    grad_x /= -orient_mag 
    grad_y /= -orient_mag
    grad_z /= -orient_mag
    grad_x[cp.isinf(grad_x) | cp.isnan(grad_x)] = 0
    grad_y[cp.isinf(grad_y) | cp.isnan(grad_y)] = 0
    grad_z[cp.isinf(grad_z) | cp.isnan(grad_z)] = 0
    grad_z[grad_z > 1] = 1
    grad_z[grad_z < -1] = -1
    psi = cp.arctan2(grad_y, grad_x)
    theta = cp.arccos(grad_z)
    return theta, psi
