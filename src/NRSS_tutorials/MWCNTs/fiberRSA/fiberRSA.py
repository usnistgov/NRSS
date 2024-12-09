import numpy as np
from numba import njit, prange
from numba import boolean, float64, float32, int32
from numba.experimental import jitclass
from numba.typed import List
from cupyx.scipy.ndimage import gaussian_filter
import cupy as cp
from numba.typed import List


# Class specification used for Numba's jitclass decorator
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
class CNT:
    '''A class to hold all values and parameters for a single MWCNT.
    Parameters
    ----------
    x_center : float32
        X-position of MWCNT center point.
    y_center : float32
        Y-position of MWCNT center point.
    z_center : float32
        Z-position of MWCNT center point.
    x : float64[:]
        Set of x-positions for points sampled along the MWCNT central axis.
    y : float64[:]
        Set of y-positions for points sampled along the MWCNT central axis.
    z : float64[:]
        Set of z-positions for points sampled along the MWCNT central axis.
    psi : float32
        Euler angle describing the MWCNT rotation from vertical into the XY-plane (about the Y-axis).
    theta : float32
        Euler angle describing the MWCNT rotation about the Z-axis.
    length : float32
        Length of the MWCNT.
    radius : float32
        Radius of the MWCNT.

    Returns
    -------
    Instance of the CNT class.
    '''
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


@njit(float64(float64, float64, float64, float64, float64, float64, int32, int32))
def calc_distance3D(x1, y1, z1, x2, y2, z2, boxsize, boxsize_z):
    '''Calculates 3D distance between two points, using periodic boundary conditions
    Parameters
    ----------
    x1 : float64
        x-position of point 1
    y1 : float64
        y-position of point 1
    z1 : float64
        z-position of point 1
    x2 : float64
        x-position of point 2
    y2 : float64
        y-position of point 2
    z2 : float64
        z-position of point 2
    boxsize : int32
        size of the periodic system in x and y
    boxsize_z : int32
        size of the periodic system in z

    Returns
    -------
    distance : float64
        The Euclidean distance between two points under periodic boundary conditions
    '''
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
    '''Determines if two sets of points are within a minimum distance of one another, using periodic boundary conditions
    Parameters
    ----------
    x1 : float64[:]
        X-positions for all points in first set.
    y1 : float64[:]
        Y-positions for all points in first set.
    z1 : float64[:]
        Z-positions for all points in first set.
    x2 : float64[:]
        X-positions for all points in second set.
    y2 : float64[:]
        Y-positions for all points in second set.
    z2 : float64[:]
        Z-positions for all points in second set.
    min_dist : int32
        Minimum distance below which two points are considered intersecting.
    boxsize : int32
        Size of periodic system in x and y.
    boxsize_z : int32
        Size of periodic system in z.

    Returns
    -------
    intersection: bool
        Returns True or False to whether the two sets of points are intersecting.
    '''
    for i in range(len(x1)):
        for j in range(len(x2)):
            test_dist = calc_distance3D(x1[i], y1[i], z1[i], x2[j], y2[j], z2[j], boxsize, boxsize_z)
            
            if test_dist < min_dist:
                return True
    return False


@njit()
def test_intersect(test_CNT, all_CNTs, boxsize, boxsize_z):
    '''Tests for intersection between a newly created CNT and all CNTs that have previously been placed into the box.
    Parameters
    ----------
    test_CNT : CNT
        New CNT which will be tested for intersection with all other CNTs.
    all_CNTs : List[CNT]
        List of CNTs that have previously been placed into the box.
    boxsize : int32
        Size of box in X and Y dimensions.
    boxsize_z : int32
        Size of box in Z dimension.

    Returns
    -------
    intersect : bool
        Returns True or False as to whether the test CNT intersects with any existing CNTs.
    '''
    intersect = False
    # if np.any(test_CNT.z < test_CNT.radius):
    #     intersect = True
    #     return intersect
    radius1 = test_CNT.radius
    for k in range(len(all_CNTs)):
        min_dist = all_CNTs[k].radius + radius1
        if _intersect(test_CNT.x, test_CNT.y, test_CNT.z, 
                      all_CNTs[k].x, all_CNTs[k].y, all_CNTs[k].z, 
                      min_dist, boxsize, boxsize_z):
            intersect = True
            return intersect
    return intersect


@njit()
def create_ball_coords(radius):
    """Generates a list of coordinates that are contained within a 3D sphere with a given radius.
    Parameters
    ----------
    radius : int32
        The radius of the sphere-shaped footprint.

    Returns
    -------
    ball_coords : List
        The footprint where elements of the neighborhood are 1 and 0 otherwise.
    """
    L = np.arange(-radius, radius + 1)
    ball_coords = List()
    for k in range(L.size):
        for j in range(L.size):
            for i in range(L.size):
                if (L[k]**2 + L[j]**2 + L[i]**2 < radius**2):
                    ball_coords.append((L[k], L[j], L[i]))
                
    return ball_coords


@njit()
def select_dilate_nb(xlist, ylist, zlist, coords, output_array, sizeXY, sizeZ, val):
    '''A dilation operation that only operates on the coordinates provided, instead of the whole array.
    Parameters
    ----------
    xlist : List[float64]
        List of x-positions.
    ylist : List[float64]
        List of y-positions.
    zlist : List[float64]
        List of z-positions.
    coords : List[int32]
        List of xyz coordinates over which to perform the dilation.
    output_array : ndarray
        Array upon which the dilation operation is performed.
    sizeXY : int32
        Size of box in x and y dimensions.
    sizeZ : int32
        Size of box in z dimension.
    val : int32
        Value to assign to the dilated coordinates.
    Returns
    -------
    None
    
    '''
    for i in range(xlist.shape[0]):
        for val_trio in coords:
            zcoord = int((val_trio[0]+round(zlist[i]))%sizeZ)
            ycoord = int((val_trio[1]+round(ylist[i]))%sizeXY)
            xcoord = int((val_trio[2]+round(xlist[i]))%sizeXY)
            output_array[zcoord, ycoord, xcoord] = val


def create_random_CNT(BoxXY, BoxZ, theta, psi, length, radius):
    '''Creates a single random CNT
    Parameters
    ----------
    BoxXY : int32
        Size of the box in the x and y dimensions.
    BoxZ : int32
        Size of the box in the z dimension.
    theta : float32
        Euler angle describing the MWCNT rotation about the Z-axis.
    psi : float32
        Euler angle describing the MWCNT rotation from vertical into the XY-plane (about the Y-axis).
    length : float32
        Length of the MWCNT.
    radius : float32
        Radius of the MWCNT.

    Returns
    -------
    random_CNT : CNT
        An instance of the CNT class created with random placement in the XYZ dimensions provided.
    '''
    r = np.linspace(-length/2, length/2, int(abs(length/2/radius+1)))
    x_center = np.random.random()*BoxXY
    y_center = np.random.random()*BoxXY
    z_center = np.random.random()*BoxZ
    x_test = (x_center + r*np.sin(theta)*np.cos(psi))%BoxXY
    y_test = (y_center + r*np.sin(theta)*np.sin(psi))%BoxXY
    z_test = (z_center + r*np.cos(theta))%BoxZ
    return CNT(x_center, y_center, z_center, 
                 x_test, y_test, z_test, psi, 
                 theta, length, radius)


def create_all_CNT(num_trials, radius_mu, radius_sigma, theta_mu, theta_sigma, length_lower, length_upper, BoxXY, BoxZ):
    '''Creates a full box of CNTs
    Parameters
    ----------
    num_trials : int32
        Number of test trials.
    radius_mu : float64
        Parameter mu for the log-normal distribution in CNT radius.
    radius_sigma : float64
        Parameter sigma for the log-normal distribution in CNT radius.
    theta_mu : float64
        Mean of the normal distribution for Euler angle theta.
    theta_sigma : float64
        Standard deviation of the normal distribution for Euler angle theta.
    length_lower : float64
        Lower limit of the uniform distribution for CNT length.
    length_upper : float64
        Upper limit of the uniform distribution for CNT length.
    BoxXY : int32
        Size of box in x and y dimensions.
    BoxZ : int32
        Size of box in z dimension.

    Returns
    -------
    all_CNTs : List[CNT]
        Returns list of all CNTs placed into the box.
    '''
    all_CNTs = List()
    psi = np.random.random()*np.pi*2
    theta = np.random.normal(theta_mu,theta_sigma)
    radius = np.random.lognormal(radius_mu, sigma=radius_sigma)
    # length= np.random.normal(length_mu, length_sigma)
    length = np.random.uniform(length_lower, length_upper)
    all_CNTs.append(create_random_CNT(BoxXY, BoxZ, theta, psi, length, radius))
    for i in range(num_trials):
        psi = np.random.random()*np.pi*2
        theta = np.random.normal(theta_mu,theta_sigma)
        radius = np.random.lognormal(radius_mu, sigma=radius_sigma)
        length = np.random.uniform(length_lower, length_upper)
        test_CNT = create_random_CNT(BoxXY, BoxZ, theta, psi, length, radius)
        if not test_intersect(test_CNT, all_CNTs, BoxXY, BoxZ):
            all_CNTs.append(test_CNT)
    return all_CNTs


def upscale_CNT(CNT,BoxXY, BoxZ):
    '''Upscales CNT central axis sampling for dilation operation. Replaces coordinates in place.
    Parameters
    ----------
    CNT : CNT
        Instance of CNT class.
    BoxXY : int32
        Size of box in x and y dimensions.
    BoxZ : int32
        Size of box in z dimension.

    Returns
    -------
    None
    '''
    r = np.linspace(-CNT.length/2, CNT.length/2, int(CNTs[i].length+1))
    CNT.x = (CNT.x_center + r*np.sin(CNT.theta)*np.cos(CNT.psi))%BoxXY
    CNT.y = (CNT.y_center + r*np.sin(CNT.theta)*np.sin(CNT.psi))%BoxXY
    CNT.z = (CNT.z_center + r*np.cos(CNT.theta))%BoxZ

@njit(parallel=True)
def dilate_all_par(CNTs, all_CNTs_dilated, BoxXY, BoxZ, val):
    '''Performs the dilation operation once to create a solid CNT. The first dilation operation assigns an arbitrary value to all voxels within the CNT radius. 
    Parameters
    ----------
    CNTs : List[CNT]
        List of CNTs
    all_CNTs_dilated : ndarray
        Array of zeros into which the dilation operation will be written.
    BoxXY : int32
        Size of box in x and y dimensions.
    BoxZ : int32
        Size of box in z dimension.
    fraction_hollow : float32
        Value between 0 and 1 denoting what fraction of the CNT radius is hollow.
    val : float32
        Value to assign to voxels.

    Returns
    -------
    None
    '''
    for i in prange(len(CNTs)):
        coords  = create_ball_coords(np.round(CNTs[i].radius))
        select_dilate_nb(CNTs[i].x, CNTs[i].y, CNTs[i].z, coords, all_CNTs_dilated, BoxXY, BoxZ, val)

@njit(parallel=True)
def dilate_all_thetapsi(CNTs, theta_dilated, psi_dilated, BoxXY, BoxZ):
    '''Dilation operation that assigns theta and psi to all voxels within the CNT radius. Used to demonstrate effects of changing uniaxial dielectric function alignment from radial to coaxial.
    Parameters
    ----------
    CNTs : List[CNT]
        List of CNTs
    theta_dilated : ndarray
        Array of zeros into which the dilation operation will be written for theta values.
    psi_dilated : ndarray
        Array of zeros into which the dilation operation will be written for psi values.
    BoxXY : int32
        Size of box in x and y dimensions.
    BoxZ : int32
        Size of box in z dimension.
    fraction_hollow : float32
        Value between 0 and 1 denoting what fraction of the CNT radius is hollow.

    Returns
    -------
    None
    '''
    for i in prange(len(CNTs)):
        coords  = create_ball_coords(np.round(CNTs[i].radius))
        select_dilate_nb(CNTs[i].x, CNTs[i].y, CNTs[i].z, coords, theta_dilated, BoxXY, BoxZ, CNTs[i].theta)
        select_dilate_nb(CNTs[i].x, CNTs[i].y, CNTs[i].z, coords, psi_dilated, BoxXY, BoxZ, CNTs[i].psi)
        

@njit(parallel=True)
def dilate_all_par_hollow(CNTs, all_CNTs_dilated, BoxXY, BoxZ, fraction_hollow, inside_value=0):
    '''Performs the dilation operation twice to create a hollow CNT. The first dilation operation assigns a value of 1 to all voxels within the CNT radius. 
    The second dilation operation assigns an arbitrary value to all voxels within some fraction of the CNT radius.
    Parameters
    ----------
    CNTs : List[CNT]
        List of CNTs
    all_CNTs_dilated : ndarray
        Array of zeros into which the dilation operation will be written.
    BoxXY : int32
        Size of box in x and y dimensions.
    BoxZ : int32
        Size of box in z dimension.
    fraction_hollow : float32
        Value between 0 and 1 denoting what fraction of the CNT radius is hollow.
    inside_value : float32
        Value to assign to the hollow interior of the CNT.

    Returns
    -------
    None
    '''
    for i in prange(len(CNTs)):
        radius_val = np.round(CNTs[i].radius)
        coords  = create_ball_coords(radius_val)
        select_dilate_nb(CNTs[i].x[radius_val:-radius_val], CNTs[i].y[radius_val:-radius_val], CNTs[i].z[radius_val:-radius_val], coords, all_CNTs_dilated, BoxXY, BoxZ, 1)
        coords = create_ball_coords(np.round(CNTs[i].radius*fraction_hollow))
        select_dilate_nb(CNTs[i].x, CNTs[i].y, CNTs[i].z, coords, all_CNTs_dilated, BoxXY, BoxZ, inside_value)


@njit(parallel=True)
def upscale_all_CNTs(CNTs, BoxXY, BoxZ):
    '''Upscales central axis of all CNTs for subsequent dilation operation. Replaces coordinates in place.
    Parameters
    ----------
    CNTs : List[CNT]
        List of CNTs
    BoxXY : int32
        Size of box in x and y dimensions.
    BoxZ : int32
        Size of box in z dimension.

    Returns
    -------
    None
    '''
    for i in prange(len(CNTs)):
        r = np.linspace(-CNTs[i].length/2, CNTs[i].length/2, int(CNTs[i].length+1))
        CNTs[i].x = (CNTs[i].x_center + r*np.sin(CNTs[i].theta)*np.cos(CNTs[i].psi))%BoxXY
        CNTs[i].y = (CNTs[i].y_center + r*np.sin(CNTs[i].theta)*np.sin(CNTs[i].psi))%BoxXY
        CNTs[i].z = (CNTs[i].z_center + r*np.cos(CNTs[i].theta))%BoxZ



def find_angles(CNTs, sigma=3):
    '''Determines the radial orientation of the uniaxial dielectric function at all points in the dilated array. Uses a Gaussian blur and gradient operation to find direction normal to all points on the CNT.
    Parameters
    ----------
    CNTs : List[CNT]
        List of CNTs.
    sigma : float32
        Standard deviation of the Gaussian used in the blurring operation.
    Returns
    -------
    theta : ndarray
        Array of theta values.
    psi : ndarray
        Array of psi values.
    '''
    try:
        blurred = gaussian_filter(CNTs, sigma=sigma)
    except TypeError:
        blurred = gaussian_filter(cp.asarray(CNTs), sigma=sigma)
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
