"""NumPy check that repeated endpoint reads can be shared across distinct energies."""
import json
import numpy as np

from probe_energy_endpoints import axis, geometry


def polar_channels(t, qx, qy, qz, k):
    # Tensor order xx, xy, xz, yy, yz as in the compact backend.
    ex, ey = t[[0,1,2]], t[[1,3,4]]
    direction = np.array([qx,qy,k+qz])
    matrix = k*k*np.eye(3) - np.outer(direction,direction)
    px, py = matrix @ ex, matrix @ ey
    return np.array([np.vdot(px,px).real, np.vdot(py,py).real, 2*np.vdot(px,py).real])


def check():
    shape = (40,23,28)
    pitch = 1.5
    nz, ny, nx = shape
    qx, qy, qz = axis(nx,pitch), axis(ny,pitch), axis(nz,pitch)
    radius = (qx[None,:]**2 + qy[:,None]**2).astype(np.float32)
    border = np.ones((ny,nx),bool)
    border[:,-1] = False
    border[-1,:] = False
    energies = [288.35,283.05,550.,283.45,282.55,287.55,285.]
    fields = 4
    rng = np.random.default_rng(17017)
    response = rng.normal(size=(len(energies),5,fields))+1j*rng.normal(size=(len(energies),5,fields))
    origin = np.array(shape)//2
    spatial = rng.normal(size=(fields, *shape))
    fourier = np.fft.fftn(spatial,axes=(1,2,3))
    neighbors = ((0,0,1), (0,0,-1), (0,1,0), (0,-1,0), (1,0,0), (-1,0,0))
    fourier[:,0,0,0] = np.stack([fourier[(slice(None),*neighbor)] for neighbor in neighbors]).real.mean(axis=0)
    def read_fields(z,y,x):
        a,b,c = (origin - [z,y,x]) % shape
        return fourier[:,a,b,c]

    input_geom=[]
    for energy in energies:
        valid, low, high = geometry(radius,border,energy,nz,pitch)
        k = 2*np.pi/(1239.84197/energy)
        root = np.sqrt(np.maximum(np.float32(k*k)-radius, np.float32(0)))
        pos = -radius/(np.float32(k)+root)/np.float32(2*np.pi/(nz*pitch))
        pos = np.maximum(pos,np.float32(-(nz//2)))
        fraction = pos-np.floor(pos)
        input_geom.append((k,valid,low,high,fraction))

    expected = np.full((len(energies),ny,nx,3),np.nan)
    for ei,(k,valid,low,high,fraction) in enumerate(input_geom):
        for y,x in np.argwhere(valid):
            bins = (low[y,x],high[y,x])
            channels = [polar_channels(response[ei]@read_fields(z,y,x),qx[x],qy[y],qz[z],k) for z in bins]
            w=float(fraction[y,x])
            expected[ei,y,x] = (1-w)*channels[0] + w*channels[1]

    order = np.argsort(energies)
    raw = stored = 0
    actual = np.full_like(expected,np.nan)
    for y,x in np.ndindex(ny,nx):
        for start in range(0,len(order),4):
            cache = {}
            for ei in order[start:start+4]:
                k, valid, low, high, fraction = input_geom[ei]
                if not valid[y,x]:
                    continue
                values = []
                for z in (low[y,x],high[y,x]):
                    raw += 1
                    if z not in cache:
                        cache[z] = read_fields(z,y,x)
                        stored += 1
                    values.append(polar_channels(response[ei]@cache[z],qx[x],qy[y],qz[z],k))
                w = float(fraction[y,x])
                actual[ei,y,x]=(1-w)*values[0]+w*values[1]
    np.testing.assert_array_equal(actual,expected)
    assert 0 < stored < raw and np.nanmax(abs(actual[...,2])) > 0
    return {"energies":len(energies),"width":4,"original_endpoint_reads":raw,
            "distinct_cached_reads":stored,"output_order_and_all_three_channels_exact":True,
            "scope":"Downstream of the basis; same-bin reuse changes loads only"}


if __name__=="__main__":
    print(json.dumps(check(),indent=2))
