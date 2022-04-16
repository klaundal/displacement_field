import numpy as np
from dipole.dipole import car_to_sph # https://github.com/klaundal/dipole
from scipy.integrate import RK45

RE = 6371.2e3
d2r = np.pi / 180


def trace_from_south_to_north(magnetic_field, lat, lon, **kwargs):
    """ 
    Trace along magnetic field lines from the southern hemisphere to the
    northern hemisphere.

    lat and lon should refer to the same coordinate system as used by the 
    magnetic_field function

    Parameters:
    -----------
    magnetic_field: function
        function of xyz [m] (ECEF coordinates) that returns the three
        ECEF componets of the magnetic field in nT
    lat: array
        array of latitudes [deg] from which to trace
    lon: array
        array of longitudes [deg] from which to trace
    height: float, optional
        height above Earth radius from / to which to map. Default is zero (ground)
    """

    # extract the mapping height from keyword arguments - if given:
    if 'height' in kwargs:
        height = kwargs.pop('height')
    else:
        height = 0

    # the rest of the keyword arguments should be related to the RK45 integration:
    if 't_bound' not in kwargs.keys():
        kwargs['t_bound'] = 130e3 * RE
    if 'first_step' not in kwargs.keys():
        kwargs['first_step'] = 1e2
    if 'atol' not in kwargs.keys():
        kwargs['atol'] = 1e-14
    if 'rtol' not in kwargs.keys():
        kwargs['rtol'] = 1e-9
    if 'vectorized' not in kwargs.keys():
        kwargs['vectorized'] = True


    # make sure that lat and lon are broacastable:
    lat, lon = np.array(lat), np.array(lon)
    lat = lat * np.ones_like(lon)
    lon = lon * np.ones_like(lat)
    shape = lat.shape

    th = (90 - lat.flatten()) * d2r
    ph = lon.flatten() * d2r
    R = (RE + height) * 1e-3

    points = R * np.vstack((np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)))

    # loop through every point and find the coordinates of the conjugate point:
    conj_lat, conj_lon = [], []
    for r0 in points.T:

        get_B = lambda t, xyz: np.array(magnetic_field(xyz)).reshape(xyz.shape)
        rk = RK45(get_B, 0, r0, **kwargs)

        rs = [r0]
        while True:
            try:
                rk.step()
            except:
                'RK step failed. Breaking out of loop and returning nan'
                rs.append(rk.y*np.nan)
                break
            rs.append(rk.y)
            if np.linalg.norm(rs[-1]) < R or rk.status == 'finished':
                break

        # Find point between the last two points which has length R. That's the intersection. 
        # This amounts to solving a quadratic equation
        rs = np.array(rs) / R

        a = np.linalg.norm(rs[-2] - rs[-1])**2
        b = 2 * np.sum(rs[-1] * (rs[-2] - rs[-1]))
        c = np.linalg.norm(rs[-1])**2 - 1

        t = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a) # choosing the root that gives the correct intesection

        intersection = rs[-1] + t * (rs[-2] - rs[-1])

        r, theta, phi = car_to_sph(intersection)
        conj_lat.append( 90 - theta )
        conj_lon.append( phi )

    return np.array(conj_lat).reshape(shape), np.array(conj_lon).reshape(shape), rs



