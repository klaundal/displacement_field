import numpy as np
from scipy.integrate import RK45
import matplotlib.pyplot as plt


RE = 6371.2e3
d2r = np.pi / 180

def dipole_B(r, theta, B0 = 2.98e-5):
    """ return dipole magnetic field at radius r [m] and colatitude theta [degrees] 

        Return values are in T, in the r (radial) and theta (geocentric south) directions
    """

    theta = theta * d2r

    C = B0 * (r / RE)**3
    Br     = -2 * C * np.cos(theta)
    Btheta =     -C * np.sin(theta)

    return(Br, Btheta)



def dipole_B_Cartesian(x, y, B0 = 2.98e-5):
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(x , y)

    Br, Btheta = dipole_B(r, theta / d2r, B0 = B0)    

    Bx = Br * np.sin(theta) + Btheta * np.cos(theta)
    By = Br * np.cos(theta) - Btheta * np.sin(theta)

    return(Bx, By)



y0 = RE * np.array([np.sin(170 * d2r), np.cos(170 * d2r)])
t0 = 0

rk = RK45(lambda t, y: dipole_B_Cartesian(y[0], y[1]), t0, y0, 130e3 * RE, first_step = 1e3, rtol = 1e-8, vectorized = True)

ys = [y0]
while True:
    rk.step()
    ys.append(rk.y)
    if np.linalg.norm(ys[-1]) <= RE or rk.status == 'finished':
        break

ys = np.array(ys).T


fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(ys[0] / RE, ys[1] / RE)
plt.show()