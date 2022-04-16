import numpy as np
import matplotlib.pyplot as plt
import dipole # https://github.com/klaundal/dipole
from pysymmetry.visualization import polarsubplot
from mapping import trace_from_south_to_north

RE = 6371.2e3
d2r = np.pi / 180



# MAKE MAGNETIC FIELD FUNCTIONS:
get_main_dipole = lambda x: np.vstack(dipole.generic_dipole_field(x.T, np.array([0, 0, -30000e-9]), r0 = np.array([0, 0, 0]))).T

def get_B(x):
    main_dipole = get_main_dipole(x)

    # set up a perturbation field in the longitudinal direction, with an amplitude defined as a 2D gaussian:
    r0 = 5 * RE * 1e-3      # distance to the 2D gaussian in the z = 0 plane
    sig_r = 4 * RE * 1e-3   # sigma of the 2D gaussian in the radial direction in z = 0 plane
    sig_Z =  1 * RE * 1e-3  # sigma of 2D gaussian in Z direction
    A = 50e-9               # amplitude at r = r0, z = 0

    xx, yy, zz = x
    rr = np.sqrt(xx**2 + yy**2)
    gaussian = A * np.exp(- ((rr - r0)**2/(2 * sig_r**2) + zz**2 / (2 * sig_Z**2)) )
    theta = np.arctan2(yy, xx)
    Bx = -np.sin(theta) * gaussian
    By =  np.cos(theta) * gaussian

    perturbation = np.zeros_like(main_dipole)
    perturbation[:, 0] = Bx
    perturbation[:, 1] = By
    B = main_dipole + perturbation
    return B


weimer = pd.read_csv('weimer.txt', sep = ' ', skipinitialspace=True, comment = '#', names = ['mlat', 'mlt', 'R_E', 'phi'])
weimer.phi = weimer.phi#*1e3
#weimer = pd.read_csv('pot_Anders_Ohma_082219_1_t300.csv', sep = ',', skipinitialspace=True, comment = '#', names = ['mlat', 'mlt', 'R_E', 'phi'])

V = np.vstack(( weimer[weimer.mlat > 0].phi.values, weimer[weimer.mlat < 0].phi.values)) * 1e-3
lat = np.abs(np.vstack((weimer[weimer.mlat > 0].mlat.values, weimer[weimer.mlat < 0].mlat.values)))
lon = np.abs(np.vstack((weimer[weimer.mlat > 0].mlt .values, weimer[weimer.mlat < 0].mlt .values))) * 15    
displacement = displacement_field.Displacement_field(V, lat, lon, theta0 = 40.01, Kmax = 20, Mmax = 15  , corotation_included = False, latlim = 80) 



fig, axes = plt.subplots(ncols = 2)
pp = polarsubplot.Polarsubplot(axes[0])

xx, yy = np.meshgrid(np.linspace(-1, 1, 12), np.linspace(-1, 1, 12))
#yy = np.linspace(-1, 1, 12)
#xx = np.zeros_like(yy)
iii = xx**2 + yy**2 < 1
xx, yy = xx[iii], yy[iii]
lat, mlt = pp._XYtomltMlat(xx, yy)
lat = -lat

pp.scatter(lat, mlt)
conj_lat_dip, conj_lon_dip  = trace_from_south_to_north(get_main_dipole, lat, mlt*15, height = 0, t_bound = 130 * RE * 1e18)
pp.scatter(conj_lat_dip, conj_lon_dip / 15, s = 3)

print('done dipole')

pp2 = polarsubplot.Polarsubplot(axes[1])

conj_lat, conj_lon  = trace_from_south_to_north(get_B, lat, mlt*15, height = 0, t_bound = 130 * RE * 1e32)

iii = conj_lat > 0

pp2.scatter(lat[iii], mlt[iii])
pp2.scatter(conj_lat[iii], conj_lon[iii] / 15, s = 3)
#print('mapped from %s, %s to %s, %s' % (lats, lons, list(conj_lat), list(conj_lon)))

plt.show()