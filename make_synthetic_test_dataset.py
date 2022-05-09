""" Map a Weimer potential from the south to the north, and save the output. For use later in the displacmeent_field.py script """


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dipole # https://github.com/klaundal/dipole
from polplot import pp
from mapping import trace_from_south_to_north
from scipy.interpolate import griddata

RE = 6371.2e3
d2r = np.pi / 180

MAKE_DATASET = False
EXPLORE = False
MAKE_FIELD_LINE_PLOT = False


# MAKE MAGNETIC FIELD FUNCTIONS:
get_main_dipole = lambda x: np.vstack(dipole.generic_dipole_field(x.T, np.array([0, 0, -30000e-9]), r0 = np.array([0, 0, 0]))).T

def get_B(x):
    main_dipole = get_main_dipole(x)

    # set up a perturbation field in the longitudinal direction, with an amplitude defined as a 2D gaussian:
    r0 = 10 * RE * 1e-3      # distance to the 2D gaussian in the z = 0 plane
    sig_r = 2 * RE * 1e-3   # sigma of the 2D gaussian in the radial direction in z = 0 plane
    sig_Z =  .5 * RE * 1e-3  # sigma of 2D gaussian in Z direction
    A = 50e-9               # amplitude at r = r0, z = 0

    xx, yy, zz = x
    rr = np.sqrt(xx**2 + yy**2)
    gaussian = -A * np.exp(- ((rr - r0)**2/(2 * sig_r**2) + zz**2 / (2 * sig_Z**2)) )
    theta = np.arctan2(yy, xx)
    Bx = -np.sin(theta) * gaussian
    By =  np.cos(theta) * gaussian

    perturbation = np.zeros_like(main_dipole)
    perturbation[:, 0] = Bx
    perturbation[:, 1] = By
    B = main_dipole + perturbation

    return B


if EXPLORE: # use this for testing

    fig, axes = plt.subplots(ncols = 2, nrows = 2)
    pax = pp(axes[0, 0])

    xx, yy = np.meshgrid(np.linspace(-1, 1, 16), np.linspace(-1, 1, 16))
    iii = xx**2 + yy**2 < 1
    xx, yy = xx[iii], yy[iii]
    lat, mlt = pax._XYtomltMlat(xx, yy)
    lon = mlt * 15

    conj_lat, conj_lon, _  = trace_from_south_to_north(get_B, -lat, lon, height = 0, t_bound = 130 * RE * 1e32)

    pax.scatter(lat, mlt)
    pax.scatter(conj_lat, conj_lon / 15, s = 3, color = 'C1')


    plt.show()

if MAKE_DATASET: # use this for making the dataset

    weimer = pd.read_csv('weimer/zero_tilt_zero_by.txt', sep = ' ', skipinitialspace=True, comment = '#', names = ['mlat', 'mlt', 'R_E', 'phi'])

    # ignore the most poleward elements because they are difficult to map, and my perturbation field will not
    # affect them anyway
    valid = np.abs(weimer.mlat) <= 87
    south = weimer.mlat < 0

    weimer['mapped_lat'] = np.nan
    weimer['mapped_mlt'] = np.nan

    # in north, close to the pole, fill in original values:
    weimer.loc[~valid & south, 'mapped_lat'] = -weimer.loc[~valid & south, 'mlat']
    weimer.loc[~valid & south, 'mapped_mlt'] = weimer.loc[~valid & south, 'mlt']

    # the rest we map - this takes a long time to run so go make dinner:
    conj_lat, conj_lon, _  = trace_from_south_to_north(get_B, weimer[valid & south].mlat.values, weimer[valid & south].mlt.values*15, height = 0, t_bound = 130 * RE * 1e32)

    weimer.loc[valid & south, 'mapped_lat'] = conj_lat
    weimer.loc[valid & south, 'mapped_mlt'] = conj_lon / 15

    fig, axes = plt.subplots(ncols = 2, nrows = 2)
    pax = pp(axes[0, 0])

    # replace the phi values in the northern hemisphere by the new potential, defined by mapping from the 
    # south to north. To do this, we must interpolate from the mapped coordinates to the fixed grid:

    xx, yy = pax._mltMlatToXY(weimer.mlt, weimer.mlat)
    xx_mapped, yy_mapped = pax._mltMlatToXY(weimer.mapped_mlt, weimer.mapped_lat)
    phi_mapped = griddata(np.vstack((xx_mapped[weimer.mlat < 0], yy_mapped[weimer.mlat < 0])).T, 
                          weimer.phi[weimer.mlat < 0].values, 
                          np.vstack((xx[weimer.mlat > 0], yy[weimer.mlat > 0])).T)
    weimer.loc[weimer.mlat > 0, 'phi'] = phi_mapped

    weimer.phi = weimer.phi.interpolate()

    weimer.to_csv('weimer_zero_tilt_zero_by_with_synthetic_displacement.csv')


    #now proceed with plots
    pax.contour(weimer[weimer.mlat < 0].mlat.values, weimer[weimer.mlat < 0].mlt.values, weimer[weimer.mlat < 0].phi.values, colors = 'black', levels = np.r_[-100:110:10])

    pax = pp(axes[0, 1])
    pax.contour(weimer[weimer.mlat < 0].mapped_lat.values, weimer[weimer.mlat < 0].mapped_mlt.values, weimer[weimer.mlat < 0].phi.values, colors = 'black', levels = np.r_[-100:110:10])

    dmlt = weimer.mlt - weimer.mapped_mlt
    dmlt = (dmlt + 12) % 24 - 12

    pax = pp(axes[1, 0])
    pax.contourf(weimer[weimer.mlat < 0].mapped_lat.values, weimer[weimer.mlat < 0].mapped_mlt.values, dmlt[weimer.mlat < 0].values, cmap = plt.cm.bwr, levels = np.linspace(-1, 1, 21))


    pax = pp(axes[1, 1])
    pax.contour(weimer[weimer.mlat > 0].mlat.values, weimer[weimer.mlat > 0].mlt.values, weimer[weimer.mlat > 0].phi.values, colors = 'black', levels = np.r_[-100:110:10])

    plt.show()


if MAKE_FIELD_LINE_PLOT:
    mlats = [-75, -70, -65, -60][::-1]
    kws = [{'color':'green'}, {'color':'red'}, {'color':'black'}, {'color':'black'}][::-1]
    fieldlines = []

    fig, ax = plt.subplots(ncols = 1)
    a = np.linspace(-np.pi, np.pi)
    ax.fill_between(np.cos(a), np.sin(a), color = 'grey')
    ax.plot(np.cos(a), np.sin(a), color = 'grey', linewidth = 3)
    ax.set_aspect('equal')


    for mlat, kw in zip(mlats, kws):
        conj_lat, conj_lon, fieldline = trace_from_south_to_north(get_B, mlat, 0, height = 0, t_bound = 130 * RE * 1e32)
        fieldline = fieldline.T
        print(mlat, conj_lat, conj_lon / 15)

        fieldlines.append(fieldline)

        ax.plot(fieldline[0], fieldline[2], **kw)


    for mlat, kw in zip(mlats, kws):
        conj_lat, conj_lon, fieldline = trace_from_south_to_north(get_B, mlat, -90, height = 0, t_bound = 130 * RE * 1e32)
        fieldline = fieldline.T
        print(mlat, conj_lat, conj_lon / 15)

        fieldlines.append(fieldline)

        #ax.plot(fieldline[0], fieldline[2], **kw)
        ax.plot(fieldline[0], fieldline[2], **kw)

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    xx, zz = np.meshgrid(np.linspace(xlim[0], xlim[1], 200), np.linspace(ylim[0], ylim[1], 200))
    x = np.vstack((xx.flatten(), np.zeros(xx.size), zz.flatten())) * RE * 1e-3
    dipoleB = get_main_dipole(x)
    totalB = get_B(x) - dipoleB
    By = totalB[:, 1].reshape(xx.shape) * 1e9
    By[(xx**2 + zz**2) <= 1] = np.nan

    ax.set_xlabel('r [$R_E$]')
    ax.set_ylabel('z [$R_E$]')


    ax.contourf(xx, zz, -By, levels = np.linspace(0, 50, 25), extend = 'both')






