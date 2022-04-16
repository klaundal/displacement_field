"""
Calculate displacement field from electric potentials in 
Northern and Southern hemispheres
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from polplot import pp, sdarngrid
import spherical
#from pysymmetry.visualization import grids
import scha
import os
from scipy.interpolate import griddata
from importlib import reload
reload(scha)

d2r = np.pi / 180

R = (6371.2 + 110) * 1e3
DAMPING = 1e1#5e-2

WEIMER_FIGURE = False
SYNTHETIC_FIGURE = True  
LATLIM = 83

def shifted_coords(theta, phi, dr, arclength = False):
    """ get shifted spherical coordinates that correspond to
        r + dr, where r is described by (theta, phi)

    parameters
    ----------
    theta: colatitude (radians)
    phi: longitude (radians)
    dr: displacement vector east/north [radians] (unit sphere)
    arclength: if True, length of dr represents arc length. By default
               it represents chord length.reshape(shape)

    returns
    -------
    theta: shifted colat (radians)
    phi: shifted longitude (radians)
        shapes of theta /phi are preserved in output
    """
    shape = theta.shape
    theta = theta.flatten()
    phi = phi.flatten()

    r = np.vstack((np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)))


    if arclength: # scale length of dr to represent chord length
        chord = np.linalg.norm(dr, axis = 0) 
        theta = 2 * np.arcsin(chord / 2)
        dr = dr / theta * chord

    # stack vertical components on dr (zeros):
    dr = np.vstack((dr, np.zeros(dr.shape[1])))
    # convert to ecef:
    dr_ecef = spherical.enu_to_ecef(dr.T, phi / d2r, 90 - theta / d2r).T
    l = np.linalg.norm(dr_ecef, axis = 0) # lengths of shift vectors
    dru = dr_ecef/l # normalize
    s = np.sqrt(1/np.cos(l)**2 - 1)

    shifted = r + dru * l
    shifted = shifted / np.linalg.norm(shifted, axis = 0)
    
    x, y, z = shifted


    # calculate spherical coordinates:
    phi = np.arctan2(y, x)
    theta = np.arccos(z )

    return theta.reshape(shape), phi.reshape(shape)


class Displacement_field(object):
    def __init__(self, V, lat, lon, 
                       Kmax = 20, Mmax = 20, theta0 = 40, 
                       latlim = LATLIM,
                       R = (6371 + 110)*1e3, corotation_included = False,
                       use_input_grid = False):
        """ 

        parameters
        ----------
        V: ndarray
            array of N potential values, in kV, in the north and south. 
            Should have shape (2, N), with upper row corresponding to northern
            hemisphere, and bottom row to southern hemisphere.
        lat: ndarray
            array of N latitutdes for potentials in the north and south. 
            Should have shape (2, N), with upper row north, and lower row south
        lon: ndarray
            array of N longitudes for potentials in the north and south. 
            Should have shape (2, N), with upper row north, and lower row south
        Kmax, Mmax: ints, optional
            spherical cap harmonic degree and order truncation level
        theta0: float, optional
            equatorward boundary of potential (colatitude). Default 40 (lat 50)
        R: scalar, optional
            radius where the potential and displacement are evaluated. Default 
            is 110 km above mean Earht radius
        corotation_included: bool, optional
            set to True if corotation electric field is included in V. Default
            is False (corotation is included in the displacement field calculation)
        use_input_grid: bool, optional
            set to True if the input grid shoudl be used in the inversion. Default is 
            False. If True, the rows in lat and lon must be equal. And the grid must
            be good enough that the inversion works.


        returns
        -------
        displacement: function
            function of (mlat, mlt) which gives the displacement vector at 
            given N coordinates. Shape will be (2, N), where first row
            is east and second north in [m]
        shift: function
            function of (mlat, mlt) which gives the coordinates in the opposite
            hemisphere, taking into account the displacement.
        """

        # check that arrays are as expected
        assert lat.shape == lon.shape == V.shape
        assert lat.shape[0] == 2

        assert np.all((90 - lat) <= theta0) # no points equatorward of boundary

        self.R = R

        lat = np.abs(lat) 

        # make spherical cap harmonic representation of potentials:
        print(lat[0], lon[0], V[0], theta0, Kmax, Mmax)
        self.V_n = scha.SCH_potential(lat[0], lon[0], V[0] * 1e3, theta0 = theta0, Kmax = Kmax, Mmax = Mmax)
        self.V_s = scha.SCH_potential(lat[1], lon[1], V[1] * 1e3, theta0 = theta0, Kmax = Kmax, Mmax = Mmax)

        # functions to get the electric field:
        self.En_n = lambda la, lo: -self.V_n.grad_n(la, lo)
        self.Ee_n = lambda la, lo: -self.V_n.grad_e(la, lo)
        self.En_s = lambda la, lo: -self.V_s.grad_n(la, lo)
        self.Ee_s = lambda la, lo: -self.V_s.grad_e(la, lo)


        # data point grid
        if use_input_grid:
            assert np.all(np.equal(*lat))
            assert np.all(np.equal(*lon))
            lat, lon = lat[0], lon[0]
        else:
            datagrid, _ = sdarngrid(dlat = 1, dlon = 5, latmin = 50)
            lat, lon = datagrid

        #############
        # Inversion #
        #############

        # prepare arrays
        dVB = (self.V_s(lat, lon) - self.V_n(lat, lon)) * self.Br(lat)
        Ee, En = self.Ee_n(lat, lon), self.En_n(lat, lon)

        if not corotation_included: # add corotation electric field
            En = En + self.B(lat) * np.cos(lat * d2r) * R * 2 * np.pi / (24 * 60**2)

        # make filter for equations with very low electric field:
        E = np.sqrt(Ee**2 + En**2)
        iii = ((E > 0)).nonzero()[0]

        # Set up SCHA matrices:
        self.SCHA = scha.SCH(Kmax, Mmax, theta0 = 40.1, R = self.R, k_minus_m = 'even')
        Ge_ =  self.SCHA.dphi(  lat, lon) / np.cos(lat.reshape((-1, 1)) * d2r) / self.SCHA.R
        Gn_ = -self.SCHA.dtheta(lat, lon) / self.SCHA.R
        Ge = -Gn_ # eastward  component of r x grad(Psi)
        Gn =  Ge_ # northward component of r x grad(Psi)

        # combine:
        Gt = -Ge * Ee[:, np.newaxis] + -Gn * En[:, np.newaxis]

        # weights
        wd = np.ones(len(dVB)) 
        wd[lat > latlim] = 0 # reduced weight poleward of limit


        # stack them together (mask equations with E~0):
        W = wd[iii][:, np.newaxis]
        G = Gt[iii] * W 
        d = dVB[iii] * W.flatten()

        # solve:
        gtg = G.T.dot(G)
        gtd = G.T.dot(d)
        RR = np.eye(gtg.shape[0])

        I0 = np.linalg.lstsq(gtg + RR*DAMPING, gtd, rcond = 0)[0]
        diff = np.linalg.norm(I0)

        I = np.zeros(gtg.shape[0]) # initialize solution in case diff = 0
        while diff > 1e-5:
            error = np.abs(G.dot(I0) - d)
            error[error < .1] = .1
            RRR = np.diag(1/error.flatten())
            gtrg = G.T.dot(RRR).dot(G)
            gtrd = G.T.dot(RRR).dot(d)
            I = np.linalg.lstsq(gtrg + RR*DAMPING, gtrd, rcond = 0)[0]
            diff = np.linalg.norm(I - I0) / (1 + np.linalg.norm(I))
            I0 = I
            print(diff)

        self.I = I



    def __call__(self, mlat, mlt):
        """ calculate displacement vector at mlat, mlt 
            units will be same as units of self.R 
        """

        Ge_ =  self.SCHA.dphi(  mlat, mlt*15) / np.cos(mlat.reshape((-1, 1)) * d2r) / self.SCHA.R
        Gn_ = -self.SCHA.dtheta(mlat, mlt*15) / self.SCHA.R
        Ge = -Gn_ # eastward  component of r x grad(Psi)
        Gn =  Ge_ # northward component of r x grad(Psi)

        return Ge.dot(self.I) / self.Br(mlat), Gn.dot(self.I) / self.Br(mlat)

    def Br(self, lat):
        """ function to calculate radial magnetic field as function of latitude """
        return 2 * np.cos(np.pi/2 - lat*d2r) * 3.12e-5 * (6371.2*1e3/self.R)**3


    def B(self, lat):
        """ function to calculate magnetic field magnitude as funciton of latitude """
        #return np.mean(np.sqrt(1 + 3 * np.sin(lat * d2r) ** 2) * 3.12e-5 * (6371.2*1e3/self.R)**3)
        return np.sqrt(1 + 3 * np.sin(lat * d2r) ** 2) * 3.12e-5 * (6371.2*1e3/self.R)**3


    def conjugate_coordinates(self, mlat, mlt):
        """ calculate the conjugate coordinates of mlat, mlt
            taking into account the displacement field
        """

        de, dn = self(mlat, mlt)
        dr = np.vstack((de, dn)) / self.R
        
        colat_s, lon_s = shifted_coords(np.pi/2 - mlat * d2r, 15*mlt * d2r, dr, arclength = False)

        return 90 - colat_s / d2r, lon_s / d2r / 15



if WEIMER_FIGURE:

    # set up figures
    fig_df     = plt.figure(figsize = (13, 13))
    fig_phi    = plt.figure(figsize = (13, 13))
    fig_phi_sh = plt.figure(figsize = (13, 13))

    # set up grids
    datagrid, _ = sdarngrid(dlat = 1, dlon = 5, latmin = 50)
    lat, lon = datagrid
    
    xxx, yyy = np.meshgrid(np.r_[55:80:1], np.r_[0:360:6])
    latv, lonv = xxx.flatten(), yyy.flatten()
    vectorgrid, _ = sdarngrid(dlat = 1, dlon = 4, latmin = 60)
    latv, lonv = vectorgrid


    path = './weimer'

    for i, tilt in enumerate(['neg', 'zero', 'pos']):
        for j, by in enumerate(['neg', 'zero', 'pos']):
            fn = path + '/' + tilt + '_tilt_' + by + '_by.txt'
            weimer = pd.read_table(fn, sep = ' ', skipinitialspace=True, comment = '#', names = ['mlat', 'mlt', 'R_E', 'phi'])

            V = np.vstack(( weimer[weimer.mlat > 0].phi.values, weimer[weimer.mlat < 0].phi.values)) 
            Vlat = np.abs(np.vstack((weimer[weimer.mlat > 0].mlat.values, weimer[weimer.mlat < 0].mlat.values)))
            Vlon = np.abs(np.vstack((weimer[weimer.mlat > 0].mlt .values, weimer[weimer.mlat < 0].mlt .values))) * 15    
            print('hey')
            displacement = Displacement_field(V, Vlat, Vlon, theta0 = 40.1, corotation_included = False, Kmax = 10, Mmax = 5) 

            Vn = displacement.V_n(lat, lon) * 1e-3
            Vs = displacement.V_s(lat, lon) * 1e-3
            dV = Vn - Vs

            # set up plots:
            pax_df = pp(fig_df    .add_subplot(3, 3, i * 3 + j + 1), linewidth = .5, linestyle = '--', minlat = 60)
            pax_ph = pp(fig_phi   .add_subplot(3, 3, i * 3 + j + 1), linewidth = .5, linestyle = '--', minlat = 60)
            pax_sh = pp(fig_phi_sh.add_subplot(3, 3, i * 3 + j + 1), linewidth = .5, linestyle = '--', minlat = 60)

            # plot potentials
            pax_ph.contour (lat, lon/15, Vn, levels = np.r_[-101:101:10], colors = 'black', linewidths = 1.5)
            pax_ph.contour (lat, lon/15, Vs, levels = np.r_[-101:101:10], colors = 'grey' , linewidths = 1.5)
            pax_ph.contourf(lat, lon/15, dV, levels = np.linspace(-41, 41, 42), extend = 'both', cmap = plt.cm.bwr)

            # plot displacement field:
            dr = displacement(latv, lonv / 15)
            pax_df.plotpins(latv, lonv / 15, dr[1]/ displacement.R / d2r, dr[0] / displacement.R / d2r, SCALE = 4, unit = None, markersize = 2, linewidth = 1)

            # plot shifted potentials
            cmlat, cmlt = displacement.conjugate_coordinates(lat, lon / 15)
            Vn_new = displacement.V_n(cmlat, cmlt * 15) * 1e-3
            dV_new = Vn_new - Vs
            pax_sh.contour (lat, lon/15, Vn_new, levels = np.r_[-101:101:10], colors = 'black', linewidths = 1.5)
            pax_sh.contour (lat, lon/15, Vs    , levels = np.r_[-101:101:10], colors = 'grey' , linewidths = 1.5)
            pax_sh.contourf(lat, lon/15, dV_new, levels = np.linspace(-41, 41, 42), extend = 'both', cmap = plt.cm.bwr)

            if i == 0:
                pax_df.ax.set_title(by + ' By', size = 16)
                pax_ph.ax.set_title(by + ' By', size = 16)
                pax_sh.ax.set_title(by + ' By', size = 16)

            if j == 0:
                pax_df.write(50, 18, tilt + ' tilt', rotation = 90, ha = 'right', va = 'center', size = 16) 
                pax_ph.write(50, 18, tilt + ' tilt', rotation = 90, ha = 'right', va = 'center', size = 16) 
                pax_sh.write(50, 18, tilt + ' tilt', rotation = 90, ha = 'right', va = 'center', size = 16) 

    for fig in [fig_df, fig_phi, fig_phi_sh]:
        fig.subplots_adjust(wspace = 0.01, hspace = 0.01, top = .95, bottom = .01, left = .05, right = .99)

    fig_df.savefig('fig_displacement.png', dpi = 250)
    fig_phi.savefig('fig_original_potentials.png', dpi = 250)
    fig_phi_sh.savefig('fig_shifted_potentials.png', dpi = 250)


    plt.show()



if SYNTHETIC_FIGURE:
    fig = plt.figure(figsize = (15, 5))


    weimer = pd.read_csv('weimer_zero_tilt_zero_by_with_synthetic_displacement.csv', sep = ',', skipinitialspace=True, comment = '#')
    weimer.phi = weimer.phi*1e3
    #weimer = pd.read_csv('pot_Anders_Ohma_082219_1_t300.csv', sep = ',', skipinitialspace=True, comment = '#', names = ['mlat', 'mlt', 'R_E', 'phi'])

    V = np.vstack(( weimer[weimer.mlat > 0].phi.values, weimer[weimer.mlat < 0].phi.values)) * 1e-3
    lat = np.abs(np.vstack((weimer[weimer.mlat > 0].mlat.values, weimer[weimer.mlat < 0].mlat.values)))
    lon = np.abs(np.vstack((weimer[weimer.mlat > 0].mlt .values, weimer[weimer.mlat < 0].mlt .values))) * 15    

    print(V.shape, lat.shape, lon.shape)

    displacement = Displacement_field(V, lat, lon, theta0 = 40.1, Kmax = 15, Mmax = 15, corotation_included = False, latlim = 88) 


    # set up evaluation grids
    datagrid, _ = sdarngrid(dlat = 1, dlon = 5, latmin = 50)
    lat, lon = datagrid
    xxx, yyy = np.meshgrid(np.r_[60:85:1], np.r_[0:360:6])
    latv, lonv = xxx.flatten(), yyy.flatten()

    Vn = displacement.V_n(lat, lon) * 1e-3
    Vs = displacement.V_s(lat, lon) * 1e-3
    dV = Vn - Vs

    # set up plots:
    pax_df = pp(fig.add_subplot(131), linewidth = .5, linestyle = '--', minlat = 60)
    pax_ph = pp(fig.add_subplot(132), linewidth = .5, linestyle = '--', minlat = 60)
    pax_sh = pp(fig.add_subplot(133), linewidth = .5, linestyle = '--', minlat = 60)

    # plot potentials
    pax_ph.contour (lat, lon/15, Vn, levels = np.r_[-101:101:10], colors = 'black', linewidths = 1.5)
    pax_ph.contour (lat, lon/15, Vs, levels = np.r_[-101:101:10], colors = 'grey' , linewidths = 1.5)
    pax_ph.contourf(lat, lon/15, dV, levels = np.linspace(-41, 41, 42), extend = 'both', cmap = plt.cm.bwr)

    # plot displacement field:
    dr = displacement(latv, lonv / 15)
    pax_df.plotpins(latv, lonv / 15, dr[1]/ displacement.R / d2r, dr[0] / displacement.R / d2r, SCALE = 4, unit = None, markersize = 2, linewidth = 1)


    # plot the *true* displacement field for comparison. To do that, we need to calculate it at the vector grid locations:
    xx, yy = pax_df._mltMlatToXY(weimer.mlt, weimer.mlat)
    x, y = pax_df._mltMlatToXY(lonv / 15, latv)
    # find where the lonv/latv grid points map to by interpolating:
    lat_mapped = griddata(np.vstack((xx[weimer.mlat < 0], yy[weimer.mlat < 0])).T, 
                          weimer.mapped_lat[weimer.mlat < 0].values, 
                          np.vstack((x, y)).T)
    mlt_mapped = griddata(np.vstack((xx[weimer.mlat < 0], yy[weimer.mlat < 0])).T, 
                          weimer.mapped_mlt[weimer.mlat < 0].values, 
                          np.vstack((x, y)).T)
    # find the displacement vector:
    dr_true = spherical.tangent_vector(latv, lonv, lat_mapped, mlt_mapped * 15, degrees=True)
    r1 = np.vstack((np.cos(latv * d2r) * np.cos(lonv * d2r), np.cos(latv * d2r) * np.sin(lonv * d2r), np.sin(latv * d2r)))
    r2 = np.vstack((np.cos(lat_mapped * d2r) * np.cos(mlt_mapped * 15 * d2r), np.cos(lat_mapped * d2r) * np.sin(mlt_mapped * 15 * d2r), np.sin(lat_mapped* d2r)))
    angle = np.arccos(np.sum(r1 * r2, axis = 0)) # this is the displacement magnitude in radians

    pax_df.plotpins(latv, lonv / 15, dr_true[1] * angle / d2r, dr_true[0] * angle / d2r, SCALE = 4, unit = None, markersize = 2, linewidth = 1, color = 'red', zorder = 0)






    # plot shifted potentials
    cmlat, cmlt = displacement.conjugate_coordinates(lat, lon / 15)
    Vn_new = displacement.V_n(cmlat, cmlt * 15) * 1e-3
    dV_new = Vn_new - Vs
    pax_sh.contour (lat, lon/15, Vn_new, levels = np.r_[-101:101:10], colors = 'black', linewidths = 1.5)
    pax_sh.contour (lat, lon/15, Vs    , levels = np.r_[-101:101:10], colors = 'grey' , linewidths = 1.5)
    pax_sh.contourf(lat, lon/15, dV_new, levels = np.linspace(-41, 41, 42), extend = 'both', cmap = plt.cm.bwr)
    fig.subplots_adjust(wspace = 0.01, hspace = 0.01, top = .95, bottom = .01, left = .05, right = .99)

    plt.show()
