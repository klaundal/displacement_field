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
from mapping import trace_from_south_to_north
from make_synthetic_test_dataset import get_B, get_main_dipole # needed to make Figure 1
from scipy.interpolate import griddata
from importlib import reload
reload(scha)

d2r = np.pi / 180
RE = 6371.2e3

R = (6371.2 + 110) * 1e3
DAMPING = 1e5#5e-2

WEIMER_FIGURES = True # set to True to make Figures 3-5
SYNTHETIC_FIGURES = False # set to True to make Figures 1-2
LATLIM = 78

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
                       Kmax = 20, Mmax = 20, theta0 = 30, 
                       latlim = LATLIM,
                       R = (6371 + 110)*1e3, corotation_included = False,
                       use_input_grid = False,
                       B0 = 28000e-9):
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

        self.R  = R
        self.B0 = B0
        self.latlim = latlim

        lat = np.abs(lat) 

        if not corotation_included: # add corotation electric field
            Vc = 2 * np.pi / (24 * 60**2) * self.B0 * RE**3 / R * np.cos(lat * d2r) **2 * 1e-3 # kV
            self.Vc = Vc
            V = V + Vc


        # make spherical cap harmonic representation of potentials:
        self.V_n = scha.SCH_potential(lat[0], lon[0], V[0] * 1e3, theta0 = theta0, Kmax = Kmax, Mmax = Mmax)
        self.V_s = scha.SCH_potential(lat[1], lon[1], V[1] * 1e3, theta0 = theta0, Kmax = Kmax, Mmax = Mmax)


        # functions to get partial_derivatives:
        self.Vlambda_n = lambda la, lo: self.V_n.d_dlambda(la, lo)
        self.Vphi_n    = lambda la, lo: self.V_n.d_dphi(   la, lo)
        self.Vlambda_s = lambda la, lo: self.V_s.d_dlambda(la, lo)
        self.Vphi_s    = lambda la, lo: self.V_s.d_dphi(   la, lo)

        # data point grid
        if use_input_grid:
            assert np.all(np.equal(*lat))
            assert np.all(np.equal(*lon))
            lat, lon = lat[0], lon[0]
        else:
            datagrid, _ = sdarngrid(dlat = .5, dlon = 5, latmin = 60)
            lat, lon = datagrid

        #############
        # Inversion #
        #############

        # prepare arrays
        dV = (self.V_n(lat, lon) - self.V_s(lat, lon))
        V_phi, V_lambda = self.Vphi_s(lat, lon), self.Vlambda_s(lat, lon)

        sinIm = 2 * np.sin(lat * d2r) / np.sqrt(4 - 3 * np.cos(lat * d2r)**2)
        Ee, En = -self.Vphi_s(lat, lon) / (self.V_s.R * np.cos(lat * d2r)), self.Vlambda_s(lat, lon) / (self.V_s.R * sinIm)

        self.E_s = np.hstack((Ee, En))
        self.grid = np.hstack((lon, lat))

        # Electric field magnitude (used below for filtering)
        E = np.sqrt(Ee**2 + En**2)

        # Set up SCHA matrices:
        self.SCHA = scha.SCH(Kmax, Mmax, theta0 = 30, R = self.R, k_minus_m = 'even')
        Gphi    =  self.SCHA.dphi(  lat, lon) 
        Glambda = -self.SCHA.dtheta(lat, lon) 

        # Boundary:
        blat = np.full(0, 60)
        blon = np.linspace(0, 360, blat.size)
        Gphi_b    =  self.SCHA.dphi(  blat, blon) 
        Glambda_b = -self.SCHA.dtheta(blat, blon) 


        # combine:
        scaling = 1 / (self.SCHA.R**2 * self.B0 * np.sin(2 * lat.reshape((-1, 1)) * d2r))
        Gt = scaling * (Gphi * V_lambda[:, np.newaxis] - Glambda * V_phi[:, np.newaxis])


        # remove equations poleward of latitude limit and with weak electric fields
        iii = ((E > 0) & (lat < self.latlim)).nonzero()[0]

        # weights
        wd = np.ones(len(dV)) 


        # stack them together (mask invalid equations   ):
        W = wd[iii][:, np.newaxis]
        #print(Gt.shape, dV.shape, W.shape, '<<< shapes')
        G = Gt[iii] * W 
        d = dV[iii] * W.flatten()

        # add boundary elements:
        G = np.vstack((G, Gphi_b, Glambda_b))
        d = np.hstack((d, np.zeros(blon.size * 2)))
        #print(d.max(), d.min())

        # solve:
        gtg = G.T.dot(G)
        gtd = G.T.dot(d)
        RR = np.eye(gtg.shape[0])

        I0 = np.linalg.lstsq(gtg + RR*DAMPING, gtd, rcond = 0)[0]
        diff = np.linalg.norm(I0)

        I = np.zeros(gtg.shape[0]) # initialize solution in case diff = 0
        while diff > 1e-3:
            error = np.abs(G.dot(I0) - d)
            error[error < 1e0] = 1e0
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

        sinIm = 2 * np.sin(mlat * d2r) / np.sqrt(4 - 3 * np.cos(mlat * d2r)**2)
        Be3 = self.B0 * np.sqrt(4 - 3 * np.cos(mlat * d2r)**2)
        Gphi    =  self.SCHA.dphi(  mlat, mlt * 15)
        Glambda = -self.SCHA.dtheta(mlat, mlt * 15)

        delta_e1 = Glambda.dot(self.I) / (Be3 * self.R * sinIm)
        delta_e2 = Gphi.dot(   self.I) / (Be3 * self.R * np.cos(mlat * d2r))

        delta_east = delta_e1
        delta_north = -sinIm * delta_e2

        return delta_east, delta_north


    def conjugate_coordinates(self, mlat, mlt):
        """ calculate the conjugate coordinates of mlat, mlt
            taking into account the displacement field
        """

        de, dn = self(mlat, mlt)
        dr = np.vstack((de, dn)) / self.R
        
        colat_s, lon_s = shifted_coords(np.pi/2 - mlat * d2r, 15*mlt * d2r, dr, arclength = False)

        return 90 - colat_s / d2r, lon_s / d2r / 15



if WEIMER_FIGURES:
    dV_scale = np.linspace(-20, 20, 22) # plotting scale for potential mismatch

    # set up figures
    fig_df     = plt.figure(figsize = (13, 13))
    fig_phi    = plt.figure(figsize = (13, 13))
    fig_phi_sh = plt.figure(figsize = (13, 13))

    # set up grids
    datagrid, _ = sdarngrid(dlat = .25, dlon = 1, latmin = 60)
    lat, lon = datagrid
    
    vectorgrid, _ = sdarngrid(dlat = 1.5, dlon = 5, latmin = 60)
    latv, lonv = vectorgrid


    path = './weimer'

    for i, tilt in enumerate(['neg', 'zero', 'pos']):
        for j, by in enumerate(['neg', 'zero', 'pos']):

            fn = path + '/' + tilt + '_tilt_' + by + '_by.txt'
            weimer = pd.read_table(fn, sep = ' ', skipinitialspace=True, comment = '#', names = ['mlat', 'mlt', 'R_E', 'phi'])
            weimer = weimer[np.abs(weimer.mlat) > 60]

            V = np.vstack(( weimer[weimer.mlat > 0].phi.values, weimer[weimer.mlat < 0].phi.values)) 
            Vlat = np.abs(np.vstack((weimer[weimer.mlat > 0].mlat.values, weimer[weimer.mlat < 0].mlat.values)))
            Vlon = np.abs(np.vstack((weimer[weimer.mlat > 0].mlt .values, weimer[weimer.mlat < 0].mlt .values))) * 15    
            displacement = Displacement_field(V, Vlat, Vlon, theta0 = 30, corotation_included = False, Kmax = 25, Mmax = 3, latlim = 78) 

            Vn = displacement.V_n(lat, lon) * 1e-3
            Vs = displacement.V_s(lat, lon) * 1e-3
            dV = Vn - Vs

            # set up plots:
            pax_df = pp(fig_df    .add_subplot(3, 3, i * 3 + j + 1), linewidth = .5, linestyle = '--', minlat = 60)
            pax_ph = pp(fig_phi   .add_subplot(3, 3, i * 3 + j + 1), linewidth = .5, linestyle = '--', minlat = 60)
            pax_sh = pp(fig_phi_sh.add_subplot(3, 3, i * 3 + j + 1), linewidth = .5, linestyle = '--', minlat = 60)

            if i == j == 0:
                for pax in [pax_df, pax_ph, pax_sh]:
                    pax.writeMLTlabels(mlat = 60)
                    for lat_ in [60, 70, 80]:
                        pax.write(lat_, 3, str(lat_) + '$^\circ$', rotation = 45, bbox=dict(facecolor='white', linewidth = 0, alpha=0.5))

            # plot potentials
            pax_ph.contour (lat, lon/15, Vn, levels = np.r_[-101:101:10], colors = 'black', linewidths = 1.5)
            pax_ph.contour (lat, lon/15, Vs, levels = np.r_[-101:101:10], colors = 'grey' , linewidths = 1.5)
            pax_ph.contourf(lat, lon/15, dV, levels = dV_scale, extend = 'both', cmap = plt.cm.bwr)

            # plot displacement field:
            dr = displacement(latv, lonv / 15)
            iii = latv < displacement.latlim
            pax_df.plotpins(latv[iii], lonv[iii] / 15, dr[1][iii]/ displacement.R / d2r, dr[0][iii] / displacement.R / d2r, SCALE = 4, unit = None, markersize = 2, linewidth = 1)

            ## plot the dot product between E and delta:
            #Ee, En = np.split(displacement.E_s, 2)
            #lon_grid, lat_grid = np.split(displacement.grid, 2)
            #dr_grid = displacement(lat_grid, lon_grid / 15)
            #iii = lat_grid < displacement.latlim
            #E_dot_delta = np.abs(Ee * dr_grid[0] + En * dr_grid[1])
            #E_dot_delta[~iii] = np.nan # mask points poleward of the latitude limit
            #pax_df.contourf(lat_grid, lon_grid/15, E_dot_delta)


            # plot shifted potentials
            cmlat, cmlt = displacement.conjugate_coordinates(lat, lon / 15)
            Vn_new = displacement.V_n(cmlat, cmlt * 15) * 1e-3
            dV_new = Vn_new - Vs
            pax_sh.contour (lat, lon/15, Vn_new, levels = np.r_[-101:101:10], colors = 'black', linewidths = 1.5)
            pax_sh.contour (lat, lon/15, Vs    , levels = np.r_[-101:101:10], colors = 'grey' , linewidths = 1.5)
            pax_sh.contourf(lat, lon/15, dV_new, levels = dV_scale, extend = 'both', cmap = plt.cm.bwr)

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

    fig_df.savefig(    './figures/figure_5.png', dpi = 250)
    fig_df.savefig(    './figures/figure_5.pdf')
    fig_phi.savefig(   './figures/figure_3.png', dpi = 250)
    fig_phi.savefig(   './figures/figure_3.pdf')
    fig_phi_sh.savefig('./figures/figure_4.png', dpi = 250)
    fig_phi_sh.savefig('./figures/figure_4.pdf')


    plt.show()



if SYNTHETIC_FIGURES:
    dV_scale = np.linspace(-20, 20, 22) # plotting scale for potential mismatch


    # set up subplots for figure 1:
    fig1 = plt.figure(figsize = (8, 9))
    ax_fieldlines   = plt.subplot2grid((2, 21), (0,  0), colspan = 20)
    ax_field_cbar   = plt.subplot2grid((2, 21), (0, 20))
    ax_displacement = plt.subplot2grid((2, 21), (1,  0), colspan = 10)
    ax_potentials   = plt.subplot2grid((2, 21), (1, 10), colspan = 10)
    ax_cbar         = plt.subplot2grid((2, 21), (1, 20))
    ax_displacement = pp(ax_displacement, linewidth = .5, linestyle = '--', minlat = 60)
    ax_potentials   = pp(ax_potentials  , linewidth = .5, linestyle = '--', minlat = 60)

    # set up subplots for figure 2:
    fig2 = plt.figure(figsize = (15,  5))
    pax_df = pp(fig2.add_subplot(131), linewidth = .5, linestyle = '--', minlat = 60)
    pax_ph = pp(fig2.add_subplot(132), linewidth = .5, linestyle = '--', minlat = 60)
    pax_sh = pp(fig2.add_subplot(133), linewidth = .5, linestyle = '--', minlat = 60)




    weimer = pd.read_csv('weimer_zero_tilt_zero_by_with_synthetic_displacement.csv', sep = ',', skipinitialspace=True, comment = '#')
    weimer = weimer[np.abs(weimer.mlat) > 60]
    weimer.phi = weimer.phi*1e3

    V = np.vstack(( weimer[weimer.mlat > 0].phi.values, weimer[weimer.mlat < 0].phi.values)) * 1e-3
    lat = np.abs(np.vstack((weimer[weimer.mlat > 0].mlat.values, weimer[weimer.mlat < 0].mlat.values)))
    lon = np.abs(np.vstack((weimer[weimer.mlat > 0].mlt .values, weimer[weimer.mlat < 0].mlt .values))) * 15    

    print(V.shape, lat.shape, lon.shape)

    displacement = Displacement_field(V, lat, lon, theta0 = 30, Kmax = 25, Mmax = 3, corotation_included = False, latlim = 90) 


    # set up evaluation grids
    datagrid, _ = sdarngrid(dlat = .25, dlon = 1, latmin = 60)
    lat, lon = datagrid
    vectorgrid, _ = sdarngrid(dlat = 1.5, dlon = 5, latmin = 60)
    latv, lonv = vectorgrid

    Vn = displacement.V_n(lat, lon) * 1e-3
    Vs = displacement.V_s(lat, lon) * 1e-3
    dV = Vn - Vs


    # plot potentials
    for ax in [pax_ph, ax_potentials]:
        ax.contour (lat, lon/15, Vn, levels = np.r_[-101:101:10], colors = 'black', linewidths = 1.5)
        ax.contour (lat, lon/15, Vs, levels = np.r_[-101:101:10], colors = 'grey' , linewidths = 1.5)
        ax.contourf(lat, lon/15, dV, levels = dV_scale, extend = 'both', cmap = plt.cm.bwr)

    # plot displacement field:
    dr = displacement(latv, lonv / 15)
    pax_df.plotpins(latv, lonv / 15, dr[1]/ displacement.R / d2r, dr[0] / displacement.R / d2r, SCALE = 4, unit = None, markersize = 2, linewidth = 1, colors = 'C1', markercolor = 'C1')


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

    for ax in [pax_df, ax_displacement]:
        ax.plotpins(latv, lonv / 15, dr_true[1] * angle / d2r, dr_true[0] * angle / d2r, SCALE = 4, unit = None, markersize = 2, linewidth = 1, color = 'C0', markercolor = 'C0', zorder = 0)




    # plot shifted potentials
    cmlat, cmlt = displacement.conjugate_coordinates(lat, lon / 15)
    Vn_new = displacement.V_n(cmlat, cmlt * 15) * 1e-3
    dV_new = Vn_new - Vs

    pax_sh.contour (lat, lon/15, Vn_new, levels = np.r_[-101:101:10], colors = 'black', linewidths = 1.5)
    pax_sh.contour (lat, lon/15, Vs    , levels = np.r_[-101:101:10], colors = 'grey' , linewidths = 1.5)
    pax_sh.contourf(lat, lon/15, dV_new, levels = dV_scale, extend = 'both', cmap = plt.cm.bwr)

    # make colorbar in figure 1:
    ax_cbar.contourf(np.vstack((np.zeros(dV_scale.size), np.ones(dV_scale.size))),
                     np.vstack((dV_scale, dV_scale)),
                     np.vstack((dV_scale, dV_scale)),
                     cmap = plt.cm.bwr, levels = dV_scale)
    ax_cbar.set_xticks([])
    ax_cbar.set_ylabel('Potential mismatch $V_n - V_s$ [kV]')
    ax_cbar.yaxis.set_label_position("right")
    ax_cbar.yaxis.tick_right()

    # make the field line panel in Figure 1:
    mlats = [-75, -70, -65, -60][::-1]
    kws = [{'color':'green'}, {'color':'red'}, {'color':'black'}, {'color':'black'}][::-1]

    a = np.linspace(-np.pi, np.pi)
    ax_fieldlines.fill_between(np.cos(a), np.sin(a), color = 'grey')
    ax_fieldlines.plot(np.cos(a), np.sin(a), color = 'grey', linewidth = 3)
    ax_fieldlines.set_aspect('equal')


    for mlat, kw in zip(mlats, kws):
        conj_lat, conj_lon, fieldline = trace_from_south_to_north(get_B, mlat, 0, height = 0, t_bound = 130 * RE * 1e32)
        fieldline = fieldline.T
        print(mlat, conj_lat, conj_lon / 15)

        ax_fieldlines.plot(fieldline[0], fieldline[2], **kw)


    for mlat, kw in zip(mlats, kws):
        conj_lat, conj_lon, fieldline = trace_from_south_to_north(get_B, mlat, -90, height = 0, t_bound = 130 * RE * 1e32)
        fieldline = fieldline.T
        print(mlat, conj_lat, conj_lon / 15)

        ax_fieldlines.plot(fieldline[0], fieldline[2], **kw)

    xlim, ylim = ax_fieldlines.get_xlim(), ax_fieldlines.get_ylim()
    xx, zz = np.meshgrid(np.linspace(xlim[0], xlim[1], 200), np.linspace(ylim[0], ylim[1], 200))
    x = np.vstack((xx.flatten(), np.zeros(xx.size), zz.flatten())) * RE * 1e-3
    dipoleB = get_main_dipole(x)
    totalB = get_B(x) - dipoleB # this is the perturbation magnetic field
    By = totalB[:, 1].reshape(xx.shape) * 1e9 # convert to nT
    By[(xx**2 + zz**2) <= 1] = np.nan

    ax_fieldlines.set_xlabel('r [$R_E$]')
    ax_fieldlines.set_ylabel('z [$R_E$]')

    l = np.linspace(0, 50, 25)
    ax_fieldlines.contourf(xx, zz, -By, levels = l, extend = 'both')

    # make the colorbar for the magnetic field
    ax_field_cbar.contourf(np.vstack((np.zeros(l.size), np.ones(l.size))), np.vstack((l, l)), np.vstack((l, l)), levels = l)
    ax_field_cbar.set_xticks([])
    ax_field_cbar.set_ylabel('Perturbation magnetic field (westward) [nT]')
    ax_field_cbar.yaxis.set_label_position("right")
    ax_field_cbar.yaxis.tick_right()


    # Titles for fig 1 panels:
    ax_fieldlines.set_title('A) Synthetic example: Dipole + perturbation field')
    ax_displacement.ax.set_title('B) Displacement field')
    ax_potentials.ax.set_title('C) Example potential mismatch')

    # Titles for fig 2 panels:
    pax_df.ax.set_title('A) Displacement field')
    pax_ph.ax.set_title('B) Potentials and mismatch')
    pax_sh.ax.set_title('C) Corrected potentials')


    fig1.subplots_adjust(top = .95, left = .05, bottom = 0.01)
    fig2.subplots_adjust(wspace = 0.01, hspace = 0.01, top = .95, bottom = .01, left = .05, right = .99)


    fig1.savefig('./figures/figure_1.png', dpi = 250)
    fig2.savefig('./figures/figure_2.png', dpi = 250)
    fig1.savefig('./figures/figure_1.pdf')
    fig2.savefig('./figures/figure_2.pdf')


    plt.show()
