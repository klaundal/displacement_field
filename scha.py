""" 
Code for working with spherical cap harmonic analysis 



"""

import numpy as np
import pandas as pd
from scipy.special import gamma, hyp2f1, factorial
from scipy.misc import derivative
from scipy.optimize import root_scalar

d2r = np.pi / 180


class SHkeys(object):

    def __init__(self, Nmax, Mmax):
        """ container for n and m in spherical harmonics

            keys = SHkeys(Nmax, Mmax)

            keys will behave as a tuple of tuples, more or less
            keys['n'] will return a list of the n's
            keys['m'] will return a list of the m's
            keys[3] will return the fourth n,m tuple

            keys is also iterable

        """

        self.Nmax = Nmax
        self.Mmax = Mmax
        keys = []
        for n in range(self.Nmax + 1):
            for m in range(self.Mmax + 1):
                keys.append((n, m))

        self.keys = tuple(keys)
        self.make_arrays()

    def __getitem__(self, index):
        if index == 'n':
            return [key[0] for key in self.keys]
        if index == 'm':
            return [key[1] for key in self.keys]

        return self.keys[index]

    def __iter__(self):
        for key in self.keys:
            yield key

    def __len__(self):
        return len(self.keys)

    def __repr__(self):
        return ''.join(['n, m\n'] + [str(key)[1:-1] + '\n' for key in self.keys])[:-1]

    def __str__(self):
        return ''.join(['n, m\n'] + [str(key)[1:-1] + '\n' for key in self.keys])[:-1]

    def setNmin(self, nmin):
        """ set minimum n """
        self.keys = tuple([key for key in self.keys if key[0] >= nmin])
        self.make_arrays()
        return self

    def MleN(self):
        """ set m <= n """
        self.keys = tuple([key for key in self.keys if abs(key[1]) <= key[0]])
        self.make_arrays()
        return self

    def Mge(self, limit):
        """ set m <= n """
        self.keys = tuple([key for key in self.keys if abs(key[1]) >= limit])
        self.make_arrays()
        return self

    def NminusModd(self):
        """ remove keys if n - m is even """
        self.keys = tuple([key for key in self.keys if (key[0] - abs(key[1])) % 2 == 1])
        self.make_arrays()
        return self

    def NminusMeven(self):
        """ remove keys if n - m is odd """
        self.keys = tuple([key for key in self.keys if (key[0] - abs(key[1])) % 2 == 0])
        self.make_arrays()
        return self

    def negative_m(self):
        """ add negative m to the keys """
        keys = []
        for key in self.keys:
            keys.append(key)
            if key[1] != 0:
                keys.append((key[0], -key[1]))
        
        self.keys = tuple(keys)
        self.make_arrays()
        
        return self


    def make_arrays(self):
        """ prepare arrays with shape ( 1, len(keys) )
            these are used when making G matrices
        """

        if len(self) > 0:
            self.m = np.array(self)[:, 1][np.newaxis, :]
            self.n = np.array(self)[:, 0][np.newaxis, :]
        else:
            self.m = np.array([])[np.newaxis, :]
            self.n = np.array([])[np.newaxis, :]


def Pnmx(n, m, theta):
    """ calculate Legendre function of n (not-necessarily integer),
        m (integer), and theta (colatitude in degrees) 
    """

    x = np.cos(theta * d2r)

    if m == 0:
        Knm = np.ones_like(n)
    else:
        Knm = np.sqrt(2) / (2**m * factorial(m)) * np.sqrt(gamma(n + m + 1) / gamma(n - m + 1)) 

    F   = hyp2f1(m - n, n + m + 1, 1 + m, (1 - x) / 2)  
    sintheta = np.sin(theta * np.pi / 180)
    pnm = F * Knm * sintheta ** m
    return pnm

def dPnmx_dtheta(n, m, theta):
    """ calculate d Pnm(theta) / dtheta evaluated at theta a """
    dp_of_x = lambda theta: Pnmx(n, m, theta)
    return  derivative(dp_of_x, theta ) / d2r


def get_root_table(theta0, Mmax, Kmax = False):
    """ NOT SURE WHAT NMAX SHOULD BE """

    if not Kmax:
        Kmax = Mmax 

    Nmax =  (np.pi / (theta0 * d2r) * (Mmax + .5) - .5) # 1.2 * equation (5) from Torta
    
    nm_of_k = {}

    for m in np.arange(0, Mmax + 1):
        n = np.linspace(m - 1 + 1e-9, Nmax, 1000)

        x = np.cos(theta0 * d2r)

        def find_roots(func, ns):
            """ find roots of func(n). ns is an array of n values to search for minima and maxima,
                which are used to define the brackets in which to search for roots

                returns list of roots in ascending order
            """ 
            f_of_n = func(ns)
            minima = n[(np.r_[True, f_of_n[1:] < f_of_n[:-1]] & np.r_[f_of_n[:-1] < f_of_n[1:], True]).nonzero()]
            maxima = n[(np.r_[True, f_of_n[1:] > f_of_n[:-1]] & np.r_[f_of_n[:-1] > f_of_n[1:], True]).nonzero()]

            # remove negative values from maxima and positive from minima - these will be at end points
            minima = [x for x in minima if func(x) <  0]
            maxima = [x for x in maxima if func(x) >= 0]

            brackets = np.sort(np.hstack((minima, maxima)))

            r = [root_scalar(func, bracket = (start, end), method = 'brentq').root for start, end in zip(brackets[:-1], brackets[1:])]
            return r


        rP  = find_roots(lambda _: Pnmx        (_, m, theta0), n)
        rdP = find_roots(lambda _: dPnmx_dtheta(_, m, theta0), n)
        r   = np.sort(np.hstack((rP, rdP)))

        roots = np.hstack((np.tile(np.nan, m), r))
        nm_of_k[m] =  dict(zip((np.arange(len(roots))), roots))


    nm_of_k = pd.DataFrame(nm_of_k)
    nm_of_k.index.name = 'k'
    nm_of_k = nm_of_k.loc[:Kmax, :]

    return nm_of_k


class SCH(object):
    def __init__(self, K, M, theta0 = 40., R = 6371.2, k_minus_m = 'even'):
        """
        Matrices are organized such that the columns correspond to
        the cos terms and sin terms, stacked in that order. The order
        of the terms is given by the SHkeys class. These details are 
        given for information only, it should not be necessary if this
        class is used for both inverse and forward problem. 


        parameters
        ----------
        theta0: float, optional
            colatitude of boundary. Default 40 (latitude 50)
        k_minus_m: 
            'even': k - m even, so that dV/dth = 0 at boundary
            'odd' : k - m odd, so that V = 0 at boundary
            'any' : k - m even and odd both included (V and dV/dth arbitrary)

        """

        # set up summation keys
        self.keys = SHkeys(K, M).MleN()
        self.keys_s = SHkeys(K, M).MleN()
        if k_minus_m == 'even':
            self.keys   = self.keys.NminusMeven()
            self.keys_s = self.keys_s.NminusMeven()
        if k_minus_m == 'odd':
            self.keys   = self.keys.NminusModd()
            self.keys_s = self.keys_s.NminusModd()
        self.keys_s = self.keys_s.Mge(1)

        # make table of non-integer ns
        self.n = get_root_table(theta0, M, K)
        print(K, M, self.n, '<---')

        # define the rest of the global variables
        self.R = R
        self.theta0 = theta0


    def __call__(self, lat, lon):
        """ return G matrix for SCHs at lat, lon [degrees] """

        theta = 90 - np.ravel(lat)
        phi = np.ravel(lon) * d2r

        P = {}
        for key in self.keys:
            P[key] = Pnmx(self.n.loc[key], key[1], theta)

        # matrices that correspond to cos and sin terms:
        Gc = np.hstack(np.array([(P[key] * np.cos(key[1] * phi)).reshape(-1, 1) for key in self.keys  ]))
        Gs = np.hstack(np.array([(P[key] * np.sin(key[1] * phi)).reshape(-1, 1) for key in self.keys_s]))
        
        # stack them horizontally and multiply by scale factor:
        return np.hstack((Gc, Gs)) * self.R


    def dtheta(self, lat, lon):
        """ return G matrix for SCHs differentiated with respect to 
            theta at lat, lon [degrees] """

        theta = 90 - np.ravel(lat)
        phi = np.ravel(lon) * d2r

        dP = {}
        for key in self.keys:
            dP[key] = dPnmx_dtheta(self.n.loc[key], key[1], theta)

        # matrices that correspond to cos and sin terms:
        Gc = np.hstack(np.array([(dP[key] * np.cos(key[1] * phi)).reshape(-1, 1) for key in self.keys  ]))
        Gs = np.hstack(np.array([(dP[key] * np.sin(key[1] * phi)).reshape(-1, 1) for key in self.keys_s]))
        
        # stack them horizontally and multiply by scale factor:
        return np.hstack((Gc, Gs)) * self.R

        

    def dphi(self, lat, lon):
        """ return G matrix for SCHs differentiated with respect to 
            phi at lat, lon [degrees] """
        
        theta = 90 - np.ravel(lat)
        phi = np.ravel(lon) * d2r

        P = {}
        for key in self.keys:
            P[key] = Pnmx(self.n.loc[key], key[1], theta)

        # matrices that correspond to cos and sin terms:
        Gc = np.hstack(-np.array([(P[key] * np.sin(key[1] * phi) * key[1]).reshape(-1, 1) for key in self.keys  ]))
        Gs = np.hstack( np.array([(P[key] * np.cos(key[1] * phi) * key[1]).reshape(-1, 1) for key in self.keys_s]))
        
        # stack them horizontally and multiply by scale factor:
        return np.hstack((Gc, Gs)) * self.R



class SCH_potential(object):
    def __init__(self, 
                 lat, lon, phi, 
                 theta0 = 40, Kmax = 20, Mmax = 10, 
                 alpha = 0, 
                 R = 6371.2e3 + 110e3):
        """ 
        Produce a spherical cap harmonic representation of electric potential
        based on a set of potential values (phi) defined on a set of points 
        (lat, lon). For use e.g., with Weimer potentials from CCMC which are 
        provided on a grid, or with MDH simulation output which come in a similar
        format. 

        Only SCH functions with k-m even are used, which implies that
        the meridional derivative of the potential (north-south E field)
        is zero at theta0

        NOTE: damping is applied equally to all wavelengths. probably not the
        best approach, so fixing it could be good. 


        parameters
        ----------
        lat: array
            latitudes of the data points. Must have matching size
        lon: array
            longitude of the data points. Must have matching size
        phi: array
            data points (potential)
        theta0: scalar, optional
            equatorward boundary colatitude. Default 40 (latitude 50)
        Kmax: int, optional
            truncation of degree - default 20
        Mmax: int, optional
            truncation of order - default 10
        alpha: scalar, optional
            damping parameter in the inversion. Increasingly positive numbers will give
            smoother solutions, and may help stabalize inversion if problem is ill-posed
            default 0 - no damping
        R: float, optional
            radius of sphere, in meters. Default 120 km above mean Earth radius

        """
        self.Kmax = Kmax
        self.Mmax = Mmax
        self.theta0 = theta0
        self.R = R

        self.sch = SCH(self.Kmax, self.Mmax, self.theta0, self.R, k_minus_m = 'even')

        G = self.sch(lat.flatten(), lon.flatten())
        GTG = G.T.dot(G)
        GTd = G.T.dot(phi.flatten())
        REG = alpha * np.eye(GTG.shape[0])

        self.m = np.linalg.lstsq(GTG + REG, GTd, rcond = 0)[0]

    def __call__(self, lat, lon):
        """ return potential at lat, lon """
        lat, lon = lat.flatten(), lon.flatten()
        G = self.sch(lat.flatten(), lon.flatten())
        return G.dot(self.m)

    def grad_e(self, lat, lon):
        """ return gradient in eastward direction """
        lat, lon = lat.flatten(), lon.flatten()
        Ge = self.sch.dphi(lat, lon) / np.cos(lat.reshape((-1, 1)) * d2r)
        return Ge.dot(self.m) / self.R


    def grad_n(self, lat, lon):
        """ return gradient in northward direction """
        lat, lon = lat.flatten(), lon.flatten()
        Gn = -self.sch.dtheta(lat, lon) 
        return Gn.dot(self.m) / self.R




if __name__ == '__main__':
    print('Reproducing figure 2a-e of Torta 2019')
    theta0 = 25
    Mmax = 10
    nm_of_k = get_root_table(theta0, Mmax)

    th, ph = np.linspace(0, theta0, 50), np.linspace(0, 360, 50)
    th, ph = np.meshgrid(th, ph)

    mk = [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]

    import matplotlib.pyplot as plt
    from polplot import pp
    fig = plt.figure(1, figsize = (8, 18))
    for i in range(len(mk)):
        pax = pp(fig.add_subplot(3, 2, i + 1), minlat = 90 - theta0)

        P = Pnmx(nm_of_k.loc[mk[i][1], mk[i][0]], mk[i][0], th.flatten()).reshape(th.shape)
        Y = P * np.cos(mk[i][0] * ph * np.pi / 180)
        pax.contourf(90 - th, ph / 15, Y, levels = np.linspace(-1, 1, 100), cmap = plt.cm.jet)

    plt.show()


