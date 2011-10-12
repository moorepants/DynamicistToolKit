from math import sin, cos, tan, atan
from scipy.optimize import newton
import numpy as np

from inertia import y_rot

def pitch_from_roll_and_steer(q4, q7, rF, rR, d1, d2, d3):
    """Returns the pitch angle of the bicycle frame for a given roll, steer and
    geometry.

    Parameters
    ----------
    q4 : float
        Roll angle.
    q5 : float
        Steer angle.
    rF : float
        Front wheel radius.
    rR : float
        Rear wheel radius.
    d1 : float
        The rear wheel offset from the steer axis.
    d2 : float
        The distance along the steer axis between the intersection of the front
        and rear offset lines.
    d3 : float
        The front wheel offset from the steer axis.

    Returns
    -------
    q5 : float
        Pitch angle.

    """
    def pitch_constraint(q5, q4, q7, rF, rR, d1, d2, d3):
        return (d2 * cos(q4) * cos(q5) + d3 * (sin(q4) * sin(q7) - sin(q5) *
                sin(q4) * sin(q7)) + rF * (1 - (sin(q4) * sin(q7) + sin(q5) *
                sin(q7) * sin(q4))**2)**(0.5) - rR * sin(q4) - d1 * sin(q5) * sin(q4))

    # guess based on steer and roll being both zero
    guess = lambda_from_abc(rF, rR, d1, d2, d3)

    args = (q4, q7, rF, rR, d1, d2, d3)

    q5 = newton(pitch_constraint, guess, args=args)

    return q5

def benchmark_whipple_to_moore_whipple(benchmarkParameters, oldMassCenter=False):
    """Returns the parameters for the Whipple model as derived by Jason K.
    Moore.

    Parameters
    ----------
    benchmarkParameters : dictionary
        Contains the set of parameters for the Whipple bicycle model as
        presented in Meijaard2007.
    oldMassCenter : boolean
        If true it returns the fork mass center dimensions, l3 and l4, with
        respect to the rear offset intersection with the steer axis, otherwise
        the dimensions are with respect to the front wheel.

    Returns
    -------
    mooreParameters : dictionary
        The parameter set for the Moore derivation of the whipple bicycle model
        as presented in Moore2012.

    """

    bP = benchmarkParameters
    mP = {}

    # geometry
    mP['rF'] = bP['rF']
    mP['rR'] = bP['rR']
    mP['d1'] =  cos(bP['lam']) * (bP['c'] + bP['w'] - bP['rR'] * tan(bP['lam']))
    mP['d3'] = -cos(bP['lam']) * (bP['c'] - bP['rF'] * tan(bP['lam']))
    mP['d2'] = (bP['rR'] + mP['d1'] * sin(bP['lam']) - bP['rF'] + mP['d3'] *
            sin(bP['lam'])) / cos(bP['lam'])

    # mass center locations
    # bicycle frame
    mP['l1'] = (bP['xB'] * cos(bP['lam']) - bP['zB'] * sin(bP['lam']) -
        bP['rR'] * sin(bP['lam']))
    mP['l2'] = (bP['xB'] * sin(bP['lam']) + bP['zB'] * cos(bP['lam']) +
        bP['rR'] * cos(bP['lam']))

    # bicycle fork
    # l3 and l4 are with reference to the front wheel center (the new way)
    mP['l4'] = ((bP['zH'] + bP['rF']) * cos(bP['lam']) + (bP['xH'] - bP['w'])
        * sin(bP['lam']))
    mP['l3'] = ((bP['xH'] - bP['w'] - mP['l4'] * sin(bP['lam'])) /
        cos(bP['lam']))

    if oldMassCenter is True:
        # l3 and l4 are with reference to the point where the rear offset line
        # intersects the steer axis (this is the old way)
        mP['l3'] = mP['d3'] + mP['l3']
        mP['l4'] = mP['d2'] + mP['l4']
    elif oldMassCenter is False:
        pass
    else:
        raise ValueError('oldMassCenter must be True or False')

    # masses
    mP['mc'] =  bP['mB']
    mP['md'] =  bP['mR']
    mP['me'] =  bP['mH']
    mP['mf'] =  bP['mF']

    # inertia
    # rear wheel inertia
    mP['id11']  =  bP['IRxx']
    mP['id22']  =  bP['IRyy']
    mP['id33']  =  bP['IRxx']

    # front wheel inertia
    mP['if11']  =  bP['IFxx']
    mP['if22']  =  bP['IFyy']
    mP['if33']  =  bP['IFxx']

    # lambda rotation matrix
    R = y_rot(bP['lam'])

    # rotate the benchmark bicycle frame inertia through the steer axis tilt,
    # lambda
    IB =  np.matrix([[bP['IBxx'], 0., bP['IBxz']],
                     [0., bP['IByy'], 0.],
                     [bP['IBxz'], 0., bP['IBzz']]])
    IBrot =  R * IB * R.T

    # bicycle frame inertia
    mP['ic11'] =  IBrot[0, 0]
    mP['ic12'] =  IBrot[0, 1]
    mP['ic22'] =  IBrot[1, 1]
    mP['ic23'] =  IBrot[1, 2]
    mP['ic31'] =  IBrot[2, 0]
    mP['ic33'] =  IBrot[2, 2]

    # rotate the benchmark bicycle fork inertia through the steer axis tilt,
    # lambda
    IH =  np.matrix([[bP['IHxx'], 0., bP['IHxz']],
                     [0., bP['IHyy'], 0.],
                     [bP['IHxz'], 0., bP['IHzz']]])
    IHrot =  R * IH * R.T

    # fork/handlebar inertia
    mP['ie11'] =  IHrot[0, 0]
    mP['ie12'] =  IHrot[0, 1]
    mP['ie22'] =  IHrot[1, 1]
    mP['ie23'] =  IHrot[1, 2]
    mP['ie31'] =  IHrot[2, 0]
    mP['ie33'] =  IHrot[2, 2]

    return mP

def lambda_from_abc(rF, rR, a, b, c):
    '''Returns the steer axis tilt, lamba, for the parameter set based on the
    offsets from the steer axis.

    Parameters
    ----------
    rF : float
        Front wheel radius.
    rR : float
        Rear wheel radius.
    a : float
        The rear wheel offset from the steer axis.
    b : float
        The distance along the steer axis between the intersection of the front
        and rear offset lines.
    c : float
        The front wheel offset from the steer axis.

    Returns
    -------
    lam : float
        The steer axis tilt as described in Meijaard2007.

    '''
    def lam_equality(lam, rF, rR, a, b, c):
        return sin(lam) - (rF - rR + c * cos(lam)) / (a + b)

    guess = atan(c / (a + b)) # guess based on equal wheel radii

    args = (rF, rR, a, b, c)

    lam = newton(lam_equality, guess, args=args)

    return lam

def trail(rF, lam, fo):
    '''Caluculate the trail and mechanical trail

    Parameters:
    -----------
    rF: float
        The front wheel radius
    lam: float
        The steer axis tilt (pi/2 - headtube angle). The angle between the
        headtube and a vertical line.
    fo: float
        The fork offset

    Returns:
    --------
    c: float
        Trail
    cm: float
        Mechanical Trail

    '''

    # trail
    c = (rF * sin(lam) - fo) / cos(lam)
    # mechanical trail
    cm = c * cos(lam)
    return c, cm

def sort_modes(evals, evecs):
    '''Sort eigenvalues and eigenvectors into weave, capsize, caster modes.

    Parameters
    ----------
    evals : ndarray, shape (n, 4)
        eigenvalues
    evecs : ndarray, shape (n, 4, 4)
        eigenvectors

    Returns
    -------
    weave['evals'] : ndarray, shape (n, 2)
        The eigen value pair associated with the weave mode.
    weave['evecs'] : ndarray, shape (n, 4, 2)
        The associated eigenvectors of the weave mode.
    capsize['evals'] : ndarray, shape (n,)
        The real eigenvalue associated with the capsize mode.
    capsize['evecs'] : ndarray, shape(n, 4, 1)
        The associated eigenvectors of the capsize mode.
    caster['evals'] : ndarray, shape (n,)
        The real eigenvalue associated with the caster mode.
    caster['evecs'] : ndarray, shape(n, 4, 1)
        The associated eigenvectors of the caster mode.

    This only works on the standard bicycle eigenvalues, not necessarily on any
    general eigenvalues for the bike model (e.g. there isn't always a distinct weave,
    capsize and caster). Some type of check unsing the derivative of the curves
    could make it more robust.

    '''
    evalsorg = np.zeros_like(evals)
    evecsorg = np.zeros_like(evecs)
    # set the first row to be the same
    evalsorg[0] = evals[0]
    evecsorg[0] = evecs[0]
    # for each speed
    for i, speed in enumerate(evals):
        if i == evals.shape[0] - 1:
            break
        # for each current eigenvalue
        used = []
        for j, e in enumerate(speed):
            try:
                x, y = np.real(evalsorg[i, j].nominal_value), np.imag(evalsorg[i, j].nominal_value)
            except:
                x, y = np.real(evalsorg[i, j]), np.imag(evalsorg[i, j])
            # for each eigenvalue at the next speed
            dist = np.zeros(4)
            for k, eignext in enumerate(evals[i + 1]):
                try:
                    xn, yn = np.real(eignext.nominal_value), np.imag(eignext.nominal_value)
                except:
                    xn, yn = np.real(eignext), np.imag(eignext)
                # distance between points in the real/imag plane
                dist[k] = np.abs(((xn - x)**2 + (yn - y)**2)**0.5)
            if np.argmin(dist) in used:
                # set the already used indice higher
                dist[np.argmin(dist)] = np.max(dist) + 1.
            else:
                pass
            evalsorg[i + 1, j] = evals[i + 1, np.argmin(dist)]
            evecsorg[i + 1, :, j] = evecs[i + 1, :, np.argmin(dist)]
            # keep track of the indices we've used
            used.append(np.argmin(dist))
    weave = {'evals' : evalsorg[:, 2:], 'evecs' : evecsorg[:, :, 2:]}
    capsize = {'evals' : evalsorg[:, 1], 'evecs' : evecsorg[:, :, 1]}
    caster = {'evals' : evalsorg[:, 0], 'evecs' : evecsorg[:, :, 0]}
    return weave, capsize, caster

def benchmark_par_to_canonical(p):
    """
    Returns the canonical matrices of the Whipple bicycle model linearized
    about the upright constant velocity configuration. It uses the parameter
    definitions from Meijaard et al. 2007.

    Parameters
    ----------
    p : dictionary
        A dictionary of the benchmark bicycle parameters. Make sure your units
        are correct, best to ue the benchmark paper's units!

    Returns
    -------
    M : ndarray, shape(2,2)
        The mass matrix.
    C1 : ndarray, shape(2,2)
        The damping like matrix that is proportional to the speed, v.
    K0 : ndarray, shape(2,2)
        The stiffness matrix proportional to gravity, g.
    K2 : ndarray, shape(2,2)
        The stiffness matrix proportional to the speed squared, v**2.

    This function handles parameters with uncertanties.

    """
    mT = p['mR'] + p['mB'] + p['mH'] + p['mF']
    xT = (p['xB'] * p['mB'] + p['xH'] * p['mH'] + p['w'] * p['mF']) / mT
    zT = (-p['rR'] * p['mR'] + p['zB'] * p['mB'] +
          p['zH'] * p['mH'] - p['rF'] * p['mF']) / mT

    ITxx = (p['IRxx'] + p['IBxx'] + p['IHxx'] + p['IFxx'] + p['mR'] *
            p['rR']**2 + p['mB'] * p['zB']**2 + p['mH'] * p['zH']**2 + p['mF']
            * p['rF']**2)
    ITxz = (p['IBxz'] + p['IHxz'] - p['mB'] * p['xB'] * p['zB'] -
            p['mH'] * p['xH'] * p['zH'] + p['mF'] * p['w'] * p['rF'])
    p['IRzz'] = p['IRxx']
    p['IFzz'] = p['IFxx']
    ITzz = (p['IRzz'] + p['IBzz'] + p['IHzz'] + p['IFzz'] +
            p['mB'] * p['xB']**2 + p['mH'] * p['xH']**2 + p['mF'] * p['w']**2)

    mA = p['mH'] + p['mF']
    xA = (p['xH'] * p['mH'] + p['w'] * p['mF']) / mA
    zA = (p['zH'] * p['mH'] - p['rF']* p['mF']) / mA

    IAxx = (p['IHxx'] + p['IFxx'] + p['mH'] * (p['zH'] - zA)**2 +
            p['mF'] * (p['rF'] + zA)**2)
    IAxz = (p['IHxz'] - p['mH'] * (p['xH'] - xA) * (p['zH'] - zA) + p['mF'] *
            (p['w'] - xA) * (p['rF'] + zA))
    IAzz = (p['IHzz'] + p['IFzz'] + p['mH'] * (p['xH'] - xA)**2 + p['mF'] *
            (p['w'] - xA)**2)
    uA = (xA - p['w'] - p['c']) * cos(p['lam']) - zA * sin(p['lam'])
    IAll = (mA * uA**2 + IAxx * sin(p['lam'])**2 +
            2 * IAxz * sin(p['lam']) * cos(p['lam']) +
            IAzz * cos(p['lam'])**2)
    IAlx = (-mA * uA * zA + IAxx * sin(p['lam']) + IAxz *
            cos(p['lam']))
    IAlz = (mA * uA * xA + IAxz * sin(p['lam']) + IAzz *
            cos(p['lam']))

    mu = p['c'] / p['w'] * cos(p['lam'])

    SR = p['IRyy'] / p['rR']
    SF = p['IFyy'] / p['rF']
    ST = SR + SF
    SA = mA * uA + mu * mT * xT

    Mpp = ITxx
    Mpd = IAlx + mu * ITxz
    Mdp = Mpd
    Mdd = IAll + 2 * mu * IAlz + mu**2 * ITzz
    M = np.array([[Mpp, Mpd], [Mdp, Mdd]])

    K0pp = mT * zT # this value only reports to 13 digit precision it seems?
    K0pd = -SA
    K0dp = K0pd
    K0dd = -SA * sin(p['lam'])
    K0 = np.array([[K0pp, K0pd], [K0dp, K0dd]])

    K2pp = 0.
    K2pd = (ST - mT * zT) / p['w'] * cos(p['lam'])
    K2dp = 0.
    K2dd = (SA + SF * sin(p['lam'])) / p['w'] * cos(p['lam'])
    K2 = np.array([[K2pp, K2pd], [K2dp, K2dd]])

    C1pp = 0.
    C1pd = (mu * ST + SF * cos(p['lam']) + ITxz / p['w'] *
            cos(p['lam']) - mu*mT*zT)
    C1dp = -(mu * ST + SF * cos(p['lam']))
    C1dd = (IAlz / p['w'] * cos(p['lam']) + mu * (SA +
            ITzz / p['w'] * cos(p['lam'])))
    C1 = np.array([[C1pp, C1pd], [C1dp, C1dd]])

    return M, C1, K0, K2

def abMatrix(M, C1, K0, K2, v, g):
    """
    Calculate the A and B matrices for the Whipple bicycle model linearized
    about the upright configuration.

    Parameters
    ----------
    M : ndarray, shape(2,2)
        The mass matrix.
    C1 : ndarray, shape(2,2)
        The damping like matrix that is proportional to the speed, v.
    K0 : ndarray, shape(2,2)
        The stiffness matrix proportional to gravity, g.
    K2 : ndarray, shape(2,2)
        The stiffness matrix proportional to the speed squared, v**2.
    v : float
        Forward speed.
    g : float
        Acceleration due to gravity.

    Returns
    -------
    A : ndarray, shape(4,4)
        System dynamic matrix.
    B : ndarray, shape(4,2)
        Input matrix.

    Notes
    -----
    The states are [roll rate, steer rate, roll angle, steer angle]
    The inputs are [roll torque, steer torque]

    """
    a11 = -v * C1
    a12 = -(g * K0 + v**2 * K2)
    a21 = np.eye(2)
    a22 = np.zeros((2, 2))
    invM = (1. / (M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]) *
           np.array([[M[1, 1], -M[0, 1]], [-M[1, 0], M[0, 0]]], dtype=M.dtype))
    A = np.vstack((np.dot(invM, np.hstack((a11, a12))),
                   np.hstack((a21, a22))))
    B = np.vstack((invM, np.zeros((2, 2))))

    return A, B
