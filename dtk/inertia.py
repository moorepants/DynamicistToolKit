#!/usr/bin/env python
# -*- coding: utf-8 -*-

# external libraries
import numpy as np


def compound_pendulum_inertia(m, g, l, T):
    '''Returns the moment of inertia for an object hung as a compound
    pendulum.

    Parameters
    ----------
    m : float
        Mass of the pendulum.
    g : float
        Acceration due to gravity.
    l : float
        Length of the pendulum.
    T : float
        The period of oscillation.

    Returns
    -------
    float
        Moment of interia of the pendulum.

    Examples
    --------

    >>> from dtk.inertia import compound_pendulum_inertia
    >>> compound_pendulum_inertia(3.0, 9.81, 0.2, 1.4)
    0.1722244785902121

    '''

    # TODO : This can give negative reseults, which is impossible. Check!

    return (T/2.0/np.pi)**2.0*m*g*l - m*l**2.0


def torsional_pendulum_inertia(k, T):
    '''Calculate the moment of inertia for an ideal torsional pendulum.

    Parameters
    ----------
    k : float
        Torsional stiffness.
    T : float
        Period of oscillation.

    Returns
    -------
    float
        Moment of inertia.

    Examples
    --------

    >>> from dtk.inertia import torsional_pendulum_inertia
    >>> torsional_pendulum_inertia(50.0, 1.0)
    1.2665147955292222

    '''

    return k*T**2/4.0/np.pi**2


def parallel_axis(Ic, m, d):
    '''Returns the moment of inertia of a body about a different point.

    Parameters
    ----------
    Ic : ndarray, shape(3,3)
        The moment of inertia about the center of mass of the body with respect
        to an orthogonal coordinate system.
    m : float
        The mass of the body.
    d : ndarray, shape(3,)
        The distances along the three ordinates that located the new point
        relative to the center of mass of the body.

    Returns
    -------
    I : ndarray, shape(3,3)
        The moment of inertia of a body about a point located by the distances
        in d.

    Examples
    --------

    >>> import numpy as np
    >>> from dtk.bicycle import benchmark_parameters
    >>> from dtk.inertia import parallel_axis
    >>> p = benchmark_parameters()
    >>> Ic = np.array([[p['IBxx'], 0.0, p['IBxz']],
    ...                [0.0, p['IByy'], 0.0],
    ...                [p['IBxz'], 0.0, p['IBzz']]])
    >>> d = np.array([-p['xB'], 0.0, -p['zB']])  # about rear wheel contact
    >>> parallel_axis(Ic, p['mB'], d)
    array([[78.05,  0.  , 25.35],
           [ 0.  , 87.5 ,  0.  ],
           [25.35,  0.  , 10.45]])

    '''
    a = d[0]
    b = d[1]
    c = d[2]
    dMat = np.zeros((3, 3), dtype=Ic.dtype)
    dMat[0] = np.array([b**2 + c**2, -a * b, -a * c])
    dMat[1] = np.array([-a * b, c**2 + a**2, -b * c])
    dMat[2] = np.array([-a * c, -b * c, a**2 + b**2])
    return Ic + m * dMat


def inertia_components(jay, beta):
    '''Returns the 2D orthogonal inertia tensor.

    When at least three moments of inertia and their axes orientations are
    known relative to a common inertial frame of a planar object, the
    orthoganal moments of inertia relative the frame are computed.

    Parameters
    ----------
    jay : ndarray, shape(n,)
        An array of at least three moments of inertia. (n >= 3)
    beta : ndarray, shape(n,)
        An array of orientation angles corresponding to the moments of inertia
        in jay.

    Returns
    -------
    ndarray, shape(3,)
        Ixx, Ixz, Izz

    Examples
    --------

    >>> import numpy as np
    >>> from dtk.inertia import inertia_components
    >>> inertia_components([1.2, 0.5, 3.1], np.deg2rad([45.0, 90.0, 135.0]))
    array([3.8 , 0.95, 0.5 ])
    >>> inertia_components([1.2, 0.5, 0.51, 3.1],
    ...                    np.deg2rad([45.0, 90.0, 90.2, 135.0]))
    array([3.79833626, 0.95000581, 0.50166378])

    '''
    sb = np.sin(beta)
    cb = np.cos(beta)
    betaMat = np.array(np.vstack((cb**2, -2*sb*cb, sb**2)).T)
    return np.linalg.lstsq(betaMat, jay, rcond=None)[0]


def tube_inertia(l, m, ro, ri):
    '''Calculate the moment of inertia for a tube (or rod) where the x axis is
    aligned with the tube's axis.

    Parameters
    ----------
    l : float
        The length of the tube.
    m : float
        The mass of the tube.
    ro : float
        The outer radius of the tube.
    ri : float
        The inner radius of the tube. Set this to zero if it is a rod instead
        of a tube.

    Returns
    -------
    Ix : float
        Moment of inertia about tube axis.
    Iy, Iz : float
        Moment of inertia about normal axis.

    Examples
    --------

    >>> from dtk.inertia import tube_inertia
    >>> tube_inertia(1.0, 0.4, 0.02, 0.015)
    (0.000125, 0.03339583333333333, 0.03339583333333333)

    '''
    Ix = m / 2. * (ro**2 + ri**2)
    Iy = m / 12. * (3 * ro**2 + 3 * ri**2 + l**2)
    Iz = Iy
    return Ix, Iy, Iz


def cylinder_inertia(l, m, ro, ri):
    """
    Calculate the moment of inertia for a hollow cylinder (or solid cylinder)
    where the x axis is aligned with the cylinder's axis.

    Parameters
    ----------
    l : float
        The length of the cylinder.
    m : float
        The mass of the cylinder.
    ro : float
        The outer radius of the cylinder.
    ri : float
        The inner radius of the cylinder. Set this to zero for a solid
        cylinder.

    Returns
    -------
    Ix : float
        Moment of inertia about cylinder axis.
    Iy, Iz : float
        Moment of inertia about axis perpendicular to cylinder axis.

    Examples
    --------

    >>> from dtk.inertia import cylinder_inertia
    >>> cylinder_inertia(1.0, 0.4, 0.02, 0.015)
    (0.000125, 0.03339583333333333, 0.03339583333333333)
    >>> cylinder_inertia(1.0, 0.4, 0.02, 0.0)
    (8e-05, 0.03337333333333334, 0.03337333333333334)

    """
    # TODO : Confused why a solid bar has less inertia about axis that hollow
    # one.
    Ix = m/2.*(ro**2 + ri**2)
    Iy = m/12.*(3*ro**2 + 3*ri**2 + l**2)
    Iz = Iy
    return Ix, Iy, Iz


def total_com(coordinates, masses):
    """
    Returns the center of mass of a group of objects if the indivdual
    centers of mass and mass is provided.

    coordinates : ndarray, shape(3,n)
        The rows are the x, y and z coordinates, respectively and the columns
        are for each object.
    masses : ndarray, shape(3,)
        An array of the masses of multiple objects, the order should correspond
        to the columns of coordinates.

    Returns
    -------
    mT : float
        Total mass of the objects.
    cT : ndarray, shape(3,)
        The x, y, and z coordinates of the total center of mass.

    Examples
    --------

    >>> import numpy as np
    >>> from dtk.bicycle import benchmark_parameters
    >>> from dtk.inertia import total_com
    >>> par = benchmark_parameters()
    >>> coordinates = np.array([[par['xB'], par['xH']],
    ...                         [0.0, 0.0],
    ...                         [par['zB'], par['zH']]])
    ...
    >>> coordinates
    array([[ 0.3,  0.9],
           [ 0. ,  0. ],
           [-0.9, -0.7]])
    >>> masses = np.array([par['mB'], par['mH']])
    >>> masses
    array([85.,  4.])
    >>> total_com(coordinates, masses)
    (89.0, array([ 0.32696629,  0.        , -0.89101124]))

    """
    products = masses * coordinates
    mT = np.sum(masses)
    cT = np.sum(products, axis=1) / mT
    return float(mT), cT


def rotate_inertia_about_y(I, angle):
    """
    Returns inertia tensor rotated through angle about the Y axis.

    Parameters
    ----------
    I : ndarray, shape(3, 3)
        An inertia tensor.
    angle : float
        Angle in radians about the positive Y axis of which to rotate the
        inertia tensor.

    Returns
    -------
    ndarray, shape(3, 3)
        Rotated inerita tensor.

    Examples
    --------

    >>> import numpy as np
    >>> from dtk.inertia import rotate_inertia_about_y
    >>> rotate_inertia_about_y(np.diag([1.0, 2.0, 3.0]), np.deg2rad(45.0))
    array([[ 2.,  0., -1.],
           [ 0.,  2.,  0.],
           [-1.,  0.,  2.]])

    """
    ca = np.cos(angle)
    sa = np.sin(angle)
    C = np.array([[ca, 0., -sa],
                  [0., 1., 0.],
                  [sa, 0., ca]])
    return C @ I @ C.T


def principal_axes(I):
    """
    Returns the principal moments of inertia and the orientation.

    Parameters
    ----------
    I : ndarray, shape(3,3)
        An inertia tensor.

    Returns
    -------
    Ip : ndarray, shape(3,)
        The principal moments of inertia. This is sorted smallest to largest.
    C : ndarray, shape(3,3)
        The rotation matrix.

    Examples
    --------

    >>> import numpy as np
    >>> from dtk.bicycle import benchmark_parameters
    >>> from dtk.inertia import principal_axes
    >>> p = benchmark_parameters()
    >>> Ic = np.array([[p['IBxx'], 0.0, p['IBxz']],
    ...                [0.0, p['IByy'], 0.0],
    ...                [p['IBxz'], 0.0, p['IBzz']]])
    >>> Ip, C = principal_axes(Ic)
    >>> Ip
    array([ 2., 10., 11.])
    >>> C
    array([[-0.31622777,  0.        ,  0.9486833 ],
           [ 0.9486833 ,  0.        ,  0.31622777],
           [ 0.        ,  1.        ,  0.        ]])
    >>> C @ Ic @ C.T
    array([[ 2.00000000e+00, -5.28515252e-17,  0.00000000e+00],
           [-3.40171594e-16,  1.00000000e+01,  0.00000000e+00],
           [ 0.00000000e+00,  0.00000000e+00,  1.10000000e+01]])

    """
    Ip, C = np.linalg.eig(I)
    indices = np.argsort(Ip)
    Ip = Ip[indices]
    C = C.T[indices]
    return Ip, C


def x_rot(angle):
    """Returns the rotation matrix for a reference frame rotated through an
    angle about the x axis.

    Parameters
    ----------
    angle : float
        The angle in radians.

    Returns
    -------
    Rx : ndarray, shape(3,3)
        The rotation matrix.

    Examples
    --------

    >>> import numpy as np
    >>> from dtk.inertia import x_rot
    >>> x_rot(np.deg2rad(45.0))
    array([[ 1.        ,  0.        ,  0.        ],
           [ 0.        ,  0.70710678,  0.70710678],
           [ 0.        , -0.70710678,  0.70710678]])

    Notes
    -----
    v' = Rx * v where v is the vector expressed the reference in the original
    reference frame and v' is the vector expressed in the new rotated reference
    frame.

    """
    sa = np.sin(angle)
    ca = np.cos(angle)
    Rx = np.array([[1., 0., 0.],
                   [0., ca, sa],
                   [0., -sa, ca]])
    return Rx


def y_rot(angle):
    """Returns the rotation matrix for a reference frame rotated through an
    angle about the y axis.

    Parameters
    ----------
    angle : float
        The angle in radians.

    Returns
    -------
    Rx : ndarray, shape(3,3)
        The rotation matrix.

    Examples
    --------

    >>> import numpy as np
    >>> from dtk.inertia import y_rot
    >>> y_rot(np.deg2rad(45.0))
    array([[ 0.70710678,  0.        , -0.70710678],
           [ 0.        ,  1.        ,  0.        ],
           [ 0.70710678,  0.        ,  0.70710678]])

    Notes
    -----
    v' = Rx * v where v is the vector expressed the reference in the original
    reference frame and v' is the vector expressed in the new rotated reference
    frame.

    """
    sa = np.sin(angle)
    ca = np.cos(angle)
    Ry = np.array([[ca, 0., -sa],
                   [0., 1., 0.],
                   [sa, 0., ca]])
    return Ry


def z_rot(angle):
    """Returns the rotation matrix for a reference frame rotated through an
    angle about the z axis.

    Parameters
    ----------
    angle : float
        The angle in radians.

    Returns
    -------
    Rx : ndarray, shape(3,3)
        The rotation matrix.

    Examples
    --------

    >>> import numpy as np
    >>> from dtk.inertia import z_rot
    >>> z_rot(np.deg2rad(45.0))
    array([[ 0.70710678,  0.70710678,  0.        ],
           [-0.70710678,  0.70710678,  0.        ],
           [ 0.        ,  0.        ,  1.        ]])

    Notes
    -----
    v' = Rx * v where v is the vector expressed the reference in the original
    reference frame and v' is the vector expressed in the new rotated reference
    frame.

    """
    sa = np.sin(angle)
    ca = np.cos(angle)
    Rz = np.array([[ca, sa, 0.],
                   [-sa, ca, 0.],
                   [0., 0., 1.]])
    return Rz


def euler_rotation(angles, order):
    """
    Returns a rotation matrix for a reference frame, B,  in another reference
    frame, A, where the B frame is rotated relative to the A frame via body
    fixed rotations (Euler angles).

    Parameters
    ----------
    angles : array_like
        An array of three angles in radians that are in order of rotation.
    order : tuple
        A three tuple containing a combination of ``1``, ``2``, and ``3`` where
        ``1`` is about the x axis of the first reference frame, ``2`` is about
        the y axis of the this new frame and ``3`` is about the z axis. Note
        that (1, 1, 1) is a valid entry and will give you correct results, but
        combinations like this are not necessarily useful for describing a
        general configuration.

    Returns
    -------
    R : ndarray, shape(3,3)
        A rotation matrix.

    Notes
    -----
    The rotation matrix is defined such that a R times a vector v equals the
    vector expressed in the rotated reference frame.

        v' = R * v

    Where v is the vector expressed in the original reference frame and v' is
    the same vector expressed in the rotated reference frame.

    Examples
    --------
    >>> import numpy as np
    >>> from dtk.inertia import euler_rotation
    >>> angles = [np.pi, np.pi / 2., -np.pi / 4.]
    >>> rotMat = euler_rotation(angles, (3, 1, 3))
    >>> rotMat
    array([[-7.07106781e-01,  1.29893408e-16, -7.07106781e-01],
           [-7.07106781e-01,  4.32978028e-17,  7.07106781e-01],
           [ 1.22464680e-16,  1.00000000e+00,  6.12323400e-17]])
    >>> v = np.array([[1.], [0.], [0.]])
    >>> vp = rotMat @ v
    >>> vp
    array([[-7.07106781e-01],
           [-7.07106781e-01],
           [ 1.22464680e-16]])

    """

    # check the length of both inputs
    if len(angles) != 3 or len(order) != 3:
        raise Exception('The length of angles and order should be 3')

    # make sure the order contains proper values
    for v in order:
        if v not in [1, 2, 3]:
            raise ValueError('The values in order have to be 1, 2 or 3')

    rot = [x_rot, y_rot, z_rot]

    mat = []

    for i, ang in enumerate(angles):
        mat.append(rot[order[i] - 1](ang))

    return mat[2] @ mat[1] @ mat[0]


def rotate3(angles):
    """
    Produces a three-dimensional rotation matrix as rotations around the
    three cartesian axes.

    Parameters
    ----------
    angles : array_like, shape(3,)
        Three angles (in units of radians) that specify the orientation of
        a new reference frame with respect to a fixed reference frame.
        The first angle is a pure rotation about the x-axis, the second about
        the y-axis, and the third about the z-axis. All rotations are with
        respect to the initial fixed frame, and they occur in the order x,
        then y, then z.

    Returns
    -------
    R : ndarray, shape(3,3)
        Three dimensional rotation matrix about three different orthogonal
        axes.

    Examples
    --------

    >>> import numpy as np
    >>> from dtk.inertia import rotate3
    >>> rotate3(np.deg2rad([12.0, 22.0, 45.0]))
    array([[ 0.65561799, -0.63658173,  0.40611422],
           [ 0.65561799,  0.74672788,  0.11208268],
           [-0.37460659,  0.19277236,  0.90692266]])

    """
    cx = np.cos(angles[0])
    sx = np.sin(angles[0])

    cy = np.cos(angles[1])
    sy = np.sin(angles[1])

    cz = np.cos(angles[2])
    sz = np.sin(angles[2])

    Rz = np.array([[ cz,-sz,  0],
                   [ sz, cz,  0],
                   [  0,  0,  1]])

    Ry = np.array([[ cy,  0, sy],
                   [  0,  1,  0],
                   [-sy,  0, cy]])

    Rx = np.array([[  1,  0,  0],
                   [  0, cx, -sx],
                   [  0, sx,  cx]])

    return Rz @ Ry @ Rx


def euler_123(angles):
    """
    Returns the direction cosine matrix as a function of the Euler 123 angles.

    Parameters
    ----------
    angles : numpy.array or list or tuple, shape(3,)
        Three angles (in units of radians) that specify the orientation of a
        new reference frame with respect to a fixed reference frame. The first
        angle, phi, is a rotation about the fixed frame's x-axis. The second
        angle, theta, is a rotation about the new y-axis (which is realized
        after the phi rotation). The third angle, psi, is a rotation about the
        new z-axis (which is realized after the theta rotation). Thus, all
        three angles are "relative" rotations with respect to the new frame.
        Note: if the rotations are viewed as occuring in the opposite direction
        (z, then y, then x), all three rotations are with respect to the
        initial fixed frame rather than "relative".

    Returns
    -------
    R : ndarray, shape(3,3)
        Three dimensional rotation matrix about three different orthogonal
        axes.

    Examples
    --------

    >>> import numpy as np
    >>> from dtk.inertia import euler_123
    >>> euler_123(np.deg2rad([12.0, 22.0, 45.0]))
    array([[ 0.65561799, -0.65561799,  0.37460659],
           [ 0.74672788,  0.63658173, -0.19277236],
           [-0.11208268,  0.40611422,  0.90692266]])

    """
    cphi = np.cos(angles[0])
    sphi = np.sin(angles[0])

    cthe = np.cos(angles[1])
    sthe = np.sin(angles[1])

    cpsi = np.cos(angles[2])
    spsi = np.sin(angles[2])

    R1 = np.array([[     1,     0,     0],
                   [     0,  cphi, -sphi],
                   [     0,  sphi,  cphi]])

    R2 = np.array([[  cthe,     0,  sthe],
                   [     0,     1,     0],
                   [ -sthe,     0,  cthe]])

    R3 = np.array([[  cpsi,  -spsi,     0],
                   [  spsi,  cpsi,     0],
                   [     0,     0,     1]])

    return R1 @ R2 @ R3


def rotate3_inertia(RotMat, relInertia):
    """
    Rotates an inertia tensor. A derivation of the formula in this function
    can be found in Crandall 1968, Dynamics of mechanical and electromechanical
    systems. This function only transforms an inertia tensor for rotations with
    respect to a fixed point. To translate an inertia tensor, one must use the
    parallel axis analogue for tensors. An inertia tensor contains both moments
    of inertia and products of inertia for a mass in a cartesian (xyz) frame.

    Parameters
    ----------
    RotMat : array_like, shape(3,3)
        Three-dimensional rotation matrix specifying the coordinate frame that
        the input inertia tensor is in, with respect to a fixed coordinate
        system in which one desires to express the inertia tensor.
    relInertia : array_like, shape(3,3)
        Three-dimensional cartesian inertia tensor describing the inertia of a
        mass in a rotated coordinate frame.

    Returns
    -------
    Inertia : ndarray, shape(3,3)
        Inertia tensor with respect to a fixed coordinate system ("unrotated").

    Examples
    --------

    >>> import numpy as np
    >>> from dtk.bicycle import benchmark_parameters
    >>> from dtk.inertia import principal_axes, rotate3_inertia
    >>> p = benchmark_parameters()
    >>> Ic = np.array([[p['IBxx'], 0.0, p['IBxz']],
    ...                [0.0, p['IByy'], 0.0],
    ...                [p['IBxz'], 0.0, p['IBzz']]])
    >>> Ip, C = principal_axes(Ic)
    >>> Ip
    array([ 2., 10., 11.])
    >>> C
    array([[-0.31622777,  0.        ,  0.9486833 ],
           [ 0.9486833 ,  0.        ,  0.31622777],
           [ 0.        ,  1.        ,  0.        ]])
    >>> rotate3_inertia(C, Ic)
    array([[ 2.00000000e+00, -5.28515252e-17,  0.00000000e+00],
           [-3.40171594e-16,  1.00000000e+01,  0.00000000e+00],
           [ 0.00000000e+00,  0.00000000e+00,  1.10000000e+01]])

    """
    return RotMat @ relInertia @ RotMat.T
