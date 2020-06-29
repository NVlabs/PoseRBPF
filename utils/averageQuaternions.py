import numpy
import numpy.matlib as npm
from transforms3d.quaternions import *
import scipy.stats as sci_stats

# Q is a Nx4 numpy matrix and contains the quaternions to average in the rows.
# The quaternions are arranged as (w,x,y,z), with w being the scalar
# The result will be the average quaternion of the input. Note that the signs
# of the output quaternion can be reversed, since q and -q describe the same orientation
def averageQuaternions(Q):
    # Number of quaternions to average
    M = Q.shape[0]
    A = npm.zeros(shape=(4,4))

    for i in range(0,M):
        q = Q[i,:]
        # multiply q with its transposed version q' and add A
        A = numpy.outer(q,q) + A

    # scale
    A = (1.0/M)*A
    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = numpy.linalg.eig(A)
    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return numpy.real(eigenVectors[:,0].A1)


# Average multiple quaternions with specific weights
# The weight vector w must be of the same length as the number of rows in the
# quaternion maxtrix Q
def weightedAverageQuaternions(Q, w):
    # Number of quaternions to average
    M = Q.shape[0]
    A = npm.zeros(shape=(4,4))
    weightSum = 0

    for i in range(0,M):
        q = Q[i,:]
        A = w[i] * numpy.outer(q,q) + A
        weightSum += w[i]

    # scale
    A = (1.0/weightSum) * A

    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = numpy.linalg.eig(A)

    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]

    # return the real part of the largest eigenvector (has only real part)
    return numpy.real(eigenVectors[:,0].A1)


def weightedAverageQuaternions_star(Q, q_star, w, rot_range, rot_var):
    # Number of quaternions to average
    M = Q.shape[0]
    A = npm.zeros(shape=(4, 4))
    weightSum = 0

    for i in range(0,M):
        q = Q[i,:]
        q_diff = qmult(qinverse(q_star), q)
        _, d_angle = quat2axangle(q_diff)
        if d_angle > np.pi:
            d_angle = d_angle - 2 * np.pi
            d_angle = - d_angle

        if np.abs(d_angle) > rot_range:
            continue

        wt_motion = 1/np.sqrt(2*np.pi*rot_var**2)*np.exp(-d_angle**2/(2*rot_var**2))

        A = w[i] * wt_motion * numpy.outer(q,q) + A
        weightSum += w[i] * wt_motion

    # scale
    if not weightSum == 0:
        A = (1.0/weightSum) * A
    else:
        A += numpy.outer(q_star,q_star)

    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = numpy.linalg.eig(A)

    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]

    # return the real part of the largest eigenvector (has only real part)
    return numpy.real(eigenVectors[:,0].A1)