import numpy as np
def distLinSeg(point1s, point1e, point2s, point2e):
    '''
    Function for fast computation of the shortest distance between two line segments
    '''
    d1 = point1e - point1s
    d2 = point2e - point2s
    d12 = point2s - point1s

    D1 = np.sum(np.power(d1,2))
    D2 = np.sum(np.power(d2,2))

    S1 = np.sum(d1*d12)
    S2 = np.sum(d2*d12)
    R = np.sum(d1*d2)

    den = D1*D2 - R**2

    if D1 == 0 or D2 == 0:
        if D1 != 0:
            u = 0
            t = S1/D1
            t = np.clip(t, 0, 1)
        elif D2 != 0:
            t = 0
            u = -S2/D2
            U = np.clip(u, 0, 1)
        else:
            t = 0
            u = 0
    elif den == 0:
        t = 0
        u = -S2/D2
        uf = np.clip(u, 0, 1)
        if uf != u:
            t = (uf*R+S1)/D1
            t = np.clip(t, 0, 1)
            u = uf 
    else:
        t = (S1*D2-S2*R)/den
        t = np.clip(t, 0, 1)
        u = (t*R-S2)/D2
        uf = np.clip(u, 0, 1)
        if uf != u:
            t = (uf*R+S1)/D1
            t = np.clip(t, 0, 1)
            u = uf
    # compute distance given parameters t and u 
    dist = np.linalg.norm(d1*t - d2*u - d12)
    # dist = np.sqrt(np.sum(np.power(d1*t-d2*u-d12,2)))
    # compute the cloest point 
    points = np.vstack((point1s + d1*t, point2s+d2*u)).transpose()
    return dist, points