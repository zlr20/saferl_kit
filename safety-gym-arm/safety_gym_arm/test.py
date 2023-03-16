#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

# link1 = np.array([[0,0,0],[0,0,0.01]]).T
# link2 = np.array([[0,0,0],[0,0,1.5]]).T
# link3 = np.array([[0,0,0],[0,0,0.01]]).T
# link4 = np.array([[0,0,0],[0,1.0,0]]).T
# link5 = np.array([[0,0,0],[0,0,0.01]]).T
# link6 = np.array([[0,0,0],[0,0.5,0]]).T
# link = np.array([link1, link2, link3, link4, link5, link6])
# print(link.shape)

data = np.loadtxt("link_pos.txt")
# data = np.unique(data,axis=0)
print(data[:6,:])
print(data.shape)
T = []
Link = []
fig = plt.figure(1)
ax = fig.gca(projection='3d')
sample = 100
pos_idx = np.array([i for i in range(sample)])*int(data.shape[0] / (7 * (sample + 5) ))
plt.ion()
# pos_idx = [0]
for i in pos_idx:
    ax.clear()
    for j in range(6):
        state = data[i*7+j,:]
        pos = state[0:3]
        mat = state[3:12]
        mat = np.reshape(mat, [3,3])
        pos = np.reshape(pos, [3,1])

        state_next = data[i*7+ j + 1,:]
        pos_next = state_next[0:3]
        mat_next = state_next[3:12]
        
        mat_next = np.reshape(mat_next, [3,3])
        pos_next = np.reshape(pos_next, [3,1])
        
        curT = np.concatenate((mat, pos), axis = 1)
        link = np.concatenate((pos, pos_next), axis = 1)
        print(link)
        curLink = link
        T.append(curT)
        Link.append(curLink)
        
        ax.plot(curLink[0,:], curLink[1,:], curLink[2,:], c='r')
        ax.scatter(curLink[0,:], curLink[1,:], curLink[2,:], c='b')
        
    ax.set_xlim(-1, 2)
    ax.set_ylim(-1, 2)
    ax.set_zlim(0, 3)
    plt.show()
    plt.pause(0.02)
plt.ioff()
plt.show()


