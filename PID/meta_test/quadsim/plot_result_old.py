from turtle import width
import matplotlib.pyplot as plt
import numpy as np
from threedeequadsim import trajectory
import matplotlib
import argparse
import os
from mpl_toolkits.mplot3d import Axes3D

# t = trajectory.fig8()
def load_data(Name,t):
    # print('logs/'+t._name+'/'+Name+'.npy')
    # assert 0
    log = np.load('logs/'+t._name+'/'+Name+'.npy', allow_pickle=True)
 
    return log

dt = 0.01

def get_ground_truth(t, len):
    seq_len = int(len/dt+1)
    gt = np.zeros((seq_len,3))
    for i in range(seq_len):
        pd = t(i*dt)[0]
        # print(i*dt,t(i*dt))
        gt[i, :] = pd
    return gt

def plot_3D_trace(Names):
    for i,name in enumerate(Names):
        fig = plt.figure()
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlabel("x[m]")
        ax.set_ylabel("y[m]")
        ax.set_zlabel("z[m]")

        colors = ['r', 'g', 'b', 'y', 'm']

        gt = get_ground_truth(t, 30)
        # print(gt)
        ax.plot(gt[:,0], gt[:,1], gt[:,2], '--',lw = 3,c = 'black')

    
        log = load_data(name,t)
        # print(log.shape)
        # ax.plot(log[:,0], log[:,1], log[:,2], c=colors[i], label=name)
        ax.scatter(log[:,0], log[:,1], log[:,2], c=1000*np.sqrt((log[:,0]-gt[:,0])**2+(log[:,2]-gt[:,2])**2+(log[:,1]-gt[:,1])**2), label=name,s=2)
    
        ax.set_xlim((0, 0.4))
        ax.set_ylim((-0.04, 0.04))
        ax.set_zlim((-0.2, 0.2))
        # ax.set_xticks([0, 1, 2])
        # ax.set_yticks([4, 5, 6])
        # ax.set_zticks([0, 4, 8])
        plt.title(f'{t._name} {name}')
        # plt.legend(loc='upper right')
        if not os.path.exists('./traces'):
            os.makedirs('./traces')
        plt.savefig("traces/"+t._name+f" {name}.jpg", dpi=150)
        plt.close()
        plt.clf()
    #plt.show()

def show_project(name,t):
    gt = get_ground_truth(t, 30)
    plt.scatter(gt[:,0], gt[:,2], s=1, c='k', label='ground truth')
    log = load_data(name,t)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.2)
    plt.scatter(log[:,0], log[:,2], c=0.1*np.sqrt((log[:,0]-gt[:,0])**2+(log[:,2]-gt[:,2])**2), cmap='rainbow', label=name, s=2, norm=norm)
    plt.colorbar()
    plt.xlim((-0.1,0.5))
    plt.ylim((-0.2, 0.2))
    #plt.legend(loc='upper right')
    if not os.path.exists('./projections'):
        os.makedirs('./projections')
    plt.savefig("projections/project_"+t._name+'_'+name+".jpg", dpi=150)
    plt.close()
    #plt.show()

parser = argparse.ArgumentParser()
if __name__=='__main__':
    parser.add_argument('--trace', type=str, default='fig-8')
    args = parser.parse_args()
    if (args.trace=='hover'):
        t = trajectory.hover()
    elif (args.trace=='fig8'):
        t = trajectory.fig8()
    elif (args.trace=='spiral'):
        t = trajectory.spiral_up()
    elif (args.trace=='sin'):
        t = trajectory.sin_forward()
        
    # Models = ['test','OODGP']
    Models = ['test']

    plot_3D_trace(Models)
    
    for model in Models:
        show_project(model,t)
