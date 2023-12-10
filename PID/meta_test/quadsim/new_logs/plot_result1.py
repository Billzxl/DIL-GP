from turtle import width
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
import argparse
import os
import sys
sys.path.append('..')
from PID.meta_test.quadsim.threedeequadsim import trajectory
from mpl_toolkits.mplot3d import Axes3D

from matplotlib import cm as mpl_cm, colors as mpl_colors
from matplotlib.transforms import Bbox
from matplotlib.colors import LinearSegmentedColormap
#不同轨迹的坐标取值范围不同，画图时同种轨迹不同方法间统一坐标范围
scale_dict = {
'hover':[(),(),()],
'fig-8':[(),(),()],
'sin':[(),(),()],
'spiral':[(),(),()]
    
}

def load_iter(data_path,Name,t):
    
    x = np.load(data_path+t._name+'/'+Name+'_iteration_x.npy', allow_pickle=True)
    y1 = np.load(data_path+t._name+'/'+Name+'_iteration_allenv.npy', allow_pickle=True)
    y2 = np.load(data_path+t._name+'/'+Name+'_iteration_newenv.npy', allow_pickle=True)
    
    return x,y1.squeeze(),y2.squeeze()


# t = trajectory.fig8()
def load_data(data_path,Name,t):
    
    log = np.load(data_path+t._name+'/'+Name+'.npy', allow_pickle=True)
    ace = np.load(data_path+t._name+'/'+Name+'_result.npy', allow_pickle=True)
 
    return log,ace

dt = 0.01

def get_ground_truth(t, len):
    seq_len = int(len/dt+1)
    gt = np.zeros((seq_len,3))
    for i in range(seq_len):
        pd = t(i*dt)[0]
        # print(i*dt,t(i*dt))
        gt[i, :] = pd
    return gt


# cm = mpl_cm.get_cmap('jet')
# errors_normed = (errors - errors_min) / (errors_max - errors_min)

# # errors_normed = 1 - (1 - errors_normed)**2
# # print('before color')
# cmap_normed = [cm(i) for i in errors_normed.reshape(-1)]
# cmap_normed = np.array(cmap_normed).reshape(6890, 4)




def plot_iter(data_path,models,t):#['oodgp','gp','rf','mlp']
    
    y1_list = []
    y2_list = []
    
    for name in models:
        _,y1,y2 =load_iter(data_path,name,t)
        y1_list.append(y1)
        y2_list.append(y2)
        
    x = np.arange(len(y1_list[0]))
    fig, ax = plt.subplots()
    ax.plot(x, y1_list[0], label='oodgp')
    ax.plot(x, y1_list[1], label='gp')
    ax.plot(x, y1_list[2], label='rf')
    ax.plot(x, y1_list[3], label='mlp')

    ax.legend()
    
    if not os.path.exists(datapath+"fig/iter_y1/"):
        os.makedirs(datapath+"fig/iter_y1/")
        
    # plt.title('iter on all domains')
    plt.savefig(datapath+"fig/iter_y1/"+t._name+".jpg", dpi=300, bbox_inches='tight')
    plt.clf()
    
    fig, ax = plt.subplots()
    ax.plot(x, y2_list[0], label='oodgp')
    ax.plot(x, y2_list[1], label='gp')
    ax.plot(x, y2_list[2], label='rf')
    ax.plot(x, y2_list[3], label='mlp')

    ax.legend()
    
    # plt.title('iter on new domains')
    if not os.path.exists(datapath+"fig/iter_y2/"):
        os.makedirs(datapath+"fig/iter_y2/")
        
    plt.savefig(datapath+"fig/iter_y2/"+t._name+".jpg", dpi=300, bbox_inches='tight')
    plt.clf()


def plot_3D_trace(datapath,name,t):
    # Models = ['oodgp','gp','rf','mlp']
    # Models = ['oodgp']
    Models = ['test']
    
    errors_list = []
    data_list = []
    
    
    for name in Models:
        log, ace = load_data(datapath,name,t)   
        data_list.append(log)
        gt = get_ground_truth(t, 15)
        errors = (log[:,0]-gt[:,0])**2+(log[:,2]-gt[:,2])**2+(log[:,1]-gt[:,1])**2
        errors_list.append(np.array(errors))
        
    gt = get_ground_truth(t, 15) 
    data_list.append(gt)
    data_all = np.vstack(data_list)
    
    min_data = np.min(data_all,axis=0)
    max_data = np.max(data_all,axis=0)
    
    # all_errors = np.concatenate(np.array(errors_list).reshape(-1,1),axis = 1)
    all_errors = np.stack(errors_list).T
    # print(all_errors.shape)
    errors_min = np.min(all_errors,axis = 1)
    errors_max = np.max(all_errors,axis = 1)
    
    
    
    # assert 0
    for i,name in enumerate(Models):
        
        cdict = {'red':   [(0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0)],
         'green': [(0.0, 0.8, 0.8),
                   (1.0, 0.0, 0.0)],
         'blue':  [(0.0, 0.2, 0.2),
                   (1.0, 0.0, 0.0)]}

        cm = LinearSegmentedColormap('GreenToRed', cdict)
        
        # cm = mpl_cm.get_cmap()
        errors_normed = (errors_list[i] - errors_min) / (errors_max - errors_min+1e-5)
        cmap_normed = [cm(i) for i in errors_normed.reshape(-1)]
        cmap_normed = np.array(cmap_normed).reshape(len(errors_normed), 4)


        fig = plt.figure()
        # ax = fig.gca(projection='3d')
        ax = fig.add_subplot(111, projection='3d')
        # ax.set_xlabel("x[m]")
        # ax.set_ylabel("y[m]")
        # ax.set_zlabel("z[m]")
        ax.view_init(elev=20, azim=-80)
        colors = ['r', 'g', 'b', 'y', 'm']

        # print(gt)
        ax.plot(gt[:,0], gt[:,1], gt[:,2], '--',lw = 2,c = 'black')


        log, ace = load_data(datapath,name,t)
        # print(log.shape)
        # ax.plot(log[:,0], log[:,1], log[:,2], c=colors[i], label=name)
        ax.scatter(log[:,0], log[:,1], log[:,2], color = cmap_normed,label=name,s=4)
        # ax.tick_params(axis='x', which='both', direction='in', pad=1)
        # ax.tick_params(axis='y', which='both', direction='in', pad=1)
        # ax.tick_params(axis='z', which='both', direction='in', pad=1)
        
        # ax.set_xticks(['0'])
        # ax.set_yticks(['0'])
        # ax.set_zticks(['0'])
        
        # ax.set_xticklabels(ax.get_xticks(), fontsize=1)
        # ax.set_yticklabels(ax.get_yticks(), fontsize=1) 
        # ax.set_zticklabels(ax.get_zticks(), fontsize=1) 
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        # ax.text(0.95, 0.95, '文本内容', ha='right', va='top', transform=ax.transAxes)
        # text = 'Error = {:.3f}'.format(ace)
        
        # ax.text(0.25, 0.04, 0.25, s = 'Error = {:.3f}'.format(ace),fontsize=12)
       
        fig.text(0.65, 0.81,s = 'Error = {:.3f}'.format(ace),fontsize=15, ha='right', va='top')
        
        #设置坐标范围
        ax.set_xlim((min_data[0], max_data[0]))
        ax.set_ylim((min_data[1], max_data[1]))
        ax.set_zlim((min_data[2], max_data[2]))
        
        # ax.set_xticks([0, 1, 2])
        # ax.set_yticks([4, 5, 6])
        # ax.set_zticks([0, 4, 8])
        # plt.title(f'{t._name} {name}')
        # plt.legend(loc='upper right')
        if not os.path.exists(datapath+'fig/traces'):
            os.makedirs(datapath+'fig/traces')
        # fig.tight_layout()
        plt.savefig(datapath+"fig/traces/"+t._name+f" {name}.jpg", dpi=500, bbox_inches=Bbox.from_bounds(1.82, 1.02, 2.9, 2.9))
        plt.close()
        plt.clf()
        #plt.show()

def show_project(datapath,name,t):
    gt = get_ground_truth(t, 30)
    plt.scatter(gt[:,0], gt[:,2], s=1, c='k', label='ground truth')
    log, ace = load_data(datapath,name,t)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.2)
    plt.scatter(log[:,0], log[:,2], c=0.1*np.sqrt((log[:,0]-gt[:,0])**2+(log[:,2]-gt[:,2])**2), cmap='rainbow', label=name, s=2, norm=norm)
    plt.colorbar()
    
    #设置坐标范围
    # plt.xlim((-0.1,0.5))
    # plt.ylim((-0.2, 0.2))
    
    
    #plt.legend(loc='upper right')
    if not os.path.exists(datapath+'fig/projections'):
        os.makedirs(datapath+'fig/projections')
    plt.savefig(datapath+"fig/projections/project_"+t._name+'_'+name+".jpg", dpi=300,bbox_inches='tight')
    plt.close()
    #plt.show()

parser = argparse.ArgumentParser()
if __name__=='__main__':
    
    t_list = ['hover','fig8','sin','spiral']
    # t_list = ['hover']

    datapath = './test0513/'


    for t_name in t_list:
    # parser.add_argument('--trace', type=str, default='hover')
        # args = parser.parse_args()
        if (t_name=='hover'):
            t = trajectory.hover()
        elif (t_name=='fig8'):
            t = trajectory.fig8()
        elif (t_name=='spiral'):
            t = trajectory.spiral_up()
        elif (t_name=='sin'):
            t = trajectory.sin_forward()
        else:
            assert 0
            
        # Models = ['oodgp']
        Models = ['oodgp','gp','rf','mlp']
        
        
        
        # plot_iter(datapath,Models,t)
        for model in Models:
            plot_3D_trace(datapath,model,t)
            # show_project(datapath,model,t)
