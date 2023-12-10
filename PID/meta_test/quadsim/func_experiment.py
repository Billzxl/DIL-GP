#########
### 导入包

import time
import os
import numpy as np
import matplotlib.pyplot as plt
plt.rcdefaults()
from threedeequadsim import quadsim, controller, trajectory, experiments


# test_name = 'hover' # {'hover', 'fig-8','sin_forward','spiral_up'}

test_name_list = ['hover', 'fig-8','sin_forward','spiral_up']
nametag = 'final'

folder = './plots/' + time.strftime('%Y-%m-%d') + '/' + nametag + '/'
if not os.path.isdir(folder):
    os.makedirs(folder)
    # print('Created data folder ' + folder)

#######
### 常量
eta_a = 0.0075

eta_A_threshold_convex = 0.0001
eta_A_convex = eta_A_threshold_convex * 1
eta_A_biconvex = 0.0002
eta_A_deep = 0.05

dim_a = 30
dim_A = 150
layer_sizes = (100, 200)

feature_freq = 0.25

q_kwargs = {
    'Vwind' : np.array((0.0, 0., 0.)),            # mean wind speed
    'wind_model': 'iid-uniform',                # {'iid', 'random-walk'}
    # 'Vwind_cov' : 7.5,                        # how quickly the wind changes
    # 'wind_constraint' : 'hard',               # 'hard' wind constraint limits wind speed to be within Vwind_gust of the mean speed
    'Vwind_gust' : np.array((5.0, 0., 2.5)),    # for hard wind constrant, wind speed is in the range Vwind +/- Vwind_gust
    'wind_update_period' : 2.0,                 # seconds between wind speed changes
    't_stop' : 15.,
}

q_kwargs['Vwind'] = np.array((3.0, 0., 1.))
# q_kwargs['Vwind_cov'] = 7.5
q_kwargs['Vwind_gust'] = np.array((2.0, 0., 1.0))
### 函数定义

def savefig(plottag):
    plt.savefig(folder + plottag + '.pdf', bbox_inches='tight')

"""
目标函数,返回值即为评价指标ACE
"""
def calculate_ACE(update_pid, q_kwargs , Name,test_name, record,opt,need_pdf=True):
    print('start cal')
    # update_pid_template = {
    #     "MC_ROLLRATE_P": 0.4,
    #     "MC_PITCHRATE_P": 0.4,
    #     "MC_YAWRATE_P": 0.1,
    #     "MC_ROLLRATE_I": 0.07,
    #     "MC_PITCHRATE_I": 0.07,
    #     "MC_YAWRATE_I": 0.0005,
    #     "MC_ROLLRATE_D": 0.0016,
    #     "MC_PITCHRATE_D": 0.0016,
    #     "MC_YAWRATE_D": 0.01
    # }

    #####目标任务
    CTRLS = [
        controller.Baseline(integral_control=False, update_pid=update_pid),
        # controller.MetaAdaptBaseline(eta_a_base=eta_a, dim_a=dim_a, dim_A=dim_A, feature_freq=feature_freq, A_type='random', update_pid=update_pid),
        # controller.Omniscient(update_pid=update_pid)
    ]
        
    if test_name == 'hover':
        T = trajectory.hover
        t_kwargs = {
            'pd' : np.zeros(3)
        }
    elif test_name == 'fig-8':
        T = trajectory.fig8
        t_kwargs = {
            'T': 5.
        }
        
    elif test_name == 'sin_forward':
        T = trajectory.sin_forward
        t_kwargs = {
    
        }
    elif test_name == 'spiral_up':
        T = trajectory.spiral_up
        t_kwargs = {
        }
    Data = []
    # seed = np.random.randint(np.iinfo(np.uint).max, dtype=np.uint)
    # seed = 120
    # print("seed:", seed)
    for c in CTRLS:
        # print("c: ", c)
        q = quadsim.QuadrotorWithSideForce(**q_kwargs, )    # create a quadrotor object
        t = T(**t_kwargs)                                   # create a trajectory object and initialize the trajectory

        data = q.run(trajectory=t, controller=c, seed=None)  # run the simulation
                                                            # note: this will set the seed, call c.reset_controller(), then reset the seed again
        Data.append(data)                                   # save the results



    # print(Data[0]['t'])
    # data = Data[0]
    # print(np.mean(data['t'][1:] - data['t'][:-1]))
    # assert 0


    ace = experiments.get_error(Data[0]['X'], Data[0]['pd'])['meanerr']

    if(record):
        
        if(not os.path.exists(('new_logs/'+opt.save_name+'/'+test_name+'/'))):
            os.makedirs('new_logs/'+opt.save_name+'/'+test_name+'/')
        
        # if(not os.path.exists('uplogs/'+test_name+'/')):
        #     os.makedirs('uplogs/'+test_name+'/')
        print('new_logs/'+opt.save_name+'/'+test_name+'/'+Name+'.npy')
        
        np.save('new_logs/'+opt.save_name+'/'+test_name+'/'+Name+'.npy',Data[0]['X'][:,0:3])
        np.save('new_logs/'+opt.save_name+'/'+test_name+'/'+Name+'_result.npy',ace)

    # print(Data[0]['X'][:,0:3].shape)
    # print(Data[0]['pd'].shape)
    # assert 0

    if need_pdf:
        # print('%15s, %6s, %15s' % ('Controller', 'ACE', 'Steady State ACE'))
        err = []
        err_ss = []
        for i, (c, data) in enumerate(zip(CTRLS, Data)):
            err.append(experiments.get_error(data['X'], data['pd']))
            err_ss.append(experiments.get_error(data['X'], data['pd'], istart=1000)) # ss=steady state
            # print('%15s, %5.4f, %5.4f' % (c._name, err[-1]['meanerr'], err_ss[-1]['meanerr']))
        color = ((1,0,0), (0,1,0), (0,0,1)) # colors for x, y and z directions

        fig = plt.figure(figsize=(12, 5))
        # gs = fig.add_gridspec(2,2)
        rows = 1
        cols = 4 # int(len(Data)/rows)
        gs = plt.GridSpec(2 * rows, cols)
        xyz = ['x', 'y', 'z']
        for j, data in enumerate(Data):
            if CTRLS[j]._name in ('pid', 'omniscient'):
                continue

            # print(CTRLS[j]._name)

            row = 0
            col = j - 1

            # Plot position tracking
            ax = fig.add_subplot(gs[row*2, col])
            for i in [0, 2]: # only plot x and z
                plt.plot(data['t'], data['X'][:,i], label=r'$p_' + xyz[i] + '$', color = color[i])
            # Plot vertical lines for each environment switch
            for t in data['t'][data['meta_adapt_trigger']]:
                plt.axvline(t, ls=':', color='k', lw=0.5)

            plt.margins(x=0)

            if j == 1: # len(Data) - 2:
                plt.legend(loc = 'upper right')
            plt.plot(data['t'], data['pd'][:,i], 'k:',)
            ax.axes.xaxis.set_ticklabels([])
            if col == 0:
                plt.ylabel('Position [m]')
            else:
                ax.axes.yaxis.set_ticklabels([])
            if row == rows-1:
                plt.xlabel('Time [s]')
            plt.ylim((-0.5, 0.5))
            plt.title(CTRLS[j].name_long + '\n$ACE=%.3f m$' % (err[j]['meanerr'], ))

            # Plot force estimation
            # ax = fig.add_subplot(gs[row*2+1, col])
            # for i in [0, 2]:
            #     plt.plot(data['t'], data['Fs'][:,i], '--', label=r'$f_' + xyz[i] + '$', color = color[i])
            #     if CTRLS[j]._name == 'baseline':
            #         plt.plot(data['t'], -data['i_term'][:,i], '-', label=r'$K_i\int\tilde{p}_' + xyz[i] + 'dt$', color = color[i])
            #     else:
            #         plt.plot(data['t'], data['f_hat'][:,0,i], '-', label=r'$\hat{f}_' + xyz[i] + '$', color = color[i])
            # # Plot vertical lines for environment changes
            # for t in data['t'][data['meta_adapt_trigger']]:
            #     plt.axvline(t, ls=':', color='k', lw=0.5)
            #
            # plt.margins(x=0)
            #
            # if j == 1 : # len(Data) - 2:
            #     plt.legend(loc = 'upper right')
            # if row == rows-1:
            #     plt.xlabel('Time [s]')
            # else:
            #     ax.axes.xaxis.set_ticklabels([])
            # if col == 0:
            #     plt.ylabel('Force [N]')
            # else:
            #     ax.axes.yaxis.set_ticklabels([])
            # plt.ylim((-8,8))

        plt.tight_layout()
        savefig('performance_combined_'+time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime()))

    return ace


if __name__ == '__main__':
    update_pid_template = {
        "MC_ROLLRATE_P": 0.4,#0.4
        "MC_PITCHRATE_P": 0.4,#0.4
        "MC_YAWRATE_P": 0.4,#0.4
        "MC_ROLLRATE_I": 0.07,#0.07
        "MC_PITCHRATE_I": 0.07,#0.07
        "MC_YAWRATE_I": 0.0005,#0.0005
        "MC_ROLLRATE_D": 0.0047,#0.0016
        "MC_PITCHRATE_D": 0.0053,#0.0016
        "MC_YAWRATE_D": 0.0058#0.01
        
    }
    import argparse
    test_name_list = ['hover', 'fig-8','sin_forward','spiral_up']

    parser = argparse.ArgumentParser(description='OODGP BO')
    opt = parser.parse_args()
    opt.save_name = 'test0514'
    ace_score = calculate_ACE(update_pid=update_pid_template, q_kwargs=q_kwargs, Name='test',test_name = 'fig-8',record = 1, opt=opt,need_pdf=False)
    print("ace_score is:", ace_score)
    debug=1





