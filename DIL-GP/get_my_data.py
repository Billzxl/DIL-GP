import numpy as np
import pandas as pd
import torch
import random
import os
import sklearn
from sklearn.datasets import load_diabetes

from sklearn.cluster import KMeans

__all__ = ["get_dateset"]

def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_dataset(dataset_name):
    setup_seed(0)
    train_names = None
    
    if dataset_name == "synthetic":
        X = torch.randn(100,1)
        f = torch.sin(X * 2 * np.pi /4).flatten()
        y = f + torch.randn_like(f) * 0.1
        y = y[:,None] * 3

        X2 = torch.randn(100,1) + 6.0
        f = -torch.sin((X2-6.0) * 2 * np.pi / 64).flatten()
        y2 = f + torch.randn_like(f) * 0.1 + 0.5
        y2 = y2[:,None]

        train_data = torch.cat([X, X2[:15]], dim=0)
        train_label = torch.cat([y, y2[:15]], dim=0)
        train_names = [1 for i in range(100)] + [0 for i in range(15)]
        valid_data = X2[20:]
        valid_label = y2[20:]
    
    
    elif dataset_name == "syn2":
        
        T1 = 0.2
        a1 = 1/T1
        T2 = 0.2
        a2 = 1/T2
        k = 1

        b1 = 2*k*np.pi+np.pi/2-0.3*a1
        b2 = 2*k*np.pi+np.pi/2-0.7*a2

        # 生成两个高斯分布
        mean1 = [0.3, 0.3]
        cov1 = [[0.01, 0], [0, 0.01]]
        x1, y1 = np.random.multivariate_normal(mean1, cov1, 20).T

        x1 = np.clip(x1, 0.05, 0.95)
        y1 = np.clip(y1, 0.05, 0.95)
        x1 = torch.Tensor(x1).unsqueeze(1)
        # print(x1.shape)
        y1 = torch.Tensor(y1).unsqueeze(1)
        z = 1.5*np.sin(x1 * a1 + b1) + 1.5*np.sin(y1 * a1 + b1) + torch.randn_like(x1) * 0.1
        X = torch.cat((x1,y1),dim=1)

        mean2 = [0.7, 0.7]
        cov2 = [[0.01, 0], [0, 0.01]]
        x2, y2 = np.random.multivariate_normal(mean2, cov2, 15).T

        x2 = np.clip(x2, 0.05, 0.95)
        y2 = np.clip(y2, 0.05, 0.95)
        x2 = torch.Tensor(x2).unsqueeze(1)
        y2 = torch.Tensor(y2).unsqueeze(1)
        z2 = 2*np.sin(x2*a2+b2) + 4*np.sin(y2*a2+b2) + torch.randn_like(x2) * 0.05
        
        X2 = torch.cat((x2,y2),dim=1)
        # 划分 domain
        # n = len(x2)
        # idx = np.random.choice(n, int(0.2*n), replace=False)
        # x2_domain1, y2_domain1 = x2[idx], y2[idx]
        # x_domain2, y_domain2 = np.delete(x2, idx), np.delete(y2, idx)

        # x_domain1 = np.concatenate((x1,x2_domain1),axis=0)
        # y_domain1 = np.concatenate((y1,y2_domain1),axis=0)

        
        train_data = torch.cat([X, X2[:5]], dim=0)
        train_label = torch.cat([z, z2[:5]], dim=0)
        # valid_data = X2[5:]
        # valid_label = z2[5:]
        
        
        n = 100
        x = torch.linspace(0, 1, n)
        y = torch.linspace(0, 1, n)

        # 使用 torch.meshgrid() 生成网格点坐标
        xx, yy = torch.meshgrid(x, y)
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)
        # print(xx)
        # print(yy)
        # assert 0
        label = 2*np.sin(xx*a2+b2) + 4*np.sin(yy*a2+b2) + torch.randn_like(yy) * 0.05
        # 将 xx 和 yy 拼接成一个 (n**2, 2) 的 Tensor
        data = torch.cat((xx, yy), dim=1)

        # # 按照步长 t 对 points 进行缩放
        # points = points * t

        # 打印生成的 Tensor
        # print(points)

        # x_domain1, y_domain1 = train_data,train_label
        # x_domain2, y_domain2 = valid_data ,valid_label


        valid_data = data
        valid_label = label
    
    elif dataset_name == "auto_mobile":
        '''
        valid: ['sedan', 'hardtop']
        Namespace(amplitude_scale=1.0, eistep=1, envlr=0.01, epoch=100, gplr=0.1, lambdae=3.0, length_scale=1.0, noise_scale=1.0, seed=0, usekmeans=True) 0.81747425
        Namespace(amplitude_scale=1.0, eistep=1, envlr=0.01, epoch=100, gplr=0.1, lambdae=1e-21, length_scale=1.0, noise_scale=1.0, seed=0, usekmeans=True) 0.83216405
        '''
        '''
        kfold:
        Namespace(amplitude_scale=1.0, eistep=1, envlr=0.01, epoch=100, gplr=0.1, lambdae=10, length_scale=1.0, noise_scale=1.0, seed=0, usekmeans=True) 0.85788155
        Namespace(amplitude_scale=1.0, eistep=1, envlr=0.01, epoch=300, gplr=0.1, lambdae=1e-21, length_scale=1.0, noise_scale=1.0, seed=0, usekmeans=True) 0.88315356
        Namespace(amplitude_scale=1.0, eistep=1, envlr=0.01, epoch=100, gplr=0.1, lambdae=1e-21, length_scale=1.0, noise_scale=1.0, seed=0, usekmeans=True) 0.8919655
        '''
        '''
        valid: ['convertible', 'sedan', 'hardtop']
        Namespace(amplitude_scale=1.0, eistep=1, envlr=0.01, epoch=100, gplr=1, lambdae=1e-10, length_scale=1.0, noise_scale=1.0, seed=0, usekmeans=True) error 0.8648686
        Namespace(amplitude_scale=1.0, eistep=1, envlr=0.01, epoch=100, gplr=1, lambdae=30.0, length_scale=1.0, noise_scale=1.0, seed=0, usekmeans=True) error 0.8073124
        '''

        car_names = []

        with open("uci_datasets/auto_mobile/imports-85.data", "r") as f:
            data = f.readlines()
        
        data = [item.split(',') for item in data]
        data = [[item.strip() for item in items] for items in data if '?' not in items]

        indices = []
        for i, item in enumerate(data):
            # print(item[6])
            if item[6] in ['convertible', 'sedan', 'hardtop']:
                indices.append(i)

        data = [[item[0]] + item[3:26] for item in data]
        data_np = list2numpy(data)

        assert np.isnan(data_np).sum() == 0

        target_np = data_np[:, 0]
        data_np = data_np[:, 1:]
        
        # train_indices = np.ones(len(data_np), dtype=bool)
        # train_indices[indices] = False
        # valid_indices = ~train_indices

        valid_indices = indices
        train_indices = [i for i in range(len(data_np)) if i not in indices]
        train_names = [item[4] for i, item in enumerate(data) if i not in indices]
        # print('train_names', train_names)
        # train_indices = indices
        # valid_indices = [i for i in range(len(data_np)) if i not in indices]
        np.random.shuffle(valid_indices)
        # train_indices = np.concatenate([train_indices, valid_indices[:15]], axis=0)
        # valid_indices = valid_indices[15:]

        train_data = torch.from_numpy(data_np[train_indices]).float()
        train_label = torch.from_numpy(target_np[train_indices]).float()
        valid_data = torch.from_numpy(data_np[valid_indices]).float()
        valid_label = torch.from_numpy(target_np[valid_indices]).float()
    
    # if dataset_name != 'synthetic':
        train_mean = train_data.mean(dim=0)
        train_std = train_data.std(dim=0)
        train_label_mean = train_label.mean()
        train_label_std = train_label.std()
        train_data = (train_data - train_mean) / (train_std + 1e-5)
        valid_data = (valid_data - train_mean) / (train_std + 1e-5)
        train_label = (train_label - train_label_mean) / (train_label_std + 1e-5)
        valid_label = (valid_label - train_label_mean) / (train_label_std + 1e-5)


    print(train_data.shape, train_label.shape, valid_data.shape, valid_label.shape)
    # shuffle
    shuffle_indices = np.arange(len(train_data))
    np.random.shuffle(shuffle_indices)
    train_data = train_data[shuffle_indices]
    train_label = train_label[shuffle_indices]
    # train_names = [train_names[idx] for idx in shuffle_indices]
    train_names = []
    # assert False
    return train_data, train_label, valid_data, valid_label, train_names
                    


def list2numpy(labels):
    label_float = []
    label_num = len(labels[0])
    for i in range(label_num):
        try_label = labels[0][i]
        labels_i = [item[i] for item in labels]
        
        try:
            float(try_label)
            label_np = np.array(labels_i, dtype=np.float32).reshape(-1, 1)
        except:
            # label_np = np.array(strlist2onehot(labels_i), dtype=np.float32)
            continue
            # print(try_label, label_np.shape)
        
        label_float.append(label_np)

    label_float = np.concatenate(label_float, axis=1)
    return label_float


def strlist2onehot(label_str):
    unique_items = list(set(label_str))
    label_1hot = np.zeros((len(label_str), len(unique_items)))
    for i in range(len(label_str)):
        raw_label = label_str[i]
        label_1hot[i, unique_items.index(raw_label)] = 1.0

    return label_1hot


# def get_dataset_wenv(dataset_name):
#     setup_seed(0)

#     if dataset_name == "housing_time_split":

#         data = pd.read_csv("data.csv") \
#             .drop(columns=["id", "date", "zipcode"])
#         # print(data.iloc[1])
#         y = data["price"]
#         x = data.drop(columns=["price"])

#         s_idx = x[x["yr_built"] <= 1919].index
#         s_idx = np.array_split(s_idx, [int(len(s_idx) * 0.8)])      ## split index
#         v_idx = s_idx[0]                                            ## validation index
#         t_idx = s_idx[1].append(x[x["yr_built"] < 1919].index)      ## train index

#         train_data = torch.from_numpy(x.loc[t_idx].to_numpy()).float()
#         train_label = torch.from_numpy(y.loc[t_idx].to_numpy()).float()
#         valid_data = torch.from_numpy(x.loc[v_idx].to_numpy()).float()
#         valid_label = torch.from_numpy(y.loc[v_idx].to_numpy()).float()

#     elif dataset_name == "foreset_fire":

#         data = pd.read_csv("uci_datasets/forest_fires/forestfires.csv") \
#             .drop(columns=["day","month"])
        
#         # train_months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
#         # train_months = ["apr", "may", "jun"]
#         # print(data.iloc[1])
#         y = data["area"]
#         x = data.drop(columns=["area"])

#         # s_idx = x[x["month"].isin(train_months)].index
#         # print('s_idx', s_idx)
#         s_idx = np.array_split(s_idx, [int(len(s_idx) * 0.8)])      ## split index
#         v_idx = s_idx[0]                                         ## validation index
#         # t_idx = s_idx[1].append(x[~x["month"].isin(train_months)].index)      ## train index

#         # x = x.drop(columns=["month"])

#         train_data = torch.from_numpy(x.loc[t_idx].to_numpy()).float()
#         train_label = torch.from_numpy(y.loc[t_idx].to_numpy()).float()
#         valid_data = torch.from_numpy(x.loc[v_idx].to_numpy()).float()
#         valid_label = torch.from_numpy(y.loc[v_idx].to_numpy()).float()
    
#     elif dataset_name == "diabetes":
#         # no imrovement

#         data = load_diabetes(scaled=False)
#         data_np = data.data
#         # ['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']
#         target_np = data.target
#         # print(data_np[:, 0], ((data_np[:, 0] < 40) * 1.0).mean())
#         train_indices = np.where(data_np[:, 0] <= 55)[0]
#         valid_indices = np.where(data_np[:, 0] > 55)[0]
#         # print(train_indices.shape, train_indices.shape)
#         np.random.shuffle(valid_indices)
#         train_indices = np.concatenate([train_indices, valid_indices[:30]], axis=0)
#         valid_indices = valid_indices[30:]
#         train_data = torch.from_numpy(data_np[train_indices]).float()
#         train_label = torch.from_numpy(target_np[train_indices]).float()
#         valid_data = torch.from_numpy(data_np[valid_indices]).float()
#         valid_label = torch.from_numpy(target_np[valid_indices]).float()

#     elif dataset_name == "auto_mgp":
#         # gp: 0.6249123
#         # gp irm: 0.86464185
#         # oodgp: 0.8157944

#         with open("uci_datasets/auto_mgp/auto-mpg.data", "r") as f:
#             data = f.readlines()
        
#         data = [item.split()[:8] for item in data]
#         data = np.array([[float(item) for item in items] for items in data if '?' not in items])
#         # print(np.unique(data[:, 6]))
        
#         train_indices = np.where(data[:, 6] <= 79)[0]
#         valid_indices = np.where(data[:, 6] > 79)[0]

#         time_split = [
#             [i for i in range(70+4*j, 74+4*j)] for j in range(5)
#         ]
#         env_w_all = np.zeros(len(data), dtype=int)
#         for i, time_int in enumerate(time_split):
#             env_w = np.isin(data[:, 6], time_int)
#             env_w_all[env_w == True] = i

#         # print('env_w_all', env_w_all)
#         data_np = data[:, 1:7]
#         target_np = data[:, 0]
#         np.random.shuffle(valid_indices)
#         train_data = torch.from_numpy(data_np[train_indices]).float()
#         train_label = torch.from_numpy(target_np[train_indices]).float()
#         valid_data = torch.from_numpy(data_np[valid_indices]).float()
#         valid_label = torch.from_numpy(target_np[valid_indices]).float()

#         train_env = torch.from_numpy(env_w_all[train_indices]).long()
#         valid_env = torch.from_numpy(env_w_all[valid_indices]).long()
    
#     elif dataset_name == "auto_mobile":

#         with open("uci_datasets/auto_mobile/imports-85.data", "r") as f:
#             data = f.readlines()
        
#         data = [item.split(',') for item in data]
#         data = [[item.strip() for item in items] for items in data if '?' not in items]

#         indices = []
#         for i, item in enumerate(data):
#             if item[6] in ['hatchback']:
#                 indices.append(i)

#         # print('?', len(indices), len(data))

#         data = [[item[0]] + item[3:26] for item in data]
#         data_np = list2numpy(data)

#         assert np.isnan(data_np).sum() == 0

#         target_np = data_np[:, 0]
#         data_np = data_np[:, 1:]
        
#         # train_indices = np.ones(len(data_np), dtype=bool)
#         # train_indices[indices] = False
#         # valid_indices = ~train_indices

#         valid_indices = indices
#         train_indices = [i for i in range(len(data_np)) if i not in indices]
#         np.random.shuffle(valid_indices)
#         train_indices = np.concatenate([train_indices, valid_indices[:15]], axis=0)
#         valid_indices = valid_indices[15:]

#         train_data = torch.from_numpy(data_np[train_indices]).float()
#         train_label = torch.from_numpy(target_np[train_indices]).float()
#         valid_data = torch.from_numpy(data_np[valid_indices]).float()
#         valid_label = torch.from_numpy(target_np[valid_indices]).float()
    

#     elif dataset_name == "parkinson_motor":

#         with open("uci_datasets/parkinson/parkinsons_updrs.data", "r") as f:
#             data = f.readlines()
        
#         data = [item.split(',') for item in data]
#         labels = [item.strip() for item in data[0]]
#         # print('labels', labels)
#         label_idx = labels.index("motor_UPDRS")

#         data = data[1:]
#         data = np.array(
#             [[float(item.strip()) for item in items] for items in data if '?' not in items]
#         )

#         # print('data[:, 1].astype(int)', data[:, 0].astype(int))
#         train_indices = np.where(
#             np.isin(data[:, 0].astype(int), [i for i in range(30, 43)])
#         )[0]
#         valid_indices = [i for i in range(len(data)) if i not in train_indices]

#         data_np = np.concatenate([data[:, :label_idx], data[:, label_idx+1:]], axis=1)
#         target_np = data[:, label_idx]
#         train_data = torch.from_numpy(data_np[train_indices]).float()
#         train_label = torch.from_numpy(target_np[train_indices]).float()
#         valid_data = torch.from_numpy(data_np[valid_indices]).float()
#         valid_label = torch.from_numpy(target_np[valid_indices]).float()

#     elif dataset_name == "parkinson_sound":
#         # gp irm wrong： 0.76386774
#         # gp irm: 1.1017772
#         # gp: 1.0132746
#         # ood irm: 1.1532074

#         with open("uci_datasets/Parkinson_Multiple_Sound_Recording/train_data.txt", "r") as f:
#             data = f.readlines()
        
#         data = [item.split(',') for item in data]
#         labels = [item.strip() for item in data[0]]

#         data = np.array(
#             [[float(item.strip()) for item in items] for items in data]
#         )

#         # print('unique', np.unique(data[:, 0].astype(int)))
#         train_indices = np.where(
#             np.isin(data[:, 0].astype(int), [i+3*i for i in range(10)])
#         )[0]
#         valid_indices = [i for i in range(len(data)) if i not in train_indices]

#         data_np = data[:, 1:27]
#         target_np = data[:, 27]
#         # print(data_np[:5], target_np[:5])
#         train_data = torch.from_numpy(data_np[train_indices]).float()
#         train_label = torch.from_numpy(target_np[train_indices]).float()
#         valid_data = torch.from_numpy(data_np[valid_indices]).float()
#         valid_label = torch.from_numpy(target_np[valid_indices]).float()

#         train_env = torch.from_numpy(data[train_indices, 0]).long()
#         valid_env = torch.from_numpy(data[train_indices, 0]).long()


#     if dataset_name != 'synthetic':
#         train_mean = train_data.mean(dim=0)
#         train_std = train_data.std(dim=0)
#         train_label_mean = train_label.mean()
#         train_label_std = train_label.std()
#         train_data = (train_data - train_mean) / (train_std + 1e-5)
#         valid_data = (valid_data - train_mean) / (train_std + 1e-5)
#         train_label = (train_label - train_label_mean) / (train_label_std + 1e-5)
#         valid_label = (valid_label - train_label_mean) / (train_label_std + 1e-5)

#     print(train_data.shape, train_label.shape, valid_data.shape, valid_label.shape)
#     # shuffle
#     shuffle_indices = np.arange(len(train_data))
#     np.random.shuffle(shuffle_indices)
#     train_data = train_data[shuffle_indices]
#     train_label = train_label[shuffle_indices]

#     return train_data, train_label, valid_data, valid_label, train_env, valid_env



# get_dataset("paris_housing")