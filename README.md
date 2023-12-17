# DIL-GP
The official implementation of **Domain Invariant Learning for Gaussian Processes and Bayesian Exploration**. We propose a domain invariant learning algorithm for Gaussian processes (DIL-GP) with a min-max optimization on the likelihood. DIL-GP discovers the heterogeneity in the data and forces invariance across partitioned subsets of data. 

Full Version: **Arxiv**


## Requirements

We'll organize **requirement.txt** shortly.



## How to Run

**We'll integrate and beautify the code shortly.**
```
cd DIL-GP/
```
---
### 1D Synthetic Dataset

**run all following methods**
```
./run_syn_1d.sh
```
**run GP, DIL-GP**
```
python train_gp_dilgp_syn_1d.py --model_name gp
python train_gp_dilgp_syn_1d.py --model_name dilgp
```

**run GP-RQ Kernel, GP-DP Kernel**
```
python train_gp_kernel_syn_1d.py --kernel RationalQuadraticKernel
python train_gp_kernel_syn_1d.py --kernel DotProductKernel
```

**run RF, MLP**
```
python train_rf_mlp_syn_1d.py --model_name rf
python train_rf_mlp_syn_1d.py --model_name mlp
```
---
### Automobile Dataset

**run all following methods**
```
./run_real.sh
```
**run GP, DIL-GP**
```
python train_gp_dilgp_real.py --model_name gp
python train_gp_dilgp_real.py --model_name dilgp
```

**run GP-RQ Kernel, GP-DP Kernel**
```
python train_gp_kernel_real.py --kernel RationalQuadraticKernel
python train_gp_kernel_real.py --kernel DotProductKernel
```

**run RF, MLP**
```
python train_rf_mlp_real.py --model_name rf
python train_rf_mlp_real.py --model_name mlp
```


