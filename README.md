# DIL-GP
The official implementation of **Domain Invariant Learning for Gaussian Processes and Bayesian Exploration**. We propose a domain invariant learning algorithm for Gaussian processes (DIL-GP) with a min-max optimization on the likelihood. DIL-GP discovers the heterogeneity in the data and forces invariance across partitioned subsets of data. 


## How to Run

```
cd DIL-GP
```
---
### 1D Synthetic Dataset


**run GP, DIL-GP**
```
python train_gp_dilgp_syn_1d.py
```

**run GP-RQ Kernel, GP-DP Kernel**
```
python train_gp_kernel_syn_1d.py
```

**run RF, MLP**
```
python train_rf_mlp_syn_1d.py
```
---
### Automobile Dataset
**run GP, DIL-GP**
```
python train_gp_dilgp_real.py
```

**run GP-RQ Kernel, GP-DP Kernel**
```
python train_gp_kernel_real.py
```

**run RF, MLP**
```
python train_rf_mlp_real.py
```



 **Result:**
