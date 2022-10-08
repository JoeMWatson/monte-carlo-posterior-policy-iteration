<p float="left">
<img src="https://monte-carlo-ppi.github.io/figs/se_policy_timeshift_smaller.gif" height="100"/>
&nbsp;
<img src="https://joemwatson.github.io/files/door-v0_se.gif" height="100"/>
</p>

## Episodic Monte Carlo Posterior Policy Iteration

Code for 'Inferring Smooth Robot Control: Monte Carlo Posterior Policy Iteration with Gaussian Proceses', presented at Conference on Robot Learning (CoRL) 2022.

If used, please cite
```
@InProceedings{Watson2022CoRL,
  author    = {Watson, Joe and Peters, Jan},
  booktitle = {Conference on Robot Learning},
  title     = {Inferring Smooth Robot Control: Monte Carlo Posterior Policy Iteration with Gaussian Processes},
  year      = {2022},
}
```
 
The MPC code was inspired by https://github.com/facebookresearch/mbrl-lib, and the policy search code builds on the work from 
'Differentiable physics models for real-world offline model-based reinforcement learning', ICRA 2021, provided by Michael Lutter. 

### Code Guide
```
code
│   README.md
│   Makefile: shows experiment commands    
│
└───ppi
│   │   algorithms.py: Gibbs posterior methods for optimization
│   │   policies.py: Kernel- and feature-based action priors 
│
└───optimization       
│   │   run_opt.py: experiment runner
│   │   functions.py: test functions
│
└───mpc
│   │   mpc.py: model predictive control agent
│   │   wrapper.py: MuJoCo interface for optimal control
│   │   metrics.py: metrics used for MPC experiments like smoothness
│   └───model_selection
│       │   extract_mavn.py: get matrix normal from demonstrations
│       │   model_selection.py: fit policy hyperparameters to estimated matrix normals
│
└───policy_search
│   │   run_policy_search.py: experiment runner
│   │   robot_descriptions/ : MuJoCo assets
│
└───viz/ scripts used for figures and gifs for the paper and website
```

### Installation

Installation is documented in the `install` target of the `Makefile`
```
	conda env create -n ppi --file ppi.yaml
	conda activate ppi
	pip install -e .
	git submodule update --init --recursive
	cd mpc/mj_envs && pip install -e .
```


### Experiments

Experiments are documented in the `run_*` Makefile targets.
For more options, see the parser help of each runner. 
