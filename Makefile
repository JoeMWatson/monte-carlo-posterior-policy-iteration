install:
	conda env create -n ppi --file ppi.yaml
	conda activate ppi
	pip install -e .
	git submodule update --init --recursive
	cd mpc/mj_envs && pip install -e .

tidy:
	black .
	isort . --profile black

run_optimization:
	python optimization/run_opt.py Reps NoisySphere --dimension 20 mc --n-samples 100

run_policy_search:
	python policy_search/run_policy_search.py Reps BallInACup RbfFeatures --epsilon 2.0 --n-iters 40 --seed 0 --dir _results MonteCarlo --n-samples 128

run_mpc_whitenoise:
	python run_mpc.py Cem door-v0 WhiteNoiseIid --n-elites 10 --dir _results MonteCarlo --n-samples 64

run_mpc_smooth_kernel:
	python run_mpc.py Lbps door-v0 SquaredExponentialKernel --delta 0.9 --n-iters 2 --anneal 0.5 --dir _results MonteCarlo --n-samples 64

run_mpc_features:
	python run_mpc.py Essps hammer-v0 RffFeatures --n-elites 10 --dir _reviewer_results MonteCarlo --n-samples 64

mpc_smoketest:
	python run_mpc.py Cem door-v0 WhiteNoiseIid --no-plots --dir _del --timesteps 5 --n-warmstart-iters 5 MonteCarlo --n-samples 5

