import pathlib as pl
from pathlib import Path

import colorednoise
import matplotlib.pyplot as plt
import numpy as np

data = np.load(
    pl.Path(__file__).parent.resolve() / "HumanoidStandup-v2_cem_mpc_data.npz"
)
act = data["actions"]
rewards = data["rewards"]

fig, ax = plt.subplots()
ax.plot(rewards[:10, :].T)

fig, axs = plt.subplots(10)
for i, ax in enumerate(axs):
    ax.plot(act[:5, :250, i].T, alpha=0.3)

plt.show()
