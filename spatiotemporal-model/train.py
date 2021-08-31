import os

from solver_spatiotemporal_wings import Solver
from config_wings import Config
import torch
if __name__ == '__main__':
    # print(os.listdir('compressai'))

    config = Config()
    solver = Solver(config)
    solver.build()
    solver.train()