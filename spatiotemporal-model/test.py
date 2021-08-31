import os

from solver_spatiotemporal_wings import Solver
from config_wings import Config

if __name__ == '__main__':
    config = Config()
    solver = Solver(config, isTrain=False)
    solver.build()
    for lmbda in config.lmbda:
        solver.test(lmbda)