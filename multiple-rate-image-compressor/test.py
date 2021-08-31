import os

from solver import Solver
from config import Config

if __name__ == '__main__':
    config = Config()
    solver = Solver(config, isTrain=False)
    solver.build()
    for lmbda in config.lmbda:
        solver.test(lmbda)