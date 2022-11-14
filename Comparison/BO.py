from bayes_opt import BayesianOptimization
import time
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning

simplefilter("ignore", category=ConvergenceWarning)
import matplotlib.pyplot as plt


from PSO.pso_function import PSOEnvironment
import numpy as np

number = 0

pbounds = {'c1': (0, 4), 'c2': (0, 4), 'c3': (0, 4), 'c4': (0, 4)}

resolution = 1
xs = 100
ys = 150
# navigation_map = np.genfromtxt('Image/ypacarai_map_bigger.csv')


def model_psogp(c1, c2, c3, c4):
    action = np.array([c1, c2, c3, c4])

    initial_position = np.array([[0, 0],
                                 [8, 56],
                                 [37, 16],
                                 [78, 81],
                                 [74, 124]])

    method = 0
    pso = PSOEnvironment(resolution, ys, method, initial_seed=1, initial_position=initial_position,
                         reward_function='inc_mse', type_error='contamination')

    last_error = []
    print('in')

    for i in range(200):

        print(i)
        done = False
        state = pso.reset()
        R_vec = []

        # Main part of the code

        while not done:
            state, reward, done, dic = pso.step(action)

            R_vec.append(-reward)

        error_data = np.array(pso.error_value())
        last_error.append(error_data[-1])

    error_total = np.mean(np.array(last_error))
    error = error_total * -1

    return error


optimizer = BayesianOptimization(
            f=model_psogp,
            pbounds=pbounds,
            random_state=1)

optimizer.maximize(init_points=10, n_iter=20, acq='ei')

print(optimizer.max)

for i, res in enumerate(optimizer.res):
    print("Iteration {}: \n\t{}".format(i, res))
