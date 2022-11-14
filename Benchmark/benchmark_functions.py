"""
Benchmark functions
author: Federico Peralta
repository: https://github.com/FedePeralta/BO_drones/blob/master/bin/Utils/utils.py#L45
"""

import numpy as np
from deap import benchmarks
from skopt.benchmarks import branin as brn
from sklearn.preprocessing import normalize
from Environment.bounds import Bounds
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from Environment.peaks_zones import ZonesPeaks
import random


class Benchmark_function():
    def __init__(self, grid, resolution, xs, ys, X_test, initial_seed, vehicles, w_ostacles=False, obstacles_on=False,
                 randomize_shekel=True):
        self.w_obstacles = w_ostacles
        self.vehicles = vehicles
        self.grid = grid
        self.X_test = X_test
        self.resolution = resolution
        self.obstacles_on = obstacles_on
        self.randomize_shekel = randomize_shekel
        self.xs = xs
        self.ys = ys
        self.a = list()
        self.a1 = list()
        self.bench = list()
        self.seed = initial_seed
        self.yukyry, self.pirayu, self.sanber, self.aregua = ZonesPeaks(self.X_test).find_zones()
        return

    def bohachevsky_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else benchmarks.bohachevsky(sol[:2])[0]

    def ackley_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else benchmarks.ackley(sol[:2])[0]

    def rosenbrock_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else benchmarks.rosenbrock(sol[:2])[0]

    def himmelblau_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else benchmarks.himmelblau(sol[:2])[0]

    def branin(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else brn(sol[:2])

    def shekel_arg(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else benchmarks.shekel(sol[:2], self.a, self.c)[0]

    def schwefel_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else benchmarks.schwefel(sol[:2])[0]

    def create_new_map(self):
        self.w_obstacles = self.obstacles_on
        xmin = -5
        xmax = 5
        ymin = 0
        ymax = 10

        if self.randomize_shekel:
            #no_maxima = np.random.randint(3, 6)
            xmin = 5
            xmax = 10
            ymin = 5
            ymax = 10

            #for i in range(no_maxima):
             #   self.a.append([1.2 + np.random.RandomState(self.seed) * 8.8, 1.2 + np.random.RandomState(self.seed) * 8.8])
              #  self.c.append(5)
            #self.a = np.array(self.a)
            #self.c = np.array(self.c).T
            ve = 4
            if self.vehicles == 2:
                num_of_peaks = 2
            else:
                num_of_peaks = np.random.RandomState(self.seed).randint(low=2, high=self.vehicles)
            #num_of_peaks = 4
            #self.a1 = np.random.RandomState(self.seed).random(size=(num_of_peaks, 2))
            #self.a = np.array([[0.5, 0.5], [0.25, 0.25], [0.25, 0.75], [0.9, 0.1]]) * 100
            #print(self.a)

            self.c = np.random.RandomState(self.seed).rand(num_of_peaks, 1) * 400 + 120
            #self.a1 = np.array(self.a1)
            #self.c = np.array(self.c).T

            index_a1 = np.random.RandomState(self.seed).random(size=(num_of_peaks, 1))
            index_a = list()
            #index_a = index_a * len(self.X_test)
            #index_a = index_a.flat
            #for i in range(len(index_a)):
                #arr = self.X_test[round(index_a[i])]
                #arr = arr[::-1]
                #self.a.append(arr)
            random.seed(self.seed)
            if self.vehicles <= 4:
                zone = random.sample(range(4), num_of_peaks)
            else:
                zone = list()
                for i in range(num_of_peaks):
                    zone.append(random.randint(0, 3))
            #print(zone)
            #zone = [random.randrange(1, 4, 1) for i in range(num_of_peaks)]
            for i in range(len(zone)):
                if zone[i] == 0:
                    id1 = index_a1[i] * len(self.yukyry) - 1
                    #print(id1)
                    id2 = self.yukyry[round(id1[0])]
                    arr = self.X_test[id2]
                    arr = arr[::-1]
                    self.a.append(arr)
                    index_a.append(id2)
                elif zone[i] == 1:
                    id1 = index_a1[i] * len(self.pirayu) - 1
                    id2 = self.pirayu[round(id1[0])]
                    arr = self.X_test[id2]
                    arr = arr[::-1]
                    self.a.append(arr)
                    index_a.append(id2)
                elif zone[i] == 2:
                    id1 = index_a1[i] * len(self.sanber) - 1
                    id2 = self.sanber[round(id1[0])]
                    arr = self.X_test[id2]
                    arr = arr[::-1]
                    self.a.append(arr)
                    index_a.append(id2)
                elif zone[i] == 3:
                    id1 = index_a1[i] * len(self.aregua) - 1
                    id2 = self.aregua[round(id1[0])]
                    arr = self.X_test[id2]
                    arr = arr[::-1]
                    self.a.append(arr)
                    index_a.append(id2)
            self.a = np.array(self.a)
            index_a = np.array(index_a)
            #print(self.a)
            #self.c = np.ones((num_of_peaks)) * 250
            #print(self.c)
        else:
            a = 1
            #self.a = np.array([[0.16, 1 / 1.5], [0.9, 0.2 / 1.5]])
            #self.c = np.array([0.15, 0.15]).T


        X1 = np.arange(0, self.grid.shape[1], 1)
        Y1 = np.arange(0, self.grid.shape[0], 1)
        X1, Y1 = np.meshgrid(X1, Y1)
        #print(X1.shape)
        # map_created1 = np.zeros(X.shape)
        #map_created = np.zeros((self.grid.shape[0], self.grid.shape[1]))

        #for i in range(map_created.shape[1]):
         #   for j in range(map_created.shape[0]):
          #      map_created[j, i] = self.shekel_arg((j, i))
        map_created1 = np.fromiter(map(self.shekel_arg, zip(X1.flat, Y1.flat)), dtype=float,
                                  count=X1.shape[0] * X1.shape[1]).reshape(X1.shape)

        #X = np.linspace(0, 1, self.grid.shape[1])
        #Y = np.linspace(0, 1, self.grid.shape[0])
        #X, Y = np.meshgrid(X, Y)
        #print(np.where(map_created1 == np.max(map_created1)))
        #map_created1 = np.zeros(X.shape)

        #for i in range(X.shape[0]):
         #   for j in range(X.shape[1]):
          #      map_created1[i,j] = self.shekel_arg((X[i,j], Y[i,j]))
        #map_created = np.fromiter(map(self.shekel_arg, zip(X.flat, Y.flat, self.grid.flat)), dtype=np.float,
         #               count=X.shape[0] * X.shape[1]).reshape(X.shape)
        #meanz = np.nanmean(map_created1)
        #stdz = np.nanstd(map_created1)
        #map_created2 = (map_created1 - meanz) / stdz
        map_max = np.max(map_created1)
        map_min = np.min(map_created1)
        map_created = list(map(lambda x: (x - map_min)/(map_max - map_min), map_created1))
        map_created = np.array(map_created)
        #print(map_created)

        #fig = plt.figure()
        #ax1 = fig.add_subplot(111)
        #im4 = ax1.imshow(map_created.T, interpolation='bilinear', origin='lower', cmap="jet")
        #plt.show()
        #print(X.shape)
        #print(map_created1[50, 67])
        #index = range(len(map_created))
        #map1 = np.reshape(map_created, (-1, 1))
        #s = sorted(index, reverse=True, key=lambda i: map1[i])
        #print(s[:num_of_peaks], self.c, map1)

        #df_bounds, X_test_or = Bounds(self.resolution, self.xs, self.ys, load_file=False).map_bound()

        #for i in range(len(X_test_or)):
         #   self.bench.append(map_created[X_test_or[i][0], X_test_or[i][1]])

        #bench_function = np.array(self.bench)  # Return solo esto de benchmark function

        bench = list()
        df_bounds_or, X_test_or, bench_list = Bounds(self.resolution, self.xs, self.ys, load_file=False).map_bound()
        for i in range(len(X_test_or)):
            bench.append(map_created[X_test_or[i][0], X_test_or[i][1]])

        bench_function_or = np.array(bench)

        return map_created, bench_function_or, num_of_peaks, index_a


class GroundTruth:

    # TODO: Implementar otras funciones de benchmark.
    # TODO: Corregir el estrechamiento cuando el navigation_map no es cuadrado

    def __init__(self, navigation_map, function_type='shekel', initial_seed=0):

        self.navigation_map = navigation_map

        self.function_type = function_type
        self.seed = initial_seed

        # Randomized parameters of Shekel Function #
        num_of_peaks = np.random.RandomState(self.seed).randint(low=1, high=5)
        self.A = np.random.RandomState(self.seed).random(size=(num_of_peaks, 2))
        self.C = np.ones((num_of_peaks)) * 0.05

    def sample_gt(self):

        X = np.linspace(0, 1, self.grid.shape[0])
        Y = np.linspace(0, 1, self.grid.shape[1])
        X, Y = np.meshgrid(X, Y)
        Z = np.fromiter(map(self.shekel_arg0, zip(X.flat, Y.flat)), dtype=np.float, count=X.shape[0] * X.shape[1]).reshape(X.shape)
        Z = (Z - np.mean(Z))/np.std(Z) # Normalize Z
        nan_mask = np.copy(self.grid)
        nan_mask[self.grid == 0] = np.nan
        z_nan = nan_mask + Z.T

        return z_nan

    def reset_gt(self):
        self.seed += 1
        num_of_peaks = np.random.RandomState(self.seed).randint(low=1, high=5)
        self.A = np.random.RandomState(self.seed).random(size=(num_of_peaks,2))
        self.C = np.ones((num_of_peaks)) * 0.05 #np.random.RandomState(self.seed).normal(0.05,0.01, size = num_of_peaks)

    def shekel_arg0(self,sol):
        return benchmarks.shekel(sol, self.A, self.C)[0]

    def read_gt_deterministically(self, my_seed):
        num_of_peaks = np.random.RandomState(my_seed).randint(low=1, high=5)
        prev_A = np.copy(self.A)
        prev_C = np.copy(self.C)
        self.A = np.random.RandomState(my_seed).random(size=(num_of_peaks, 2))
        self.C = np.random.RandomState(my_seed).normal(0.1, 0.05, size=num_of_peaks)

        # Sample with the provided seed #
        z_nan = self.sample_gt()

        # Restore previous states
        self.A = prev_A
        self.C = prev_C

        return z_nan