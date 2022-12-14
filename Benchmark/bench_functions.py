"""
Benchmark functions
author: Federico Peralta
repository: https://github.com/FedePeralta/BO_drones/blob/master/bin/Utils/utils.py#L45
"""

import random

from deap import benchmarks
from deap.benchmarks.tools import translate, rotate, scale
from skopt.benchmarks import branin as brn

from Environment.bounds import Bounds
from Environment.peaks_zones import ZonesPeaks
from Environment.map import Map
import matplotlib.image as img
import numpy as np
import matplotlib.pyplot as plt


@translate([10, 10])
@scale([0.98, 0.98])
@rotate(np.linalg.qr(np.random.random((2, 2)))[0])
def ackley_modified(sol):
    return benchmarks.ackley(sol)


@translate([10, 10])
@scale([0.98, 0.98])
@rotate(np.linalg.qr(np.random.random((2, 2)))[0])
def ackley_modified(sol):
    return benchmarks.ackley(sol)


@translate([10, 10])
@scale([0.98, 0.98])
@rotate(np.linalg.qr(np.random.random((2, 2)))[0])
def bohachevsky_modified(sol):
    return benchmarks.bohachevsky(sol)


@translate([10, 10])
@scale([0.98, 0.98])
@rotate(np.linalg.qr(np.random.random((2, 2)))[0])
def griewank_modified(sol):
    return benchmarks.griewank(sol)


@translate([10, 10])
@scale([0.98, 0.98])
@rotate(np.linalg.qr(np.random.random((2, 2)))[0])
def h1_modified(sol):
    return benchmarks.h1(sol)


@translate([10, 10])
@scale([0.98, 0.98])
@rotate(np.linalg.qr(np.random.random((2, 2)))[0])
def himmelblau_modified(sol):
    return benchmarks.himmelblau(sol)


@translate([10, 10])
@scale([0.98, 0.98])
@rotate(np.linalg.qr(np.random.random((2, 2)))[0])
def rastrigin_modified(sol):
    return benchmarks.rastrigin(sol)


@translate([10, 10])
@scale([0.98, 0.98])
@rotate(np.linalg.qr(np.random.random((2, 2)))[0])
def rosenbrock_modified(sol):
    return benchmarks.rosenbrock(sol)


@translate([10, 10])
@scale([0.98, 0.98])
@rotate(np.linalg.qr(np.random.random((2, 2)))[0])
def schaffer_modified(sol):
    return benchmarks.schaffer(sol)


@translate([10, 10])
@scale([0.98, 0.98])
@rotate(np.linalg.qr(np.random.random((2, 2)))[0])
def schwefel_modified(sol):
    return benchmarks.schwefel(sol)


class Benchmark_function():
    def __init__(self, grid, resolution, xs, ys, X_test, initial_seed, vehicles, w_ostacles=False, obstacles_on=False,
                 randomize_shekel=True, base_benchmark="shekel"):
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
        # if base_benchmark == "shekel":
        #     self.yukyry, self.pirayu, self.sanber, self.aregua = ZonesPeaks(self.X_test).find_zones()
        self.base_benchmark = base_benchmark
        return

    def ackley_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else ackley_modified(sol[:2])[0]

    def bohachevsky_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else bohachevsky_modified(sol[:2])[0]

    def griewank_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else griewank_modified(sol[:2])[0]

    def h1_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else h1_modified(sol[:2])[0]

    def himmelblau_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else himmelblau_modified(sol[:2])[0]

    def rastrigin_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else rastrigin_modified(sol[:2])[0]

    def rosenbrock_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else rosenbrock_modified(sol[:2])[0]

    def schaffer_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else schaffer_modified(sol[:2])[0]

    def schwefel_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else schwefel_modified(sol[:2])[0]

    def shekel_arg(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else benchmarks.shekel(sol[:2], self.a, self.c)[0]

    def branin(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else brn(sol[:2])

    def create_new_map(self):
        self.w_obstacles = self.obstacles_on

        if self.randomize_shekel:

            # if self.vehicles == 2:
            #     num_of_peaks = 2
            # else:
            #     num_of_peaks = np.random.RandomState(self.seed).randint(low=2, high=self.vehicles)
            #
            # self.c = np.random.RandomState(self.seed).rand(num_of_peaks, 1) * 400 + 120
            #
            # index_a1 = np.random.RandomState(self.seed).random(size=(num_of_peaks, 1))
            # index_a = list()
            #
            # random.seed(self.seed)
            # if self.vehicles <= 4:
            #     zone = random.sample(range(4), num_of_peaks)
            # else:
            #     zone = list()
            #     for i in range(num_of_peaks):
            #         zone.append(random.randint(0, 3))
            # for i in range(len(zone)):
            #     if zone[i] == 0:
            #         id1 = index_a1[i] * len(self.yukyry) - 1
            #         id2 = self.yukyry[round(id1[0])]
            #         arr = self.X_test[id2]
            #         arr = arr[::-1]
            #         self.a.append(arr)
            #         index_a.append(id2)
            #     elif zone[i] == 1:
            #         id1 = index_a1[i] * len(self.pirayu) - 1
            #         id2 = self.pirayu[round(id1[0])]
            #         arr = self.X_test[id2]
            #         arr = arr[::-1]
            #         self.a.append(arr)
            #         index_a.append(id2)
            #     elif zone[i] == 2:
            #         id1 = index_a1[i] * len(self.sanber) - 1
            #         id2 = self.sanber[round(id1[0])]
            #         arr = self.X_test[id2]
            #         arr = arr[::-1]
            #         self.a.append(arr)
            #         index_a.append(id2)
            #     elif zone[i] == 3:
            #         id1 = index_a1[i] * len(self.aregua) - 1
            #         id2 = self.aregua[round(id1[0])]
            #         arr = self.X_test[id2]
            #         arr = arr[::-1]
            #         self.a.append(arr)
            #         index_a.append(id2)
            # self.a = np.array(self.a)
            # index_a = np.array(index_a)

            no_maxima = np.random.RandomState(self.seed).randint(4, 6)
            xmin = 0
            xmax = 10
            ymin = 0
            ymax = 10
            self.a = []
            self.c = []
            seed_a = self.seed
            for i in range(no_maxima):
                seed_a += 1000
                seed_b = seed_a + 10
                self.a.append([np.random.RandomState(seed_a).rand() * 100, np.random.RandomState(seed_b).rand() * 150])
                self.c.append(1)
            self.a = np.array(self.a)
            self.c = np.array(self.c).T
        else:
            self.a = np.array([[0.16, 1 / 1.5], [0.9, 0.2 / 1.5]])
            self.c = np.array([0.15, 0.15]).T

        self.c = self.c * 400 + 120

        # Como los distintos benchmarks(BM) tienen distintos bounds, vamos a cambiar X1 e Y1 de acuerdo al benchmark
        # seleccionado
        random.seed(self.seed)

        if self.base_benchmark == "ackley":
            stepx, stepy = 60 / self.grid.shape[1], 60 / self.grid.shape[0]
            print(self.grid.shape)
            xmin, xmax, ymin, ymax = -30 + stepx / 2, 30, -30 + stepy / 2, 30

            a1 = np.random.RandomState(self.seed).random(size=2) * 10
            ackley_modified.translate(a1)  # TODO reemplazar 0.0 0.0 por literalmente cualquier random
            ackley_modified.scale(
                [1,
                 1])  # TODO reemplazar cuidadosamente estos valores, tienen que ser cercanos a uno (hasta 5 funciona bien creo)
            rot_matrix_quat, _ = np.linalg.qr(np.identity(2))
            ackley_modified.rotate(rot_matrix_quat)  # TODO reemplazar por cualquier matriz de rotacion

            bm_func = self.ackley_arg0
        elif self.base_benchmark == "bohachevsky":
            stepx, stepy = 30 / self.grid.shape[1], 30 / self.grid.shape[0]
            xmin, xmax, ymin, ymax = -15 + stepx / 2, 15, -15 + stepy / 2, 15

            a2 = np.random.RandomState(self.seed).random(size=2) * 10
            bohachevsky_modified.translate(a2)  # TODO reemplazar 0.0 0.0 por literalmente cualquier random
            bohachevsky_modified.scale(
                [1,
                 1])  # TODO reemplazar cuidadosamente estos valores, tienen que ser cercanos a uno (hasta 5 funciona bien creo)
            rot_matrix_quat, _ = np.linalg.qr(np.identity(2))
            bohachevsky_modified.rotate(rot_matrix_quat)  # TODO reemplazar por cualquier matriz de rotacion

            bm_func = self.bohachevsky_arg0
        elif self.base_benchmark == "griewank":
            stepx, stepy = 100 / self.grid.shape[1], 100 / self.grid.shape[0]
            xmin, xmax, ymin, ymax = -50 + stepx / 2, 50, -50 + stepy / 2, 50

            a3 = np.random.RandomState(self.seed).random(size=2) * 10
            griewank_modified.translate(a3)  # TODO reemplazar 0.0 0.0 por literalmente cualquier random
            griewank_modified.scale(
                [1,
                 1])  # TODO reemplazar cuidadosamente estos valores, tienen que ser cercanos a uno (hasta 5 funciona bien creo)
            rot_matrix_quat, _ = np.linalg.qr(np.identity(2))
            griewank_modified.rotate(rot_matrix_quat)  # TODO reemplazar por cualquier matriz de rotacion

            bm_func = self.griewank_arg0
        elif self.base_benchmark == "h1":
            stepx, stepy = 50 / self.grid.shape[1], 50 / self.grid.shape[0]
            xmin, xmax, ymin, ymax = -25 + stepx / 2, 25, -25 + stepy / 2, 25

            a4 = np.random.RandomState(self.seed).random(size=2) * 10
            h1_modified.translate(a4)  # TODO reemplazar 0.0 0.0 por literalmente cualquier random
            h1_modified.scale(
                [1,
                 1])  # TODO reemplazar cuidadosamente estos valores, tienen que ser cercanos a uno (hasta 5 funciona bien creo)
            rot_matrix_quat, _ = np.linalg.qr(np.identity(2))
            h1_modified.rotate(rot_matrix_quat)  # TODO reemplazar por cualquier matriz de rotacion

            bm_func = self.h1_arg0
        elif self.base_benchmark == "himmelblau":
            stepx, stepy = 12 / self.grid.shape[1], 12 / self.grid.shape[0]
            xmin, xmax, ymin, ymax = -6 + stepx / 2, 6, -6 + stepy / 2, 6

            a5 = np.random.RandomState(self.seed).random(size=2) * 10
            himmelblau_modified.translate(a5)  # TODO reemplazar 0.0 0.0 por literalmente cualquier random
            himmelblau_modified.scale(
                [1,
                 1])  # TODO reemplazar cuidadosamente estos valores, tienen que ser cercanos a uno (hasta 5 funciona bien creo)
            rot_matrix_quat, _ = np.linalg.qr(np.identity(2))
            himmelblau_modified.rotate(rot_matrix_quat)  # TODO reemplazar por cualquier matriz de rotacion

            bm_func = self.himmelblau_arg0
        elif self.base_benchmark == "rastrigin":
            stepx, stepy = 10 / self.grid.shape[1], 10 / self.grid.shape[0]
            xmin, xmax, ymin, ymax = -5 + stepx / 2, 5, -5 + stepy / 2, 5

            a6 = np.random.RandomState(self.seed).random(size=2) * 10
            rastrigin_modified.translate(a6)  # TODO reemplazar 0.0 0.0 por literalmente cualquier random
            rastrigin_modified.scale(
                [1,
                 1])  # TODO reemplazar cuidadosamente estos valores, tienen que ser cercanos a uno (hasta 5 funciona bien creo)
            rot_matrix_quat, _ = np.linalg.qr(np.identity(2))
            rastrigin_modified.rotate(rot_matrix_quat)  # TODO reemplazar por cualquier matriz de rotacion

            bm_func = self.rastrigin_arg0
        elif self.base_benchmark == "rosenbrock":
            stepx, stepy = 4 / self.grid.shape[1], 4 / self.grid.shape[0]
            xmin, xmax, ymin, ymax = -2 + stepx / 2, 2, -1 + stepy / 2, 3

            a7 = np.random.RandomState(self.seed).random(size=2) * 10
            rosenbrock_modified.translate(a7)  # TODO reemplazar 0.0 0.0 por literalmente cualquier random
            rosenbrock_modified.scale(
                [1,
                 1])  # TODO reemplazar cuidadosamente estos valores, tienen que ser cercanos a uno (hasta 5 funciona bien creo)
            rot_matrix_quat, _ = np.linalg.qr(np.identity(2))
            rosenbrock_modified.rotate(rot_matrix_quat)  # TODO reemplazar por cualquier matriz de rotacion

            bm_func = self.rosenbrock_arg0
        elif self.base_benchmark == "schaffer":
            stepx, stepy = 40 / self.grid.shape[1], 40 / self.grid.shape[0]
            xmin, xmax, ymin, ymax = -20 + stepx / 2, 20, -20 + stepy / 2, 20

            a8 = np.random.RandomState(self.seed).random(size=2) * 10
            schaffer_modified.translate(a8)  # TODO reemplazar 0.0 0.0 por literalmente cualquier random
            schaffer_modified.scale(
                [1,
                 1])  # TODO reemplazar cuidadosamente estos valores, tienen que ser cercanos a uno (hasta 5 funciona bien creo)
            rot_matrix_quat, _ = np.linalg.qr(np.identity(2))
            schaffer_modified.rotate(rot_matrix_quat)  # TODO reemplazar por cualquier matriz de rotacion

            bm_func = self.schaffer_arg0
        elif self.base_benchmark == "schwefel":
            stepx, stepy = 800 / self.grid.shape[1], 800 / self.grid.shape[0]
            xmin, xmax, ymin, ymax = -400 + stepx / 2, 400, -400 + stepy / 2, 400

            a9 = np.random.RandomState(self.seed).random(size=2) * 10
            schwefel_modified.translate(a9)  # TODO reemplazar 0.0 0.0 por literalmente cualquier random
            schwefel_modified.scale(
                [1,
                 1])  # TODO reemplazar cuidadosamente estos valores, tienen que ser cercanos a uno (hasta 5 funciona bien creo)
            rot_matrix_quat, _ = np.linalg.qr(np.identity(2))
            schwefel_modified.rotate(rot_matrix_quat)  # TODO reemplazar por cualquier matriz de rotacion

            bm_func = self.schwefel_arg0
        else:
            xmin, xmax, ymin, ymax = 0, self.grid.shape[1], 0, self.grid.shape[0]
            stepx, stepy = 1, 1
            bm_func = self.shekel_arg

        X1 = np.arange(xmin, xmax, stepx)
        Y1 = np.arange(ymin, ymax, stepy)
        X1, Y1 = np.meshgrid(X1, Y1)
        print('Creating', self.base_benchmark, stepx, stepy)

        map_created1 = np.fromiter(map(bm_func, zip(X1.flat, Y1.flat)), dtype=float,
                                   count=X1.shape[0] * X1.shape[1]).reshape(X1.shape)

        map_max = np.max(map_created1)
        map_min = np.min(map_created1)
        map_created = list(map(lambda x: (x - map_min) / (map_max - map_min), map_created1))
        map_created = np.array(map_created)

        bench = list()
        df_bounds_or, X_test_or, bench_list = Bounds(self.resolution, self.xs, self.ys, load_file=False).map_bound()
        for i in range(len(X_test_or)):
            bench.append(map_created[X_test_or[i][0], X_test_or[i][1]])

        bench_function_or = np.array(bench)
        return map_created, bench_function_or, None, None

class Benchmark_function_het():
    def __init__(self, grid, resolution, xs, ys, X_test, initial_seed, vehicles, w_ostacles=False, obstacles_on=False,
                 randomize_shekel=True, base_benchmark="shekel"):
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
        # if base_benchmark == "shekel":
        #     self.yukyry, self.pirayu, self.sanber, self.aregua = ZonesPeaks(self.X_test).find_zones()
        self.base_benchmark = base_benchmark
        return

    def ackley_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else ackley_modified(sol[:2])[0]

    def bohachevsky_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else bohachevsky_modified(sol[:2])[0]

    def griewank_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else griewank_modified(sol[:2])[0]

    def h1_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else h1_modified(sol[:2])[0]

    def himmelblau_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else himmelblau_modified(sol[:2])[0]

    def rastrigin_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else rastrigin_modified(sol[:2])[0]

    def rosenbrock_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else rosenbrock_modified(sol[:2])[0]

    def schaffer_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else schaffer_modified(sol[:2])[0]

    def schwefel_arg0(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else schwefel_modified(sol[:2])[0]

    def shekel_arg(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else benchmarks.shekel(sol[:2], self.a, self.c)[0]

    def branin(self, sol):
        return np.nan if self.w_obstacles and sol[2] == 1 else brn(sol[:2])

    def create_new_map(self):
        self.w_obstacles = self.obstacles_on

        if self.randomize_shekel:

            # if self.vehicles == 2:
            #     num_of_peaks = 2
            # else:
            #     num_of_peaks = np.random.RandomState(self.seed).randint(low=2, high=self.vehicles)
            #
            # self.c = np.random.RandomState(self.seed).rand(num_of_peaks, 1) * 400 + 120
            #
            # index_a1 = np.random.RandomState(self.seed).random(size=(num_of_peaks, 1))
            # index_a = list()
            #
            # random.seed(self.seed)
            # if self.vehicles <= 4:
            #     zone = random.sample(range(4), num_of_peaks)
            # else:
            #     zone = list()
            #     for i in range(num_of_peaks):
            #         zone.append(random.randint(0, 3))
            # for i in range(len(zone)):
            #     if zone[i] == 0:
            #         id1 = index_a1[i] * len(self.yukyry) - 1
            #         id2 = self.yukyry[round(id1[0])]
            #         arr = self.X_test[id2]
            #         arr = arr[::-1]
            #         self.a.append(arr)
            #         index_a.append(id2)
            #     elif zone[i] == 1:
            #         id1 = index_a1[i] * len(self.pirayu) - 1
            #         id2 = self.pirayu[round(id1[0])]
            #         arr = self.X_test[id2]
            #         arr = arr[::-1]
            #         self.a.append(arr)
            #         index_a.append(id2)
            #     elif zone[i] == 2:
            #         id1 = index_a1[i] * len(self.sanber) - 1
            #         id2 = self.sanber[round(id1[0])]
            #         arr = self.X_test[id2]
            #         arr = arr[::-1]
            #         self.a.append(arr)
            #         index_a.append(id2)
            #     elif zone[i] == 3:
            #         id1 = index_a1[i] * len(self.aregua) - 1
            #         id2 = self.aregua[round(id1[0])]
            #         arr = self.X_test[id2]
            #         arr = arr[::-1]
            #         self.a.append(arr)
            #         index_a.append(id2)
            # self.a = np.array(self.a)
            # index_a = np.array(index_a)

            no_maxima = np.random.RandomState(self.seed).randint(4, 6)
            xmin = 0
            xmax = 10
            ymin = 0
            ymax = 10
            self.a = []
            self.c = []
            seed_a = self.seed
            for i in range(no_maxima):
                seed_a += 1000
                seed_b = seed_a + 10
                self.a.append([np.random.RandomState(seed_a).rand() * 100, np.random.RandomState(seed_b).rand() * 150])
                self.c.append(1)
            self.a = np.array(self.a)
            self.c = np.array(self.c).T
        else:
            self.a = np.array([[0.16, 1 / 1.5], [0.9, 0.2 / 1.5]])
            self.c = np.array([0.15, 0.15]).T

        self.c = self.c * 400 + 120

        # Como los distintos benchmarks(BM) tienen distintos bounds, vamos a cambiar X1 e Y1 de acuerdo al benchmark
        # seleccionado
        random.seed(self.seed)

        if self.base_benchmark == "ackley":
            stepx, stepy = 60 / self.grid.shape[1], 60 / self.grid.shape[0]
            print(self.grid.shape)
            xmin, xmax, ymin, ymax = -30 + stepx / 2, 30, -30 + stepy / 2, 30

            a1 = np.random.RandomState(self.seed).random(size=2) * 10
            ackley_modified.translate(a1)  # TODO reemplazar 0.0 0.0 por literalmente cualquier random
            ackley_modified.scale(
                [1,
                 1])  # TODO reemplazar cuidadosamente estos valores, tienen que ser cercanos a uno (hasta 5 funciona bien creo)
            rot_matrix_quat, _ = np.linalg.qr(np.identity(2))
            ackley_modified.rotate(rot_matrix_quat)  # TODO reemplazar por cualquier matriz de rotacion

            bm_func = self.ackley_arg0
        elif self.base_benchmark == "bohachevsky":
            stepx, stepy = 30 / self.grid.shape[1], 30 / self.grid.shape[0]
            xmin, xmax, ymin, ymax = -15 + stepx / 2, 15, -15 + stepy / 2, 15

            a2 = np.random.RandomState(self.seed).random(size=2) * 10
            bohachevsky_modified.translate(a2)  # TODO reemplazar 0.0 0.0 por literalmente cualquier random
            bohachevsky_modified.scale(
                [1,
                 1])  # TODO reemplazar cuidadosamente estos valores, tienen que ser cercanos a uno (hasta 5 funciona bien creo)
            rot_matrix_quat, _ = np.linalg.qr(np.identity(2))
            bohachevsky_modified.rotate(rot_matrix_quat)  # TODO reemplazar por cualquier matriz de rotacion

            bm_func = self.bohachevsky_arg0
        elif self.base_benchmark == "griewank":
            stepx, stepy = 100 / self.grid.shape[1], 100 / self.grid.shape[0]
            xmin, xmax, ymin, ymax = -50 + stepx / 2, 50, -50 + stepy / 2, 50

            a3 = np.random.RandomState(self.seed).random(size=2) * 10
            griewank_modified.translate(a3)  # TODO reemplazar 0.0 0.0 por literalmente cualquier random
            griewank_modified.scale(
                [1,
                 1])  # TODO reemplazar cuidadosamente estos valores, tienen que ser cercanos a uno (hasta 5 funciona bien creo)
            rot_matrix_quat, _ = np.linalg.qr(np.identity(2))
            griewank_modified.rotate(rot_matrix_quat)  # TODO reemplazar por cualquier matriz de rotacion

            bm_func = self.griewank_arg0
        elif self.base_benchmark == "h1":
            stepx, stepy = 50 / self.grid.shape[1], 50 / self.grid.shape[0]
            xmin, xmax, ymin, ymax = -25 + stepx / 2, 25, -25 + stepy / 2, 25

            a4 = np.random.RandomState(self.seed).random(size=2) * 10
            h1_modified.translate(a4)  # TODO reemplazar 0.0 0.0 por literalmente cualquier random
            h1_modified.scale(
                [1,
                 1])  # TODO reemplazar cuidadosamente estos valores, tienen que ser cercanos a uno (hasta 5 funciona bien creo)
            rot_matrix_quat, _ = np.linalg.qr(np.identity(2))
            h1_modified.rotate(rot_matrix_quat)  # TODO reemplazar por cualquier matriz de rotacion

            bm_func = self.h1_arg0
        elif self.base_benchmark == "himmelblau":
            stepx, stepy = 12 / self.grid.shape[1], 12 / self.grid.shape[0]
            xmin, xmax, ymin, ymax = -6 + stepx / 2, 6, -6 + stepy / 2, 6

            a5 = np.random.RandomState(self.seed).random(size=2) * 10
            himmelblau_modified.translate(a5)  # TODO reemplazar 0.0 0.0 por literalmente cualquier random
            himmelblau_modified.scale(
                [1,
                 1])  # TODO reemplazar cuidadosamente estos valores, tienen que ser cercanos a uno (hasta 5 funciona bien creo)
            rot_matrix_quat, _ = np.linalg.qr(np.identity(2))
            himmelblau_modified.rotate(rot_matrix_quat)  # TODO reemplazar por cualquier matriz de rotacion

            bm_func = self.himmelblau_arg0
        elif self.base_benchmark == "rastrigin":
            stepx, stepy = 10 / self.grid.shape[1], 10 / self.grid.shape[0]
            xmin, xmax, ymin, ymax = -5 + stepx / 2, 5, -5 + stepy / 2, 5

            a6 = np.random.RandomState(self.seed).random(size=2) * 10
            rastrigin_modified.translate(a6)  # TODO reemplazar 0.0 0.0 por literalmente cualquier random
            rastrigin_modified.scale(
                [1,
                 1])  # TODO reemplazar cuidadosamente estos valores, tienen que ser cercanos a uno (hasta 5 funciona bien creo)
            rot_matrix_quat, _ = np.linalg.qr(np.identity(2))
            rastrigin_modified.rotate(rot_matrix_quat)  # TODO reemplazar por cualquier matriz de rotacion

            bm_func = self.rastrigin_arg0
        elif self.base_benchmark == "rosenbrock":
            stepx, stepy = 4 / self.grid.shape[1], 4 / self.grid.shape[0]
            xmin, xmax, ymin, ymax = -2 + stepx / 2, 2, -1 + stepy / 2, 3

            a7 = np.random.RandomState(self.seed).random(size=2) * 10
            rosenbrock_modified.translate(a7)  # TODO reemplazar 0.0 0.0 por literalmente cualquier random
            rosenbrock_modified.scale(
                [1,
                 1])  # TODO reemplazar cuidadosamente estos valores, tienen que ser cercanos a uno (hasta 5 funciona bien creo)
            rot_matrix_quat, _ = np.linalg.qr(np.identity(2))
            rosenbrock_modified.rotate(rot_matrix_quat)  # TODO reemplazar por cualquier matriz de rotacion

            bm_func = self.rosenbrock_arg0
        elif self.base_benchmark == "schaffer":
            stepx, stepy = 40 / self.grid.shape[1], 40 / self.grid.shape[0]
            xmin, xmax, ymin, ymax = -20 + stepx / 2, 20, -20 + stepy / 2, 20

            a8 = np.random.RandomState(self.seed).random(size=2) * 10
            schaffer_modified.translate(a8)  # TODO reemplazar 0.0 0.0 por literalmente cualquier random
            schaffer_modified.scale(
                [1,
                 1])  # TODO reemplazar cuidadosamente estos valores, tienen que ser cercanos a uno (hasta 5 funciona bien creo)
            rot_matrix_quat, _ = np.linalg.qr(np.identity(2))
            schaffer_modified.rotate(rot_matrix_quat)  # TODO reemplazar por cualquier matriz de rotacion

            bm_func = self.schaffer_arg0
        elif self.base_benchmark == "schwefel":
            stepx, stepy = 800 / self.grid.shape[1], 800 / self.grid.shape[0]
            xmin, xmax, ymin, ymax = -400 + stepx / 2, 400, -400 + stepy / 2, 400

            a9 = np.random.RandomState(self.seed).random(size=2) * 10
            schwefel_modified.translate(a9)  # TODO reemplazar 0.0 0.0 por literalmente cualquier random
            schwefel_modified.scale(
                [1,
                 1])  # TODO reemplazar cuidadosamente estos valores, tienen que ser cercanos a uno (hasta 5 funciona bien creo)
            rot_matrix_quat, _ = np.linalg.qr(np.identity(2))
            schwefel_modified.rotate(rot_matrix_quat)  # TODO reemplazar por cualquier matriz de rotacion

            bm_func = self.schwefel_arg0
        else:
            xmin, xmax, ymin, ymax = 0, self.grid.shape[1], 0, self.grid.shape[0]
            stepx, stepy = 1, 1
            bm_func = self.shekel_arg

        X1 = np.arange(xmin, xmax, stepx)
        Y1 = np.arange(ymin, ymax, stepy)
        X1, Y1 = np.meshgrid(X1, Y1)
        print('Creating', self.base_benchmark, stepx, stepy)

        map_created1 = np.fromiter(map(bm_func, zip(X1.flat, Y1.flat)), dtype=float,
                                   count=X1.shape[0] * X1.shape[1]).reshape(X1.shape)

        map_max = np.max(map_created1)
        map_min = np.min(map_created1)
        map_created = list(map(lambda x: (x - map_min) / (map_max - map_min), map_created1))
        map_created = np.array(map_created)

        bench = list()
        df_bounds_or, X_test_or, bench_list = Bounds(self.resolution, self.xs, self.ys, load_file=False).map_bound()
        for i in range(len(X_test_or)):
            bench.append(map_created[X_test_or[i][0], X_test_or[i][1]])

        bench_function_or = np.array(bench)
        return map_created, bench_function_or

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
        Z = np.fromiter(map(self.shekel_arg0, zip(X.flat, Y.flat)), dtype=np.float,
                        count=X.shape[0] * X.shape[1]).reshape(X.shape)
        Z = (Z - np.mean(Z)) / np.std(Z)  # Normalize Z
        nan_mask = np.copy(self.grid)
        nan_mask[self.grid == 0] = np.nan
        z_nan = nan_mask + Z.T

        return z_nan

    def reset_gt(self):
        self.seed += 1
        num_of_peaks = np.random.RandomState(self.seed).randint(low=1, high=5)
        self.A = np.random.RandomState(self.seed).random(size=(num_of_peaks, 2))
        self.C = np.ones(
            (num_of_peaks)) * 0.05  # np.random.RandomState(self.seed).normal(0.05,0.01, size = num_of_peaks)

    def shekel_arg0(self, sol):
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

# if __name__ == '__main__':
#     import matplotlib.image as img
#     import numpy as np
#     import matplotlib.pyplot as plt
#
#     my_benchmarks = [
#         "ackley",
#         "bohachevsky",
#         "griewank",
#         "h1",
#         "himmelblau",
#         "rastrigin",
#         "rosenbrock",
#         "schaffer",
#         "schwefel",
#         # "shekel"
#     ]
#
#     xs = 150
#     ys = 100
#     #grid = np.flipud(img.imread("../Image/snazzy-image-prueba.png")[:, :, 0])
#     grid_or = Map( xs, ys).black_white()
#     grid = np.flipud(grid_or)
#
#     _, X_test_or, _ = Bounds(1, ys, xs, load_file=False).map_bound()
#     for idx, benchmark in enumerate(my_benchmarks):
#         m_map, bFunction, no_peaks, index_a = Benchmark_function(grid_or, 1, xs, ys, None, 42, 0, base_benchmark=benchmark,
#                                                                  randomize_shekel=False).create_new_map()
#         m_nan_map = np.full_like(m_map, np.nan)
#         for point in X_test_or:
#             m_nan_map[point[1], point[0]] = m_map[point[1], point[0]]
#
#         plt.subplot(331 + idx)
#         plt.title(benchmark)
#         plt.imshow(m_nan_map)
#         # plt.colorbar()
#     plt.show()
