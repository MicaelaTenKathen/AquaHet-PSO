from Environment.map import Map
from Data.data_path import *
import pandas as pd
import numpy as np


class Bounds():
    def __init__(self, resolution, xs, ys, load_file=False, file=0):
        self.resolution = resolution
        self.xs = xs
        self.ys = ys
        self.load_file = load_file
        self.file = file

        return

    def map_bound(self):
        if self.load_file:
            with open('./GroundTruth/bounds.npy'.format(self.file), 'rb') as bn:
                df_bounds = np.load(bn)

            with open('./GroundTruth/grid.npy'.format(self.file), 'rb') as gd:
                grid = np.load(gd)

            with open('./GroundTruth/available.npy'.format(self.file), 'rb') as av:
                available = np.load(av)

            return df_bounds, available

        else:
            available, first, last, y_first, y_last = list(), list(), list(), list(), list()
            confirm = list()
            index, first_x, last_x, all_y = list(), list(), list(), list()
            grid = Map(self.xs, self.ys).black_white()
            bound = True

            f, o = True, False
            for j in range(len(grid[1])):
                for i in range(len(grid)):
                    if grid[i, j] == 1:
                        if bound:
                            first.append(i)
                            y_first.append(j)
                            u = 4 + int(y_first[0])
                            if f:
                                if j > u:
                                    if y_first[-1] == y_last[-1]:
                                        first[-5] = first[-2]
                                        first.insert(-4, first[-1])
                                        y_first.insert(-4, y_first[-5])
                                        first[-4] = first[-2]
                                        first.insert(-3, first[-1])
                                        y_first.insert(-3, y_first[-4])
                                        first[-3] = first[-2]
                                        first.insert(-2, first[-1])
                                        y_first.insert(-2, y_first[-3])
                                        o = True
                                        f = False
                            bound = False
                        available.append([i, j])
                        grid_ant = i
                        grid_y = j
                    else:
                        if not bound:
                            last.append(grid_ant)
                            y_last.append(grid_y)
                            bound = True
                            if o:
                                last[-5] = last[-2]
                                last.insert(-4, last[-1])
                                last[-4] = last[-2]
                                last.insert(-3, last[-1])
                                last[-3] = last[-2]
                                last.insert(-2, last[-1])
                                o = False

            for i in range(len(first)):
                if first[i] == last[i]:
                    confirm.append(True)

            if np.array(confirm).all():
                for i in range(len(first)):
                    first_x.append(first[i] + 2)
                    last_x.append(last[i] - 2)
                    all_y.append(y_first[i])

                for x in range(2):
                    first_x.pop(0), last_x.pop(0), all_y.pop(0)
                    first_x.pop(-1), last_x.pop(-1), all_y.pop(-1)
                bounds = {'First X': first_x, 'Last X': last_x, 'Y': all_y}
                df_bounds = pd.DataFrame(data=bounds)
            else:
                print('An error occurred. Map bound, y array')

            bp = bounds_path_classic()
            gp = grid_path_classic()
            ap = available_path_classic()
            # bp = bounds_path()
            # gp = grid_path()
            # ap = available_path()
            x = list()
            y = list()
            for i in range(len(available)):
                coord = available[i]
                x_av = coord[0]
                y_av = coord[1]
                x.append(x_av)
                y.append(y_av)
            min_x = min(x)
            max_x = max(x)
            min_y = min(y)
            max_y = max(y)
            bench_limits = [min_x, max_x, min_y, max_y]

            with open(bp, 'wb') as bn:
                np.save(bn, df_bounds)

            with open(gp, 'wb') as gd:
                np.save(gd, grid)

            with open(ap, 'wb') as av:
                np.save(av, available)

            return df_bounds, available, bench_limits

    def interest_area(self):
        if self.load_file:
            with open('./GroundTruth/secure_grid.npy'.format(self.file), 'rb') as sg:
                secure_grid = np.load(sg)

            with open('./GroundTruth/secure_av.npy'.format(self.file), 'rb') as sa:
                se_available = np.load(sa)
            return secure_grid, se_available
        else:
            df_bounds, available, bench_limits = Bounds(self.resolution, self.xs, self.ys).map_bound()
            secure_grid = np.zeros((self.xs, self.ys))

            for i in range(len(df_bounds)):
                secure_grid[np.array(df_bounds)[i, 0], np.array(df_bounds)[i, 2]] = 1
                secure_grid[np.array(df_bounds)[i, 1], np.array(df_bounds)[i, 2]] = 1

            se_first = list()
            se_last = list()
            se_available = list()

            for j in range(len(secure_grid[1])):
                con = False
                uno = 0
                for i in range(len(secure_grid)):
                    if secure_grid[i, j] == 1:
                        con = True
                        uno += 1
                    if con and uno == 1:
                        secure_grid[i, j] = 1

            bound = True

            for j in range(len(secure_grid[1])):
                for i in range(len(secure_grid)):
                    if secure_grid[i, j] == 1:
                        if bound:
                            se_first.append([i, j])
                            bound = False
                        se_available.append([i, j])
                        grid_ant = [i, j]
                    else:
                        if not bound:
                            se_last.append(grid_ant)
                            bound = True

            sp = secure_path_classic()
            seap = se_available_path_classic()
            # sp = secure_path()
            # seap = se_available_path()

            with open(sp, 'wb') as sg:
                np.save(sg, secure_grid)

            with open(seap, 'wb') as sa:
                np.save(sa, se_available)

            return secure_grid, df_bounds

    def bounds_y(self):
        available, x_first, last_y, y_first, x_last, y_last = list(), list(), list(), list(), list(), list()
        confirm = list()
        index, first_y, last_x, all_x, y_last = list(), list(), list(), list(), list()
        grid = Map(self.xs, self.ys).black_white()
        bound = True

        f, o = True, False
        for i in range(self.xs):
            for j in range(self.ys):
                if grid[i, j] == 1:
                    if bound:
                        x_first.append(i)
                        y_first.append(j)
                        bound = False
                    grid_ant = i
                    grid_y = j
                else:
                    if not bound:
                        x_last.append(grid_ant)
                        y_last.append(grid_y)
                        bound = True

        for i in range(len(x_first)):
            if y_first[i] == y_last[i]:
                confirm.append(True)

        if np.array(confirm).all():
            for i in range(len(y_first)):
                first_y.append(y_first[i] + 2)
                last_y.append(y_last[i] - 2)
                all_x.append(x_first[i])

            for x in range(2):
                del first_y[0]
                del last_y[0]
                del all_x[0]
                del first_y[-1]
                del last_y[-1]
                del all_x[-1]
            bounds = {'First Y': first_y, 'Last Y': last_y, 'X': all_x}
            df_bounds_x = pd.DataFrame(data=bounds)
        else:
            print('An error occurred. Map bound, y array')

        bp = boundsy_path_classic()
        with open(bp, 'wb') as bn:
            np.save(bn, df_bounds_x)

        return df_bounds_x
