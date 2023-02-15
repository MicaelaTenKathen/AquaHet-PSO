import numpy as np
from sys import path


class Limits:
    def __init__(self, grid, xs, ys, vehicles, file=0):
        self.file = file
        self.secure = grid
        self.xs = xs
        self.ys = ys
        self.bn_ant = {}
        i = 1
        with open('../GroundTruth/bounds.npy'.format(self.file), 'rb') as bn:
            self.df_bounds_x = np.load(bn)
        return

    def ratio_s(self, x_int, y_int, part):
        x_int = int(x_int)
        y_int = int(y_int)
        x_left = x_int + 2
        x_right = x_int - 2
        y_up = y_int + 2
        y_down = y_int - 2
        x_i = int(part[0])
        y_i = int(part[1])
        if self.secure[x_right, y_down] == 1:
            part[0] = x_right
            part[1] = y_down
        else:
            if self.secure[x_int, y_down] == 1:
                part[1] = y_down
                part[0] = x_int
            else:
                if self.secure[x_left, y_i] == 1:
                    part[0] = x_left
                    part[1] = y_int
                else:
                    if self.secure[x_right, y_i] == 1:
                        part[0] = x_right
                        part[1] = y_int
                    else:
                        if self.secure[x_i, y_up] == 1:
                            part[1] = y_up
                            part[0] = x_int
                        else:
                            part[0] = x_i
                            part[1] = y_i
        return part

    def new_limit(self, g, part, s_n_x, n_data, s_ant_x, part_ant):
        n_dat = n_data + 1
        x_int = int(part[0])
        y_int = int(part[1])
        if x_int >= self.xs:
            part[0] = self.xs - 1
            x_int = int(part[0])
        if y_int >= self.ys:
            part[1] = self.ys - 1
            y_int = (part[1])
        if self.secure[x_int, y_int] == 0:
            s_x, n_x = 0, 0
            bn_x = list()
            for i in range(len(self.df_bounds_x)):
                if int(y_int) == self.df_bounds_x[i, 2]:
                    s_x += 1
                    bn_x.append(self.df_bounds_x[i, :])
            if s_x == 0:
                if part[1] < self.df_bounds_x[0, 2]:
                    part[1] = self.df_bounds_x[0, 2] + 2
                    for i in range(len(self.df_bounds_x)):
                        if self.df_bounds_x[i, 2] == int(part[1]):
                            s_x += 1
                            bn_x.append(self.df_bounds_x[i, :])
                else:
                    part[1] = self.df_bounds_x[-1, 2] - 2
                    for i in range(len(self.df_bounds_x)):
                        if self.df_bounds_x[i, 2] == int(part[1]):
                            s_x += 1
                            bn_x.append(self.df_bounds_x[i, :])
            bn = np.array(bn_x)
            if s_ant_x[n_data] > 1 and s_n_x[n_data]:
                part = self.ratio_s(part_ant[g, 2 * n_data], part_ant[g, 2 * n_data + 1], part)
                s_n_x[n_data] = False
            else:
                if part[0] <= bn[0, 0]:
                    part[0] = bn[0, 0] + 2
                else:
                    part[0] = bn[0, 1] - 2
            s_ant_x[n_data] = s_x


            #if n_dat == 1.0:
             #   if s_ant[0] > 1 and s_1:
              #      part = limit.ratio_s(part_ant[g, 0], part_ant[g, 1], part)
               #     s_1 = False
                #else:
                 #   if part[0] <= bn[0, 0]:
                  #      part[0] = bn[0, 0] + 2
                   # else:
                    #    part[0] = bn[0, 1] - 2
                #s_ant[0] = s
            #elif n_dat == 2.0:
             #   if s_ant[1] > 1 and s_2:
              #      part = limit.ratio_s(part_ant[g, 2], part_ant[g, 3], part)
               #     s_2 = False
                #else:
                 #   if part[0] <= bn[0, 0]:
                  #      part[0] = bn[0, 0] + 2
                   # else:
                    #    part[0] = bn[0, 1] - 2
                #s_ant[1] = s
            #elif n_dat == 3.0:
             #   if s_ant[2] > 1 and s_3:
              #      part = limit.ratio_s(part_ant[g, 4], part_ant[g, 5], part)
               #     s_3 = False
                #else:
                 #   if part[0] <= bn[0, 0]:
                  #      part[0] = bn[0, 0] + 2
                   # else:
                    #    part[0] = bn[0, 1] - 2
                #s_ant[2] = s
            #elif n_dat == 4.0:
             #   if s_ant[3] > 1 and s_4:
              #      part = limit.ratio_s(part_ant[g, 6], part_ant[g, 7], part)
               #     s_4 = False

                #else:
                 #   if part[0] <= bn[0, 0]:
                  #      part[0] = bn[0, 0] + 2
                   # else:
                    #    part[0] = bn[0, 1] - 2
                #s_ant[3] = s
        #s_n = [s_1, s_2, s_3, s_4]
        return part, s_n_x

    def check_lm_limits(self, n_pos, vehicle):
        with open('../GroundTruth/boundsy.npy'.format(self.file), 'rb') as bn:
            df_boundsy = np.load(bn)
        check = False
        x_int = n_pos[0]
        y_int = n_pos[1]
        if x_int >= self.xs:
            n_pos[0] = self.xs - 1
            x_int = int(n_pos[0])
        elif x_int <= 0:
            n_pos[0] = 0
        if y_int >= self.ys:
            n_pos[1] = self.ys - 1
            y_int = (n_pos[1])
        elif y_int <= 0:
            y_int = 0
        ch = 0
        # if vehicle % 2 == 0:
        for i in range(len(self.df_bounds_x)):
            if int(y_int) == self.df_bounds_x[i, 2]:
                if int(x_int) < (self.df_bounds_x[i, 0] + 1) or int(x_int) > (self.df_bounds_x[i, 1] - 1):
                    check = False
                    break
                else:
                    check = True
                    break
        return check

    def Z_var_mean(self, mu, sigma, X_test, grid):
        Z_var = np.zeros([grid.shape[0], grid.shape[1]])
        Z_mean = np.zeros([grid.shape[0], grid.shape[1]])
        for i in range(len(X_test)):
            Z_var[X_test[i][0], X_test[i][1]] = sigma[i]
            Z_mean[X_test[i][0], X_test[i][1]] = mu[i]
        Z_var[Z_var == 0] = np.nan
        Z_mean[Z_mean == 0] = np.nan
        return Z_var, Z_mean