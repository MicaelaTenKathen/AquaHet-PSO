import math
import numpy as np
import openpyxl


class Utils:
    def __init__(self, vehicles):
        self.MSE_data = list()
        self.it = list()
        self.vehicles = vehicles

    def distance_part(self, g, n_data, part, part_ant, distances, array_part, dfirst=False):

        """
        Calculates the distance between (x,y)^t and (x,y)^(t-1).
        """
        if dfirst:
            part_ant[0, 2 * n_data] = part[0]
            part_ant[0, 2 * n_data + 1] = part[1]
            #if n_data == 1:
            #    part_ant[0, 0] = part[0]
             #   part_ant[0, 1] = part[1]
            #if n_data == 2:
             #   part_ant[0, 2] = part[0]
              #  part_ant[0, 3] = part[1]
            #if n_data == 3:
             #   part_ant[0, 4] = part[0]
              #  part_ant[0, 5] = part[1]
            #if n_data == 4:
             #   part_ant[0, 6] = part[0]
              #  part_ant[0, 7] = part[1]
        else:
            array_part[0, 2 * n_data] = part[0]
            array_part[0, 2 * n_data + 1] = part[1]
            distances[n_data] = math.sqrt(
                    (array_part[0, 2 * n_data] - part_ant[g, 2 * n_data]) ** 2 + (array_part[0, 2 * n_data + 1] - part_ant[g, 2 * n_data + 1])
                    ** 2) + distances[n_data]

            #if n_data == 1:
             #   array_part[0, 0] = part[0]
              #  array_part[0, 1] = part[1]
               # distances[0] = math.sqrt(
                #    (array_part[0, 0] - part_ant[g, 0]) ** 2 + (array_part[0, 1] - part_ant[g, 1])
                 #   ** 2) + distances[0]
            #elif n_data == 2:
             #   array_part[0, 2] = part[0]
              #  array_part[0, 3] = part[1]
               # distances[1] = math.sqrt(
                #    (array_part[0, 2] - part_ant[g, 2]) ** 2 + (array_part[0, 3] - part_ant[g, 3])
                 #   ** 2) + distances[1]
            #elif n_data == 3:
             #   array_part[0, 4] = part[0]
              #  array_part[0, 5] = part[1]
               # distances[2] = math.sqrt(
                #    (array_part[0, 4] - part_ant[g, 4]) ** 2 + (array_part[0, 5] - part_ant[g, 5])
                 #   ** 2) + distances[2]
            #elif n_data == 4:
             #   array_part[0, 6] = part[0]
              #  array_part[0, 7] = part[1]
               # distances[3] = math.sqrt(
                #    (array_part[0, 6] - part_ant[g, 6]) ** 2 + (array_part[0, 7] - part_ant[g, 7])
                 #   ** 2) + distances[3]
            if n_data == self.vehicles - 1:
                part_ant = np.append(part_ant, array_part, axis=0)

        return part_ant, distances

    def mse(self, g, y_data, mu_data):

        """
        Calculates the mean square error (MSE) between the collected data and the estimated data.
        """

        total_suma = 0
        y_array = np.array(y_data)
        mu_array = np.array(mu_data)
        for i in range(len(mu_array)):
            total_suma = ((float(y_array[i]) - float(mu_array[i])) ** 2) + total_suma
        MSE = total_suma / y_data.shape[0]
        self.MSE_data.append(MSE)
        self.it.append(g)
        return self.MSE_data, self.it

    def savexlsx(self, sigma_data, mu_data, seed, file):

        """
        Save the data in excel documents.
        """

        for i in range(len(mu_data)):
            mu_data[i] = float(mu_data[i])

        wb1 = openpyxl.Workbook()
        hoja1 = wb1.active
        hoja1.append(self.MSE_data)
        wb1.save('Test/' + file + '/ALLCONError' + str(seed[0]) + '.xlsx')

        wb2 = openpyxl.Workbook()
        hoja2 = wb2.active
        hoja2.append(sigma_data)
        wb2.save('Test/' + file + '/ALLCONSigma' + str(seed[0]) + '.xlsx')

        wb3 = openpyxl.Workbook()
        hoja3 = wb3.active
        hoja3.append(mu_data)
        wb3.save('Test/' + file + '/ALLCONMu' + str(seed[0]) + '.xlsx')

        wb4 = openpyxl.Workbook()
        hoja4 = wb4.active
        hoja4.append(list(self.distances))
        wb4.save('Test/' + file + '/ALLCONDistance' + str(seed[0]) + '.xlsx')

        wb5 = openpyxl.Workbook()
        hoja5 = wb5.active
        hoja5.append(self.it)
        wb5.save('Test/' + file + '/ALLCONData' + str(seed[0]) + '.xlsx')
