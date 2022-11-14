import math


class ZonesPeaks():
    def __init__(self, X_test):
        self.X_test = X_test
        self.yukyry_zone = [15, 35]
        self.pirayu_zone = [85, 135]
        self.sanber_zone = [83, 83]
        self.aregua_zone = [23, 83]
        self.d = 10

    def find_zones(self):
        yukyry_index = list()
        pirayu_index = list()
        sanber_index = list()
        aregua_index = list()
        for i in range(len(self.X_test)):
            point = self.X_test[i]
            if math.sqrt((self.yukyry_zone[0] - point[0])**2 + (self.yukyry_zone[1] - point[1])**2) <= self.d:
                yukyry_index.append(i)
            elif math.sqrt((self.pirayu_zone[0] - point[0])**2 + (self.pirayu_zone[1] - point[1])**2) <= self.d:
                pirayu_index.append(i)
            elif math.sqrt((self.sanber_zone[0] - point[0]) ** 2 + (
                    self.sanber_zone[1] - point[1]) ** 2) <= self.d:
                sanber_index.append(i)
            elif math.sqrt((self.aregua_zone[0] - point[0]) ** 2 + (
                    self.aregua_zone[1] - point[1]) ** 2) <= self.d:
                aregua_index.append(i)
        return yukyry_index, pirayu_index, sanber_index, aregua_index