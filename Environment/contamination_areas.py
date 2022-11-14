import numpy as np
import math
import copy


class DetectContaminationAreas():
    def __init__(self, X_test, benchmark, vehicles=4, area=100):
        self.coord = copy.copy(X_test)
        self.coord_bench = copy.copy(X_test)
        self.coord_real = copy.copy(X_test)
        self.radio = area / vehicles
        self.benchmark = copy.copy(benchmark)
        self.ava = np.array(X_test)
        self.vehicles = vehicles

    def real_peaks(self):
        array_max_x = list()
        array_max_y = list()
        array_action_zones = list()
        coordinate_action_zones = list()
        for i in range(len(self.benchmark)):
            if self.benchmark[i] >= 0.33:
                array_action_zones.append(self.benchmark[i])
                coordinate_action_zones.append(self.coord[i])
        while True:
            max_action_zone = max(array_action_zones)
            max_index = array_action_zones.index(max_action_zone)
            max_coordinate = coordinate_action_zones[max_index]
            x_max = max_coordinate[0]
            array_max_x.append(x_max)
            y_max = max_coordinate[1]
            array_max_y.append(y_max)
            coordinate_array = np.array(coordinate_action_zones)
            m = 0
            for i in range(len(array_action_zones)):
                if math.sqrt(
                        (x_max - coordinate_array[i, 0]) ** 2 + (y_max - coordinate_array[i, 1]) ** 2) <= self.radio:
                    index_del = i - m
                    del array_action_zones[index_del]
                    del coordinate_action_zones[index_del]
                    m += 1
            if len(array_action_zones) == 0:
                break
        max_peaks = np.column_stack((array_max_x, array_max_y))

        return max_peaks

    def areas_levels(self, mu):
        dict_ = {}
        dict_coord_ = {}
        dict_impor_ = {}
        dict_index_ = {}
        dict_limits = {}
        array_max_x = list()
        array_max_y = list()
        array_action_zones = list()
        coordinate_action_zones = list()
        mean = mu.flat
        warning_level = max(mean) * 0.33
        j = 0
        impo = self.vehicles * 10 + 10
        action_zones = list()
        cen = 0
        for i in range(len(mean)):
            if mean[i] >= warning_level:
                array_action_zones.append(mean[i])
                coordinate_action_zones.append(self.coord[i])
        while cen < self.vehicles:
            max_action_zone = max(array_action_zones)
            max_index = array_action_zones.index(max_action_zone)
            max_coordinate = coordinate_action_zones[max_index]
            x_max = max_coordinate[0]
            array_max_x.append(x_max)
            y_max = max_coordinate[1]
            array_max_y.append(y_max)
            coordinate_array = np.array(coordinate_action_zones)
            m = 0
            for i in range(len(array_action_zones)):
                if math.sqrt(
                        (x_max - coordinate_array[i, 0]) ** 2 + (y_max - coordinate_array[i, 1]) ** 2) <= self.radio:
                    index_del = i - m
                    del array_action_zones[index_del]
                    del coordinate_action_zones[index_del]
                    m += 1
            if len(array_action_zones) == 0:
                break
            cen += 1
        center_peaks = np.column_stack((array_max_x, array_max_y))
        for w in range(len(array_max_x)):
            list_zone = list()
            list_coord = list()
            list_impo = list()
            del_list = list()
            coordinate_array = np.array(self.coord)
            for i in range(len(self.coord)):
                if math.sqrt(
                        (array_max_x[w] - coordinate_array[i, 0]) ** 2 + (
                                array_max_y[w] - coordinate_array[i, 1]) ** 2) <= self.radio:
                    list_zone.append(mu[i])
                    list_coord.append(self.coord[i])
                    list_impo.append(impo)
                    del_list.append(i)
            m = 0
            for i in range(len(del_list)):
                index_del = del_list[i] - m
                del self.coord[index_del]
                m += 1
            array_list_coord = np.array(list_coord)
            x_coord = array_list_coord[:, 0]
            y_coord = array_list_coord[:, 1]
            max_x_coord = max(x_coord)
            min_x_coord = min(x_coord)
            max_y_coord = max(y_coord)
            min_y_coord = min(y_coord)
            dict_limits["action_zone%s" % j] = [min_x_coord, max_y_coord, max_x_coord, min_y_coord]
            index = list()
            for i in range(len(array_list_coord)):
                x = array_list_coord[i, 0]
                y = array_list_coord[i, 1]
                for p in range(len(self.ava)):
                    if x == self.ava[p, 0] and y == self.ava[p, 1]:
                        index.append(p)
                        break
            dict_["action_zone%s" % j] = list_zone
            dict_coord_["action_zone%s" % j] = list_coord
            dict_impor_["action_zone%s" % j] = list_impo
            dict_index_["action_zone%s" % j] = index
            impo -= 10
            j += 1
        return dict_, dict_coord_, dict_impor_, j, center_peaks, dict_index_, action_zones, \
               dict_limits

    def benchmark_areas(self):
        dict_coord_bench = {}
        dict_index_bench = {}
        dict_impor_bench = {}
        dict_bench_ = {}
        dict_limits_bench = {}
        array_action_zones_bench = list()
        coordinate_action_zones_bench = list()
        bench = copy.copy(self.benchmark)
        index_xtest = list()
        index_center_bench = list()
        j = 0
        array_max_x_bench = list()
        array_max_y_bench = list()
        max_bench_list = list()
        action_zone_bench = list()
        warning_bench = max(bench) * 0.33
        impo = self.vehicles * 10 + 10
        cen = 0

        for i in range(len(bench)):
            if bench[i] >= warning_bench:
                array_action_zones_bench.append(bench[i])
                coordinate_action_zones_bench.append(self.coord_bench[i])
                index_xtest.append(i)
        while cen < self.vehicles:
            max_action_zone_bench = max(array_action_zones_bench)
            max_index_bench = array_action_zones_bench.index(max_action_zone_bench)
            index_center_bench.append(index_xtest[max_index_bench])
            max_coordinate_bench = coordinate_action_zones_bench[max_index_bench]
            max_bench_list.append(max_action_zone_bench)
            x_max_bench = max_coordinate_bench[0]
            array_max_x_bench.append(x_max_bench)
            y_max_bench = max_coordinate_bench[1]
            array_max_y_bench.append(y_max_bench)
            coordinate_array = np.array(coordinate_action_zones_bench)
            m = 0
            for i in range(len(array_action_zones_bench)):
                if math.sqrt(
                        (x_max_bench - coordinate_array[i, 0]) ** 2 + (y_max_bench - coordinate_array[i, 1]) ** 2) <= self.radio:
                    index_del = i - m
                    del array_action_zones_bench[index_del]
                    del coordinate_action_zones_bench[index_del]
                    del index_xtest[index_del]
                    m += 1
            if len(array_action_zones_bench) == 0:
                break
            cen += 1
        center_peaks_bench = np.column_stack((array_max_x_bench, array_max_y_bench))
        for w in range(len(array_max_x_bench)):
            list_zone_bench = list()
            list_coord_bench = list()
            list_impo = list()
            del_list = list()
            coordinate_array = np.array(self.coord_bench)
            for i in range(len(self.coord_bench)):
                if math.sqrt(
                        (array_max_x_bench[w] - coordinate_array[i, 0]) ** 2 + (
                                array_max_y_bench[w] - coordinate_array[i, 1]) ** 2) <= self.radio:
                    list_zone_bench.append(bench[i])
                    list_coord_bench.append(self.coord_bench[i])
                    list_impo.append(impo)
                    del_list.append(i)
            m = 0
            for i in range(len(del_list)):
                index_del = del_list[i] - m
                del self.coord_bench[index_del]
                m += 1
            array_list_coord = np.array(list_coord_bench)
            x_coord = array_list_coord[:, 0]
            y_coord = array_list_coord[:, 1]
            max_x_coord = max(x_coord)
            min_x_coord = min(x_coord)
            max_y_coord = max(y_coord)
            min_y_coord = min(y_coord)
            dict_limits_bench["action_zone%s" % j] = [min_x_coord, max_y_coord, max_x_coord, min_y_coord]
            index = list()
            for i in range(len(array_list_coord)):
                x = array_list_coord[i, 0]
                y = array_list_coord[i, 1]
                for p in range(len(self.ava)):
                    if x == self.ava[p, 0] and y == self.ava[p, 1]:
                        index.append(p)
                        action_zone_bench.append(self.benchmark[p])
                        break
            dict_bench_["action_zone%s" % j] = list_zone_bench
            dict_coord_bench["action_zone%s" % j] = list_coord_bench
            dict_index_bench["action_zone%s" % j] = index
            dict_impor_bench["action_zone%s" % j] = list_impo
            impo -= 10
            j += 1
        return j, dict_index_bench, dict_bench_, dict_coord_bench, center_peaks_bench, max_bench_list, \
               dict_limits_bench, action_zone_bench, dict_impor_bench, index_center_bench
