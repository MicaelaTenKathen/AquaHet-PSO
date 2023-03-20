import numpy as np
import math
import copy
from scipy.spatial.distance import euclidean as d


class DetectContaminationAreas():
    def __init__(self, X_test):
        self.coord = copy.copy(X_test)
        self.X_test = X_test
        self.coord_bench = copy.copy(X_test)
        self.coord_real = copy.copy(X_test)
        self.ava = np.array(X_test)
        self.action_zone = 0
        self.action_bench = 0

    def real_peaks(self, benchmark, radio):
        array_max_x = list()
        array_max_y = list()
        array_action_zones = list()
        coordinate_action_zones = list()
        for i in range(len(benchmark)):
            if benchmark[i] >= 0.33:
                array_action_zones.append(benchmark[i])
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
                        (x_max - coordinate_array[i, 0]) ** 2 + (y_max - coordinate_array[i, 1]) ** 2) <= radio:
                    index_del = i - m
                    del array_action_zones[index_del]
                    del coordinate_action_zones[index_del]
                    m += 1
            if len(array_action_zones) == 0:
                break
        max_peaks = np.column_stack((array_max_x, array_max_y))

        return max_peaks

    def areas_levels(self, sensor, vehicles, radio):
        d_ = {}
        coord = copy.copy(self.coord)
        sensor['action_zones'] = {}
        mu = sensor['mu']['data']
        # dict_coord_ = {}
        dict_index_ = {}
        # dict_limits = {}
        array_max_x = list()
        array_max_y = list()
        array_action_zones = list()
        coordinate_action_zones = list()
        mean = mu.flat
        warning_level = max(mean) * 0.33
        j = 0
        impo = vehicles * 10 + 10
        # action_zones = list()
        cen = 0
        for i in range(len(mean)):
            if mean[i] >= warning_level:
                array_action_zones.append(mean[i])
                coordinate_action_zones.append(coord[i])
        while cen < vehicles:
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
                        (x_max - coordinate_array[i, 0]) ** 2 + (y_max - coordinate_array[i, 1]) ** 2) <= radio:
                    index_del = i - m
                    del array_action_zones[index_del]
                    del coordinate_action_zones[index_del]
                    m += 1
            if len(array_action_zones) == 0:
                break
            cen += 1
        sensor['action_zones']['peaks_coord'] = np.column_stack((array_max_x, array_max_y))
        for w in range(len(array_max_x)):
            list_zone = list()
            list_coord = list()
            list_impo = list()
            del_list = list()
            coordinate_array = np.array(coord)
            for i in range(len(coord)):
                if math.sqrt(
                        (array_max_x[w] - coordinate_array[i, 0]) ** 2 + (
                                array_max_y[w] - coordinate_array[i, 1]) ** 2) <= radio:
                    list_zone.append(mu[i])
                    list_coord.append(coord[i])
                    list_impo.append(impo)
                    del_list.append(i)
            m = 0
            for i in range(len(del_list)):
                index_del = del_list[i] - m
                del coord[index_del]
                m += 1
            array_list_coord = np.array(list_coord)
            # x_coord = array_list_coord[:, 0]
            # y_coord = array_list_coord[:, 1]
            # max_x_coord = max(x_coord)
            # min_x_coord = min(x_coord)
            # max_y_coord = max(y_coord)
            # min_y_coord = min(y_coord)
            # dict_limits["action_zone%s" % j] = [min_x_coord, max_y_coord, max_x_coord, min_y_coord]
            index = list()
            for i in range(len(array_list_coord)):
                x = array_list_coord[i, 0]
                y = array_list_coord[i, 1]
                for p in range(len(self.ava)):
                    if x == self.ava[p, 0] and y == self.ava[p, 1]:
                        index.append(p)
                        break
            # dict_["action_zone%s" % j] = list_zone
            # dict_coord_["action_zone%s" % j] = list_coord
            sensor['action_zones']["action_zone%s" % j] = {}
            sensor['action_zones']["action_zone%s" % j]['center'] = [array_max_x[w], array_max_y[w]]
            sensor['action_zones']["action_zone%s" % j]['number'] = self.action_zone
            sensor['action_zones']["action_zone%s" % j]['priority'] = list_impo
            sensor['action_zones']["action_zone%s" % j]['coord'] = list_coord
            sensor['action_zones']["action_zone%s" % j]['index'] = index
            sensor['action_zones']["action_zone%s" % j]['radio'] = radio
            impo -= 10
            j += 1
            self.action_zone += 1
        sensor['action_zones']['number'] = j
        return sensor

    def benchmark_areas(self, sensor, vehicles, radio):
        # dict_coord_bench = {}
        d_ = {}
        coord = copy.copy(self.coord)
        sensor['action_zones'] = {}
        dict_impor_bench = {}
        # dict_bench_ = {}
        # dict_limits_bench = {}
        array_action_zones_bench = list()
        benchmark = sensor['original']
        coordinate_action_zones_bench = list()
        bench = copy.copy(benchmark)
        index_xtest = list()
        index_center_bench = list()
        j = 0
        array_max_x_bench = list()
        array_max_y_bench = list()
        max_bench_list = list()
        # action_zone_bench = list()
        warning_bench = max(bench) * 0.33
        impo = vehicles * 10 + 10
        cen = 0

        for i in range(len(bench)):
            if bench[i] >= warning_bench:
                array_action_zones_bench.append(bench[i])
                coordinate_action_zones_bench.append(coord[i])
                index_xtest.append(i)
        while cen < vehicles:
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
                        (x_max_bench - coordinate_array[i, 0]) ** 2 + (
                                y_max_bench - coordinate_array[i, 1]) ** 2) <= radio:
                    index_del = i - m
                    del array_action_zones_bench[index_del]
                    del coordinate_action_zones_bench[index_del]
                    del index_xtest[index_del]
                    m += 1
            if len(array_action_zones_bench) == 0:
                break
            cen += 1
        sensor['action_zones']['peaks_coord'] = np.column_stack((array_max_x_bench, array_max_y_bench))
        peaks_coord = np.column_stack((array_max_x_bench, array_max_y_bench))
        sensor['action_zones']['peaks'] = max_bench_list
        for w in range(len(array_max_x_bench)):
            list_zone_bench = list()
            list_coord_bench = list()
            list_impo = list()
            del_list = list()
            coordinate_array = np.array(coord)
            for i in range(len(coord)):
                if math.sqrt(
                        (array_max_x_bench[w] - coordinate_array[i, 0]) ** 2 + (
                                array_max_y_bench[w] - coordinate_array[i, 1]) ** 2) <= radio:
                    list_zone_bench.append(bench[i])
                    list_coord_bench.append(coord[i])
                    list_impo.append(impo)
                    del_list.append(i)
            m = 0
            for i in range(len(del_list)):
                index_del = del_list[i] - m
                del coord[index_del]
                m += 1
            array_list_coord = np.array(list_coord_bench)
            # x_coord = array_list_coord[:, 0]
            # y_coord = array_list_coord[:, 1]
            # max_x_coord = max(x_coord)
            # min_x_coord = min(x_coord)
            # max_y_coord = max(y_coord)
            # min_y_coord = min(y_coord)
            # dict_limits_bench["action_zone%s" % j] = [min_x_coord, max_y_coord, max_x_coord, min_y_coord]
            index = list()
            for i in range(len(array_list_coord)):
                x = array_list_coord[i, 0]
                y = array_list_coord[i, 1]
                for p in range(len(self.ava)):
                    if x == self.ava[p, 0] and y == self.ava[p, 1]:
                        index.append(p)
                        # action_zone_bench.append(benchmark[p])
                        break
            # dict_bench_["action_zone%s" % j] = list_zone_bench
            # dict_coord_bench["action_zone%s" % j] = list_coord_bench
            index_peaks = []
            for p in range(len(peaks_coord)):
                peak = peaks_coord[p]
                for k in range(len(self.X_test)):
                    coord_ = self.X_test[k]
                    if peak[0] == coord_[0] and peak[1] == coord_[1]:
                        index_peaks.append(k)
            sensor['action_zones']["action_zone%s" % j] = {}
            sensor['action_zones']["action_zone%s" % j]['number'] = self.action_bench
            sensor['action_zones']["action_zone%s" % j]['index'] = index
            sensor['action_zones']["action_zone%s" % j]['priority'] = list_impo
            sensor['action_zones']["action_zone%s" % j]['coord'] = array_list_coord
            sensor['action_zones']['peaks_index'] = index_peaks
            impo -= 10
            self.action_bench += 1
            j += 1
        sensor['action_zones']['number'] = j
        sensor['action_zones']['radio'] = radio
        return sensor

    def overlapping_areas(self, sensors, dict, check):
        sz = 0
        d_ = {}
        z_ = {}
        az = 0
        for s, sensor in enumerate(sensors):
            center1 = dict[sensor]['action_zones']['peaks_coord']
            sen = s + 1
            for c, center in enumerate(center1):
                r1 = dict[sensor]['action_zones']["action_zone%s" % c]['radio']
                if not check:
                    sen = s
                else:
                    sen = s + 1
                for e in range(sen, len(sensors)):
                    center2 = dict[sensors[e]]['action_zones']['peaks_coord']
                    if e == s:
                        h = c + 1
                    else:
                        h = 0
                    for n in range(h, len(center2)):
                        r2 = dict[sensors[e]]['action_zones']["action_zone%s" % n]['radio']
                        # print(r2)
                        dist = d(center, center2[n])
                        if dist < (r1 + r2):
                            d_['subzone%s' % sz] = {}
                            d_['subzone%s' % sz]['index'] = np.union1d(
                                dict[sensor]['action_zones']["action_zone%s" % c]['index'],
                                dict[sensors[e]]['action_zones']["action_zone%s" % n]['index'])
                            d_['subzone%s' % sz]['zones'] = np.union1d(
                                dict[sensor]['action_zones']["action_zone%s" % c]['number'],
                                dict[sensors[e]]['action_zones']["action_zone%s" % n]['number'])
                            d_['subzone%s' % sz]['sensors'] = np.union1d(sensor, sensors[e])
                            d_['subzone%s' % sz]['peaks'] = np.union1d(
                                dict[sensor]['action_zones']["action_zone%s" % c]['center'],
                                dict[sensors[e]]['action_zones']["action_zone%s" % n]['center'])
                            # print('subzones', d_['subzone%s' % sz]['zones'])
                            sz += 1
                            az += 1
        s_ = copy.copy(d_)
        r = 0
        while r < 2:
            for s in range(sz):
                y = s + 1
                for t in range(y, sz):
                    sensor = s_['subzone%s' % s]['zones']
                    sensor1 = s_['subzone%s' % t]['zones']
                    if len(sensor) > 0 and len(sensor1) > 0:
                        inter = list(set(sensor) & set(sensor1))
                        # print(sensor, 'sen1', sensor1,'inter', inter)
                        if len(inter) > 0:
                            s_['subzone%s' % s]['index'] = np.union1d(s_['subzone%s' % s]['index'],
                                                                      s_['subzone%s' % t]['index'])
                            s_['subzone%s' % s]['zones'] = np.union1d(s_['subzone%s' % s]['zones'],
                                                                      s_['subzone%s' % t]['zones'])
                            s_['subzone%s' % s]['sensors'] = np.union1d(s_['subzone%s' % s]['sensors'],
                                                                         s_['subzone%s' % t]['sensors'])
                            s_['subzone%s' % s]['peaks'] = np.union1d(s_['subzone%s' % s]['peaks'],
                                                                        s_['subzone%s' % t]['peaks'])
                            s_['subzone%s' % t]['zones'] = []
                            s_['subzone%s' % t]['index'] = []
                            s_['subzone%s' % t]['sensors'] = []
                            s_['subzone%s' % t]['peaks'] = []
            r += 1
        p = 0
        for s in range(sz):
            sensor = s_['subzone%s' % s]['zones']
            if len(sensor) > 0:
                coord = list()
                z_['zone%s' % p] = {}
                z_['zone%s' % p]['number'] = p
                z_['zone%s' % p]['index'] = s_['subzone%s' % s]['index']
                z_['zone%s' % p]['act_zones'] = s_['subzone%s' % s]['zones']
                z_['zone%s' % p]['sensors'] = dict.fromkeys(list(s_['subzone%s' % s]['sensors']), [])
                z_['zone%s' % p]['peaks'] = s_['subzone%s' % s]['peaks']
                # z_['zone%s' % p]['rad'] = s_['subzone%s' % s]['rad']
                z_['zone%s' % p]['priority'] = [len(s_['subzone%s' % s]['sensors'])]
                # print(z_['zone%s' % p]['sensors'])
                index = z_['zone%s' % p]['index']
                for i in range(len(index)):
                    coord.append(self.coord[index[i]])
                z_['zone%s' % p]['coord'] = copy.copy(coord)
                p += 1
        o = p
        for s, sensor in enumerate(sensors):
            center1 = dict[sensor]['action_zones']['peaks_coord']
            for c, center in enumerate(center1):
                zon = False
                for y in range(p):
                    numb = z_['zone%s' % y]['act_zones']
                    for u in range(len(numb)):
                        if dict[sensor]['action_zones']["action_zone%s" % c]['number'] == numb[u]:
                            zon = True
                if not zon:
                    coord = list()
                    z_['zone%s' % o] = {}
                    z_['zone%s' % o]['number'] = o
                    z_['zone%s' % o]['index'] = dict[sensor]['action_zones']["action_zone%s" % c]['index']
                    z_['zone%s' % o]['act_zones'] = [dict[sensor]['action_zones']["action_zone%s" % c]['number']]
                    z_['zone%s' % o]['sensors'] = dict.fromkeys([sensor], [])
                    z_['zone%s' % o]['peaks'] = [dict[sensor]['action_zones']["action_zone%s" % c]['center']]
                    z_['zone%s' % o]['priority'] = [len(sensor)]
                    index = z_['zone%s' % o]['index']
                    # print(z_['zone%s' % o]['sensors'])
                    for i in range(len(index)):
                        coord.append(self.coord[index[i]])
                    z_['zone%s' % o]['coord'] = copy.copy(coord)
                    o += 1
        return z_

    def re_overlap(self, z_, no_assigned, dict_, s_sf):
        name = []
        for i in range(len(no_assigned)):
            # print(no_assigned)
            name.append('zone%s' % no_assigned[i])
        for z, nzone in enumerate(name):
            keys = copy.copy(list(z_.keys()))
            # print(keys, 'so', nzone)
            keys.remove(nzone)
            data = z_[nzone]['act_zones']
            # print(data)
            sensors = []
            number = []
            centers = []
            for n in range(len(data)):
                for s, sen in enumerate(s_sf):
                    numbers = dict_[sen]['action_zones']['number']
                    for j in range(numbers):
                        value = dict_[sen]['action_zones']['action_zone%s' % j]['number']
                        if value == data[n]:
                            sensors.append(sen)
                            number.append(j)
                            centers.append(dict_[sen]['action_zones']['action_zone%s' % j]['center'])
            min_dist = 10000000
            for t in range(len(centers)):
                for k, key in enumerate(keys):
                    data1 = z_[key]['act_zones']
                    sensors1 = []
                    number1 = []
                    centers1 = []
                    rad1 = []
                    for m in range(len(data1)):
                        for e, sens in enumerate(s_sf):
                            numbers1 = dict_[sens]['action_zones']['number']
                            for l in range(numbers1):
                                value = dict_[sens]['action_zones']['action_zone%s' % l]['number']
                                if value == data1[m]:
                                    sensors1.append(sens)
                                    number1.append(l)
                                    centers1.append(dict_[sens]['action_zones']['action_zone%s' % l]['center'])
                                    rad1.append(dict_[sens]['action_zones']['action_zone%s' % l]['radio'])
                    for c in range(len(centers1)):
                        distance = d(centers[t], centers1[c])
                        if min_dist > distance:
                            min_dist = distance
                            coupled = [name[z], centers[t], sensors[t], number[t], key, centers1[c], sensors1[c], number1[c], rad1[c]]
                # print(coupled)
                new_rad = int(math.floor(min_dist - (coupled[8] - 2)))
                list_coord = list()
                list_impo = list()
                impo = dict_[coupled[6]]['action_zones']['action_zone%s' % coupled[7]]['priority'][0]
                center = coupled[1]
                x = center[0]
                y = center[1]
                coordinate_array = np.array(self.coord)
                for i in range(len(coordinate_array)):
                    if math.sqrt(
                            (x - coordinate_array[i, 0]) ** 2 + (
                                    y - coordinate_array[i, 1]) ** 2) <= new_rad:
                        list_coord.append(self.coord[i])
                        list_impo.append(impo)
                index = list()
                array_list_coord = np.array(list_coord)
                for i in range(len(array_list_coord)):
                    x = array_list_coord[i, 0]
                    y = array_list_coord[i, 1]
                    for p in range(len(self.ava)):
                        if x == self.ava[p, 0] and y == self.ava[p, 1]:
                            index.append(p)
                            break
                dict_[coupled[2]]['action_zones']['action_zone%s' % coupled[3]]['priority'] = list_impo
                dict_[coupled[2]]['action_zones']['action_zone%s' % coupled[3]]['index'] = index
                dict_[coupled[2]]['action_zones']['action_zone%s' % coupled[3]]['coord'] = list_coord
                dict_[coupled[2]]['action_zones']['action_zone%s' % coupled[3]]['radio'] = new_rad
        return dict_


