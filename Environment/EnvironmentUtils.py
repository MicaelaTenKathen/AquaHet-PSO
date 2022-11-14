from deap import benchmarks
import numpy as np
import math


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

        X = np.linspace(0, 1, self.navigation_map.shape[0])
        Y = np.linspace(0, 1, self.navigation_map.shape[1])
        X, Y = np.meshgrid(X, Y)
        Z = np.fromiter(map(self.shekel_arg0, zip(X.flat, Y.flat)), dtype=np.float, count=X.shape[0] * X.shape[1]).reshape(X.shape)
        Z = (Z - np.mean(Z))/np.std(Z) # Normalize Z
        nan_mask = np.copy(self.navigation_map)
        nan_mask[self.navigation_map == 0] = np.nan
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


class boundaries_generator:

    def __init__(self, shape, max_occupation, radius, dt=1, theta=0.15):

        self.noise = self.OrnsteinUhlenbeckActionNoise(np.array([0,0]), np.array([1,1]), dt=dt, theta=theta)
        self.noise.reset()
        self.shape = shape
        self.max_occupation = max_occupation
        self.radius = radius

    def reset(self):

        self.noise.reset()

    def sample(self):

        self.noise.reset()
        return self.generate_navigation_map(self.shape, self.max_occupation, self.radius)


    class OrnsteinUhlenbeckActionNoise:
        def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
            self.theta = theta
            self.mu = mu
            self.sigma = sigma
            self.dt = dt
            self.x0 = x0
            self.reset()

        def __call__(self):
            x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
            self.x_prev = x
            return x

        def reset(self):
            self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

        def __repr__(self):
            return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)

    def generate_navigation_map(self, shape, max_ocupation, radius = 5):
        """ Generate a ranfom navigation map with given shape"""

        # Base map #
        base_map = np.zeros(shape)

        # Initial random point #
        position = np.floor(np.array(shape)/2).astype(int)

        ocupation_rate = 0.0

        x = np.arange(0, shape[0])
        y = np.arange(0, shape[1])

        while ocupation_rate < max_ocupation:


            mask = (x[np.newaxis, :] - position[0]) ** 2 + (y[:, np.newaxis] - position[1]) ** 2 < radius ** 2
            base_map[mask] = 1

            movement = np.floor(self.noise()).astype(int)
            movement = np.clip(movement, a_min=-radius, a_max=radius)


            position = position + movement

            position = np.clip(position, a_min=(0, 0), a_max=shape)

            ocupation_rate = np.count_nonzero(base_map)/np.prod(shape)

        return base_map

class AStarPlanner:

    def __init__(self, obstacle_map, resolution, rr):
        """
        Initialize grid map for a star planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        """

        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        ox, oy = np.where(obstacle_map == 0)
        self.calc_obstacle_map(ox, oy)

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x  # index of grid
            self.y = y  # index of grid
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return str(self.x) + "," + str(self.y) + "," + str(
                self.cost) + "," + str(self.parent_index)

    def planning(self, start, goal):

        """
        A star path search

        output:
            rx: x position list of the final path
            ry: y position list of the final path

        """

        sx = start[0]
        sy = start[1]

        gx = goal[0]
        gy = goal[1]

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), 0.0, -1)

        open_set, closed_set = dict(), dict()
        open_set[self.calc_grid_index(start_node)] = start_node

        while 1:
            if len(open_set) == 0:
                break

            c_id = min(
                open_set,
                key=lambda o: open_set[o].cost + self.calc_heuristic(goal_node,
                                                                     open_set[
                                                                         o]))
            current = open_set[c_id]

            if current.x == goal_node.x and current.y == goal_node.y:
                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # Remove the item from the open set
            del open_set[c_id]

            # Add it to the closed set
            closed_set[c_id] = current

            # expand_grid search grid based on motion model
            for i, _ in enumerate(self.motion):
                node = self.Node(current.x + self.motion[i][0],
                                 current.y + self.motion[i][1],
                                 current.cost + self.motion[i][2], c_id)
                n_id = self.calc_grid_index(node)

                # If the node is not safe, do nothing
                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                if n_id not in open_set:
                    open_set[n_id] = node  # discovered a new node
                else:
                    if open_set[n_id].cost > node.cost:
                        # This path is the best until now. record it
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)

        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        # generate final course
        rx, ry = [self.calc_grid_position(goal_node.x, self.min_x)], [
            self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index

        return rx, ry

    @staticmethod
    def calc_heuristic(n1, n2):
        w = 5.0  # weight of heuristic
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d

    def calc_grid_position(self, index, min_position):
        """
        calc grid position

        :param index:
        :param min_position:
        :return:
        """
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):

        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)

        # obstacle map generation
        self.obstacle_map = [[False for _ in range(self.y_width)]
                             for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        # dx, dy, cost
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]

        return motion

