from PIL import Image
import numpy as np
from Data.data_path import map_path_classic, map_path


class Map():
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys
        return

    def black_white(self):
        im = Image.open(map_path_classic())
        # im = Image.open(map_path())
        nim = im.resize((self.xs, self.ys))
        array = np.zeros((self.xs, self.ys))
        img = Image.new('RGB', (self.xs, self.ys))
        j = 0
        while j < self.ys:
            i = 0
            while i < self.xs:
                r, g, b = nim.getpixel((i, j))
                p = (r * 0.3 + g * 0.59 + b * 0.11)
                gray = int(p)
                if gray < 225:
                    color = 255
                    bit = 0
                else:
                    color = 0
                    bit = 1
                pixel = tuple([color, color, color])
                img.putpixel((i, j), pixel)
                array[i, j] = int(bit)
                i += 1
            j += 1

        return array

    def map_values(self):
        grid_max_x = self.xs
        grid_max_y = self.ys

        grid_min = 0

        if grid_max_x < grid_max_y:
            grid_max = grid_max_y
        else:
            grid_max = grid_max_x

        return grid_min, grid_max, grid_max_x, grid_max_y
