#####################################################
# (C) 2016 Koh Young Research America (KYRA)        #
#                                                   #
# Proprietary and confidential                      #
#####################################################


class CoordinateTransform(object):
    def __init__(self, fov_x, fov_y, width, height, scale_x, scale_y):
        """
        Transforms coordinates from the physical dimensions of the board to
        pixel coordinates in the given field of view
        :param fov_x:
        :param fov_y:
        :param width:
        :param height:
        :param scale_x:
        :param scale_y:
        """
        self._fov_x = fov_x
        self._fov_y = fov_y
        self._width = width
        self._height = height
        self._scale_x = scale_x
        self._scale_y = scale_y

    def convert(self, x, y):
        """
        Converts coordinates, returns a tuple (x,y)
        :param x: x in absolute physical coordinates
        :param y: y in absolute physical coordinates
        :return: (x,y) in pixel coordinates
        :rtype: tuple
        """
        r_x = int((float(x) - self._fov_x + self._width) * self._scale_x)
        r_y = int((float(y) - self._fov_y + self._height) * self._scale_y)
        return r_x, r_y

    def scale(self, w, h):
        return int(w*self._scale_x), int(h*self._scale_y)

    def inv_scale(self, w, h):
        return (w/self._scale_x), (h/self._scale_y)
