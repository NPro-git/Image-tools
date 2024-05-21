
import os
import cv2
import numpy as np

dir = 'C:/FILEs/data/nails output/'

def getContours(data, th):
    """zone: zone[i] is a collection of coordinates belonging to connected
    component i. then zone[i][j] is one coordinate pair in zone i, consisting
    of (zone[i][j][0],zone[i][j][1])
    """

    def zone_mark(mark, edge):
        h, w = mark.shape
        tmp = np.zeros(mark.shape)
        ls, rs = np.ones(h).astype(np.uint32) * w, np.zeros(h).astype(np.uint32)
        ts, bs = np.ones(w).astype(np.uint32) * h, np.zeros(w).astype(np.uint32)
        for x, y in edge:
            ls[y], rs[y] = min(ls[y], x), max(rs[y], x)
            ts[x], bs[x] = min(ts[x], y), max(bs[x], y)
        for y in range(mark.shape[0]):
            for x in range(ls[y], rs[y] + 1):
                tmp[y, x] = tmp[y, x] + 1
        for x in range(mark.shape[1]):
            for y in range(ts[x], bs[x] + 1):
                tmp[y, x] = tmp[y, x] + 1
        tmp = (tmp > 1) * 255
        tmp = np.stack((mark, tmp))
        mark = np.max(tmp, axis=0)
        return mark

    # return all points that belong to the zone that starts from point coor
    def getZone(data, mark, th, coor):
        cir = [(-1, 0), (-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1)]
        # uplimit of the number of edge points
        edgeUplim = data.shape[0] * 2 + data.shape[1] * 4

        def within(data, coor):  # verdict if a point is within the image domain
            return coor[0] >= 0 and coor[0] < data.shape[1] and \
                coor[1] >= 0 and coor[1] < data.shape[0]

        # First step, extract the contour of the zone
        edge = [coor]
        lastDir = 2
        isolated = False
        while len(edge) < edgeUplim:
            for k in range(-2, 6):
                dir = (lastDir + k) & 0x07
                coor = [edge[-1][0] + cir[dir][0], edge[-1][1] + cir[dir][1]]
                if within(data, coor) and data[coor[1], coor[0]] > th:
                    edge.append(coor)
                    lastDir = dir
                    break
                if k == 5:
                    isolated = True
            if isolated or (len(edge) > 7 and edge[-1] == edge[1] and edge[-2] == edge[0]):
                break
        '''
        for x, y in edge:
            mark[y,x] = 255
        '''

        mark = zone_mark(mark, edge)

        return edge, mark
    contours = []
    mark = np.zeros(data.shape)
    for y in range(0, data.shape[0], 10):
        for x in range(1, data.shape[1]):
            if data[y,x-1] <= th and data[y,x] > th and mark[y][x] == 0:
                edge, mark = getZone(data, mark, th, [x, y])
                contours.append(edge)
    return contours