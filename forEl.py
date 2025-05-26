import numpy as np


def euclidean_distance(data_point, centroids):
    return np.sqrt(np.sum((centroids - data_point) ** 2, axis=0))

def get_clusters(points, radius):
    center_points = []
    cnt = 0
    labels = [0] * len(points)

    while len(points) != 0:
        # here choose random point and get its neighbors
        cur_point = get_random_point(points)
        neighbors = get_neighbors(cur_point, radius, points)
        center_point = get_center_point(neighbors)
        # change data untill range beetween points will be higher than accuracy
        # maybe change to some const number
        while euclidean_distance(cur_point,  center_point) > radius:
            cur_point = center_point
            neighbors = get_neighbors(cur_point, radius, points)
            center_point = get_center_point(neighbors)

        points, labels, cnt = remove_points(neighbors, points, labels, cnt)
        center_points.append(cur_point)

    return np.array(center_points), labels


# get point neighbors obv, which lays in circle
def get_neighbors(cur_point, radius, points):
    neighbors = [point for point in points if euclidean_distance(cur_point,  point) <= radius]
    return np.array(neighbors)


def get_center_point(points):
    pts = []
    # go threw np array collumns and chose avg point like: sum(point.value)/number of points
    for i in range(len(points[0])):
        pts.append(np.mean(points[:, i]))
    # pts.append(points[len(points)/2])
    return np.array(pts)


def get_random_point(points):
    rnd_index = np.random.choice(len(points), 1)[0]
    return points[rnd_index]


# removing points that lies in circle, means that points will be in that cluster
def remove_points(sub_points, pointis, labels, cnt):
    for i in range(len(sub_points)):
        for j in range(len(pointis)):
            if pointis[j].all() == sub_points[i].all():
                labels[j] = cnt
                j = len(pointis)

    points = [point for point in pointis if point not in sub_points]
    cnt = cnt + 1
    return points, labels, cnt
