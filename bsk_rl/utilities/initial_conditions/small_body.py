import numpy as np
import math as m


def generate_waypoints(num_spheres, num_lat, num_lon, radius):
    """
    Generates a number of spheres of waypoints, each sphere is at [1...n]*radius,
    where n is the num_spheres
    """

    for idx in range(num_spheres):
        u1, v1 = np.mgrid[0 : 2 * np.pi : (num_lon * 1j), 0 : np.pi : (num_lat * 1j)]
        x1 = (idx + 1) * radius * np.cos(u1) * np.sin(v1)
        y1 = (idx + 1) * radius * np.sin(u1) * np.sin(v1)
        z1 = (idx + 1) * radius * np.cos(v1)

        x1 = x1.reshape(x1.size)
        y1 = y1.reshape(y1.size)
        z1 = z1.reshape(z1.size)

        if idx == 0:
            waypoints = np.array([x1, y1, z1])
        else:
            waypoints = np.concatenate((waypoints, [x1, y1, z1]), axis=1)

    waypoints = waypoints.T
    repeats = []

    # Check for duplicates
    for idx1, row1 in enumerate(waypoints):
        for idx2, row2 in enumerate(waypoints):
            if idx1 == idx2:
                continue
            else:
                if np.linalg.norm(row1 - row2) < 1e-8:
                    if idx2 > idx1:
                        repeats.append(idx2)

    waypoints = np.delete(waypoints, repeats, axis=0)

    return waypoints


def generate_mapping_points(num_points, radius):
    """Generates a number of mapping points on the surface of the body using a
    Fibonnaci sphere Algorithm from:
    https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    """

    points = []
    phi = m.pi * (3.0 - m.sqrt(5.0))  # golden angle in radians

    for i in range(num_points):
        y = 1 - (i / float(num_points - 1)) * 2  # y goes from 1 to -1
        r = m.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = m.cos(theta) * r
        z = m.sin(theta) * r

        points.append(
            radius
            * np.matmul(
                np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]), np.array([x, y, z])
            )
        )

    return np.array(points)


def generate_imaging_points(num_points, radius):
    """Generates a number of random imaging points on the surface of the body"""
    points = np.random.uniform(-1.0, 1.0, size=(num_points, 3))

    for idx in range(num_points):
        points[idx, :] = radius * points[idx, :] / np.linalg.norm(points[idx, :])

    return points
