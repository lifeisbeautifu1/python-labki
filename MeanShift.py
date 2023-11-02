import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

np.random.seed(42)
GROUP_DISTANCE_TOLERANCE = 0.1
MIN_DISTANCE = 1e-3

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class PointGrouper(object):
    def group_points(self, points):
        group_assignment = []
        groups = []
        group_index = 0
        for point in points:
            nearest_group_index = self._determine_nearest_group(point, groups)
            if nearest_group_index is None:
                # create new group
                groups.append([point])
                group_assignment.append(group_index)
                group_index += 1
            else:
                group_assignment.append(nearest_group_index)
                groups[nearest_group_index].append(point)
        return np.array(group_assignment)

    def _determine_nearest_group(self, point, groups):
        nearest_group_index = None
        index = 0
        for group in groups:
            distance_to_group = self._distance_to_group(point, group)
            if distance_to_group < GROUP_DISTANCE_TOLERANCE:
                nearest_group_index = index
            index += 1
        return nearest_group_index

    def _distance_to_group(self, point, group):
        min_distance = sys.float_info.max
        for pt in group:
            dist = euclidean_distance(point, pt)
            if dist < min_distance:
                min_distance = dist
        return min_distance

class MeanShift(object):
    def cluster(self, points, σ=1):
        shifted_points = points.copy()

        max_min_dist = 1

        done_shifting = [False] * points.shape[0]

        # untill maximum of all distances reach epsilon
        while max_min_dist > MIN_DISTANCE:
            max_min_dist = 0

            for i in range(0, len(shifted_points)):

                if done_shifting[i]:
                    continue

                p_new_start = shifted_points[i]
                p_new = self._shift_point(p_new_start, points, σ)
                dist = euclidean_distance(p_new, p_new_start)

                if dist > max_min_dist:
                    max_min_dist = dist

                if dist < MIN_DISTANCE:
                    done_shifting[i] = True

                shifted_points[i] = p_new
            
        point_grouper = PointGrouper()
        group_assignments = point_grouper.group_points(shifted_points)
        return points, shifted_points, group_assignments

    def _shift_point(self, point, points, σ):
        # from http://en.wikipedia.org/wiki/Mean-shift

        # numerator
        point_weights = self._gaussian_kernel(point - points, σ)
        tiled_weights = np.tile(point_weights, [2, 1])
        # denominator
        denominator = sum(point_weights)
        shifted_point = np.multiply(tiled_weights.transpose(), points).sum(axis=0) / denominator
        return shifted_point
    
        # shift_x = 0.0
        # shift_y = 0.0
        # scale_factor = 0.0

        # for other_point in points:
        #     # numerator
        #     dist = euclidean_distance(point, other_point)
        #     weight = self._gaussian_kernel(dist, σ)
        #     shift_x += other_point[0] * weight
        #     shift_y += other_point[1] * weight
        #     # denominator
        #     scale_factor += weight
        # shift_x = shift_x / scale_factor
        # shift_y = shift_y / scale_factor
        # return [shift_x, shift_y]

    def _gaussian_kernel(self, x, σ):
        # https://en.wikipedia.org/wiki/Radial_basis_function_kernel
        squared_euclidean_distance = np.sqrt(((x)**2).sum(axis=1))
        return  (1 / (σ * (2 * np.pi) ** 0.5)) * np.exp(-0.5 *((squared_euclidean_distance / σ)) ** 2)


if __name__ == "__main__":
    # data = np.genfromtxt('data.csv', delimiter=',')

    data, y = make_blobs(centers=5, n_samples=1000, random_state=40)

    ms = MeanShift()
    original_points, shifted_points, cluster_assignments = ms.cluster(data)

    x = original_points[:, 0]
    y = original_points[:, 1]
    centers = shifted_points


    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = ax.scatter(x, y, c=cluster_assignments, s=50)

    for x, y in centers:
        ax.scatter(x, y, c='red', marker='x', linewidth=2)

    plt.show()