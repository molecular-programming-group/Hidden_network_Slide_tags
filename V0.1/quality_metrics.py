from scipy.spatial import KDTree
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import numpy as np
import scipy

class QualityMetrics:
    """usage:
            qm = QualityMetrics(original_points, reconstructed_points)
            og_metrics_dict = qm.evaluate_metrics()
    """
    def __init__(self, original_points, reconstructed_points, k=15, threshold=1000, compute_distortion=False):
        """
        :param k: the "k" of the knn metric
        :param threshold: the cpd threshold, mainly due to memory restrictions
        """
        self.k = k
        self.threshold = threshold
        
        self.original_points = original_points
        self.reconstructed_points = reconstructed_points


        self.knn = None
        self.cpd = None
        self.distortion = None
        self.knn_individual = None
        self.distortion_individual = None


    def knn_metric(self, visualize=False):
        num_points = len(self.original_points)
        original_tree = KDTree(self.original_points)
        reconstructed_tree = KDTree(self.reconstructed_points)
        original_neighbors = original_tree.query(self.original_points, self.k + 1)[1][:, 1:]
        reconstructed_neighbors = reconstructed_tree.query(self.reconstructed_points, self.k + 1)[1][:, 1:]
        
        # This does the same as below but in one line
        # shared_neighbors = sum([len(set(original).intersection(set(reconstructed))) for original, reconstructed in zip(original_neighbors, reconstructed_neighbors)])


        individual_knn = []
        count = 0
        for original, reconstructed in zip(original_neighbors, reconstructed_neighbors):
            n = len(original)
            # TODO: this probably raises error if len(reconstructed) < len(original)
            if count==0:
                count+=1
                # print("original neighbors", original)
                # print("reconstructed neighbors", reconstructed)
            individual_knn.append(len(set(original).intersection(set(reconstructed[:n]))) / n)

        self.knn_individual = individual_knn
        self.knn = sum(individual_knn) / len(individual_knn)


        # TODO: delete this or get in another function
        point_index = 50  # dummy index
        if visualize and point_index < num_points:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Original points and neighbors
            axes[0].scatter(self.original_points[:, 0], self.original_points[:, 1], c='grey')
            axes[0].scatter(self.original_points[point_index, 0], self.original_points[point_index, 1], c='red')
            axes[0].scatter(self.original_points[original_neighbors[point_index], 0],
                            self.original_points[original_neighbors[point_index], 1], c='blue')
            axes[0].text(self.original_points[point_index, 0], self.original_points[point_index, 1], str(point_index),
                         fontsize=12, color='red')

            for neighbor_index in original_neighbors[point_index]:
                axes[0].text(self.original_points[neighbor_index, 0], self.original_points[neighbor_index, 1],
                             str(neighbor_index), fontsize=12, color='blue')

            axes[0].set_title('Original Point and Neighbors')

            # Reconstructed points and neighbors
            axes[1].scatter(self.reconstructed_points[:, 0], self.reconstructed_points[:, 1], c='grey')
            axes[1].scatter(self.reconstructed_points[point_index, 0], self.reconstructed_points[point_index, 1], c='red')
            axes[1].scatter(self.reconstructed_points[reconstructed_neighbors[point_index], 0],
                            self.reconstructed_points[reconstructed_neighbors[point_index], 1], c='blue')
            axes[1].text(self.reconstructed_points[point_index, 0], self.reconstructed_points[point_index, 1], str(point_index),
                         fontsize=12, color='red')

            for neighbor_index in reconstructed_neighbors[point_index]:
                axes[1].text(self.reconstructed_points[neighbor_index, 0], self.reconstructed_points[neighbor_index, 1],
                             str(neighbor_index), fontsize=12, color='blue')

            axes[1].set_title('Reconstructed Point and Neighbors')

            plt.show()

        return self.knn
        # return shared_neighbors / (self.k * num_points)

    def cpd_metric(self):
        num_points = len(self.original_points)
        if num_points > self.threshold:
            indices = np.random.choice(num_points, self.threshold, replace=False)
            self.original_points = self.original_points[indices]
            self.reconstructed_points = self.reconstructed_points[indices]
        original_distances = pdist(self.original_points)
        reconstructed_distances = pdist(self.reconstructed_points)
        correlation, _ = pearsonr(original_distances, reconstructed_distances)
        # R_squared
        r_squared = correlation**2
        return r_squared, [original_distances, reconstructed_distances]

    def compute_distortion(self, original_positions, reconstructed_positions):
        """
        Compute the distortion between the original and reconstructed 2D point clouds.

        :param original_positions: NumPy array of shape (N, 2) representing the original points
        :param reconstructed_positions: NumPy array of shape (N, 2) representing the reconstructed points
        :return: The distortion, a median value of the distances between original and reconstructed points
        """
        distances = [np.linalg.norm(original - reconstructed) for original, reconstructed in
                     zip(original_positions, reconstructed_positions)]
        distortion = np.median(distances)
        return distortion

    def get_distortion(self, original_positions, reconstructed_positions):
        # TODO: how to account for reflection and scale? Use old pipeline? But that requires artificial positioning of nodes
        mtx1, mtx2, disparity = scipy.spatial.procrustes(reconstructed_positions, original_positions)
        distances = np.linalg.norm(mtx1 - mtx2, axis=1)
    def evaluate_metrics(self,  distortion=False):
        knn_result = self.knn_metric()
        cpd_result, distances_for_plotting = self.cpd_metric()
        quality_metrics = {'KNN': knn_result, 'CPD': cpd_result}
        if distortion:
            distortion_result = self.compute_distortion()
            quality_metrics.update({"Distortion": distortion_result, "original_distances":distances_for_plotting[0], "reconstructed_distances":distances_for_plotting[0]})
        # print(quality_metrics)
        quality_metrics.update({"original_distances":distances_for_plotting[0], "reconstructed_distances":distances_for_plotting[1]})
        return quality_metrics
