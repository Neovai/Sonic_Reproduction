import datetime
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import hdbscan
import warnings
import random
import sys
from torch_kmeans import KMeans
from models.fishdbc import FISHDBC
from torchmetrics.clustering import AdjustedMutualInfoScore
from datasets import Dataset_by_label, MNIST_cluster
from sonic import Sonic
from attack.constrained_poisoning import ConstrainedAdvPoisoningGlobal

"""
hierarchical algo:
- Hsl

density-based:
- DBSCAN (most-famous) - single epsilon distance parameter used for entire space

hybrid:
- HDBSCAN* (evolution of DBSCAN) - uses hueristic to find epsilon value for diff. parts of the space
    - supports hierarchical via recognizing clusters of diff densities within dataset

Incremental Density-Based Clustering:
- FISHDBC (used in Sonic); evolves HDBSCAN and approximates it (ef param used to control search cost - raising it increases computational cost)
    - ef parameter controls HNSW (Hierarchical Navigable Small Worlds)
    - used so don't have to re-compute entire clusters when using the sonic algorithm during genetic evolution - SAVES BIG TIME

HDBSCAN or DBSCAN are non-incremental; require whole clustering recomputed if new data added.

TODO: for noise:
we assign a unique label to each noise
sample to ensure that the presence of noise does not artificially
inflate the AMI score. This change ensures that noise points
do not contribute positively to the AMI score, making it more
sensitive to actual clustering performance changes.
"""

def distance(x, y):
    """
    distance for FISHDBC clustering (l2 norm)
    """
    return np.linalg.norm(x - y)


class Model():
    """
    Wrapper class for clustering algorithms to override their fit_predict function
    """
    def __init__(self, model):
        self.model = model
        pass

    def fit_predict(self, X):
        """
        fits prediction on entire X
        """
        if isinstance(self.model, hdbscan.HDBSCAN):
            labels = torch.IntTensor(self.model.fit_predict(X.squeeze(2))) # squeeze makes X 2D instead of 3D
        elif isinstance(self.model, KMeans):
            labels = self.model.fit_predict(X.view(1, X.size(0), X.size(1))).view(-1)
        elif isinstance(self.model, FISHDBC):
             torch.IntTensor(self.model.update(X.squeeze(2))) # squeeze makes X 2D instead of 3D
             labels = self.model.cluster()
        return labels
    
    def update(self, x):
        """
        approximates new fit based on previous model fit and new point(s) x

        returns prediction of new labels.
        """
        labels = None
        if isinstance(self.model, hdbscan.HDBSCAN):
            new_labels, _ = hdbscan.approximate_predict(self.model, x.squeeze(2)) # squeeze makes X 2D instead of 3D
            labels = torch.from_numpy(np.concatenate((self.model.labels_, new_labels), axis=0))
        elif isinstance(self.model, FISHDBC):
            torch.IntTensor(self.model.update(x.squeeze(2))) # squeeze makes X 2D instead of 3D
            labels = self.model.cluster()
        return labels


def relabel_noise(y_hat):
    """
    changes noise values in y_hat (-1) to unique values that aren't already classes in y_hat
    """
    k_classes = len(torch.unique(y_hat)) - 1 # Get # of classes not including noise (-1)
    num_nois_samples = y_hat[y_hat == -1].size(0)
    y_hat[y_hat == -1] = torch.tensor(range(k_classes, num_nois_samples + k_classes), dtype=y_hat.dtype)
    return y_hat


def main(out_file):
    # General Setup:
    np.random.seed(10)
    torch.manual_seed(10)
    random.seed(10)
    torch.backends.cudnn.deterministic = True
    warnings.simplefilter(action='ignore', category=FutureWarning)
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(DEVICE)

    data_path = "./data"
    ami = AdjustedMutualInfoScore()
    C = Model(hdbscan.HDBSCAN(min_cluster_size=5, prediction_data=True))
    # C = Model(flexible_clustering.FISHDBC(distance, ef=50))
    # C = Model(KMeans(n_clusters=2, max_iter=10))

    # Attack Hyper-params:
    s_range = [0.01, 0.05, 0.1, 0.2] # [0.01 - 0.2]
    delta_range = [0.05, 0.2, 0.4, 0.6] # [0.05 - 0.6]
    lambda_penalty = 0.1
    P_cr = 0.85 # cross-over rate
    P_mu = 0.15 # mutation rate
    P_zero = 0.05 # zero rate
    niters = 110
    ef = 50
    method = "sonic"

    # run expirement on MNIST:
    trainset = torchvision.datasets.MNIST(root=data_path, train=True, download=True, transform=transforms.Compose([transforms.ToTensor(), ]))
    MNIST_X, MNIST_Y = Dataset_by_label(trainset, labels=[0, 4], n_samples=800) # 800 used in Cina et. al. paper
    MNIST_X.to(DEVICE)
    MNIST_Y.to(DEVICE)

    # run expirement on FASHIONMNIST:
    # trans = transforms.Compose([transforms.ToTensor(),])
    # trainset = torchvision.datasets.FashionMNIST(root=data_path, train=True, transform=trans, download=True)
    # MNIST_X, MNIST_Y = Dataset_by_label(trainset, labels=[3, 9], n_samples=800) # 800 used in Cina et. al. paper, 3=dresss and 9=ankleboot
    # MNIST_X.to(DEVICE)
    # MNIST_Y.to(DEVICE)

    # create target mode:
    target_model = Model(hdbscan.HDBSCAN(min_cluster_size=50))
    # target_model = Model(KMeans(n_clusters=2, max_iter=10))

    # get clean clustering results
    print("Getting clean results")
    clean_results = target_model.fit_predict(MNIST_X)
    print(torch.unique(clean_results))
    print(torch.sum(clean_results == 0))
    print(torch.sum(clean_results == 1))
    
    with open(out_file + ".csv", "w+") as writer:
        writer.write("method s delta #_poisoned AMI l2 time\n")

    # run experiments:
    for s in s_range:
        for delta in delta_range:
            # Time attack:
            start_time = datetime.datetime.now()
            Attack = ConstrainedAdvPoisoningGlobal(delta, s, C, 
                                                lb=lambda_penalty, 
                                                G=niters,
                                                mutation_rate=P_mu,
                                                crossover_rate=P_cr,
                                                zero_rate=P_zero, 
                                                objective="AMI")
            adv_X, best_poison_mask, ts_idx, _ = Attack.forward(MNIST_X, clean_results, method=method, from_to=[1,0]) # binary clustering
            end_time = datetime.datetime.now()
            total_time = end_time - start_time

            # get eps L2 norm to measure how aggressive attack was:
            eps_l2 = torch.dist(MNIST_X, adv_X, p=2)
            num_poisoned = len(ts_idx) / 784

            print("Getting poisoned results")
            poisoned_results = target_model.fit_predict(adv_X)
            print(torch.unique(poisoned_results))
            print(torch.sum(poisoned_results == 0))
            print(torch.sum(poisoned_results == 1))

            # relabel noise points before evaluation as directed by paper:
            relabel_noise(clean_results)
            relabel_noise(poisoned_results)

            # Get and print results
            posion_diff = ami(clean_results, poisoned_results)
            print(f"Results for {method}: s = {s} || δ = {delta} || AMI = {posion_diff:.5f} || eps_l2 = {eps_l2:.5f} || # poisoned = {num_poisoned} || time = {total_time}")

            # write results to file:
            with open(out_file + ".csv", "a") as writer:
                writer.write(f"{method} {s} {delta} {num_poisoned} {posion_diff:.5f} {eps_l2:.5f} {total_time}\n")


if __name__ == "__main__":
    main(sys.argv[1])
