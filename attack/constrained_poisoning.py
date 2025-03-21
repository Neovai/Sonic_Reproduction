import math, torch
import copy
from attack.adversarial_base import AdversarialBase
from attack.evolve import SingleGenOptimization
from heapq import *

sign = lambda x: 0.0 if x == 0.0 else math.copysign(1, x)


class HeapItem:
    def __init__(self, score, eps):
        self.score = score
        self.eps = eps

    def __lt__(self, other):
        return self.score <= other.score

    def getattributes(self):
        return self.score, self.eps


class ConstrainedAdvPoisoningGlobal(SingleGenOptimization, AdversarialBase):
    def __init__(
        self,
        delta,
        s,
        clst_model,
        lb=1.0,
        G=70,
        mutation_rate=0.001,
        crossover_rate=0.8,
        zero_rate=0.0001,
        mode="guided",
        link="centroids",
        domain_cons=(0.0, 1.0),
        box_cons=None,
        objective="AMI",
    ):

        super(ConstrainedAdvPoisoningGlobal, self).__init__(
            G, delta, mutation_rate, crossover_rate, zero_rate=zero_rate, mode=mode
        )
        AdversarialBase.__init__(self, lb, domain_cons, box_cons, objective)

        ## Clustering parameters
        self.clst_model = clst_model  # clustering model to attack

        ## Algorithm parameters
        self.s = s  # number of data points to attack (percentage of original dataset)

        ## Perturbation Parameters and precision
        self.sign = sign(delta)
        self.delta = math.fabs(delta)  # noise threshold

        ## Constraints and Optimization Parameters
        self.objective = objective
        self.mode = mode
        self.link = link

    def get_fitness(self, Xadv, eps, Y):
        Yadv = self.clst_model.fit_predict(Xadv)
        miss_clust = self.eval_adv_efficiency(Y, Yadv)
        fit_miss = self.eval_objective_cost(miss_clust, eps) # main oblective function
        return fit_miss

    def inject_noise(self, X, ts_idx, eps):
        Xadv = X.clone()
        Xadv[ts_idx, :] = X[ts_idx, :] + eps[ts_idx, :] * self.direction
        Xadv = self.fix_to_constraints(Xadv)
        return Xadv

    def sonic(self, X, ts_idx, poison_idxs):
        """
        TODO: split X into clean and poison set. Wont work currently as we approximate entire dataset (X)

        Proposed Sonic algorithm in paper. Very similar to slowP.
        Returns optimal noise mask (eps*)
        X - full dataset
        ts_idx - indices of samples in target cluster
        """
        # split clean and touchable samples in X:
        targets = torch.unique(ts_idx)
        clean_idxs = ~torch.isin(torch.tensor(range(X.size(0))), targets)
        D_clean = X[clean_idxs, :].clone()
        D_poison = X[targets, :].clone()

        # Copy base state of clustering model
        self.clst_model.fit_predict(D_clean) # M is self.clst_model
        M = copy.deepcopy(self.clst_model)
        P = self.clst_model.update(D_poison)
    
        eps = torch.zeros(X.shape, dtype=self.dtype, device=self.device) # instead of random, initialize to 0
        candidates = [] # Theta: list of score candidates and eps pairs

        # For each generation g: update candidates with eps_g then evolve current e_g to get eps_g+1
        for g in range(self.G):
            Xadv = self.inject_noise(X, ts_idx, eps) # Xadv contains clean + poisoned
            D_poison = Xadv[targets, :] # extract only poisoned samples
            Yadv = self.clst_model.update(D_poison) # includes previously labeled clean points
            miss_clust = self.eval_adv_efficiency(P, Yadv) # phi
            theta = self.eval_objective_cost(miss_clust, eps) # main oblective function (equation 1.)
            heappush(candidates, HeapItem(theta, eps)) # update list of epsilons
            eps = self.evolve(eps, ts_idx, candidates) # run genetic algorithm (ch, cr, mut) on current eps 

        candidate = nsmallest(1, candidates)[0]
        opt_score, opt_eps = candidate.getattributes()
        return opt_eps

    def slowP(self, X, ts_idx):
        """
        Original black-box threat algorithm

        X - full clean dataset
        ts_idx - indices of touchable samples to poison

        Previously named global_evolve_perturbation
        """
        Y = self.clst_model.fit_predict(X)
        eps = self.rand_eps_mask(X, ts_idx)
        eps0 = eps.clone()
        candidates = []
        for g in range(self.G):  # for each generation g
            Xadv = self.inject_noise(X, ts_idx, eps)
            fit_miss = self.get_fitness(Xadv, eps, Y)
            heappush(candidates, HeapItem(fit_miss, eps))
            eps = self.evolve(eps, ts_idx, candidates)
        candidate = nsmallest(1, candidates)[0]
        opt_score, opt_eps = candidate.getattributes() # opt_score contains best mask's AMI score
        return opt_eps

    def retrieve_sensitive_entities(
        self, X, Y, origin_clst, origin_cluster_lb, target_c, s
    ):
        """
        Get entities from the origin cluster that are sensitive to be moved towards the target centroid
        :param X: Training dataset from which cluster are obtained
        :param Y: Labeling
        :param origin_clst: cluster to perturb
        :param origin_cluster_lb: origin cluster label
        :param target_c: target centroid
        :param s: number of sensitive entities to return
        :return: indexes of sensitive entities in X, cluster's centroids,
        """
        sensitive_entities_in_tgclust = self.nearest_idxpoints(origin_clst, target_c, s) # Select flattened images in dataset
        tg_sensitive_idx_X = torch.nonzero(Y == origin_cluster_lb, as_tuple=False) # get sample indices in original clustering

        # clamp points in target cluster to not exceed max index value in tg_sensitive_idx_X:
        sensitive_entities_in_tgclust = sensitive_entities_in_tgclust.clamp(min=0, max=tg_sensitive_idx_X.size(0)-1)

        # Flatten all touchable points in images (s * imageSize):
        # print(torch.unique(tg_sensitive_idx_X).shape)
        # print(sensitive_entities_in_tgclust)
        # print(tg_sensitive_idx_X[sensitive_entities_in_tgclust])
        tg_sensitive_idx_X = tg_sensitive_idx_X[sensitive_entities_in_tgclust][:s].view(-1)
        return tg_sensitive_idx_X, sensitive_entities_in_tgclust

    def get_sensitive_entities(self, X, Y, from_to, s):
        origin_clst = self.get_components_of_cluster(X, Y, from_to[0])
        target_clst = self.get_components_of_cluster(X, Y, from_to[1])
        t_centroid = self.get_centroid(target_clst)

        if 0 <= s < 1:
            s = int(origin_clst.shape[0] * s)

        nn_sensitive_entities_idx, touchable_sample_idxs = self.retrieve_sensitive_entities(
            X, Y, origin_clst, from_to[0], t_centroid, s
        )
        return nn_sensitive_entities_idx, touchable_sample_idxs

    def generate_adv_noise(self, X, tg_idx, poison_idxs, method="slowP"):
        Xadv = X.clone()
        if method == "slowP":
            opt_eps = self.slowP(X, tg_idx)
        elif method == "sonic":
            opt_eps = self.sonic(X, tg_idx, poison_idxs)
        else:
            raise ValueError(f"argument method = \"{method}\" (expecting \"slowP\" or \"sonic\")")
        Xadv[tg_idx, :] = X[tg_idx, :] + self.direction * opt_eps[tg_idx, :]
        Xadv = self.fix_to_constraints(Xadv)
        return Xadv, opt_eps

    def get_likage_direction(self, X, Y, ts_idx, from_to):
        cl_t = self.get_components_of_cluster(X, Y, from_to[1])
        ct = self.get_centroid(cl_t)
        if self.link == "centroids":
            cl_o = self.get_components_of_cluster(X, Y, from_to[0])
            co = self.get_centroid(cl_o)
            difference = ct - co
        elif self.link == "single":
            D = (X[ts_idx, :].unsqueeze(1) - cl_t.unsqueeze(0)).norm(2, dim=2)
            closest = D.argmin(dim=1).squeeze(1)
            difference = cl_t[closest] - X[ts_idx]
        elif self.link == "complete":
            D = (X[ts_idx, :].unsqueeze(1) - cl_t.unsqueeze(0)).norm(2, dim=2)
            farthest = D.argmax(dim=1).squeeze(1)
            difference = cl_t[farthest] - X[ts_idx]
        else:
            difference = ct - X[ts_idx]
        return torch.sign(difference)

    def forward(self, X, Y, method="slowP", from_to=[1, 0]):
        """
        Original black-box threat algorithm

        Manipulate entries in an adversarial way for reaching a desired miss-clustering
        :param X: pure/estimated training set
        :param Y: pure/estimated labelling for entries in X
        :param from_to: [origin cluster, destination cluster]
        :return: Xadv, poisoned dataset obtained from X
        """
        self.set_device(X.device)
        self.set_dtype(X.dtype)

        # Get sample indices from the origin cluster that are sensitive to be moved towards the target centroid:
        tg_sensitive_idx, touchable_D = self.get_sensitive_entities(X, Y, from_to, self.s)
        # Get direction to move sensitive samples from original centroid to target centroid:
        self.direction = self.get_likage_direction(X, Y, tg_sensitive_idx, from_to)
        # Run black-box genetic poisoning algorithm to generate poison mask (epsilon):
        Xadv, eps = self.generate_adv_noise(X, tg_sensitive_idx, touchable_D, method)
        return Xadv, eps, tg_sensitive_idx, self.direction

    def multi_forward(self, X, Y, y_target=1):
        Xadv = X.clone()
        ts_idx = torch.tensor([], dtype=torch.long, device=X.device)

        for i in Y.unique():
            if i != y_target:
                Xadv, _, ts, _ = self.forward(Xadv, Y, from_to=[i, y_target])
                ts = ts.to(ts_idx.device)
                ts_idx = torch.cat([ts_idx, ts], dim=0)
        return Xadv, ts_idx
