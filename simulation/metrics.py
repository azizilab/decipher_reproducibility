import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import scanpy as sc
from scipy import stats
from sklearn.metrics import pairwise_distances


def compute_distances_between_clusters(data, cluster_ids, metric="euclidean"):
    avg_dists = []
    unique_labels = np.unique(cluster_ids)
    unique_labels.sort()
    n_labels = len(unique_labels)
    distances = np.zeros((n_labels, n_labels))

    points_distances = pairwise_distances(data, data, metric=metric)
    for i, label_1 in enumerate(unique_labels):
        for j, label_2 in enumerate(unique_labels):
            distances[i, j] = points_distances[cluster_ids == label_1, :][
                :, cluster_ids == label_2
            ].mean()
    return distances


def get_obsm(adata, key):
    if key is None:
        return adata.X
    return adata.obsm[key]


def metric_global_distortion(adata, obsm_key, cluster_key, obsm_key_ref=None):
    cluster_rank_ambient = compute_distances_between_clusters(
        get_obsm(adata, obsm_key_ref), adata.obs[cluster_key]
    )
    cluster_rank_obsm = compute_distances_between_clusters(
        get_obsm(adata, obsm_key), adata.obs[cluster_key]
    )
    res = []
    for i in range(cluster_rank_ambient.shape[0]):
        tau, p_value = stats.kendalltau(cluster_rank_obsm[i, :], cluster_rank_ambient[i, :])
        res.append(tau)
    return np.array(res)


def compute_plot_metrics(adata, keys, ref_key=None, show_plots=True, color=None):
    res = dict()
    for k in keys:
        res[k] = metric_global_distortion(adata, k, "cluster_latent", obsm_key_ref=ref_key)

    res = pd.DataFrame(res)
    return res
