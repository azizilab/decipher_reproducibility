import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phate
import scanpy as sc
import scvi
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

import decipher as dc

sc.set_figure_params(figsize=[3, 3])


class RandomNet(nn.Module):
    def __init__(self, n_in, n_out, seed=0):
        super().__init__()
        torch.manual_seed(seed)

        self.fc1 = nn.Linear(n_in, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_out)
        self.n_out = n_out

        # initialize weights
        nn.init.normal_(self.fc1.weight, 0, 10)
        nn.init.normal_(self.fc1.bias, 0, 0)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc2.bias, 0, 0)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.normal_(self.fc3.bias, 0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=-1)
        l = torch.normal(np.log(10_000), 0.1, size=(self.n_out,)).exp()
        x = l * x
        return x


def simulation_basic(n_samples=1_000, n_genes=200, seed=0, k_clusters=20) -> sc.AnnData:
    np.random.seed(seed)
    torch.manual_seed(seed)

    # latent time between 0 and 1
    latent_t = np.random.uniform(0, 1, size=(n_samples, 1))

    branching_time = 0.6
    branch = np.random.binomial(1, 0.5, size=(n_samples, 1)) * 2 - 1
    branch = branch * (latent_t > branching_time)
    latent_z = np.random.normal(
        latent_t * np.array([[1, 0]]) + branch * (latent_t - branching_time) * np.array([[0, 1]]),
        0.05,
    )

    net = RandomNet(2, n_genes)
    data = net(torch.tensor(latent_z).float()).detach().numpy().astype(int)

    sim_adata = sc.AnnData(data)
    sim_adata.obs["latent_t"] = latent_t
    sim_adata.obs["branch_id"] = branch
    sim_adata.obs["latent_z1"] = latent_z[:, 0]
    sim_adata.obs["latent_z2"] = latent_z[:, 1]
    sim_adata.obsm["latent"] = latent_z

    sim_adata.obs["cluster_latent"] = (
        KMeans(
            k_clusters,
        )
        .fit(latent_z[:, :2])
        .labels_
    )

    return sim_adata


def simulation_correlated_1(
    n_samples=500,
    n_genes=50,
    seed=0,
    sigma=0.03,
    branch_prob=0.7,
    k_clusters=20,
    hole_size=0,
    n_holes=0,
    hole_density=0.0,
) -> sc.AnnData:
    np.random.seed(seed=seed)

    chunk_size = np.array([1, hole_size] * n_holes + [1])
    chunk_offset = np.cumsum(chunk_size) - chunk_size[0]
    chunk_prob = np.array([1, hole_size * hole_density] * n_holes + [1])
    chunk_prob = chunk_prob / chunk_prob.sum()
    total_size = sum(chunk_size)
    # sample which chunk the latent time is in
    chunk = np.random.choice(np.arange(len(chunk_prob)), size=(n_samples,), p=chunk_prob)
    # sample the latent time within the chunk
    latent_t_chunk = np.random.uniform(0, 1, size=(n_samples,))
    latent_t = latent_t_chunk * chunk_size[chunk] + chunk_offset[chunk]
    latent_t = latent_t / total_size
    latent_t = latent_t[:, None]

    # branching
    branching_t1 = 0.3
    branching_t2 = 0.5
    branch_id = np.random.binomial(1, branch_prob, size=(n_samples, 1)) * 2 - 1
    branch_id = branch_id * (latent_t > branching_t1)

    z1 = latent_t * total_size
    z5 = branch_id * (latent_t - branching_t1)
    z3 = (branch_id == 1) * np.clip(latent_t - 0.3, 0, 0.5)
    z4 = (branch_id == 1) * np.clip(latent_t - branching_t2, 0, branching_t1)
    z2 = (branch_id >= 0) * (np.clip(latent_t - 0.2, 0, 0.5) - np.clip(latent_t - 0.5, 0, 0.5))

    latent_z = np.concatenate([z1, z3, z5], axis=1)

    latent_z_sampled = np.random.normal(latent_z, sigma)
    latent_z_sampled[:, 0] /= total_size
    net = RandomNet(
        latent_z_sampled.shape[1],
        n_genes,
    )
    data = net(torch.tensor(latent_z_sampled).float()).detach().numpy().astype(int)

    adata_sim = sc.AnnData(data)
    adata_sim.obs["latent_t"] = latent_t
    adata_sim.obs["branch_id"] = branch_id
    k_means = KMeans(k_clusters)
    k_means.fit(latent_z)

    adata_sim.obs["cluster_latent"] = k_means.labels_

    for i in range(latent_z.shape[1]):
        adata_sim.obs[f"latent_z{i}"] = latent_z[:, i]
    adata_sim.obsm["latent"] = latent_z_sampled
    latent_names = [f"latent_z{i}" for i in range(latent_z.shape[1])]
    adata_sim.uns["latent_z_names"] = latent_names

    # for each cluster, order the other clusters by distance of their kmeans center
    cluster_centers = k_means.cluster_centers_
    cluster_rank = np.argsort(
        np.linalg.norm(cluster_centers[:, None] - cluster_centers[None, :], axis=2)
    )
    adata_sim.uns["cluster_rank"] = cluster_rank[:, 1:]

    return adata_sim


_LOGGER = logging.getLogger(__name__)


def run_methods(adata, seed=0):
    latent_spaces = []

    # Compute UMAP
    _LOGGER.info("Computing UMAP")
    sc.pp.neighbors(adata, random_state=seed)
    sc.tl.umap(adata, random_state=seed)
    adata.obsm["X_default_umap"] = adata.obsm["X_umap"]
    latent_spaces.append("X_default_umap")
    _LOGGER.info("UMAP computed")

    # Compute scVI
    _LOGGER.info("Computing scVI")
    scvi.settings.seed = seed
    scvi.model.SCVI.setup_anndata(adata)
    sim_model = scvi.model.SCVI(adata, gene_likelihood="nb", n_latent=10)
    sim_model.train()
    adata.obsm["X_scVI"] = sim_model.get_latent_representation()
    latent_spaces.append("X_scVI")
    _LOGGER.info("scVI computed")

    # Compute scVI-umap
    _LOGGER.info("Computing scVI-UMAP")
    sc.pp.neighbors(adata, use_rep="X_scVI", key_added="scVI_neighbors", random_state=seed)
    sc.tl.umap(adata, neighbors_key="scVI_neighbors", random_state=seed)
    adata.obsm["X_scVI_umap"] = adata.obsm["X_umap"]
    latent_spaces.append("X_scVI_umap")
    _LOGGER.info("scVI-UMAP computed")

    # Compute scVI-linear
    _LOGGER.info("Computing scVI-linear")
    scvi.settings.seed = seed
    scvi.model.LinearSCVI.setup_anndata(adata)
    sim_model = scvi.model.LinearSCVI(adata, gene_likelihood="nb", n_latent=10)
    sim_model.train()
    adata.obsm["X_scVI_linear"] = sim_model.get_latent_representation()
    latent_spaces.append("X_scVI_linear")
    _LOGGER.info("scVI-linear computed")

    # Compute scVI-linear-umap
    _LOGGER.info("Computing scVI-linear-UMAP")
    sc.pp.neighbors(
        adata, use_rep="X_scVI_linear", key_added="scVI_linear_neighbors", random_state=seed
    )
    sc.tl.umap(adata, neighbors_key="scVI_linear_neighbors", random_state=seed)
    adata.obsm["X_scVI_linear_umap"] = adata.obsm["X_umap"]
    latent_spaces.append("X_scVI_linear_umap")
    _LOGGER.info("scVI-linear-UMAP computed")

    # Compute PCA
    _LOGGER.info("Computing PCA")
    sc.tl.pca(adata, n_comps=10, random_state=seed)
    latent_spaces.append("X_pca")
    _LOGGER.info("PCA computed")

    # Compute Fruchterman-Reingold
    _LOGGER.info("Computing Force Atlas 2")
    sc.tl.draw_graph(adata, layout="fa", random_state=seed)
    adata.obsm["X_fd"] = adata.obsm["X_draw_graph_fa"]
    latent_spaces.append("X_fd")
    _LOGGER.info("Force Atlas 2 computed")

    # Compute Phate
    _LOGGER.info("Computing Phate")
    phate_op = phate.PHATE(n_components=10, random_state=seed)
    adata.obsm["X_phate"] = phate_op.fit_transform(adata.X)
    latent_spaces.append("X_phate")
    _LOGGER.info("Phate computed")
    #
    _LOGGER.info("Computing Phate-UMAP")
    sc.pp.neighbors(adata, use_rep="X_phate", key_added="phate_neighbors", random_state=seed)
    sc.tl.umap(adata, neighbors_key="phate_neighbors", random_state=seed)
    adata.obsm["X_phate_umap"] = adata.obsm["X_umap"]
    latent_spaces.append("X_phate_umap")
    _LOGGER.info("Phate-UMAP computed")

    configuration = {"dim_v": 2, "dim_z": 10}
    config = dc.tl.DecipherConfig(
        **configuration, n_epochs=1000, early_stopping_patience=10, seed=seed
    )
    a = adata.copy()
    res = dc.tl.decipher_train(a, config, plot_every_k_epochs=-2)
    k = "decipher"
    if config.dim_v > 0:
        adata.obsm[k + "_decipher_v"] = a.obsm["decipher_v"]
        latent_spaces.append(k + "_decipher_v")
    adata.obsm[k + "_decipher_z"] = a.obsm["decipher_z"]
    latent_spaces.append(k + "_decipher_z")

    return latent_spaces


def repeated_benchmark(n_repeats=10, seed=0):
    results = []
    seed_init = seed
    for i in range(n_repeats):
        seed = seed_init + i
        adata_sim = simulation_correlated_1(
            n_samples=500,
            n_genes=50,
            seed=seed + i,
            sigma=0.03,
            branch_prob=0.7,
            k_clusters=20,
            hole_size=1,
            n_holes=3,
        )
        latent_spaces = run_methods(adata_sim, seed=seed + i)
        # latent_spaces = [x for x in latent_spaces if adata_sim.obsm[x].shape[1] == 2]

        local_distortion = compute_plot_metrics(
            adata_sim, latent_spaces, ref_key="latent", show_plots=False
        )["Cluster local distortion"]
        local_distortion = local_distortion.melt(var_name="method", value_name="value")
        local_distortion["seed"] = i
        results.append(local_distortion)

    results = pd.concat(results)

    compute_plot_metrics(
        adata_sim, latent_spaces, ref_key="latent", color=["latent_t", "branch_id"], show_plots=True
    )
    return results


if __name__ == "__main__":
    from metrics import compute_plot_metrics

    folder = "figures"
    os.makedirs(folder, exist_ok=True)

    # v2: n_genes=100, n_samples=2000, sigma=0.05, n_holes=3, hole_size=1
    # v3: n_genes=50, n_samples=1000, sigma=0.05, n_holes=3, hole_size=1

    metrics = []
    for seed in range(5):
        for hole_density in [0.0, 0.025, 0.05, 0.075, 0.1]:
            adata_sim = simulation_correlated_1(
                n_samples=1000,
                n_genes=50,
                seed=seed,
                sigma=0.05,
                branch_prob=0.7,
                k_clusters=20,
                hole_size=1,
                n_holes=3,
                hole_density=hole_density,
            )

            sc.pl.embedding(
                adata_sim,
                basis="latent",
                color=["latent_t", "branch_id", *adata_sim.uns["latent_z_names"]],
                ncols=2,
                show=False,
            )
            plt.savefig(f"{folder}/latent_{hole_density}_seed_{seed}.pdf", bbox_inches="tight")
            #
            latent_spaces = run_methods(adata_sim)
            #
            latent_spaces = [x for x in latent_spaces if adata_sim.obsm[x].shape[1] == 2]
            metrics_local = compute_plot_metrics(
                adata_sim,
                latent_spaces,
                ref_key="latent",
                color=["latent_t", "branch_id"],
                file_name=f"{folder}/density_{hole_density}_seed_{seed}",
            )["Global distortion"]
            metrics_local = metrics_local.melt(var_name="method", value_name="global_distortion")
            metrics_local["hole_density"] = hole_density
            metrics_local["seed"] = seed
            metrics.append(metrics_local)

            adata_sim.uns["distortion"] = metrics_local
            # save adata
            adata_sim.write(f"{folder}/adata_density_{hole_density}_seed_{seed}.h5ad")

    metrics = pd.concat(metrics)

    metrics.to_csv(f"{folder}/metrics-range-of-densities.csv", index=False)
