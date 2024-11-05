import pandas as pd
import scanpy as sc
import os

from matplotlib import pyplot as plt
import seaborn as sns

import matplotlib as mpl

mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42


def plot_embedding(adata, latent_space, figsize, folder, file_suffix="", title=None):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    adata.obs["branch_id"] = adata.obs["branch_id"].astype(str)
    adata.obs["branch_id"].replace(
        {
            "0": "Origin",
            "1": "Branch 1",
            "-1": "Branch 2",
        },
        inplace=True,
    )
    if latent_space == "decipher_decipher_v":
        import decipher as dc

        dc.tl.decipher_rotate_space(
            adata,
            v1_col="latent_t",
            obsm_key_v="decipher_decipher_v",
            auto_flip_decipher_z=False,
        )

    sc.pl.embedding(
        adata,
        basis=latent_space,
        color="latent_t",
        wspace=0.5,
        hspace=0.5,
        ax=ax,
        size=30,
        show=False,
    )
    # adjust colorbar ticks to just 0 and 1
    cbar = fig.get_axes()[1]
    cbar.set_yticks([0, 1])
    sns.despine()
    # remove x and y axis and their ticks/labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    titles = dict(
        latent="Ground truth",
        X_umap="UMAP",
        X_fd="Force-directed",
        X_scVI_umap="scVI UMAP",
        decipher_decipher_v="Decipher $v$",
    )
    # add margin below title
    if title is None:
        title = titles[latent_space]
    ax.set_title(title, fontsize=12, pad=5)

    save_path = f"{folder}/{latent_space}{file_suffix}.pdf"
    plt.savefig(save_path, bbox_inches="tight")
    plt.show()


folder = "figures/paper"
os.makedirs(folder, exist_ok=True)

# just pick one seed (here the last run)
adata = sc.read("figures/adata_density_0.0_seed_4.h5ad")

latent_spaces = ["latent", "X_umap", "X_fd", "X_scVI_umap", "decipher_decipher_v"]

figsize = [2.5, 1.5]
for latent_space in latent_spaces:
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_embedding(adata, latent_space, figsize, folder)

figsize = [2.5, 2]
poster_folder = "figures/poster"
os.makedirs(poster_folder, exist_ok=True)
for latent_space in latent_spaces:
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plot_embedding(adata, latent_space, figsize, poster_folder)

# Now look at higher densities

for density in [0, 0.05, 0.1]:
    adata = sc.read(f"figures/adata_density_{density:.1f}_seed_4.h5ad")
    latent_space = "latent"
    plot_embedding(
        adata,
        latent_space,
        figsize,
        folder,
        f"_density_{density}",
        title=f"Ground truth - density {int(density*100)}%",
    )

# %%
# Plot global distortion for each seed (not shown in the paper)

global_distortion = []
for seed in range(5):
    adata = sc.read(f"figures/adata_density_0.0_seed_{seed}.h5ad")
    global_distortion.append(adata.uns["distortion"])

global_distortion = pd.concat(global_distortion, axis=0).reset_index(drop=True)
global_distortion = global_distortion.replace(
    {
        "decipher_decipher_v": "Decipher $v$",
        "X_scVI_umap": "scVI (UMAP)",
        "X_scVI_linear_umap": "linear scVI\n(UMAP)",
        "X_default_umap": "UMAP",
        "X_fd": "Force directed",
        "X_phate_umap": "PHATE (UMAP)",
    }
)

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
sns.boxplot(
    data=global_distortion,
    y="method",
    x="global_distortion",
    hue="method",
    showfliers=False,
    orient="h",
    ax=ax,
)

plt.ylabel("")
plt.yticks(fontsize=12)
plt.xlabel("  Global preservation   \n(density=0%)     ", fontsize=14, loc="right")
plt.xticks([0, 1], ["0", "1"])
sns.despine()

plt.tight_layout()
plt.savefig(f"{folder}/global_distortion.pdf", bbox_inches="tight")
plt.show()


# %%
# Plot the metrics for a range of densities

distortions_range_densities = pd.read_csv("figures/metrics-range-of-densities.csv")
distortions_range_densities = distortions_range_densities.replace(
    {
        "decipher_decipher_v": "Decipher $v$",
        "X_scVI_umap": "scVI (UMAP)",
        "X_scVI_linear_umap": "linear scVI\n(UMAP)",
        "X_default_umap": "UMAP",
        "X_fd": "Force directed",
        "X_phate_umap": "PHATE (UMAP)",
    }
)

fig, ax = plt.subplots(1, 1, figsize=(6, 3))
sns.lineplot(
    data=distortions_range_densities,
    x="hole_density",
    y="global_distortion",
    hue="method",
    style="method",
    markers=True,
    dashes=False,
    err_style="bars",
    err_kws={"capsize": 2},
)
plt.ylabel("Global preservation", fontsize=12)
plt.xlabel("Density of cells in cell-state transitions.", fontsize=12)
plt.xticks([0, 0.025, 0.05, 0.075, 0.1], ["0%", "2.5%", "5%", "7.5%", "10%"])
plt.legend(title="Method", bbox_to_anchor=(1, 1))
# set Decipher in bold in the legend
for t in ax.get_legend().texts:
    if "decipher" in t.get_text().lower():
        t.set_fontweight("bold")
plt.tight_layout()
plt.savefig(f"{folder}/global_distortion_range_of_densities.pdf", bbox_inches="tight")
plt.show()
