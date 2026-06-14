import numpy as np
import torch
import matplotlib.pyplot as plt

from methods import MDS, Contrastive
from distances import distance_matrix, euclidean_distance, poincare_distance, hamming_distance
from optimizers import PoincareRiemannianAdamOptim, PoincareOptim
from SyntheticTreeDataset import hasone, get_tree_data


# -----------------------------
# Reproducibility
# -----------------------------
torch.manual_seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Build dataset
# -----------------------------
depth = 5
X_np, _ = get_tree_data(depth=depth)   # shape (n, n)
X = torch.tensor(X_np, dtype=torch.float32, device=device)

# Pairwise distances in the original space
#data_dist_matrix = distance_matrix(X, euclidean_distance).detach()
data_dist_matrix = distance_matrix(X, hamming_distance).detach()

# # Pairwise distances (train.py: distance_method='hamming')
# data_dist_matrix = distance_matrix(X, hamming_distance).detach().cpu().numpy()
# data_binary_dist_matrix = (data_dist_matrix <= 1.01).astype(int)
# data_binary_dist_matrix = torch.tensor(data_binary_dist_matrix)
# data_dist_matrix = torch.tensor(data_dist_matrix, dtype=torch.float32, device=device)

#temperature = 0.01  # train.py default
normalize = True   # train.py default; required for hypergaussian + Poincaré

n = X.shape[0]
idx = torch.arange(n, device=device)


# -----------------------------
# Create 2D hyperbolic embedder
# -----------------------------
embedder = MDS(
    data_size=n,
    latent_dim=2,
    latent_dist_fun=poincare_distance,
    distr="hypergaussian",
).to(device)
# embedder = Contrastive(
#     data_size=n,
#     latent_dim=2,
#     latent_dist_fun=poincare_distance,
#     distr="hypergaussian",
# ).to(device)

# Optional: start safely inside the disk
with torch.no_grad():
    embedder.embeddings.data *= 0.1

# optimizer = PoincareRiemannianAdamOptim(
#     embedder=embedder,
#     lr=0.03,
#     beta1=0.9,
#     beta2=0.999,
#     epsilon=1e-8,
# )

optimizer = PoincareOptim(
    embedder=embedder,
    lr=0.03,
)

# -----------------------------
# Training loop
# -----------------------------
num_epochs = 7000
loss_history = []

for epoch in range(num_epochs):
    optimizer.zero_grad()

    if normalize:
        embedder.normalize()
    else:
        embedder.normalize(normalization=False)

    loss = embedder.loss_fun(data_dist_matrix, idx)
    # loss = embedder.loss_fun(
    #     data_dist_matrix, idx, data_binary_dist_matrix, temperature
    # )
    loss.backward()
    optimizer.step(idx)

    loss_history.append(loss.item())

    if epoch % 100 == 0:
        with torch.no_grad():
            norms = torch.linalg.vector_norm(embedder.embeddings, dim=-1)
            print(
                f"epoch={epoch:4d} | loss={loss.item():.6f} | "
                f"max_norm={norms.max().item():.6f}"
            )


# -----------------------------
# Visualization
# -----------------------------
emb = embedder.embeddings.detach().cpu().numpy()

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Loss curve
axes[0].plot(loss_history)
axes[0].set_title("Training loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")

# Poincare disk embedding
theta = np.linspace(0, 2 * np.pi, 400)
axes[1].plot(np.cos(theta), np.sin(theta), "k--", linewidth=1)  # unit circle
axes[1].scatter(emb[:, 0], emb[:, 1], s=30)

# annotate node ids
for i, (x, y) in enumerate(emb):
    axes[1].text(x, y, str(i + 1), fontsize=8)

axes[1].set_title("2D Poincaré embedding")
axes[1].set_aspect("equal", "box")
axes[1].set_xlim(-1.05, 1.05)
axes[1].set_ylim(-1.05, 1.05)

plt.tight_layout()
plt.savefig("figs/synthetic_tree_PoincareOptim_hamming_mds.svg")  
#plt.savefig("figs/synthetic_tree_PoincareRiemannianAdamOptim_hamming_mds.svg")  

plt.show()

