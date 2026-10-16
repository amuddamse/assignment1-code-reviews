"""CSE 242 Assignment 1, Part 3 (Q2-Q4): PCA from scratch with NumPy.

Run:  python3 Assign1_variance_numpy_matplotlib.py
Requires: numpy, matplotlib
Output: prints the PCA results and saves the figure to pca_plots.png.

Background (Q1): for a unit vector u and random vector X with mean mu and
covariance Sigma = E[(X - mu)(X - mu)^T], the projection u^T X has
Var(u^T X) = E[(u^T (X - mu))^2] = u^T E[(X - mu)(X - mu)^T] u = u^T Sigma u.
The PCA steps below are the sample version of that result.
"""
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# =====================================================================
# Q2: Generate Synthetic 2D Correlated Gaussian Dataset
# =====================================================================
n_samples = 300
mean = np.array([0, 0])
# Define a highly correlated covariance structure
covariance = np.array([
    [3.0, 2.5],
    [2.5, 3.0]
])

# Generate dataset X of shape (n_samples, n_features)
X = np.random.multivariate_normal(mean, covariance, size=n_samples)

# Q2 discussion: correlation structure.
# Both features have variance 3 and covariance 2.5, so the true correlation is
# rho = 2.5 / sqrt(3 * 3) = 0.833, a strong positive correlation. In the
# scatter plot (left panel) the points form a narrow ellipse tilted along the
# line x2 = x1: when feature 1 is large, feature 2 tends to be large too.
# The sample correlation printed below is close to the true value.
true_corr = covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1])
sample_corr = np.corrcoef(X, rowvar=False)[0, 1]

# =====================================================================
# Q3: Implement PCA from Scratch using NumPy
# =====================================================================
# Step 1: Center your dataset (mean subtraction)
# The Q1 derivation is written in terms of X - mu. Without centering, the
# matrix below would include mu mu^T and would not be the covariance.
X_mean = np.mean(X, axis=0)
X_centered = X - X_mean

# Step 2: Compute the covariance matrix
# Sample estimate of Sigma = E[(X - mu)(X - mu)^T]:
# Sigma_hat = X_centered^T X_centered / (n - 1)   (np.cov uses n - 1)
cov_matrix = np.cov(X_centered, rowvar=False)

# Step 3: Find eigenvalues and eigenvectors
# PCA picks the unit vector u that maximizes Var(u^T X) = u^T Sigma u.
# With the constraint u^T u = 1, the Lagrangian u^T Sigma u - lambda(u^T u - 1)
# gives Sigma u = lambda u, so the best directions are eigenvectors of Sigma
# and the variance along each one equals its eigenvalue lambda.
# eigh is used because the covariance matrix is symmetric.
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

# Step 4: Select the principal components (sort, highest eigenvalues first)
# PC1 = eigenvector with the largest eigenvalue = direction of maximum variance.
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# Eigenvector signs are arbitrary (u and -u span the same direction);
# flip each so its largest entry is positive, for reproducible output.
signs = np.sign(eigenvectors[np.argmax(np.abs(eigenvectors), axis=0), [0, 1]])
eigenvectors = eigenvectors * signs

# Projections of the data onto the components (the u^T (x - mu) from Q1)
scores = X_centered @ eigenvectors

# =====================================================================
# Q4: Proportion of Total Variance Explained
# =====================================================================
# Total variance = trace(Sigma) = sum of eigenvalues, so each eigenvalue
# divided by the sum is the fraction of variance that component captures.
total_variance = np.sum(eigenvalues)
prop_variance_explained = eigenvalues / total_variance

# Q4 discussion: results and the dimensionality-reduction trade-off.
# PC1 explains about 92% of the total variance and PC2 about 8% (printed
# below). PC1 points along the diagonal (about [0.71, 0.71]), the direction
# the strong correlation stretches the data. Keeping only PC1 reduces the data
# from 2D to 1D while keeping ~92% of the variance; the ~8% lost is the
# spread perpendicular to the diagonal. The trade-off: fewer dimensions give
# simpler, cheaper models and less noise, but each dropped component throws
# away its share of variance. A common rule is to keep enough components to
# reach a target such as 90-95% cumulative variance. If the features were
# uncorrelated with equal variance, each PC would explain 50% and no dimension
# could be dropped without losing half the information.

print(f"Sample mean: {X_mean}")
print(f"Covariance matrix:\n{cov_matrix}")
print(f"Correlation: sample = {sample_corr:.3f}, true = {true_corr:.3f}")
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors (columns = PC1, PC2):\n{eigenvectors}")
print(f"Proportion of Variance Explained: {prop_variance_explained}")
print(f"Check: variance of PC scores = {scores.var(axis=0, ddof=1)} (equals eigenvalues)")

# =====================================================================
# Plots (Q2 scatter + Q3 principal directions, Q4 explained variance)
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: data with principal component directions drawn from the mean.
# Arrow length = 2 * sqrt(eigenvalue), i.e. two standard deviations.
ax1.scatter(X[:, 0], X[:, 1], s=10, alpha=0.5, label=f"samples (n = {n_samples})")
colors = ["tab:red", "tab:green"]
for j in range(2):
    vec = eigenvectors[:, j] * 2 * np.sqrt(eigenvalues[j])
    ax1.annotate("", xy=X_mean + vec, xytext=X_mean,
                 arrowprops=dict(arrowstyle="->", color=colors[j], lw=2.5))
    ax1.plot([], [], color=colors[j], lw=2.5,
             label=f"PC{j + 1} ({prop_variance_explained[j]:.1%} of variance)")
ax1.plot(*X_mean, "kx", ms=10, mew=2, label="mean")
ax1.set_aspect("equal")
ax1.set_xlabel("Feature 1 ($x_1$)")
ax1.set_ylabel("Feature 2 ($x_2$)")
ax1.set_title(f"Correlated Gaussian data (r = {sample_corr:.2f})\n"
              "with principal directions (arrows = 2 std dev)")
ax1.grid(alpha=0.3)
ax1.legend(loc="lower right")

# Right: explained variance per component and cumulative.
labels = ["PC1", "PC2"]
bars = ax2.bar(labels, prop_variance_explained, color=colors, alpha=0.8,
               label="individual")
for bar, frac in zip(bars, prop_variance_explained):
    ax2.text(bar.get_x() + bar.get_width() / 2, frac + 0.02, f"{frac:.1%}",
             ha="center")
ax2.plot(labels, np.cumsum(prop_variance_explained), "ko-", label="cumulative")
ax2.set_ylim(0, 1.1)
ax2.set_xlabel("Principal component")
ax2.set_ylabel("Fraction of total variance explained")
ax2.set_title("Explained variance")
ax2.grid(alpha=0.3, axis="y")
ax2.legend(loc="center right")

fig.tight_layout()
fig.savefig("pca_plots.png", dpi=120)
plt.show()
