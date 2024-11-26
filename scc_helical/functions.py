import numpy as np
from scipy.sparse import csr_matrix

# This function takes as inputs anndata (the data which will be subjected to noise), the type of noise and a scaling factor.
# It returns the data with noise.
def add_noise(anndata, noise_distribution='logNormal', scaling_factor=1):
    # Compute mean for each gene
    gene_means = np.array(anndata.X.mean(axis=0)).flatten()
    gene_means = np.where(gene_means == 0, 1e-6, gene_means)  # change zeros by 1e-6 to avoid division errors

    # Compute std for each gene
    gene_stds = np.sqrt((anndata.X.power(2).mean(axis=0) - np.power(gene_means, 2))).flatten()  # Sparse std computation

    # Add noise gaussian noise
    if noise_distribution == 'normal':
        noise = np.abs(np.random.normal(loc=1, scale=scaling_factor * gene_stds,
                                        size=anndata.X.shape))  # abs used to avoid negative values

    # Add logNormal noise (avoids negative numbers)
    elif noise_distribution == 'logNormal':
        gene_cv = gene_stds / gene_means  # Coefficient of variation used in logNormal
        noise = np.random.lognormal(mean=0, sigma=scaling_factor * gene_cv, size=anndata.X.shape)

        # both distributions have a scaling factor to introduce more or less noise.

    # Apply noise to the sparse matrix
    anndata.X = anndata.X.multiply(noise)  # Element-wise multiplication for sparse matrices
    anndata.X = csr_matrix(np.round(anndata.X).astype(int))  # Ensure the result is a sparse integer matrix
    assert not np.any(anndata.X.toarray() < 0)  # test to check no negative numbers generated

    return anndata




# This function takes as inputs anndata (dataset which will be subjected to gene perturbations) and the specified perturbation
# Both upregulation, downregulation, deletino and gene saturation can be simulated
# Multiple gene perturbations can be simulated
def add_perturbations(anndata_subset, anndata, gene_perturbations):
    anndata_subset_perturbed = anndata_subset.copy()
    for gene, perturbation in gene_perturbations.items():
        if perturbation == 'down':
            min_value = anndata[:, gene].X.min()
            anndata_subset_perturbed[:, gene].X = int(min_value)

        if perturbation == 'up':
            max_value = anndata[:, gene].X.max()
            anndata_subset_perturbed[:, gene].X = int(max_value)

        if perturbation == 'deletion':
            anndata_subset_perturbed[:, gene].X = 0

        if perturbation == 'saturation':
            anndata_subset_perturbed[:, gene].X = 1000
    return anndata_subset_perturbed







