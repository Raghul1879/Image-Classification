import numpy as np

class GaussianMixtureModel:
    def __init__(self, n_components, n_iterations=100, tol=1e-4):
        self.n_components = n_components
        self.n_iterations = n_iterations
        self.tol = tol

    def fit(self, X):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.full(shape=self.n_components, fill_value=1 / self.n_components)
        self.means = np.random.rand(self.n_components, n_features)
        self.covariances = np.array([np.eye(n_features)] * self.n_components)

        # E-M algorithm
        for _ in range(self.n_iterations):
            # E-step: Compute responsibilities
            responsibilities = self._compute_responsibilities(X)

            # M-step: Update parameters
            self._update_parameters(X, responsibilities)

    def _compute_responsibilities(self, X):
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        for i in range(self.n_components):
            # Compute the probability of each sample under the i-th Gaussian
            gaussian_pdf = self._multivariate_normal_pdf(X, self.means[i], self.covariances[i])
            responsibilities[:, i] = self.weights[i] * gaussian_pdf

        # Normalize responsibilities
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
        return responsibilities

    def _update_parameters(self, X, responsibilities):
        n_samples = X.shape[0]

        # Update weights
        self.weights = np.mean(responsibilities, axis=0)

        # Update means
        weighted_sum = np.dot(responsibilities.T, X)
        self.means = weighted_sum / np.sum(responsibilities, axis=0, keepdims=True)

        # Update covariances
        for i in range(self.n_components):
            diff = X - self.means[i]
            weighted_diff = responsibilities[:, i, None] * diff
            cov_matrix = np.dot(diff.T, weighted_diff)
            self.covariances[i] = cov_matrix / np.sum(responsibilities[:, i])

    def _multivariate_normal_pdf(self, X, mean, covariance):
        n_features = X.shape[1]
        det = np.linalg.det(covariance)
        norm_const = 1.0 / (np.power((2 * np.pi), n_features / 2) * np.sqrt(det))
        inv_covariance = np.linalg.inv(covariance)
        exp_val = np.exp(-0.5 * np.einsum('ij, ij -> i', X - mean, np.dot(X - mean, inv_covariance.T)))
        return norm_const * exp_val
# Create an instance of GaussianMixtureModel
gmm = GaussianMixtureModel(n_components=3)

# Generate some random data
np.random.seed(0)
n_samples = 1000
n_features = 2
X = np.random.randn(n_samples, n_features)

# Fit the model to the data
gmm.fit(X)

# Retrieve the learned parameters
weights = gmm.weights
means = gmm.means
covariances = gmm.covariances
