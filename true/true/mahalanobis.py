import numpy as np
import torch
from tqdm import tqdm

DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [10**exp for exp in range(-15, 0, 1)]


def _compute_centroid(train_features, train_labels, label, zero_vector=None):
    label_features = train_features[train_labels == label]
    if len(label_features):
        return label_features.mean(dim=0), False
    return zero_vector, True


def compute_centroids(train_features, train_labels, num_labels=None):
    labels = (
        np.sort(np.unique(train_labels))
        if num_labels is None
        else np.arange(num_labels)
    )
    device = train_features.device
    centroids = torch.empty(
        len(labels), train_features.shape[1], dtype=torch.float32, device=device
    )
    centroids_mask = torch.empty(len(labels), dtype=torch.bool, device="cpu")
    zero_vector = torch.zeros(train_features.shape[1], device=device)

    for i, label in enumerate(labels):
        centroid, centroid_mask = _compute_centroid(
            train_features, train_labels, label, zero_vector
        )
        centroids[i].copy_(centroid, non_blocking=True)
        centroids_mask[i] = centroid_mask

    return centroids, centroids_mask


def compute_inv_covariance(centroids, train_features, train_labels, jitters=None):
    if jitters is None:
        jitters = JITTERS
    jitter = 0
    jitter_eps = None

    cov = torch.zeros(
        centroids.shape[1], centroids.shape[1], device=centroids.device
    ).float()
    for c, mu_c in tqdm(enumerate(centroids)):
        for x in train_features[train_labels == c]:
            d = (x - mu_c).unsqueeze(1)
            cov += d @ d.T
    cov_scaled = cov / (train_features.shape[0] - 1)

    for i, jitter_eps in enumerate(jitters):
        jitter = jitter_eps * torch.eye(
            cov_scaled.shape[1],
            device=cov_scaled.device,
        )
        cov_scaled_update = cov_scaled + jitter
        eigenvalues = torch.symeig(cov_scaled_update).eigenvalues
        if (eigenvalues >= 0).all():
            break
    cov_scaled = cov_scaled + jitter
    cov_inv = torch.inverse(cov_scaled.to(torch.float64)).float()
    return cov_inv, jitter_eps


def mahalanobis_distance(train_features, train_labels, eval_features):
    centroids, centroids_mask = compute_centroids(train_features, train_labels)
    sigma = compute_inv_covariance(centroids, train_features, train_labels)
    diff = eval_features[:, None, :] - centroids[None, :, :]
    sigma_inv = np.linalg.pinv(sigma)

    dists = np.matmul(np.matmul(diff, sigma_inv), diff.transpose(0, 2, 1))
    dists = np.asarray([np.diag(dist) for dist in dists])
    return np.min(dists, axis=1)


def mahalanobis_distance_with_known_centroids_sigma_inv(
    centroids, centroids_mask, sigma_inv, eval_features
):
    diff = eval_features.unsqueeze(1) - centroids.unsqueeze(
        0
    )  # bs (b), num_labels (c / s), dim (d / a)
    dists = torch.sqrt(torch.einsum("bcd,da,bsa->bcs", diff, sigma_inv, diff))
    device = dists.device
    dists = torch.stack([torch.diag(dist).cpu() for dist in dists], dim=0)
    if centroids_mask is not None:
        dists = dists.masked_fill_(centroids_mask, float("inf")).to(device)
    return dists  # np.min(dists, axis=1)


def centered_cov(x):
    return x.T @ x / (len(x) - 1)


def compute_density(log_logits, label_probs):
    return torch.logsumexp(log_logits, dim=1)


def get_gmm_log_probs(gaussians_model, embeddings):
    return gaussians_model.log_prob(embeddings[:, None, :])


def gmm_fit(embeddings, labels):
    num_classes = len(set(labels))
    with torch.no_grad():
        centroids = torch.stack(
            [torch.mean(embeddings[labels == c], dim=0) for c in range(num_classes)]
        )
        cov_matrix = torch.stack(
            [
                centered_cov(embeddings[labels == c] - centroids[c])
                for c in range(num_classes)
            ]
        )

    with torch.no_grad():
        gmm = None
        for jitter_eps in JITTERS:
            try:
                jitter = jitter_eps * torch.eye(
                    cov_matrix.shape[1],
                    device=cov_matrix.device,
                ).unsqueeze(0)
                gmm = torch.distributions.MultivariateNormal(
                    loc=centroids,
                    covariance_matrix=(cov_matrix + jitter),
                )
                break
            except RuntimeError as e:
                if "cholesky" in str(e):
                    continue
            except ValueError as e:
                if "The parameter covariance_matrix has invalid values" in str(e):
                    continue
        if gmm is None:
            cov_matrix_eye = torch.eye(
                cov_matrix.shape[1],
                device=cov_matrix.device,
            ).unsqueeze(0)
            gmm = torch.distributions.MultivariateNormal(
                loc=centroids,
                covariance_matrix=cov_matrix_eye,
            )

    return gmm, jitter_eps
