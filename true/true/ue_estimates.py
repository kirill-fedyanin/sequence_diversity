import logging
from copy import deepcopy
from typing import Union, Tuple

from scipy.stats import entropy
import numpy as np
import torch
from torch import nn
import torch.optim as optim
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

from true.generate import generate
from true.mahalanobis import (
    compute_inv_covariance,
    mahalanobis_distance_with_known_centroids_sigma_inv,
    compute_density,
    get_gmm_log_probs,
    gmm_fit,
)
from true.seq2seq_metrics import (
    calculate_top_1_acc
)
from true.ue_estimates_utils import (
    calculate_bleuvar_scores,
    calculate_bleuvar_with_deterministic_scores,
    calculate_metricvar_scores,
    get_token_level_data,
)
from true.ue_mlp import Net
from true.mc_utils import get_mc_output, get_mc_forward_output
from true.data.mlp_dataset import Dataset

log = logging.getLogger()
DEVICE = torch.cuda.is_available()


class StrategyManager:
    strategy: str

    def __call__(self, strategy=None, **strategy_kwargs):
        strategy = strategy or self.strategy
        return getattr(self, strategy)(**strategy_kwargs)

    @staticmethod
    def nsp(inference_output=None, **generate_kwargs):
        log.info("Calculating NSP scores...")
        if inference_output is None:
            inference_output = generate(**generate_kwargs)
        return -inference_output["scores_unbiased"]

    @staticmethod
    def nsp_biased(inference_output=None, **generate_kwargs):
        log.info("Calculating NSP scores...")
        if inference_output is None:
            inference_output = generate(**generate_kwargs)
        return -inference_output["scores"]

    @staticmethod
    def msp(inference_output=None, **generate_kwargs):
        log.info("Calculating NSP scores...")
        if inference_output is None:
            inference_output = generate(**generate_kwargs)
        return -inference_output["max_scores"]

    @staticmethod
    def entropy(inference_output=None, **generate_kwargs):
        log.info("Calculating ENTROPY scores...")
        if inference_output is None:
            inference_output = generate(**generate_kwargs)
        return inference_output["entropy"]

    @staticmethod
    def entropy_top5(inference_output=None, **generate_kwargs):
        log.info("Calculating ENTROPY scores...")
        if inference_output is None:
            inference_output = generate(**generate_kwargs)
        return inference_output["entropy_top5"]

    @staticmethod
    def entropy_top10(inference_output=None, **generate_kwargs):
        log.info("Calculating ENTROPY scores...")
        if inference_output is None:
            inference_output = generate(**generate_kwargs)
        return inference_output["entropy_top10"]

    @staticmethod
    def entropy_top15(inference_output=None, **generate_kwargs):
        log.info("Calculating ENTROPY scores...")
        if inference_output is None:
            inference_output = generate(**generate_kwargs)
        return inference_output["entropy_top15"]

    @staticmethod
    def entropy_s(inference_output=None, **generate_kwargs):
        log.info("Calculating ENTROPY-S scores...")
        if inference_output is None:
            inference_output = generate(**generate_kwargs)
        ue = inference_output['token_level_scores']['entropy_s']
        return ue

    @staticmethod
    def entropy_s_u(inference_output=None, **generate_kwargs):
        log.info("Calculating ENTROPY-S-U scores...")
        if inference_output is None:
            inference_output = generate(**generate_kwargs)
        ue = inference_output['token_level_scores']['entropy_s_u']
        return ue

    @staticmethod
    def usp(inference_output=None, **generate_kwargs):
        log.info("Calculating USP scores...")
        if inference_output is None:
            inference_output = generate(**generate_kwargs, length_penalty=0.0)
        return -inference_output["scores"]

    @staticmethod
    def ensp(inference_output=None, **mc_output_kwargs):
        """
        Expected Normalized Sequence Probability
        """
        log.info("Calculating ENSP scores...")
        if inference_output is None:
            inference_output = get_mc_output(**mc_output_kwargs)
        scores = inference_output["scores"]
        uncertainty_estimates = -np.mean(scores, axis=1)
        return uncertainty_estimates

    @staticmethod
    def esp(inference_output=None, **mc_output_kwargs):
        """
        Expected Sequence Probability
        """
        log.info("Calculating ENSP scores...")
        if inference_output is None:
            inference_output = get_mc_output(**mc_output_kwargs, length_penalty=0.0)
        scores = inference_output["scores"]
        uncertainty_estimates = -np.mean(scores, axis=1)
        return uncertainty_estimates

    @staticmethod
    def mnsp(inference_output=None, **mc_output_kwargs):
        """
        Median Normalized Sequence Probability
        :return:
        """
        log.info("Calculating MNSP scores...")
        if inference_output is None:
            inference_output = get_mc_output(**mc_output_kwargs)
        scores = inference_output["scores"]
        uncertainty_estimates = -np.median(scores, axis=1)
        return uncertainty_estimates

    @staticmethod
    def ensv(inference_output=None, **mc_output_kwargs):
        """
        Expected Normalized Sequence Variance
        """
        log.info("Calculating ENSV scores...")
        if inference_output is None:
            inference_output = get_mc_output(**mc_output_kwargs)
        scores = inference_output["scores"]
        uncertainty_estimates = np.var(scores, axis=1)
        return uncertainty_estimates

    @staticmethod
    def oracle(inference_output=None, **mc_output_kwargs):
        """
        Oracle strategy, unavailable in real life
        """
        log.info("Calculating Oracle scores...")
        if inference_output is None:
            labels = mc_output_kwargs.pop("labels")
            inference_output = get_mc_forward_output(**mc_output_kwargs)
        losses = inference_output["losses"]
        uncertainty_estimates = np.var(losses, axis=1)
        return uncertainty_estimates

    @staticmethod
    def edslv(inference_output=None, **mc_output_kwargs):
        """
        Expected Deterministic Sequence Loss Variance
        """
        log.info("Calculating EDSLV scores...")
        if inference_output is None:
            inference_output = get_mc_forward_output(**mc_output_kwargs)
        losses = inference_output["losses"]
        uncertainty_estimates = np.var(losses, axis=1)
        return uncertainty_estimates

    @staticmethod
    def edssv(inference_output=None, **mc_output_kwargs):
        """
        Expected Deterministic Sequence Score Variance
        """
        log.info("Calculating EDSSV scores...")
        if inference_output is None:
            inference_output = get_mc_forward_output(**mc_output_kwargs)
        scores = inference_output["scores"]
        uncertainty_estimates = np.var(-scores, axis=1)
        return uncertainty_estimates

    @staticmethod
    def edsl(inference_output=None, **mc_output_kwargs):
        """
        Expected Deterministic Sequence Loss
        """
        log.info("Calculating EDSL scores...")
        if inference_output is None:
            inference_output = get_mc_forward_output(**mc_output_kwargs)
        losses = inference_output["losses"]
        uncertainty_estimates = np.mean(losses, axis=1)
        return uncertainty_estimates

    @staticmethod
    def edss(inference_output=None, **mc_output_kwargs):
        """
        Expected Deterministic Sequence Score
        """
        log.info("Calculating EDSS scores...")
        if inference_output is None:
            inference_output = get_mc_forward_output(**mc_output_kwargs)
        scores = inference_output["scores"]
        uncertainty_estimates = np.mean(-scores, axis=1)
        return uncertainty_estimates

    @staticmethod
    def bald(inference_output=None, **mc_output_kwargs):
        """
        Avg loss - avg probas loss
        """
        log.info("Calculating BALD scores...")
        if inference_output is None:
            inference_output = get_mc_forward_output(**mc_output_kwargs)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        # for i in mc_runs: \sum_j log p_ij / n ;
        max_num_labels = np.max(inference_output["num_labels"])
        losses = inference_output["losses"]
        # for i in mc_runs: (\prod_j p_ij) ^ (1/n)
        avg_loss = torch.Tensor(np.mean(losses, axis=1)).to(device)

        log_probas = inference_output["log_probas"][:, :, :max_num_labels]
        labels = torch.Tensor(inference_output["labels"][:, :max_num_labels]).to(device)
        avg_log_probas = torch.Tensor(np.mean(log_probas, axis=1)).to(device)
        avg_probas_loss = (avg_log_probas * (labels != -100).int()).mean(-1)

        uncertainty_estimates = avg_loss - avg_probas_loss
        return uncertainty_estimates.cpu().detach().numpy()

    @staticmethod
    def avgloss(inference_output=None, **mc_output_kwargs):
        """
        Avg loss + avg probas loss
        """
        log.info("Calculating BALD-a scores...")
        if inference_output is None:
            inference_output = get_mc_forward_output(**mc_output_kwargs)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        # for i in mc_runs: \sum_j log p_ij / n ;
        max_num_labels = np.max(inference_output["num_labels"])
        losses = inference_output["losses"]
        # for i in mc_runs: (\prod_j p_ij) ^ (1/n)
        avg_loss = torch.Tensor(np.mean(losses, axis=1)).to(device)

        log_probas = inference_output["log_probas"][:, :, :max_num_labels]
        labels = torch.Tensor(inference_output["labels"][:, :max_num_labels]).to(device)
        avg_log_probas = torch.Tensor(np.mean(log_probas, axis=1)).to(device)
        avg_probas_loss = (avg_log_probas * (labels != -100).int()).mean(-1)

        uncertainty_estimates = avg_probas_loss + avg_loss
        return uncertainty_estimates.cpu().detach().numpy()

    @staticmethod
    def eloss(inference_output=None, **mc_output_kwargs):
        """
        Avg loss
        """
        log.info("Calculating BALD-a scores...")
        if inference_output is None:
            inference_output = get_mc_forward_output(**mc_output_kwargs)
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        # for i in mc_runs: \sum_j log p_ij / n ;
        losses = inference_output["losses"]
        # for i in mc_runs: (\prod_j p_ij) ^ (1/n)
        avg_loss = torch.Tensor(np.mean(losses, axis=1)).to(device)

        uncertainty_estimates = avg_loss
        return uncertainty_estimates.cpu().detach().numpy()

    @staticmethod
    def bleuvar(inference_output=None, **mc_output_kwargs):
        """
        BLEU Variance
        """
        log.info("Calculating BLEUVAR scores...")
        if inference_output is None:
            inference_output = get_mc_output(**mc_output_kwargs)
        hypotheses = inference_output["hypotheses"]
        uncertainty_estimates = calculate_bleuvar_scores(hypotheses=hypotheses)[0]
        return uncertainty_estimates

    @staticmethod
    def metricvar(inference_output=None, metric="rouge1", **mc_output_kwargs):
        """
        Metric Variance for ROUGE / sacrebleu
        """
        log.info(f"Calculating {metric.upper()}-VAR scores...")
        if inference_output is None:
            inference_output = get_mc_output(**mc_output_kwargs)
        hypotheses = inference_output["hypotheses"]
        uncertainty_estimates = calculate_metricvar_scores(
            metric=metric, hypotheses=hypotheses
        )[0]
        return uncertainty_estimates

    @staticmethod
    def bleuvar_deterministic(
        stochastic_output=None,
        deterministic_output=None,
        symmetric: bool = True,
        **mc_output_kwargs,
    ):
        """
        BLEU Variance with deterministic hypothesis instead of pair-wise stochastic
        """
        log.info("Calculating BLEUVARDET scores...")
        if stochastic_output is None:
            stochastic_output = get_mc_output(**mc_output_kwargs)
        if deterministic_output is None:
            kwargs = {
                key: val
                for key, val in mc_output_kwargs.items()
                if key
                in [
                    "model",
                    "data",
                    "tokenizer",
                    "generation_max_length",
                    "batch_size",
                    "data_config",
                    "is_tokenized",
                    "to_numpy",
                ]
            }
            deterministic_output = generate(**kwargs)
        stochastic_hypotheses = stochastic_output["hypotheses"]
        deterministic_hypotheses = deterministic_output["hypotheses"]
        uncertainty_estimates = calculate_bleuvar_with_deterministic_scores(
            stochastic_hypotheses=stochastic_hypotheses,
            deterministic_hypotheses=deterministic_hypotheses,
            symmetric=symmetric,
        )
        return uncertainty_estimates

    @staticmethod
    def mahalanobis_distance(
        train_embeddings: torch.Tensor, test_embeddings: torch.Tensor, ue_dict=None
    ):
        log.info("Calculating MD scores...")
        train_labels = np.zeros(train_embeddings.shape[0])
        centroid = train_embeddings.mean(dim=0)
        sigma_inv, _ = compute_inv_covariance(
            centroid.unsqueeze(0), train_embeddings, train_labels
        )
        dists = mahalanobis_distance_with_known_centroids_sigma_inv(
            centroid.unsqueeze(0),
            None,
            sigma_inv,
            test_embeddings,
        )[:, 0]
        return dists.cpu().detach().numpy()

    @staticmethod
    def rde(
        train_embeddings: torch.Tensor, test_embeddings: torch.Tensor, ue_dict=None
    ):
        log.info("Calculating RDE scores...")
        from sklearn.decomposition import KernelPCA
        from sklearn.covariance import MinCovDet

        pca = KernelPCA(n_components=100, kernel="rbf", random_state=42)
        X_pca_train = pca.fit_transform(train_embeddings.cpu().detach().numpy())
        X_pca_test = pca.transform(test_embeddings.cpu().detach().numpy())

        covariance = MinCovDet(random_state=42).fit(X_pca_train)
        rde = covariance.mahalanobis(X_pca_test)

        return rde

    @staticmethod
    def ddu(
        train_embeddings: torch.Tensor, test_embeddings: torch.Tensor, ue_dict=None
    ):
        log.info("Calculating DDU scores...")
        train_labels = np.zeros(train_embeddings.shape[0])
        gmm, jitter = gmm_fit(train_embeddings, train_labels)
        log_probs = get_gmm_log_probs(gmm, test_embeddings)
        scores = -compute_density(log_probs, None)

        return scores.detach().cpu().numpy()

    @staticmethod
    def nuq(
        train_embeddings: torch.Tensor, test_embeddings: torch.Tensor, ue_dict=None
    ):
        import ray
        from true.packages.NUQ.nuq import NuqClassifier

        log.info("Calculating NUQ scores...")
        train_labels = np.zeros(train_embeddings.shape[0])

        nuq_classifier = NuqClassifier(
            tune_bandwidth=None,
            n_neighbors=200,
            log_pN=-100,
        )
        nuq_classifier.fit(X=train_embeddings.detach().cpu().numpy(), y=train_labels)
        _, squared_dists = ray.get(
            nuq_classifier.index_.knn_query.remote(
                nuq_classifier.X_ref_, return_dist=True
            )
        )
        dists = np.sqrt(squared_dists)[:, 1:]
        min_dists = dists[:, 0]
        left, right = min_dists[min_dists != 0].min(), dists.max()
        bandwidth = np.sqrt(left * right)
        nuq_classifier.bandwidth_ref_ = ray.put(np.array(bandwidth))
        _, log_epistemic_uncs = nuq_classifier.predict_proba(
            test_embeddings.detach().cpu().numpy(), return_uncertainty="epistemic"
        )

        return log_epistemic_uncs

    @staticmethod
    def emb_mlp(
        enc_train_embeddings: torch.Tensor, enc_test_embeddings: torch.Tensor, 
        dec_train_embeddings: torch.Tensor, dec_test_embeddings: torch.Tensor, 
        train_preds, test_preds, train_labels, test_labels,
        ue_dict=None
    ):
        cat_train_embeddings = torch.cat([enc_train_embeddings, dec_train_embeddings], dim=-1)
        cat_test_embeddings = torch.cat([enc_test_embeddings, dec_test_embeddings], dim=-1)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        X_train = cat_train_embeddings.to(device)
        y_train = calculate_top_1_acc(train_preds, train_labels)['Top 1 Acc']

        X_val = cat_train_embeddings.to(device)
        y_val = calculate_top_1_acc(train_preds, train_labels)['Top 1 Acc']

        X_test = cat_test_embeddings.to(device)
        y_test = calculate_top_1_acc(test_preds, test_labels)['Top 1 Acc']

        trainset = Dataset(X, y)
        testset = Dataset(X_test, y_test)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=512)
        testloader = torch.utils.data.DataLoader(testset, batch_size=512)

        seed = 1
        set_seed(seed)

        net = Net().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(net.parameters(), lr=5e-6, weight_decay=1e-4)

        while True:  # loop over the dataset until val loss stops decreasing
            running_loss = 0.0
            net.train()
            n = 0
            for i, data in enumerate(trainloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(inputs.to(device))
                loss = criterion(outputs, labels.to(device))
                print(loss)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                n += 1

        net.eval()
        preds = []
        for i, data in enumerate(testloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs.to(device))
            preds.append(outputs.cpu())

        preds = torch.cat(preds)[:, 0].flatten().detach().cpu()

        return preds

    @staticmethod
    def enc_diff(
        embedding_diffs: torch.Tensor, ue_dict=None
    ):
        log.info("Calculating ENC DIFF scores...")
        ues = embedding_diffs.norm(p=2, dim=-1)
        return ues.cpu().detach().numpy()

    @staticmethod
    def hue(
        train_embeddings: torch.Tensor,
        dev_embeddings: torch.Tensor,
        test_embeddings: torch.Tensor,
        inference_output=None,
        inference_output_dev=None,
        epistemic_method=None,
        ue_dict=None,
        **generate_kwargs,
    ):
        pass

    def deep_ensemble(
        self,
        strategy="ensp",
        inference_output=None,
        models_paths=None,
        strategy_kwargs=None,
        **generate_kwargs,
    ):
        log.info(f"Calculating DE+{strategy} scores...")
        de_generate_kwargs = deepcopy(generate_kwargs)
        if strategy_kwargs is None:
            strategy_kwargs = {}
        log.info(f"Calculating MD + {strategy} scores...")
        if inference_output is None:
            device = generate_kwargs["model"].device
            inference_output = {}
            for model_path in models_paths:
                de_generate_kwargs["model"] = AutoModelForSeq2SeqLM.from_pretrained(
                    model_path
                ).to(device)
                de_generate_kwargs["tokenizer"] = AutoTokenizer.from_pretrained(
                    model_path
                )
                inference_output[model_path] = generate(**de_generate_kwargs)
        scores = np.r_[[x["scores"] for x in inference_output.values()]].T
        hypotheses = np.r_[
            [x["hypotheses"] for x in inference_output.values()]
        ].T.tolist()
        return getattr(self, strategy)(
            inference_output=dict(scores=scores, hypotheses=hypotheses),
            **strategy_kwargs,
        )

    @staticmethod
    def geom_nsp(inference_output=None, **mc_output_kwargs):
        """ """
        log.info("Calculating ENSP scores...")
        if inference_output is None:
            inference_output = get_mc_output(**mc_output_kwargs)
        scores = inference_output["scores"]
        uncertainty_estimates = -np.prod(scores, axis=1) ** (1 / scores.shape[1])
        return uncertainty_estimates

    @staticmethod
    def garmonic_nsp(inference_output=None, **mc_output_kwargs):
        """ """
        log.info("Calculating ENSP scores...")
        if inference_output is None:
            inference_output = get_mc_output(**mc_output_kwargs)
        scores = inference_output["scores"]
        uncertainty_estimates = scores.shape[1] / np.sum(1 / scores, axis=1)
        return uncertainty_estimates

    @staticmethod
    def square_nsp(inference_output=None, **mc_output_kwargs):
        """ """
        log.info("Calculating ENSP scores...")
        if inference_output is None:
            inference_output = get_mc_output(**mc_output_kwargs)
        scores = inference_output["scores"]
        uncertainty_estimates = ((scores**2).sum(1) / scores.shape[1]) ** (0.5)
        return uncertainty_estimates

    @staticmethod
    def pe_token_unc(inference_output=None, token_level_data=None, **token_calc_kwargs):
        """
        PE token-level uncertainties (refer to Malinin Uncertainty)
        """
        log.info("Calculating PE token uncertainties...")
        if token_level_data is None:
            token_level_data = get_token_level_data(
                inference_output, **token_calc_kwargs
            )
        weights = token_level_data["weights"]
        uncertainty_estimates = {}

        for key in [
            "total_uncertainty",
            "mutual_information",
            "data_uncertainty",
            "epkl_total_uncertainty",
            "epkl",
            "rmi",
        ] + [f"entropy_top{k}" for k in token_calc_kwargs.pop("topk", (5, 10, 15))]:
            unc = token_level_data[key]
            uncertainty_estimates["PE-T-" + key.replace("_", "-")] = (
                unc * weights
            ).sum(-1)

        return uncertainty_estimates

    @staticmethod
    def pe_seq_unc(
        inference_output=None,
        sequence_level_data=None,
        softmax_t: Union[int, float] = 1,
        **useless_kwargs,
    ):
        """
        PE sequence-level uncertainties (refer to Malinin Uncertainty)
        """
        log.info("Calculating PE sequence WTU scores...")
        assert (inference_output is not None) and (
            sequence_level_data is not None
        ), "Sequence-level scores must be obtained with UEGenerator!"
        if softmax_t != 1:
            raise NotImplementedError()
        else:
            weights = inference_output["weights"]

        ens_probas = inference_output["scores"]  # num_obs x num_beams
        ens_log_probas = inference_output["log_scores"]  # num_obs x num_beams
        model_log_probas = sequence_level_data[
            "log_probas"
        ]  # num_models x num_obs x num_beams
        tu = (ens_probas * weights).sum(-1)  # num_obs
        rmi = (
            ((ens_log_probas - model_log_probas) * weights[None, :, :])
            .sum(-1)  # sum across beams
            .mean(0)  # mean by models
        )  # num_obs
        rmi_abs = (
            (np.abs(ens_log_probas - model_log_probas) * weights[None, :, :])
            .sum(-1)  # sum across beams
            .mean(0)  # mean by models
        )  # num_obs
        # Unbiased scores
        unbiased_tu = (inference_output["scores_unbiased"] * weights).sum(
            -1
        )  # num_obs x num_beams

        uncertainty_estimates = {
            "PE-S-total-uncertainty": -tu,
            "PE-S-tu-unbiased": -unbiased_tu,
            "PE-S-rmi": rmi,
            "PE-S-rmi-abs": rmi_abs,
        }

        return uncertainty_estimates

    @staticmethod
    def ep_token_unc(
        inference_output=None,
        token_level_data=None,
        topk: Tuple[int] = (5, 10, 15),  # TODO: add
        softmax_t: Union[int, float] = 1,
        is_based_on_single_output: bool = False,
        **_,
    ):
        """
        EP token-level uncertainties (refer to Malinin Uncertainty)
        """
        log.info("Calculating EP token uncertainties...")
        assert (
            token_level_data is not None
        ), "Token-level scores must be obtained with UEGenerator!"
        if softmax_t != 1:
            raise NotImplementedError()
        weights = inference_output["weights"]

        uncertainty_estimates = {}
        keys = [
            "total_uncertainty",
            "mutual_information",
            "data_uncertainty",
            "epkl_total_uncertainty",
            "epkl",
            "rmi",
        ] + [f"entropy_top{k}" for k in topk]

        prefix = "EP" if not is_based_on_single_output else "SEP"
        for key in keys:
            unc = token_level_data[key]
            uncertainty_estimates[f"{prefix}-T-" + key.replace("_", "-")] = (
                unc * weights
            ).sum(-1)

        return uncertainty_estimates

    @staticmethod
    def ep_seq_unc(
        inference_output=None,
        sequence_level_data=None,
        softmax_t: Union[int, float] = 1,
        is_based_on_single_output: bool = False,
        **useless_kwargs,
    ):
        """
        EP sequence-level uncertainties (refer to Malinin Uncertainty)
        """
        log.info("Calculating EP sequence WTU scores...")
        assert (
            sequence_level_data is not None
        ), "Sequence-level scores must be obtained with UEGenerator!"

        model_log_probas = sequence_level_data['log_probas']
        ens_log_probas = torch.tensor(model_log_probas).logsumexp(0) - torch.tensor(len(model_log_probas)).log()  # num_obs x num_beams
        ens_log_probas = ens_log_probas.numpy()
        ens_probas = np.exp(ens_log_probas)

        ens_probas_exp = ens_probas**softmax_t
        weights = ens_probas_exp / ens_probas_exp.sum(-1, keepdims=True)

        tu = (ens_probas * weights).sum(-1)  # num_obs
        rmi = (
            ((ens_log_probas - model_log_probas) * weights[None, :, :]).sum(-1).mean(0)
        )  # num_obs
        rmi_abs = (
            (np.abs(ens_log_probas - model_log_probas) * weights[None, :, :])
            .sum(-1)
            .mean(0)
        )  # num_obs

        prefix = "EP" if not is_based_on_single_output else "SEP"
        uncertainty_estimates = {
            f"{prefix}-S-total-uncertainty": -tu,
            f"{prefix}-S-rmi": rmi,
            f"{prefix}-S-rmi-abs": rmi_abs,
        }

        return uncertainty_estimates
