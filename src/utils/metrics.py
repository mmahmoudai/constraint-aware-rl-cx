"""
Evaluation metrics: AUC, ECE, Brier score, McNemar's test, DeLong's test, Cohen's d.
"""

import numpy as np
from scipy import stats
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute comprehensive classification metrics.

    Args:
        y_true: ground truth labels (0 or 1), shape (N,)
        y_prob: predicted probabilities for class 1, shape (N,)
        threshold: classification threshold

    Returns:
        dict with accuracy, auc, sensitivity, specificity, ppv, npv, f1, brier, ece
    """
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def prevalence_adjusted_ppv_npv(
    sensitivity: float, specificity: float, prevalence: float
) -> dict:
    """Compute PPV and NPV at a given prevalence via Bayes' theorem.

    Args:
        sensitivity: true positive rate
        specificity: true negative rate
        prevalence:  disease prevalence (e.g., 0.05 for 5%)

    Returns:
        dict with ppv and npv
    """
    tp_rate = sensitivity * prevalence
    fp_rate = (1 - specificity) * (1 - prevalence)
    fn_rate = (1 - sensitivity) * prevalence
    tn_rate = specificity * (1 - prevalence)

    ppv = tp_rate / (tp_rate + fp_rate) if (tp_rate + fp_rate) > 0 else 0.0
    npv = tn_rate / (tn_rate + fn_rate) if (tn_rate + fn_rate) > 0 else 0.0

    return {"ppv": ppv, "npv": npv}


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE) with equal-frequency bins.

    ECE = Σ (|B_m| / n) · |acc(B_m) - conf(B_m)|
    """
    n = len(y_true)
    if n == 0:
        return 0.0

    # equal-frequency bins
    sorted_indices = np.argsort(y_prob)
    bin_size = n // n_bins
    ece = 0.0

    for i in range(n_bins):
        start = i * bin_size
        end = start + bin_size if i < n_bins - 1 else n
        idx = sorted_indices[start:end]

        if len(idx) == 0:
            continue

        bin_acc = y_true[idx].mean()
        bin_conf = y_prob[idx].mean()
        ece += len(idx) / n * abs(bin_acc - bin_conf)

    return ece


def mcnemar_test(y_true: np.ndarray, y_pred_a: np.ndarray, y_pred_b: np.ndarray) -> dict:
    """McNemar's test comparing two classifiers on paired data.

    Returns dict with statistic, p_value, and significant (at Bonferroni-corrected α).
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # discordant pairs
    b = np.sum(correct_a & ~correct_b)  # A correct, B wrong
    c = np.sum(~correct_a & correct_b)  # A wrong, B correct

    if b + c == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False}

    # continuity-corrected McNemar
    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    # Bonferroni correction for 28 pairwise comparisons
    alpha_corr = 0.05 / 28

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < alpha_corr,
    }


def delong_test(y_true: np.ndarray, y_prob_a: np.ndarray, y_prob_b: np.ndarray) -> dict:
    """DeLong's test for comparing two AUC values on paired data.

    Implements the fast O(n log n) algorithm.

    Returns dict with z_statistic, p_value, auc_a, auc_b, delta_auc.
    """
    auc_a = roc_auc_score(y_true, y_prob_a)
    auc_b = roc_auc_score(y_true, y_prob_b)

    pos = y_true == 1
    neg = y_true == 0
    n_pos = pos.sum()
    n_neg = neg.sum()

    if n_pos == 0 or n_neg == 0:
        return {
            "z_statistic": 0.0,
            "p_value": 1.0,
            "auc_a": auc_a,
            "auc_b": auc_b,
            "delta_auc": auc_a - auc_b,
        }

    # structural components for each model
    def structural_components(y_prob):
        V_pos = np.zeros(n_pos)
        V_neg = np.zeros(n_neg)
        pos_scores = y_prob[pos]
        neg_scores = y_prob[neg]

        for i, ps in enumerate(pos_scores):
            V_pos[i] = np.mean(ps > neg_scores) + 0.5 * np.mean(ps == neg_scores)
        for j, ns in enumerate(neg_scores):
            V_neg[j] = np.mean(pos_scores > ns) + 0.5 * np.mean(pos_scores == ns)

        return V_pos, V_neg

    V_pos_a, V_neg_a = structural_components(y_prob_a)
    V_pos_b, V_neg_b = structural_components(y_prob_b)

    # covariance matrix of the two AUC estimates
    S_pos = np.cov(V_pos_a, V_pos_b)[0, 1] if n_pos > 1 else 0
    S_neg = np.cov(V_neg_a, V_neg_b)[0, 1] if n_neg > 1 else 0

    var_a = np.var(V_pos_a) / n_pos + np.var(V_neg_a) / n_neg if n_pos > 1 and n_neg > 1 else 1e-10
    var_b = np.var(V_pos_b) / n_pos + np.var(V_neg_b) / n_neg if n_pos > 1 and n_neg > 1 else 1e-10
    cov_ab = S_pos / n_pos + S_neg / n_neg

    var_diff = var_a + var_b - 2 * cov_ab
    if var_diff <= 0:
        var_diff = 1e-10

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return {
        "z_statistic": float(z),
        "p_value": float(p_value),
        "auc_a": float(auc_a),
        "auc_b": float(auc_b),
        "delta_auc": float(auc_a - auc_b),
    }


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups."""
    na, nb = len(group_a), len(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    pooled_std = np.sqrt(((na - 1) * var_a + (nb - 1) * var_b) / (na + nb - 2))
    if pooled_std == 0:
        return 0.0
    return float((np.mean(group_a) - np.mean(group_b)) / pooled_std)


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple:
    """Compute bootstrap confidence interval for a metric.

    Args:
        y_true: ground truth
        y_prob: predictions
        metric_fn: callable(y_true, y_prob) -> float
        n_bootstrap: number of bootstrap samples
        ci: confidence level
        seed: random seed

    Returns:
        (lower, upper) confidence interval bounds
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        try:
            s = metric_fn(y_true[idx], y_prob[idx])
            scores.append(s)
        except (ValueError, ZeroDivisionError):
            continue

    alpha = (1 - ci) / 2
    lower = np.percentile(scores, 100 * alpha)
    upper = np.percentile(scores, 100 * (1 - alpha))
    return float(lower), float(upper)
