"""
Fast DeLong test for comparing two correlated (paired, same-subjects) ROC AUCs.
Standard algorithm (Sun & Xu, 2014, "Fast Implementation of DeLong's Algorithm").
"""
import numpy as np
from scipy import stats


def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive_examples[r, :])
        ty[r, :] = _compute_midrank(negative_examples[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_roc_test(y_true, y_prob_a, y_prob_b):
    """
    Paired DeLong test comparing two classifiers' predicted probabilities on the
    SAME subjects/labels. Returns (auc_a, auc_b, z_stat, p_value, ci95_diff).
    """
    y_true = np.asarray(y_true)
    order = np.argsort(-y_true, kind="mergesort")
    y_true_sorted = y_true[order]
    m = int(np.sum(y_true_sorted == 1))

    preds = np.vstack([np.asarray(y_prob_a)[order], np.asarray(y_prob_b)[order]])
    aucs, delongcov = _fast_delong(preds, m)

    auc_a, auc_b = aucs[0], aucs[1]
    diff = auc_a - auc_b
    var = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    var = max(var, 1e-12)
    z = diff / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    se = np.sqrt(var)
    ci = (diff - 1.96 * se, diff + 1.96 * se)
    return auc_a, auc_b, z, p, ci
