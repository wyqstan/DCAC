import torch
from collections import deque
import numpy as np
import sklearn.metrics as sk
import math
def get_entropy(loss, num_class):
    max_entropy = math.log2(num_class)
    return loss / max_entropy
def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    """Update cache using FIFO strategy with deque: remove oldest if over capacity."""
    with torch.no_grad():
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]] + [features_loss[3]]

        if pred not in cache:
            cache[pred] = deque(maxlen=shot_capacity)  # create new deque with fixed capacity

        cache[pred].append(item)  # automatically removes oldest if full

def compute_cache_logits(image_features, cache, alpha, topk=None):
    cache_keys = [item[0] for class_index in sorted(cache.keys()) for item in cache[class_index]]
    cache_values = [item[2] for class_index in sorted(cache.keys()) for item in cache[class_index]]
    cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
    cache_values = torch.cat(cache_values, dim=0)
    topk_values, topk_indices = torch.topk(cache_values, k=topk, dim=1)
    masked_probs = torch.zeros_like(cache_values)
    cache_values = masked_probs.scatter_(1, topk_indices, topk_values)
    affinity = image_features @ cache_keys
    cache_logits = affinity @ cache_values
    return alpha * cache_logits

def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out