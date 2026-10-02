"""Small binary evaluation core; all scores use probabilities of class index 1."""
from __future__ import annotations

import math


def validate_probabilities(values):
    values = [float(x) for x in values]
    if not values or any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError("Expected nonempty finite probabilities in [0,1]")
    return values


def binary_metrics(labels, probabilities, threshold=.5):
    labels = list(labels)
    probabilities = validate_probabilities(probabilities)
    if len(labels) != len(probabilities) or any(type(y) is not int or y not in (0, 1) for y in labels):
        raise ValueError("Labels must align with probabilities and be integer 0/1")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0,1]")
    predictions = [int(p >= threshold) for p in probabilities]
    matrix = [[sum(y == i and p == j for y, p in zip(labels, predictions)) for j in (0, 1)] for i in (0, 1)]
    tn, fp = matrix[0]; fn, tp = matrix[1]
    def ratio(a, b):
        return a / b if b else None
    by_class = []
    for index in (0, 1):
        correct = matrix[index][index]
        predicted = sum(row[index] for row in matrix)
        support = sum(matrix[index])
        by_class.append(dict(precision=ratio(correct, predicted), recall=ratio(correct, support),
                             f1=ratio(2 * correct, predicted + support), support=support))
    positives, negatives = sum(labels), len(labels) - sum(labels)
    auc = None
    if positives and negatives:
        ordered = sorted(zip(probabilities, labels))
        rank_sum, start = 0., 0
        while start < len(ordered):
            end = start + 1
            while end < len(ordered) and ordered[end][0] == ordered[start][0]:
                end += 1
            rank_sum += ((start + 1 + end) / 2) * sum(y for _, y in ordered[start:end])
            start = end
        auc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    clipped = [min(1 - 1e-15, max(1e-15, p)) for p in probabilities]
    return dict(sample_count=len(labels), threshold=threshold, confusion_matrix=matrix,
                confusion_matrix_axes="rows=true, columns=predicted; class order=[0,1]",
                accuracy=(tp + tn) / len(labels), by_class=by_class,
                sensitivity=ratio(tp, tp + fn), specificity=ratio(tn, tn + fp),
                roc_auc=auc, roc_auc_note=None if auc is not None else "Undefined: only one true class present",
                brier_score=sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(labels),
                log_loss=-sum(y * math.log(p) + (1-y) * math.log(1-p) for p, y in zip(clipped, labels)) / len(labels),
                undefined_metric_policy="null; no invented zeros")


def evaluation_report(rows, model_predictions, class_names, *, threshold=.5):
    """Preserve manifest order and per-model outputs; equally weight an ensemble."""
    if len(class_names) != 2 or len(set(class_names)) != 2 or not model_predictions:
        raise ValueError("Need two class names and at least one model")
    if len({r["source"] for r in rows}) != len(rows):
        raise ValueError("Repeated sample identity")
    predictions = {name: validate_probabilities(p) for name, p in model_predictions.items()}
    if any(len(p) != len(rows) for p in predictions.values()):
        raise ValueError("Prediction count does not match the manifest")
    if "ensemble" in predictions:
        raise ValueError("ensemble is a reserved name")
    if len(predictions) > 1:
        predictions["ensemble"] = [sum(p[i] for p in predictions.values()) / len(predictions) for i in range(len(rows))]
    labels = [r["label"] for r in rows]
    return dict(schema_version=1, class_names=class_names, positive_class=class_names[1],
                ensemble_policy="unweighted arithmetic mean; threshold selected before test evaluation",
                metrics={name: binary_metrics(labels, p, threshold) for name, p in predictions.items()},
                samples=[dict(sample_id=r["source"], sha256=r["sha256"], group=r["group"],
                              true_label=r["label"], true_class=class_names[r["label"]],
                              probabilities={name: p[i] for name, p in predictions.items()})
                         for i, r in enumerate(rows)])
