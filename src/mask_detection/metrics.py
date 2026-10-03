"""Binary and categorical scores with explicit, shared decision policies."""

from __future__ import annotations

import math


def validate_probabilities(values):
    values = [float(x) for x in values]
    if not values or any(not math.isfinite(x) or not 0 <= x <= 1 for x in values):
        raise ValueError("Expected nonempty finite probabilities in [0,1]")
    return values


def binary_metrics(labels, probabilities, threshold=0.5):
    labels = list(labels)
    probabilities = validate_probabilities(probabilities)
    if len(labels) != len(probabilities) or any(
        type(y) is not int or y not in (0, 1) for y in labels
    ):
        raise ValueError("Labels must align with probabilities and be integer 0/1")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be in [0,1]")
    predictions = [int(p >= threshold) for p in probabilities]
    matrix = [
        [sum(y == i and p == j for y, p in zip(labels, predictions)) for j in (0, 1)]
        for i in (0, 1)
    ]
    tn, fp = matrix[0]
    fn, tp = matrix[1]

    def ratio(a, b):
        return a / b if b else None

    by_class = []
    for index in (0, 1):
        correct = matrix[index][index]
        predicted = sum(row[index] for row in matrix)
        support = sum(matrix[index])
        by_class.append(
            dict(
                precision=ratio(correct, predicted),
                recall=ratio(correct, support),
                f1=ratio(2 * correct, predicted + support),
                support=support,
            )
        )
    positives, negatives = sum(labels), len(labels) - sum(labels)
    auc = None
    if positives and negatives:
        ordered = sorted(zip(probabilities, labels))
        rank_sum, start = 0.0, 0
        while start < len(ordered):
            end = start + 1
            while end < len(ordered) and ordered[end][0] == ordered[start][0]:
                end += 1
            rank_sum += ((start + 1 + end) / 2) * sum(y for _, y in ordered[start:end])
            start = end
        auc = (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)
    clipped = [min(1 - 1e-15, max(1e-15, p)) for p in probabilities]
    return dict(
        sample_count=len(labels),
        threshold=threshold,
        confusion_matrix=matrix,
        confusion_matrix_axes="rows=true, columns=predicted; class order=[0,1]",
        accuracy=(tp + tn) / len(labels),
        by_class=by_class,
        sensitivity=ratio(tp, tp + fn),
        specificity=ratio(tn, tn + fp),
        roc_auc=auc,
        roc_auc_note=None if auc is not None else "Undefined: only one true class present",
        brier_score=sum((p - y) ** 2 for p, y in zip(probabilities, labels)) / len(labels),
        log_loss=-sum(y * math.log(p) + (1 - y) * math.log(1 - p) for p, y in zip(clipped, labels))
        / len(labels),
        undefined_metric_policy="null; no invented zeros",
    )


def _binary_evaluation_report(rows, model_predictions, class_names, *, threshold=0.5):
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
        predictions["ensemble"] = [
            sum(p[i] for p in predictions.values()) / len(predictions) for i in range(len(rows))
        ]
    labels = [r["label"] for r in rows]
    return dict(
        schema_version=1,
        class_names=class_names,
        positive_class=class_names[1],
        ensemble_policy="unweighted arithmetic mean; threshold selected before test evaluation",
        metrics={name: binary_metrics(labels, p, threshold) for name, p in predictions.items()},
        samples=[
            dict(
                sample_id=r["source"],
                sha256=r["sha256"],
                group=r["group"],
                true_label=r["label"],
                true_class=class_names[r["label"]],
                probabilities={name: p[i] for name, p in predictions.items()},
            )
            for i, r in enumerate(rows)
        ],
    )


DECISION_POLICY = {"kind": "argmax", "tie_break": "lowest_class_index"}
PROBABILITY_SUM_TOLERANCE = 1e-6


def validate_probability_matrix(values, num_classes):
    """Validate ordered categorical rows without renormalizing model outputs."""
    if type(num_classes) is not int or num_classes < 2:
        raise ValueError("Need at least two ordered classes")
    try:
        if isinstance(values, (str, bytes)):
            raise ValueError("Expected probability rows, not text")
        values = list(values)
        if any(isinstance(row, (str, bytes)) for row in values):
            raise ValueError("Expected probability rows, not text")
        rows = [[float(value) for value in row] for row in values]
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("Expected a probability matrix") from error
    if not rows or any(len(row) != num_classes for row in rows):
        raise ValueError("Expected nonempty probability rows matching class count")
    for row in rows:
        if any(not math.isfinite(value) or not 0 <= value <= 1 for value in row):
            raise ValueError("Categorical probabilities must be finite and in [0,1]")
        if not math.isclose(math.fsum(row), 1.0, rel_tol=0, abs_tol=PROBABILITY_SUM_TOLERANCE):
            raise ValueError("Categorical probabilities must sum to one within 1e-6")
    return rows


def categorical_predictions(probabilities, num_classes):
    """Argmax chooses the first recorded class in an exact probability tie."""
    rows = validate_probability_matrix(probabilities, num_classes)
    return [max(range(num_classes), key=row.__getitem__) for row in rows]


def categorical_metrics(labels, probabilities, *, num_classes=3):
    """Evaluate one ordered categorical distribution per example.

    Missing true classes have undefined recall/F1. Strict macro recall and F1
    are null if any required class has no true support. An unpredicted class
    with true support has precision null, recall zero, and F1 zero.
    """
    labels = list(labels)
    probabilities = validate_probability_matrix(probabilities, num_classes)
    if len(labels) != len(probabilities) or any(
        type(label) is not int or not 0 <= label < num_classes for label in labels
    ):
        raise ValueError("Labels must align with probabilities and be integer class indices")
    predictions = categorical_predictions(probabilities, num_classes)
    matrix = [[0] * num_classes for _ in range(num_classes)]
    for label, predicted in zip(labels, predictions):
        matrix[label][predicted] += 1
    by_class = []
    for index in range(num_classes):
        correct = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        by_class.append(
            dict(
                precision=correct / predicted if predicted else None,
                recall=correct / support if support else None,
                f1=2 * correct / (support + predicted) if support else None,
                support=support,
                predicted_count=predicted,
            )
        )

    def strict_mean(key):
        values = [row[key] for row in by_class]
        return (
            math.fsum(values) / num_classes if all(value is not None for value in values) else None
        )

    macro_recall = strict_mean("recall")
    return dict(
        sample_count=len(labels),
        decision_policy=dict(DECISION_POLICY),
        confusion_matrix=matrix,
        confusion_matrix_axes=f"rows=true, columns=predicted; class order={list(range(num_classes))}",
        accuracy=sum(matrix[i][i] for i in range(num_classes)) / len(labels),
        by_class=by_class,
        macro_recall=macro_recall,
        balanced_accuracy=macro_recall,
        macro_f1=strict_mean("f1"),
        log_loss=-math.fsum(
            math.log(max(1e-15, row[label])) for row, label in zip(probabilities, labels)
        )
        / len(labels),
        log_loss_policy="natural logarithm; true-class probability floored at 1e-15",
        brier_score=math.fsum(
            math.fsum((value - int(index == label)) ** 2 for index, value in enumerate(row))
            for row, label in zip(probabilities, labels)
        )
        / len(labels),
        brier_score_convention="mean of sum over all classes of squared probability error; no division by class count; range [0,2]",
        probability_validation="finite [0,1] entries; row sum within absolute 1e-6 of one; no renormalization",
        undefined_metric_policy="precision null without predicted support; recall and F1 null without true support; F1 zero for supported but unpredicted classes; macro recall and F1 null if any constituent is undefined",
    )


def evaluation_report(rows, model_predictions, class_names, *, threshold=None):
    """Dispatch legacy binary vectors or categorical ordered probability rows."""
    if len(class_names) == 2:
        return _binary_evaluation_report(
            rows, model_predictions, class_names, threshold=0.5 if threshold is None else threshold
        )
    num_classes = len(class_names)
    if (
        num_classes < 3
        or any(not isinstance(name, str) or not name.strip() for name in class_names)
        or len(set(class_names)) != num_classes
        or not model_predictions
    ):
        raise ValueError("Need ordered unique class names and at least one model")
    if threshold is not None:
        raise ValueError("Categorical predictions use argmax; a threshold is not applicable")
    if len({row["source"] for row in rows}) != len(rows):
        raise ValueError("Repeated sample identity")
    predictions = {
        name: validate_probability_matrix(values, num_classes)
        for name, values in model_predictions.items()
    }
    if any(len(values) != len(rows) for values in predictions.values()):
        raise ValueError("Prediction count does not match the manifest")
    if "ensemble" in predictions:
        raise ValueError("ensemble is a reserved name")
    if len(predictions) > 1:
        predictions["ensemble"] = [
            [
                math.fsum(values[i][j] for values in predictions.values()) / len(predictions)
                for j in range(num_classes)
            ]
            for i in range(len(rows))
        ]
    labels = [row["label"] for row in rows]
    # Validate labels before using them to index the ordered class names.
    metrics = {
        name: categorical_metrics(labels, values, num_classes=num_classes)
        for name, values in predictions.items()
    }
    predicted_classes = {
        name: categorical_predictions(values, num_classes) for name, values in predictions.items()
    }
    return dict(
        schema_version=2,
        class_names=list(class_names),
        output_mode="categorical_softmax",
        decision_policy=dict(DECISION_POLICY),
        ensemble_policy="unweighted arithmetic mean of aligned probability vectors, then the same argmax decision policy",
        metrics=metrics,
        samples=[
            dict(
                sample_id=row["source"],
                sha256=row["sha256"],
                group=row["group"],
                true_label=row["label"],
                true_class=class_names[row["label"]],
                probabilities={name: values[i] for name, values in predictions.items()},
                predicted_class_indices={
                    name: values[i] for name, values in predicted_classes.items()
                },
            )
            for i, row in enumerate(rows)
        ],
    )
