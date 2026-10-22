"""
IC-272 — Introduction to Machine Learning
Lab 7 — Logistic Regression
Scratch implementation using only numpy, pandas, and matplotlib.

Expected files in the same folder:
  - this script: ic272_lab7.py
  - ic272_lab7_loans.csv (provided separately by the course)

Run:
  python ic272_lab7.py

Optional homework H3 additionally needs:
  - ic272_lab6_vibration.csv
  - your Lab 6 split / k-NN result for comparison
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Reusable functions (Lab 4 / Lab 1 equivalents)
# ---------------------------------------------------------------------------

def design_matrix(X):
    """Add an intercept column of ones to a feature matrix."""
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    return np.column_stack((np.ones(X.shape[0]), X))


def standardise_fit(X):
    """Return standardised X and the training-set mean and standard deviation."""
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    # Prevent division by zero if a feature is constant.
    sigma[sigma == 0] = 1.0
    return (X - mu) / sigma, mu, sigma


def standardise_transform(X, mu, sigma):
    """Standardise data using training-set mean and standard deviation."""
    sigma_safe = np.asarray(sigma, dtype=float).copy()
    sigma_safe[sigma_safe == 0] = 1.0
    return (np.asarray(X, dtype=float) - mu) / sigma_safe


def confusion_matrix_scratch(y_true, y_pred):
    """Return [[TN, FP], [FN, TP]] for binary labels 0 and 1."""
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    return np.array([[tn, fp], [fn, tp]], dtype=int)


def precision_recall_f1_scratch(y_true, y_pred):
    """Return precision, recall, and F1; undefined ratios are reported as 0."""
    cm = confusion_matrix_scratch(y_true, y_pred)
    tn, fp = cm[0]
    fn, tp = cm[1]
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return precision, recall, f1


def accuracy_scratch(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int).ravel()
    y_pred = np.asarray(y_pred, dtype=int).ravel()
    return float(np.mean(y_true == y_pred))


# ---------------------------------------------------------------------------
# Logistic regression (Lab 7 tasks 1–3)
# ---------------------------------------------------------------------------

def sigmoid(z):
    """Vectorised sigmoid, numerically stable for large positive/negative z."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    positive = z >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    out[~positive] = exp_z / (1.0 + exp_z)
    return out


def log_likelihood(X, y, theta):
    """
    Logistic-regression log likelihood:
    sum(y*log(p) + (1-y)*log(1-p)).
    X is supplied WITHOUT an intercept column.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    A = design_matrix(X)
    p = sigmoid(A @ np.asarray(theta, dtype=float).ravel())
    eps = 1e-12
    return float(np.sum(y * np.log(p + eps) + (1.0 - y) * np.log(1.0 - p + eps)))


def fit_logistic(X, y, lr=0.1, n_iters=2000):
    """
    Fit logistic regression using gradient ascent.
    Returns (theta, history), where history has one log-likelihood per iteration.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float).ravel()
    A = design_matrix(X)
    m = len(y)
    theta = np.zeros(A.shape[1], dtype=float)
    history = []

    for _ in range(n_iters):
        p = sigmoid(A @ theta)
        theta += (lr / m) * (A.T @ (y - p))
        history.append(log_likelihood(X, y, theta))

    return theta, history


def roc_curve_scratch(y_true, p):
    """
    Homework H1: compute FPR and TPR over probability thresholds.
    Includes endpoints and returns points in increasing FPR order.
    """
    y_true = np.asarray(y_true, dtype=int).ravel()
    p = np.asarray(p, dtype=float).ravel()
    thresholds = np.r_[np.inf, np.sort(np.unique(p))[::-1], -np.inf]

    fpr_values = []
    tpr_values = []
    positives = np.sum(y_true == 1)
    negatives = np.sum(y_true == 0)

    for threshold in thresholds:
        pred = (p >= threshold).astype(int)
        cm = confusion_matrix_scratch(y_true, pred)
        tn, fp = cm[0]
        fn, tp = cm[1]
        fpr_values.append(fp / negatives if negatives else 0.0)
        tpr_values.append(tp / positives if positives else 0.0)

    order = np.argsort(fpr_values)
    return np.asarray(fpr_values)[order], np.asarray(tpr_values)[order]


def auc_trapezoid(fpr, tpr):
    """Homework H1: area under the ROC curve using the trapezoid rule."""
    fpr = np.asarray(fpr, dtype=float)
    tpr = np.asarray(tpr, dtype=float)
    order = np.argsort(fpr)
    return float(np.trapezoid(tpr[order], fpr[order]))


def plot_roc(fpr, tpr, auc_value):
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {auc_value:.3f})")
    plt.plot([0, 1], [0, 1], "--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def run_main_lab():
    print("=" * 70)
    print("IC-272 Lab 7 — Logistic Regression")
    print("=" * 70)

    # Setting up: keep seed and split exactly as specified in the handout.
    df = pd.read_csv("ic272_lab7_loans.csv")
    features = ["income_k", "utilisation"]
    X = df[features].to_numpy(dtype=float)
    y = df["defaulted"].to_numpy(dtype=int)

    idx = np.random.default_rng(0).permutation(len(y))
    te, tr = idx[:100], idx[100:]
    X_train, y_train = X[tr], y[tr]
    X_test, y_test = X[te], y[te]

    print("train:", X_train.shape, " test:", X_test.shape)

    # Demonstration: linear regression is not a valid probability model.
    print("\n--- Why not linear regression? ---")
    # This is the equivalent of the Lab 4 normal equation, used only for comparison.
    A_train = design_matrix(X_train)
    theta_lin = np.linalg.pinv(A_train) @ y_train.astype(float)
    pred_lin = design_matrix(X) @ theta_lin
    print("smallest prediction:", round(float(pred_lin.min()), 3))
    print("largest prediction: ", round(float(pred_lin.max()), 3))
    print("predictions below 0:", int(np.sum(pred_lin < 0)))

    # Standardise using training data only; apply the same parameters to test data.
    X_train_s, mu, sigma = standardise_fit(X_train)
    X_test_s = standardise_transform(X_test, mu, sigma)

    # Task 1: sigmoid and log likelihood.
    print("\n--- Task 1: Sigmoid and log-likelihood ---")
    theta_zero = np.zeros(X_train_s.shape[1] + 1)
    ll_zero = log_likelihood(X_train_s, y_train, theta_zero)
    print("log likelihood at theta=0 (total):", round(ll_zero, 6))
    print("log likelihood per training example:", round(ll_zero / len(y_train), 6))
    print("Expected per-example value near:", round(float(-np.log(2)), 6))

    # Task 2: gradient ascent.
    print("\n--- Task 2: Fit by gradient ascent ---")
    theta, history = fit_logistic(X_train_s, y_train, lr=0.1, n_iters=2000)
    print("theta [intercept, standardised income_k, standardised utilisation]:")
    print(theta)
    plt.figure()
    plt.plot(np.arange(1, len(history) + 1), history)
    plt.xlabel("Iteration")
    plt.ylabel("Log-likelihood")
    plt.title("Log-likelihood during gradient ascent")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Task 3: probability and metrics at thresholds 0.5 and 0.3.
    print("\n--- Task 3: Thresholds and evaluation ---")
    p_test = sigmoid(design_matrix(X_test_s) @ theta)
    print("Baseline accuracy (always predict repaid):", round(float(np.mean(y_test == 0)), 4))

    for threshold in (0.5, 0.3):
        y_pred = (p_test >= threshold).astype(int)
        cm = confusion_matrix_scratch(y_test, y_pred)
        precision, recall, f1 = precision_recall_f1_scratch(y_test, y_pred)
        accuracy = accuracy_scratch(y_test, y_pred)
        print(f"\nThreshold = {threshold}")
        print("Confusion matrix [[TN, FP], [FN, TP]]:\n", cm)
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1       : {f1:.4f}")

    # Homework H1: ROC curve and AUC.
    print("\n--- Homework H1: ROC and AUC ---")
    fpr, tpr = roc_curve_scratch(y_test, p_test)
    auc_value = auc_trapezoid(fpr, tpr)
    print(f"AUC: {auc_value:.4f}")
    plot_roc(fpr, tpr, auc_value)

    # Homework H2: decision boundary in original feature units.
    print("\n--- Homework H2: Decision boundary ---")
    plt.figure()
    for label, name in ((0, "Repaid (0)"), (1, "Defaulted (1)")):
        mask = y == label
        plt.scatter(X[mask, 0], X[mask, 1], label=name, alpha=0.7)

    # theta[0] + theta[1]*(income-mu0)/sigma0
    #          + theta[2]*(utilisation-mu1)/sigma1 = 0
    income_line = np.linspace(X[:, 0].min(), X[:, 0].max(), 200)
    if abs(theta[2]) > 1e-12:
        utilisation_line = (
            -theta[0]
            - theta[1] * ((income_line - mu[0]) / sigma[0])
        ) * sigma[1] / theta[2] + mu[1]
        plt.plot(income_line, utilisation_line, label="p(default) = 0.5")
    else:
        print("Cannot draw utilisation boundary: fitted utilisation coefficient is approximately zero.")

    plt.xlabel("Monthly income (thousands of rupees)")
    plt.ylabel("Credit utilisation (fraction)")
    plt.title("Loan data and logistic-regression decision boundary")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return X_train, y_train, X_test, y_test, X_train_s, X_test_s, theta, p_test


def run_homework_h3():
    """
    Optional H3. Requires ic272_lab6_vibration.csv with columns vib_x, vib_y.
    The handout says to use the same split as Lab 6, but that split is not specified
    in the Lab 7 PDF; adapt the split below to match your Lab 6 notebook.
    """
    path = "ic272_lab6_vibration.csv"
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"\nH3 skipped: {path} not found.")
        return

    required = {"vib_x", "vib_y"}
    if not required.issubset(df.columns):
        print(f"H3 skipped: expected columns {sorted(required)} in {path}.")
        return

    # Infer a binary label column only if exactly one suitable column is present.
    feature_cols = ["vib_x", "vib_y"]
    label_candidates = [
        c for c in df.columns
        if c not in feature_cols and df[c].nunique(dropna=True) == 2
    ]
    if len(label_candidates) != 1:
        print("H3 skipped: cannot identify one unambiguous binary label column.")
        print("Set the correct label column and Lab 6 split manually.")
        return

    label_col = label_candidates[0]
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[label_col].to_numpy(dtype=int)

    # Placeholder split; replace with the exact Lab 6 split if different.
    idx = np.random.default_rng(0).permutation(len(y))
    te, tr = idx[: max(1, int(0.2 * len(y)))], idx[max(1, int(0.2 * len(y))):]
    X_train, y_train = X[tr], y[tr]
    X_test, y_test = X[te], y[te]

    X_train_s, mu, sigma = standardise_fit(X_train)
    X_test_s = standardise_transform(X_test, mu, sigma)

    theta, _ = fit_logistic(X_train_s, y_train, lr=0.1, n_iters=2000)
    p = sigmoid(design_matrix(X_test_s) @ theta)
    pred = (p >= 0.5).astype(int)
    print(f"\nH3 scratch logistic test accuracy (label column {label_col!r}): {accuracy_scratch(y_test, pred):.4f}")

    X_train_r2 = np.column_stack((X_train, np.sum(X_train ** 2, axis=1)))
    X_test_r2 = np.column_stack((X_test, np.sum(X_test ** 2, axis=1)))
    X_train_r2_s, mu2, sigma2 = standardise_fit(X_train_r2)
    X_test_r2_s = standardise_transform(X_test_r2, mu2, sigma2)
    theta2, _ = fit_logistic(X_train_r2_s, y_train, lr=0.1, n_iters=2000)
    pred2 = (sigmoid(design_matrix(X_test_r2_s) @ theta2) >= 0.5).astype(int)
    print(f"H3 with r2 feature test accuracy: {accuracy_scratch(y_test, pred2):.4f}")
    print("Compare with your Lab 6 k-NN result using the same exact split.")


def run_homework_h4():
    """Homework H4: compare gradient ascent on raw, unstandardised loan features."""
    df = pd.read_csv("ic272_lab7_loans.csv")
    X = df[["income_k", "utilisation"]].to_numpy(dtype=float)
    y = df["defaulted"].to_numpy(dtype=int)

    idx = np.random.default_rng(0).permutation(len(y))
    te, tr = idx[:100], idx[100:]
    X_train, y_train = X[tr], y[tr]

    plt.figure()
    for lr in (0.001, 0.01):
        _, history = fit_logistic(X_train, y_train, lr=lr, n_iters=2000)
        plt.plot(np.arange(1, len(history) + 1), history, label=f"lr={lr}")
    plt.xlabel("Iteration")
    plt.ylabel("Log-likelihood")
    plt.title("Homework H4: raw features and learning rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_main_lab()
    run_homework_h4()
    # H3 is deliberately not run automatically because its label/split must match Lab 6.
    # To run H3, uncomment the next line after checking the CSV columns and Lab 6 split:
    # run_homework_h3()
