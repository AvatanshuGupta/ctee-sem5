import pickle
import argparse
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])


class Node:
    def __init__(
        self, feature=None, threshold=None, left=None, right=None, *, value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_feats=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        # greedily select the best split according to information gain
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

        # grow the children that result from the split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        # parent loss
        parent_entropy = entropy(y)

        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # compute the weighted avg. of the loss for the children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
        



def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, n_samples, replace=True)
    return X[idxs], y[idxs]


def most_common_label(y):
    counter = Counter(y)
    most_common = counter.most_common(1)[0][0]
    return most_common



class RandomForest:
    def __init__(self, n_trees=10, min_samples_split=2, max_depth=100, n_feats=None):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_feats = n_feats
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_feats=self.n_feats,
            )
            X_samp, y_samp = bootstrap_sample(X, y)
            tree.fit(X_samp, y_samp)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred)



LABEL_COLS = ["behavior", "composition"]
SPLITS = ["train", "calibration", "test_noisy", "test_clean"]
SHORTCUT_COLS = ["snr_v_db", "snr_i_db", "f_est"]  
NAN_FILL = -1e9 


def load_splits(data_dir, drop_shortcut):
    data_dir = Path(data_dir)
    dfs = {}
    for name in SPLITS:
        path = data_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Use --data-dir to point to dataset/features")
        dfs[name] = pd.read_csv(path)

    feat_cols = [c for c in dfs["train"].columns if c not in LABEL_COLS]
    if drop_shortcut:
        feat_cols = [c for c in feat_cols if c not in SHORTCUT_COLS]
    return dfs, feat_cols


def to_matrix(df, feat_cols):
    X = df[feat_cols].to_numpy(dtype=np.float64)
    X = np.nan_to_num(X, nan=NAN_FILL, posinf=1e12, neginf=-1e12)
    return X


def accuracy(y_true, y_pred):
    return float(np.mean(y_true == y_pred))


def macro_f1(y_true, y_pred, n_classes):
    f1s = []
    for c in range(n_classes):
        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s))


def confusion(y_true, y_pred, classes):
    return pd.crosstab(
        pd.Series(classes[y_true], name="true"),
        pd.Series(classes[y_pred], name="pred"),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="dataset/")
    ap.add_argument("--n-trees", type=int, default=25)
    ap.add_argument("--max-depth", type=int, default=15)
    ap.add_argument("--min-samples-split", type=int, default=5)
    ap.add_argument(
        "--n-feats",
        type=int,
        default=None,
        help="features tried per split (default: sqrt(n_features)"
    )
    ap.add_argument(
        "--max-thresholds",
        type=int,
        default=32,
        help="candidate thresholds per feature (0 = try all unique values, very slow)"
    )
    ap.add_argument(
        "--drop-shortcut",
        action="store_true",
        help="drop snr_v_db, snr_i_db, f_est"
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default="rf_results")
    args = ap.parse_args()


    np.random.seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # Load datasets
    dfs, feat_cols = load_splits(
        args.data_dir,
        args.drop_shortcut
    )

    n_feats = args.n_feats or int(np.sqrt(len(feat_cols)))

    print(
        f"Features used: {len(feat_cols)}   "
        f"features per split: {n_feats}"
    )

    print(
        "Rows: " +
        ", ".join(
            f"{k}={len(v)}"
            for k, v in dfs.items()
        )
    )

    # Convert datasets to matrices
    X = {
        name: to_matrix(df, feat_cols)
        for name, df in dfs.items()
    }

    results = []

    preds = {}
    truth = {}
    class_names = {}

    # This will store both trained Random Forests
    trained_models = {}

    # This will store the label encoding information
    label_mappings = {}

    for label in LABEL_COLS:

        # Get classes from training data
        classes = np.array(
            sorted(dfs["train"][label].unique())
        )

        class_names[label] = classes

        # Convert original labels → integer labels
        mapping = {
            c: i
            for i, c in enumerate(classes)
        }

        label_mappings[label] = {
            "classes": classes.tolist(),
            "mapping": mapping
        }

        y = {
            name: df[label]
                .map(mapping)
                .to_numpy(dtype=int)
            for name, df in dfs.items()
        }

        print(
            f"\n=== Training head: {label} "
            f"({len(classes)} classes: {list(classes)}) ==="
        )


        rf = RandomForest(
            n_trees=args.n_trees,
            min_samples_split=args.min_samples_split,
            max_depth=args.max_depth,
            n_feats=n_feats,
            
        )

        t0 = time.time()

        rf.fit(
            X["train"],
            y["train"]
        )

        print(
            f"  training time: "
            f"{time.time() - t0:.1f}s"
        )

        trained_models[label] = rf

        for split in SPLITS:

            p = rf.predict(X[split])

            preds.setdefault(
                split, {}
            )[label] = p

            truth.setdefault(
                split, {}
            )[label] = y[split]

            results.append({
                "head": label,
                "split": split,
                "accuracy": accuracy(
                    y[split],
                    p
                ),
                "macro_f1": macro_f1(
                    y[split],
                    p,
                    len(classes)
                ),
            })



    for split in SPLITS:

        ok = np.ones(
            len(dfs[split]),
            dtype=bool
        )

        for label in LABEL_COLS:

            ok &= (
                preds[split][label]
                == truth[split][label]
            )

        results.append({
            "head": "joint (21-class)",
            "split": split,
            "accuracy": float(ok.mean()),
            "macro_f1": np.nan
        })


    res = pd.DataFrame(results)

    summary = (
        res
        .pivot(
            index="head",
            columns="split",
            values="accuracy"
        )[SPLITS]
    )

    print(
        "\nACCURACY"
    )

    print(
        summary
        .round(4)
        .to_string()
    )

    print(
        "\nMACRO-F1"
    )

    f1 = (
        res
        .dropna(subset=["macro_f1"])
        .pivot(
            index="head",
            columns="split",
            values="macro_f1"
        )[SPLITS]
    )

    print(
        f1
        .round(4)
        .to_string()
    )



    for split in ["test_noisy", "test_clean"]:

        for label in LABEL_COLS:

            cm = confusion(
                truth[split][label],
                preds[split][label],
                class_names[label]
            )

            print(
                f"\nConfusion matrix - {label} "
                f"on {split} "
                f"(rows = true, cols = predicted)"
            )

            print(cm.to_string())

            cm.to_csv(
                out_dir /
                f"confusion_{label}_{split}.csv"
            )


    res.to_csv(
        out_dir / "metrics.csv",
        index=False
    )


    model_artifact = {
        "models": trained_models,

        # Exact feature order used during training
        "feature_columns": feat_cols,

        # Label encoding information
        "label_mappings": label_mappings,

        # Same NaN handling used during training
        "nan_fill": NAN_FILL,

        # Number of features considered at each split
        "n_feats": n_feats,

        # Training parameters
        "parameters": {
            "n_trees": args.n_trees,
            "max_depth": args.max_depth,
            "min_samples_split": args.min_samples_split,
            "seed": args.seed,
        }
    }

    # Save directly in project root
    model_path = Path("random_forest_model.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(
            model_artifact,
            f
        )

    print(
        f"\nModel dumped successfully to:"
        f"\n{model_path.resolve()}"
    )

    print(
        f"\nSaved metrics and confusion matrices to:"
        f"\n{out_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
