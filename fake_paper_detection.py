import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import umap
from sentence_transformers import SentenceTransformer

# 1. Config
os.environ["TOKENIZERS_PARALLELISM"] = "false"
RANDOM_STATE = 42

# -------------------------
# Ablation Configuration
# -------------------------
# TEXT_MODE = "title"
TEXT_MODE = "title_abstract"
# "title" or "title_abstract"

# -------------------------
# Embedding Configuration
# -------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
# Alternative:


def load_clean_data():
    print("Loading datasets...")
    arxiv_texts = []
    with open("arxiv_cs.json", "r") as f:
        for line in f:
            p = json.loads(line)
            # Prefer combined title+abstract text; fallback to title for safety
            if TEXT_MODE == "title":
                arxiv_texts.append(
                    p.get("title", "").replace("\n", " ").strip()
                )
            else:  # title + abstract
                arxiv_texts.append(
                    p.get("text", p.get("title", "")).replace("\n", " ").strip()
                )

    df_norm = pd.DataFrame({"text": arxiv_texts, "label": 0})

    rw = pd.read_csv("retraction_watch_cs.csv")
    ret_texts = (
        rw["Title"]
        .fillna("")
        .astype(str)
        .str.replace(r"(?i)retracted:", "", regex=True)
        .str.strip()
    )

    df_ret = pd.DataFrame({"text": ret_texts, "label": 1})

    return df_norm, df_ret


def experiment_iforest_sensitivity(X_train, X_test, y_test):
    print("\nRunning Isolation Forest Sensitivity Analysis (n_estimators)...")
    estimator_settings = [50, 100, 200, 400, 600]
    results = []

    for n in estimator_settings:
        model = IsolationForest(n_estimators=n, contamination=0.5, random_state=RANDOM_STATE)
        model.fit(X_train)
        scores = -model.score_samples(X_test)
        roc_auc = auc(*roc_curve(y_test, scores)[:2])
        ap = average_precision_score(y_test, scores)
        results.append((n, roc_auc, ap))
        print(f"iForest (n={n}): AUC={roc_auc:.3f}, AP={ap:.3f}")

    ns, aucs, aps = zip(*results)
    plt.figure(figsize=(7, 5))
    plt.plot(ns, aucs, marker="o", color='teal', label="ROC-AUC")
    plt.plot(ns, aps, marker="s", color='orange', label="Avg Precision")
    plt.xlabel("Number of Estimators (Trees)")
    plt.ylabel("Score")
    plt.title("Isolation Forest Sensitivity to Tree Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("iforest_sensitivity.png")


def experiment_lof_sensitivity(X_train, X_test, y_test):
    print("\nRunning LOF Sensitivity Analysis (k-neighbors)...")
    neighbor_settings = [10, 20, 35, 50, 75]
    results = []

    for k in neighbor_settings:
        lof = LocalOutlierFactor(n_neighbors=k, novelty=True)
        lof.fit(X_train)
        scores = -lof.decision_function(X_test)
        roc_auc = auc(*roc_curve(y_test, scores)[:2])
        ap = average_precision_score(y_test, scores)
        results.append((k, roc_auc, ap))
        print(f"LOF (k={k}): AUC={roc_auc:.3f}, AP={ap:.3f}")

    ks, aucs, aps = zip(*results)
    plt.figure(figsize=(7, 5))
    plt.plot(ks, aucs, marker="o", color='blue', label="ROC-AUC")
    plt.plot(ks, aps, marker="s", color='red', label="Avg Precision")
    plt.xlabel("Number of Neighbors (k)")
    plt.ylabel("Score")
    plt.title("LOF Sensitivity to Neighborhood Size")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("lof_sensitivity.png")


def generate_interactive_umap(X, y, texts=None,
                              output_path="figures/umap_semantic_manifold_interactive.html"):
    import os
    import pandas as pd
    import plotly.express as px

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=RANDOM_STATE
    )

    X_umap = reducer.fit_transform(X)

    df_plot = pd.DataFrame({
        "UMAP-1": X_umap[:, 0],
        "UMAP-2": X_umap[:, 1],
        "Label": y.astype(str)
    })

    if texts is not None:
        df_plot["Title"] = texts

    fig = px.scatter(
        df_plot,
        x="UMAP-1",
        y="UMAP-2",
        color="Label",
        hover_data=["Title"] if texts is not None else None,
        title="Interactive UMAP: Semantic Manifold of Scientific Papers",
        opacity=0.6
    )

    # Save locally as HTML (NO fig.show())
    fig.write_html(output_path, include_plotlyjs="cdn")

    print(f"[SUCCESS] Interactive UMAP saved to {output_path}")



def main():
    df_norm, df_ret = load_clean_data()

    # 2. Sampling
    n_test_each = min(len(df_ret) // 2, 300)
    test_norm = df_norm.sample(n=n_test_each, random_state=RANDOM_STATE)
    train_norm = df_norm.drop(test_norm.index).sample(
    n=1200,
    random_state=RANDOM_STATE
    )

    test_ret = df_ret.sample(n=n_test_each, random_state=RANDOM_STATE)
    # Balanced test set used for evaluation stability, not real-world prevalence
    test_df = pd.concat([test_norm, test_ret]).reset_index(drop=True)
    y_test = test_df["label"].values

    print(f"Using embedding model: {EMBEDDING_MODEL}")
    print(f"Text mode: {TEXT_MODE}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    # 3. Embeddings & Scaling
    print("Generating embeddings...")

    X_train_raw = embedder.encode(train_norm["text"].tolist(), show_progress_bar=True)
    X_test_raw = embedder.encode(test_df["text"].tolist(), show_progress_bar=True)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    # 4. Sensitivity Experiments
    experiment_iforest_sensitivity(X_train, X_test, y_test)
    experiment_lof_sensitivity(X_train, X_test, y_test)

    # 5. Final Evaluation (SEPARATE FIGURES)
    models = {
        "Isolation Forest": IsolationForest(
            n_estimators=200, contamination=0.5, random_state=RANDOM_STATE
        ),
        "Local Outlier Factor": LocalOutlierFactor(
            n_neighbors=75, novelty=True
        )
    }

    roc_fig, roc_ax = plt.subplots(figsize=(7, 6))
    pr_fig, pr_ax = plt.subplots(figsize=(7, 6))
    score_fig, score_ax = plt.subplots(figsize=(7, 6))

    for name, clf in models.items():
        print(f"Final Evaluation for {name}...")
        # Train only on non-retracted arXiv papers (labels NOT used)
        clf.fit(X_train)

        scores = (
            -clf.decision_function(X_test)
            if "Local" in name
            else -clf.score_samples(X_test)
        )

        test_df[f"score_{name.replace(' ', '_')}"] = scores

        # ROC
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_ax.plot(fpr, tpr, lw=2, label=f'{name} (AUC={auc(fpr, tpr):.2f})')

        # Precision–Recall
        prec, rec, _ = precision_recall_curve(y_test, scores)
        pr_ax.plot(rec, prec, lw=2,
                   label=f'{name} (AP={average_precision_score(y_test, scores):.2f})')

        # Score distribution
        sns.kdeplot(scores[y_test == 0], ax=score_ax,
                    label=f"{name} - Normal", fill=True, alpha=0.25)
        sns.kdeplot(scores[y_test == 1], ax=score_ax,
                    label=f"{name} - Retracted", fill=True, alpha=0.25)

    # --- ROC Styling ---
    roc_ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    roc_ax.set_title("ROC Curve Comparison")
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.legend()
    roc_fig.tight_layout()
    roc_fig.savefig(f"roc_comparison_{TEXT_MODE}_{EMBEDDING_MODEL}.png")


    # --- PR Styling ---
    pr_ax.set_title("Precision–Recall Curve Comparison")
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.legend()
    pr_fig.tight_layout()
    pr_fig.savefig(f"pr_comparison_{TEXT_MODE}_{EMBEDDING_MODEL}.png")

    # --- Score Distribution Styling ---
    score_ax.set_title("Anomaly Score Distribution")
    score_ax.set_xlabel("Anomaly Score")
    score_ax.legend()
    score_fig.tight_layout()
    score_fig.savefig(f"score_distribution_{TEXT_MODE}_{EMBEDDING_MODEL}.png")

    # Save error analysis table
    test_df["embedding_model"] = EMBEDDING_MODEL
    test_df["text_mode"] = TEXT_MODE
    test_df.to_csv(f"error_analysis_results_{TEXT_MODE}_{EMBEDDING_MODEL}.csv", index=False)

    # 6. UMAP Semantic Manifold Visualization
    from sklearn.metrics import silhouette_score

    sil_score = silhouette_score(X_test, y_test)

    plt.figure(figsize=(9, 7))

    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=RANDOM_STATE
    )

    X_umap = reducer.fit_transform(X_test)

    sns.scatterplot(
        x=X_umap[:, 0],
        y=X_umap[:, 1],
        hue=y_test,
        palette={0: "blue", 1: "red"},
        alpha=0.5
    )

    plt.title(
        f"UMAP Semantic Manifold of Scientific Papers\n"
        f"(Silhouette Score = {sil_score:.3f})"
    )
    plt.xlabel("UMAP-1")
    plt.ylabel("UMAP-2")
    plt.legend(title="Label")
    plt.tight_layout()
    plt.savefig("umap_semantic_manifold.png")

    generate_interactive_umap(
        X=X_test,
        y=y_test,
        texts=test_df["text"].tolist()
    )

    print("\n[SUCCESS] Separate figures generated for paper-ready results.")


if __name__ == "__main__":
    main()
