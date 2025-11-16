"""main.py
Entry point to run the full Fake News detection pipeline.

Usage example:
    python main.py --data ./data/news.csv --output ./output
"""
import argparse
import os
import sys
import logging
import pandas as pd

from src import preprocessing, embedder, train, evaluate, visualize, utils


def _setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main(args):
    _setup_logging()
    utils.set_seed(42)

    data_path = args.data
    output_dir = args.output

    logging.info(f"Loading dataset from {data_path}")
    df = preprocessing.load_dataset(data_path)

    logging.info("Creating stratified splits")
    # In quick mode use larger test/val fractions to ensure stratified split works on tiny samples
    if getattr(args, "quick", False):
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.stratified_splits(df, test_size=0.34, val_size=0.17, random_state=42)
    else:
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessing.stratified_splits(df, test_size=args.test_size, val_size=args.val_size)

    # Encode embeddings
    logging.info("Encoding train embeddings")
    emb_train = embedder.encode_texts(X_train, batch_size=args.batch_size, quick=args.quick)
    logging.info("Encoding validation embeddings")
    emb_val = embedder.encode_texts(X_val, batch_size=args.batch_size, quick=args.quick)
    logging.info("Encoding test embeddings")
    emb_test = embedder.encode_texts(X_test, batch_size=args.batch_size, quick=args.quick)

    emb_path = os.path.join(output_dir, "embeddings_and_labels.joblib")
    logging.info(f"Saving embeddings to {emb_path}")
    embedder.save_embeddings(emb_path, emb_train, labels=y_train.values)

    # Train embedding-based classifiers
    logging.info("Training embedding-based classifiers")
    models_emb = train.train_embedding_models(emb_train, y_train.values, emb_val, y_val.values, output_dir=output_dir, apply_pca=args.pca, pca_components=args.pca_components, quick=args.quick)

    # Train TF-IDF + NB baseline
    logging.info("Training TF-IDF + MultinomialNB baseline")
    models_nb = train.train_tfidf_nb(X_train.tolist(), y_train.values, X_val.tolist(), y_val.values, output_dir=output_dir, quick=args.quick)

    # Combine models for evaluation: for embedding models we use their best_estimator_ when present
    eval_models = {}
    for k, gs in models_emb.items():
        if hasattr(gs, "best_estimator_"):
            eval_models[k] = gs.best_estimator_
        else:
            eval_models[k] = gs
    for k, gs in models_nb.items():
        if hasattr(gs, "best_estimator_"):
            eval_models[k] = gs.best_estimator_
        else:
            eval_models[k] = gs

    # Save all trained model objects (grids + best estimators) into a single file for reproducibility
    trained_models_path = os.path.join(output_dir, "trained_models.joblib")
    try:
        import joblib
        trained_payload = {
            "grids_embedding": models_emb,
            "grids_nb": models_nb,
            "best_estimators": eval_models,
        }
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(trained_payload, trained_models_path)
        logging.info(f"Saved trained models to {trained_models_path}")
    except Exception as e:
        logging.warning(f"Could not save trained models to {trained_models_path}: {e}")

    # Evaluate on test: embedding models use embeddings, text models use raw text
    logging.info("Evaluating models on test set")
    emb_model_names = [k for k in eval_models.keys() if k not in models_nb]
    text_model_names = [k for k in eval_models.keys() if k in models_nb]

    metrics_parts = []
    if emb_model_names:
        emb_models = {k: eval_models[k] for k in emb_model_names}
        emb_metrics = evaluate.evaluate_models(emb_models, emb_test, y_test.values)
        metrics_parts.append(emb_metrics)
    if text_model_names:
        text_models = {k: eval_models[k] for k in text_model_names}
        text_metrics = evaluate.evaluate_models(text_models, X_test.tolist(), y_test.values)
        metrics_parts.append(text_metrics)

    if metrics_parts:
        metrics_df = metrics_parts[0].copy()
        for part in metrics_parts[1:]:
            metrics_df = pd.concat([metrics_df, part])
    else:
        metrics_df = pd.DataFrame()
    metrics_path = os.path.join(output_dir, "metrics_table.csv")
    evaluate.save_metrics_table(metrics_df, metrics_path)
    logging.info(f"Saved metrics to {metrics_path}")

    # Visualizations
    logging.info("Generating visualizations")
    roc_path = os.path.join(output_dir, "roc_comparison.png")
    # Build per-model test inputs mapping
    test_inputs = {}
    for name in eval_models.keys():
        test_inputs[name] = X_test.tolist() if name in models_nb else emb_test
    visualize.plot_roc_comparison(eval_models, test_inputs, y_test.values, roc_path)

    # Choose best model by F1
    best_model_name = metrics_df["f1"].idxmax()
    best_model = eval_models[best_model_name]
    cm_path = os.path.join(output_dir, "confusion_matrix_best.png")
    best_X = X_test.tolist() if best_model_name in models_nb else emb_test
    visualize.plot_confusion_matrix(best_model, best_X, y_test.values, cm_path)

    bar_path = os.path.join(output_dir, "accuracy_f1_comparison.png")
    visualize.plot_accuracy_f1(metrics_df, bar_path)

    logging.info("Pipeline finished. Summary:")
    print(metrics_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake News Detection pipeline")
    parser.add_argument("--data", type=str, required=True, help="Path to data CSV or folder")
    parser.add_argument("--output", type=str, default="./output", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size")
    parser.add_argument("--pca", action="store_true", help="Apply PCA before classifiers")
    parser.add_argument("--pca-components", type=int, default=50, help="PCA components if enabled")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set proportion")
    parser.add_argument("--val-size", type=float, default=0.1, help="Validation set proportion")
    parser.add_argument("--nb-only-text", action="store_true", help="Evaluation for NB uses raw text features (internal switch)")
    parser.add_argument("--quick", action="store_true", help="Quick dev mode: smaller grids and deterministic fake embeddings")
    args = parser.parse_args()
    main(args)
