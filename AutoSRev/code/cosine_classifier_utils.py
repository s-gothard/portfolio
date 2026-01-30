
import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize

def apply_cosine_logreg_to_unlabeled(df, clf, vectorizer, train_df=None, test_df=None, text_col="proc_text", prefix="th", threshold=0.4):
    """
    Applies a cosine-normalized logistic regression classifier to unlabeled documents.

    Args:
        df (DataFrame): Full corpus.
        clf (LogisticRegression): Trained classifier.
        vectorizer (TfidfVectorizer): Fitted vectorizer.
        train_df (DataFrame): Optional. Training set to exclude from prediction.
        test_df (DataFrame): Optional. Test set to exclude from prediction.
        text_col (str): Column name for processed text.
        prefix (str): Prefix for score and prediction columns.
        threshold (float): Decision threshold.

    Returns:
        df with new columns: f"{prefix}_prob", f"{prefix}_predicted", "source"
    """
    df = df.copy()

    exclude_pmcs = set()
    if train_df is not None:
        exclude_pmcs.update(train_df["pmcid"])
    if test_df is not None:
        exclude_pmcs.update(test_df["pmcid"])

    df["is_excluded"] = df["pmcid"].isin(exclude_pmcs)
    predict_df = df[~df["is_excluded"]].dropna(subset=[text_col]).copy()
    test_only_df = df[df["pmcid"].isin(getattr(test_df, "pmcid", []))].dropna(subset=[text_col]).copy()

    X_pred = normalize(vectorizer.transform(predict_df[text_col]), norm='l2')
    predict_df[f"{prefix}_prob"] = clf.predict_proba(X_pred)[:, 1]
    predict_df[f"{prefix}_predicted"] = (predict_df[f"{prefix}_prob"] > threshold).astype(int)
    predict_df["source"] = "predicted"

    X_test = normalize(vectorizer.transform(test_only_df[text_col]), norm='l2')
    test_only_df[f"{prefix}_prob"] = clf.predict_proba(X_test)[:, 1]
    test_only_df[f"{prefix}_predicted"] = (test_only_df[f"{prefix}_prob"] > threshold).astype(int)
    test_only_df["source"] = "test"

    final_df = pd.concat([test_only_df, predict_df], axis=0)
    return final_df.drop(columns=["is_excluded"])


def show_top_snippets(df, score_col, label_col="sg_query", top_n=5, char_limit=500):
    print(f"\nTop {top_n} Results for: {score_col}\n" + "-"*50)
    top_docs = df.sort_values(score_col, ascending=False).head(top_n)
    for idx, row in top_docs.iterrows():
        print(f"PMCID: {row['pmcid']}")
        print(f"Title: {row['title']}")
        print(f"Score: {row[score_col]:.8f}")
        print(f"Label: {row.get(label_col, 'N/A')}")
        print("Snippet:")
        print(row.get("raw_text", "")[:char_limit].strip(), "\n" + "-"*80 + "\n")
