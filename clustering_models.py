import joblib
import numpy as np

import joblib
import numpy as np
import pandas as pd

# Load models and data
vectorizer = joblib.load("tfidf.pkl")
svd = joblib.load("svd50.pkl")
birch_model = joblib.load("birch_model.pkl")
X_reduced = np.load("X_reduced.npy")
X = joblib.load("tfidf_matrix.pkl")

df = pd.read_csv("clustered_data.csv")  # or combined_cleaned_social_media.csv if needed



def predict_cluster(new_texts):
    """Predict cluster for new posts"""
    X_new = vectorizer.transform(new_texts)
    X_new_reduced = svd.transform(X_new)
    labels = birch_model.predict(X_new_reduced)
    return labels

def get_top_words(cluster_label, top_n=10):
    """Get top words of a cluster"""
    import pandas as pd
    df = pd.read_csv("clustered_data.csv")
    feature_names = vectorizer.get_feature_names_out()
    cluster_idx = df[df['cluster'] == cluster_label].index
    cluster_vec_sum = X[cluster_idx].sum(axis=0).A1
    top_idx = cluster_vec_sum.argsort()[-top_n:][::-1]
    return [feature_names[i] for i in top_idx]

def sample_posts(cluster_label, n=5):
    """Get sample posts from a cluster"""
    import pandas as pd
    df = pd.read_csv("clustered_data.csv")
    cluster_df = df[df['cluster'] == cluster_label]
    return cluster_df['clean_text'].sample(min(n, len(cluster_df)), random_state=42).tolist()