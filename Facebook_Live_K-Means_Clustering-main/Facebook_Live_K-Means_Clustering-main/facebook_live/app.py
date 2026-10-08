import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Facebook Live K-Means Clustering",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Facebook Live Posts - K-Means Clustering")
st.write(
    "This application uses **K-Means Clustering** to group Facebook Live "
    "posts according to their engagement characteristics."
)


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Live.csv")


try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ Live.csv not found. Please keep Live.csv in the same folder as app.py.")
    st.stop()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("⚙️ Settings")

st.sidebar.write("Dataset Information")
st.sidebar.write(f"Rows: {df.shape[0]}")
st.sidebar.write(f"Columns: {df.shape[1]}")


# ---------------------------------------------------------
# DATASET PREVIEW
# ---------------------------------------------------------
st.header("1️⃣ Dataset Overview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Posts", df.shape[0])

with col2:
    st.metric("Total Columns", df.shape[1])

with col3:
    st.metric(
        "Post Types",
        df["status_type"].nunique() if "status_type" in df.columns else 0
    )

st.subheader("Dataset Preview")
st.dataframe(df.head(10), use_container_width=True)


# ---------------------------------------------------------
# MISSING VALUES
# ---------------------------------------------------------
st.subheader("Missing Values")

missing = df.isnull().sum()

missing_df = pd.DataFrame({
    "Column": missing.index,
    "Missing Values": missing.values
})

st.dataframe(missing_df, use_container_width=True)


# ---------------------------------------------------------
# POST TYPE DISTRIBUTION
# ---------------------------------------------------------
st.subheader("Post Type Distribution")

if "status_type" in df.columns:

    counts = df["status_type"].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(counts.index, counts.values)

    ax.set_title("Distribution of Facebook Post Types")
    ax.set_xlabel("Status Type")
    ax.set_ylabel("Number of Posts")

    plt.xticks(rotation=45)
    plt.tight_layout()

    st.pyplot(fig)


# ---------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------
st.header("2️⃣ Data Preparation")

drop_cols = [
    "status_id",
    "status_published",
    "Column1",
    "Column2",
    "Column3",
    "Column4"
]

data = df.copy()

# Drop only columns that actually exist
existing_drop_cols = [col for col in drop_cols if col in data.columns]

data.drop(
    columns=existing_drop_cols,
    inplace=True
)


# ---------------------------------------------------------
# SELECT FEATURES
# ---------------------------------------------------------
features = [
    "num_reactions",
    "num_comments",
    "num_shares",
    "num_likes",
    "num_loves",
    "num_wows",
    "num_hahas",
    "num_sads",
    "num_angrys"
]

# Check features
missing_features = [
    feature for feature in features
    if feature not in data.columns
]

if missing_features:
    st.error(
        f"❌ Required features are missing: {missing_features}"
    )
    st.stop()


st.write("### Features used for clustering")

st.write(features)


X = data[features].copy()


# ---------------------------------------------------------
# HANDLE MISSING VALUES
# ---------------------------------------------------------
X = X.fillna(X.median())


# ---------------------------------------------------------
# LOG TRANSFORMATION
# ---------------------------------------------------------
st.header("3️⃣ Feature Transformation")

st.write(
    "Engagement values are highly skewed. Therefore, "
    "`log1p()` transformation is applied before scaling."
)

X_log = np.log1p(X)


# ---------------------------------------------------------
# STANDARDIZATION
# ---------------------------------------------------------
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_log)

X_scaled = pd.DataFrame(
    X_scaled,
    columns=features
)

st.write("### Scaled Feature Data")

st.dataframe(
    X_scaled.head(),
    use_container_width=True
)


# ---------------------------------------------------------
# ELBOW METHOD
# ---------------------------------------------------------
st.header("4️⃣ Elbow Method")

k_values = range(2, 11)

inertia = []

for k in k_values:

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    model.fit(X_scaled)

    inertia.append(model.inertia_)


fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    list(k_values),
    inertia,
    marker="o"
)

ax.set_title("Elbow Method for Selecting K")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Inertia")
ax.set_xticks(list(k_values))

plt.tight_layout()

st.pyplot(fig)


# ---------------------------------------------------------
# SILHOUETTE SCORE
# ---------------------------------------------------------
st.header("5️⃣ Silhouette Score")

silhouette_scores = {}

for k in k_values:

    model = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    silhouette_scores[k] = silhouette_score(
        X_scaled,
        labels
    )


fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(
    list(silhouette_scores.keys()),
    list(silhouette_scores.values()),
    marker="o"
)

ax.set_title("Silhouette Score for Different K Values")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Silhouette Score")
ax.set_xticks(list(k_values))

plt.tight_layout()

st.pyplot(fig)


# ---------------------------------------------------------
# OPTIMAL K
# ---------------------------------------------------------
optimal_k = max(
    silhouette_scores,
    key=silhouette_scores.get
)

best_silhouette = silhouette_scores[optimal_k]


col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Optimal Number of Clusters",
        optimal_k
    )

with col2:
    st.metric(
        "Best Silhouette Score",
        round(best_silhouette, 4)
    )


st.success(
    f"✅ Optimal K based on the highest silhouette score = {optimal_k}"
)


# ---------------------------------------------------------
# K-MEANS MODEL
# ---------------------------------------------------------
st.header("6️⃣ Final K-Means Clustering")

kmeans = KMeans(
    n_clusters=optimal_k,
    init="k-means++",
    n_init=10,
    random_state=42
)

cluster_labels = kmeans.fit_predict(X_scaled)


# ---------------------------------------------------------
# ADD CLUSTERS
# ---------------------------------------------------------
result = data.copy()

result["Cluster"] = cluster_labels


# ---------------------------------------------------------
# FINAL MODEL RESULTS
# ---------------------------------------------------------
final_inertia = kmeans.inertia_

final_silhouette = silhouette_score(
    X_scaled,
    cluster_labels
)

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "Final Inertia",
        round(final_inertia, 2)
    )

with col2:
    st.metric(
        "Final Silhouette Score",
        round(final_silhouette, 4)
    )


# ---------------------------------------------------------
# CLUSTER COUNTS
# ---------------------------------------------------------
st.header("7️⃣ Cluster Distribution")

cluster_counts = (
    result["Cluster"]
    .value_counts()
    .sort_index()
)

cluster_count_df = pd.DataFrame({
    "Cluster": cluster_counts.index,
    "Number of Posts": cluster_counts.values
})

st.dataframe(
    cluster_count_df,
    use_container_width=True
)


fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

ax.set_title("Number of Posts in Each Cluster")
ax.set_xlabel("Cluster")
ax.set_ylabel("Number of Posts")

plt.tight_layout()

st.pyplot(fig)


# ---------------------------------------------------------
# CLUSTER PROFILE
# ---------------------------------------------------------
st.header("8️⃣ Cluster Profile")

cluster_profile = (
    result
    .groupby("Cluster")[features]
    .mean()
    .round(2)
)

st.write("### Mean Engagement Values")

st.dataframe(
    cluster_profile,
    use_container_width=True
)


# ---------------------------------------------------------
# MEDIAN PROFILE
# ---------------------------------------------------------
st.write("### Median Engagement Values")

cluster_median = (
    result
    .groupby("Cluster")[features]
    .median()
    .round(2)
)

st.dataframe(
    cluster_median,
    use_container_width=True
)


# ---------------------------------------------------------
# PCA
# ---------------------------------------------------------
st.header("9️⃣ PCA Visualization")

pca = PCA(
    n_components=2,
    random_state=42
)

X_pca = pca.fit_transform(X_scaled)


pca_df = pd.DataFrame({
    "PC1": X_pca[:, 0],
    "PC2": X_pca[:, 1],
    "Cluster": cluster_labels
})


explained_1 = pca.explained_variance_ratio_[0]
explained_2 = pca.explained_variance_ratio_[1]
total_explained = explained_1 + explained_2


col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "PC1 Variance",
        f"{explained_1:.2%}"
    )

with col2:
    st.metric(
        "PC2 Variance",
        f"{explained_2:.2%}"
    )

with col3:
    st.metric(
        "Total Variance",
        f"{total_explained:.2%}"
    )


# PCA scatter plot
fig, ax = plt.subplots(figsize=(9, 6))

for cluster in sorted(pca_df["Cluster"].unique()):

    cluster_data = pca_df[
        pca_df["Cluster"] == cluster
    ]

    ax.scatter(
        cluster_data["PC1"],
        cluster_data["PC2"],
        label=f"Cluster {cluster}",
        alpha=0.6
    )


ax.set_title("K-Means Clusters Visualized Using PCA")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.legend()

plt.tight_layout()

st.pyplot(fig)


# ---------------------------------------------------------
# STATUS TYPE INTERPRETATION
# ---------------------------------------------------------
st.header("🔟 Post-Type Composition by Cluster")

if "status_type" in result.columns:

    cluster_status = pd.crosstab(
        result["Cluster"],
        result["status_type"],
        normalize="index"
    ).round(3)

    st.dataframe(
        cluster_status,
        use_container_width=True
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    cluster_status.plot(
        kind="bar",
        stacked=True,
        ax=ax
    )

    ax.set_title(
        "Post-Type Composition of Each Cluster"
    )

    ax.set_xlabel("Cluster")
    ax.set_ylabel("Proportion")

    plt.xticks(rotation=0)
    plt.tight_layout()

    st.pyplot(fig)


# ---------------------------------------------------------
# CLUSTER INTERPRETATION
# ---------------------------------------------------------
st.header("📌 Cluster Interpretation")

st.write(
    """
    The cluster numbers themselves do not have a fixed meaning.
    They should be interpreted using their engagement profiles.
    A cluster having high values of reactions, comments and shares
    can be considered a high-engagement group, while a cluster
    with lower engagement values can be considered a low-engagement group.
    """
)


# ---------------------------------------------------------
# SHOW FINAL DATA
# ---------------------------------------------------------
st.header("📋 Final Clustered Dataset")

st.dataframe(
    result.head(100),
    use_container_width=True
)


# ---------------------------------------------------------
# DOWNLOAD RESULT
# ---------------------------------------------------------
csv_output = result.to_csv(index=False).encode("utf-8")

st.download_button(
    label="⬇️ Download Clustered Dataset",
    data=csv_output,
    file_name="Facebook_Live_KMeans_Results.csv",
    mime="text/csv"
)


# ---------------------------------------------------------
# CONCLUSION
# ---------------------------------------------------------
st.header("✅ Conclusion")

st.write(
    f"""
    The Facebook Live dataset was successfully clustered using
    the K-Means algorithm. Engagement features such as reactions,
    comments, shares and different reaction types were used.

    Log transformation and standardization were applied to improve
    clustering performance.

    Based on the silhouette score, the selected number of clusters
    is **{optimal_k}**.

    The clusters were visualized using PCA, and cluster profiles
    were generated to understand the engagement characteristics
    of different groups of Facebook posts.
    """
)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")

st.caption(
    "Facebook Live K-Means Clustering | Streamlit Machine Learning Project"
)