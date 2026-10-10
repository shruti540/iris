import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from scipy.stats import mode


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Iris K-Means Clustering",
    page_icon="🌸",
    layout="wide"
)

st.title("🌸 Iris Dataset - K-Means Clustering")
st.write("Interactive Machine Learning application using K-Means Clustering")


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

@st.cache_data
def load_data():
    return pd.read_csv("Iris.csv")


try:
    df = load_data()
except FileNotFoundError:
    st.error("Iris.csv file not found. Please keep Iris.csv in the same folder as app.py.")
    st.stop()


# ---------------------------------------------------------
# DATASET INFORMATION
# ---------------------------------------------------------

st.sidebar.header("⚙️ Settings")

st.sidebar.success("Dataset loaded successfully")

st.sidebar.write("### Dataset Information")
st.sidebar.write(f"Rows: {df.shape[0]}")
st.sidebar.write(f"Columns: {df.shape[1]}")


# Remove Id from clustering
features = [
    "SepalLengthCm",
    "SepalWidthCm",
    "PetalLengthCm",
    "PetalWidthCm"
]

X = df[features].copy()

# Species labels are used only for evaluation
y = df["Species"].astype("category").cat.codes
class_names = list(df["Species"].astype("category").cat.categories)


# ---------------------------------------------------------
# SIDEBAR K VALUE
# ---------------------------------------------------------

k = st.sidebar.slider(
    "Select Number of Clusters (K)",
    min_value=2,
    max_value=6,
    value=3
)


# ---------------------------------------------------------
# DATASET PREVIEW
# ---------------------------------------------------------

st.header("📊 Dataset Preview")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Samples", len(df))

with col2:
    st.metric("Features", len(features))

with col3:
    st.metric("Selected K", k)

st.dataframe(df.head(10), use_container_width=True)


# ---------------------------------------------------------
# DATASET STATISTICS
# ---------------------------------------------------------

with st.expander("🔍 Dataset Statistics"):

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Missing Values")
        st.dataframe(
            df.isnull().sum().to_frame("Missing Values"),
            use_container_width=True
        )

    with col2:
        st.subheader("Species Distribution")
        st.dataframe(
            df["Species"].value_counts().to_frame("Count"),
            use_container_width=True
        )

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)


# ---------------------------------------------------------
# STANDARDIZATION
# ---------------------------------------------------------

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ---------------------------------------------------------
# ELBOW METHOD
# ---------------------------------------------------------

st.header("📈 Elbow Method")

k_values = range(2, 7)
inertia = []

for i in k_values:

    model = KMeans(
        n_clusters=i,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    model.fit(X_scaled)
    inertia.append(model.inertia_)


fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    list(k_values),
    inertia,
    marker="o"
)

ax.set_title("Elbow Method")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Inertia")
ax.set_xticks(list(k_values))
ax.grid(True)

st.pyplot(fig)


# ---------------------------------------------------------
# SILHOUETTE SCORE
# ---------------------------------------------------------

st.header("📊 Silhouette Score")

silhouette_scores = {}

for i in k_values:

    model = KMeans(
        n_clusters=i,
        init="k-means++",
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(X_scaled)

    silhouette_scores[i] = silhouette_score(
        X_scaled,
        labels
    )


fig, ax = plt.subplots(figsize=(8, 4))

ax.plot(
    list(silhouette_scores.keys()),
    list(silhouette_scores.values()),
    marker="o"
)

ax.set_title("Silhouette Score for Different K Values")
ax.set_xlabel("Number of Clusters (K)")
ax.set_ylabel("Silhouette Score")
ax.set_xticks(list(k_values))
ax.grid(True)

st.pyplot(fig)


# ---------------------------------------------------------
# DISPLAY SILHOUETTE TABLE
# ---------------------------------------------------------

silhouette_table = pd.DataFrame({
    "K": list(silhouette_scores.keys()),
    "Silhouette Score": [
        round(value, 4)
        for value in silhouette_scores.values()
    ]
})

st.dataframe(
    silhouette_table,
    use_container_width=True
)


# ---------------------------------------------------------
# K-MEANS MODEL
# ---------------------------------------------------------

st.header("🤖 K-Means Clustering")

kmeans = KMeans(
    n_clusters=k,
    init="k-means++",
    n_init=10,
    random_state=42
)

cluster_labels = kmeans.fit_predict(X_scaled)


# ---------------------------------------------------------
# SILHOUETTE SCORE FOR SELECTED K
# ---------------------------------------------------------

final_silhouette = silhouette_score(
    X_scaled,
    cluster_labels
)

st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Number of Clusters",
        k
    )

with col2:
    st.metric(
        "Silhouette Score",
        round(final_silhouette, 4)
    )

with col3:
    st.metric(
        "Inertia",
        round(kmeans.inertia_, 2)
    )


# ---------------------------------------------------------
# MAP CLUSTERS TO SPECIES
# ---------------------------------------------------------

mapped_labels = np.zeros_like(cluster_labels)

for cluster in range(k):

    mask = cluster_labels == cluster

    if np.sum(mask) > 0:

        mapped_labels[mask] = mode(
            y[mask],
            keepdims=True
        ).mode[0]


accuracy = accuracy_score(
    y,
    mapped_labels
)


# ---------------------------------------------------------
# ACCURACY
# ---------------------------------------------------------

st.subheader("🎯 Clustering Accuracy")

st.metric(
    "Accuracy",
    f"{accuracy * 100:.2f}%"
)

st.info(
    "Note: K-Means is an unsupervised algorithm. "
    "The accuracy shown here is calculated only by mapping "
    "clusters to the majority Iris species for evaluation."
)


# ---------------------------------------------------------
# CLUSTER RESULTS
# ---------------------------------------------------------

result = df.copy()

result["Cluster"] = cluster_labels

result["Mapped_Species"] = [
    class_names[i]
    for i in mapped_labels
]


st.header("📋 Clustering Results")

st.dataframe(
    result.head(20),
    use_container_width=True
)


# ---------------------------------------------------------
# CLUSTER COUNTS
# ---------------------------------------------------------

st.header("📌 Samples in Each Cluster")

cluster_counts = (
    result["Cluster"]
    .value_counts()
    .sort_index()
)

fig, ax = plt.subplots(figsize=(8, 4))

ax.bar(
    cluster_counts.index.astype(str),
    cluster_counts.values
)

ax.set_title("Number of Samples in Each Cluster")
ax.set_xlabel("Cluster")
ax.set_ylabel("Number of Samples")

st.pyplot(fig)


# ---------------------------------------------------------
# PCA VISUALIZATION
# ---------------------------------------------------------

st.header("🎨 PCA Visualization of Clusters")

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(
    X_pca,
    columns=["PC1", "PC2"]
)

pca_df["Cluster"] = cluster_labels
pca_df["Species"] = df["Species"].values


fig, ax = plt.subplots(figsize=(9, 6))

sns.scatterplot(
    data=pca_df,
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="tab10",
    s=80,
    ax=ax
)

ax.set_title(
    f"K-Means Clusters Visualized Using PCA (K = {k})"
)

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")

st.pyplot(fig)


# ---------------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------------

st.header("🔲 Confusion Matrix")

cm = confusion_matrix(
    y,
    mapped_labels
)

fig, ax = plt.subplots(figsize=(7, 5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_names,
    yticklabels=class_names,
    cmap="Blues",
    ax=ax
)

ax.set_xlabel("Predicted Species")
ax.set_ylabel("Actual Species")
ax.set_title(
    f"Confusion Matrix - K-Means (K = {k})"
)

st.pyplot(fig)


# ---------------------------------------------------------
# SPECIES VS CLUSTER
# ---------------------------------------------------------

st.header("🔄 Species vs Cluster Comparison")

comparison = pd.crosstab(
    result["Species"],
    result["Cluster"]
)

st.dataframe(
    comparison,
    use_container_width=True
)


# ---------------------------------------------------------
# FEATURE DISTRIBUTION
# ---------------------------------------------------------

st.header("🌸 Feature Relationships")

selected_x = st.selectbox(
    "Select X-axis Feature",
    features,
    index=0
)

selected_y = st.selectbox(
    "Select Y-axis Feature",
    features,
    index=2
)

fig, ax = plt.subplots(figsize=(9, 6))

sns.scatterplot(
    data=result,
    x=selected_x,
    y=selected_y,
    hue="Cluster",
    style="Species",
    palette="tab10",
    s=90,
    ax=ax
)

ax.set_title(
    f"{selected_x} vs {selected_y}"
)

st.pyplot(fig)


# ---------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------

st.header("📌 Final Model Summary")

summary = pd.DataFrame({
    "Parameter": [
        "Dataset",
        "Number of Samples",
        "Number of Features",
        "Number of Clusters (K)",
        "Accuracy",
        "Silhouette Score",
        "Inertia"
    ],

    "Value": [
        "Iris Dataset",
        len(df),
        len(features),
        k,
        f"{accuracy * 100:.2f}%",
        round(final_silhouette, 4),
        round(kmeans.inertia_, 4)
    ]
})

st.table(summary)


# ---------------------------------------------------------
# DOWNLOAD RESULTS
# ---------------------------------------------------------

st.header("⬇️ Download Results")

csv_data = result.to_csv(index=False)

st.download_button(
    label="Download Clustering Results CSV",
    data=csv_data,
    file_name="iris_kmeans_results.csv",
    mime="text/csv"
)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.markdown("---")

st.caption(
    "Iris K-Means Clustering | Machine Learning Project"
)