# iris
# 🌸 Iris Dataset – K-Means Clustering

## 📌 Project Overview

This project applies **K-Means Clustering**, an unsupervised machine learning algorithm, to group Iris flowers based on their sepal and petal measurements.

The project includes an interactive **Streamlit web application** that visualizes clusters, evaluates clustering performance, and compares the results with actual Iris species.

## 🎯 Objectives

- Analyze the Iris dataset using machine learning.
- Group flowers based on their physical measurements.
- Explore different numbers of clusters.
- Evaluate clustering using the Silhouette Score.
- Visualize clusters using PCA.
- Compare clusters with actual species labels.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Jupyter Notebook

## 📂 Project Structure

```text
Iris-KMeans-Clustering/
│
├── app.py
├── Iris.csv
├── Iris_KMeans_Clustering_Corrected.ipynb
├── K_Means_Clustering_Live_Modified.ipynb
├── requirements.txt
└── README.md
```

## 📊 Dataset Description

The project uses the Iris dataset, which contains flower measurements and species labels.

**Features used for clustering:**
- `SepalLengthCm`
- `SepalWidthCm`
- `PetalLengthCm`
- `PetalWidthCm`

The `Species` column is used for evaluation and comparison, not as an input feature for clustering.

## ⚙️ Methodology

1. **Data Loading:** Load the Iris dataset using Pandas.
2. **Data Preprocessing:** Select the four numerical flower measurements.
3. **Feature Scaling:** Apply `StandardScaler` to standardize the measurements.
4. **Elbow Method:** Analyze inertia for different cluster counts.
5. **Silhouette Score:** Compare clustering quality for different values of K.
6. **K-Means Clustering:** Group flowers using the selected number of clusters.
7. **Evaluation:** Compare cluster assignments with actual species using majority-label mapping.
8. **PCA Visualization:** Reduce the features to two dimensions for plotting.
9. **Result Analysis:** Display cluster counts, confusion matrix, and species comparisons.

## ✨ Key Features

- Interactive Streamlit dashboard
- Dataset preview and statistical summary
- Missing-value analysis
- Adjustable number of clusters (K)
- Elbow Method visualization
- Silhouette Score comparison
- Clustering accuracy estimate
- PCA cluster visualization
- Confusion matrix
- Species-versus-cluster comparison
- Interactive feature relationship plots
- Downloadable clustering results as CSV

## 🚀 Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Iris-KMeans-Clustering.git
```

### Step 2: Navigate to the Project Folder

```bash
cd Iris-KMeans-Clustering
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

Alternatively, install the packages directly:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn scipy jupyter
```

### Step 4: Run the Application

Ensure `app.py` and `Iris.csv` are in the same directory.

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

`http://localhost:8501`

### Step 5: Run the Jupyter Notebook (Optional)

```bash
jupyter notebook
```

Open either notebook to explore the clustering analysis.

## 📈 Expected Results

The application displays:

- Dataset statistics and flower measurements
- Cluster evaluation scores
- Cluster distribution charts
- PCA-based cluster visualization
- Confusion matrix and species comparison
- A summary of model performance
- Downloadable clustering results

Results vary depending on the selected value of K.

## ⚠️ Important Note

K-Means is an **unsupervised learning algorithm**. The accuracy displayed in this application is calculated by mapping each cluster to its majority Iris species. It is an evaluation measure, not the conventional supervised classification accuracy.

## 🔮 Future Enhancements

- Compare K-Means with hierarchical clustering and DBSCAN.
- Add interactive filters for flower measurements.
- Improve cluster interpretation.
- Deploy the Streamlit application online.
- Add downloadable visualization reports.
