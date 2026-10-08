# Facebook_Live_K-Means_Clustering
Machine Learning project using K-Means Clustering to analyze and group Facebook Live posts based on their engagement characteristics.
# 📊 Facebook Live Posts - K-Means Clustering

## 📌 Project Overview

This project uses **Machine Learning and K-Means Clustering** to group Facebook Live posts according to their engagement characteristics.

The project analyzes different engagement features such as **reactions, comments, shares, likes, loves, wows, hahas, sads, and angrys** to identify groups of Facebook posts with similar engagement patterns.

A **Streamlit web application** is used to provide an interactive interface for dataset analysis, clustering, visualization, and downloading the final clustered dataset.

---

## 🎯 Objectives

* Analyze Facebook Live post engagement data.
* Perform data preprocessing and transformation.
* Apply **K-Means Clustering**.
* Select the optimal number of clusters using the **Elbow Method** and **Silhouette Score**.
* Visualize clusters using **PCA**.
* Understand the engagement characteristics of different post groups.
* Provide the clustered dataset for download.

---

## 🛠️ Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Matplotlib
* Scikit-learn
* K-Means Clustering
* PCA (Principal Component Analysis)

---

## 📂 Project Files

```text
Facebook-Live-KMeans/
│
├── app.py
├── Live.csv
├── facebook_live_project.ipynb
└── README.md
```

> **Note:** The application expects `Live.csv` to be in the same folder as `app.py`.

---

## 🔍 Features

### 1. Dataset Overview

The application displays:

* Total number of Facebook posts
* Total number of columns
* Number of post types
* Dataset preview
* Missing value information
* Post type distribution

### 2. Data Preparation

The project removes unnecessary columns such as:

* `status_id`
* `status_published`
* `Column1`
* `Column2`
* `Column3`
* `Column4`

The clustering features include:

```text
num_reactions
num_comments
num_shares
num_likes
num_loves
num_wows
num_hahas
num_sads
num_angrys
```

### 3. Data Transformation

Missing values are handled using median values.

Since engagement values can be highly skewed, **log1p transformation** is applied before standardization.

The project then uses `StandardScaler` to standardize the features.

### 4. Elbow Method

The Elbow Method is used to calculate inertia for different values of **K from 2 to 10**.

This helps in understanding a suitable number of clusters.

### 5. Silhouette Score

Silhouette scores are calculated for different K values.

The project selects the K value with the **highest silhouette score** as the optimal number of clusters.

### 6. K-Means Clustering

The final K-Means model uses:

```text
init = k-means++
n_init = 10
random_state = 42
```

Each Facebook Live post is assigned to a cluster.

### 7. Cluster Analysis

The application displays:

* Cluster distribution
* Number of posts in each cluster
* Mean engagement values
* Median engagement values
* Cluster profiles
* Post-type composition

### 8. PCA Visualization

**Principal Component Analysis (PCA)** is used to reduce the standardized data to two dimensions.

The clusters are displayed using a PCA scatter plot.

### 9. Download Results

The final clustered dataset can be downloaded as:

```text
Facebook_Live_KMeans_Results.csv
```

---

## 📊 Machine Learning Workflow

```text
Facebook Live Dataset
        ↓
Data Loading
        ↓
Data Cleaning
        ↓
Handle Missing Values
        ↓
Feature Selection
        ↓
Log Transformation
        ↓
Standardization
        ↓
Elbow Method
        ↓
Silhouette Score
        ↓
Select Optimal K
        ↓
K-Means Clustering
        ↓
Cluster Analysis
        ↓
PCA Visualization
        ↓
Download Results
```

---

## 🚀 How to Run the Project

### Step 1: Install Python

Make sure Python is installed on your computer.

### Step 2: Install Required Libraries

Open Command Prompt or PowerShell and run:

```bash
pip install streamlit pandas numpy matplotlib scikit-learn
```

If `pip` is not recognized, use:

```bash
py -m pip install streamlit pandas numpy matplotlib scikit-learn
```

### Step 3: Keep the Files Together

Place these files in the same folder:

```text
app.py
Live.csv
```

### Step 4: Open the Project Folder

Open Command Prompt or PowerShell inside the project folder.

### Step 5: Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your web browser.

---

## 📈 Results

The project creates clusters of Facebook Live posts based on their engagement behavior.

Clusters can be interpreted according to their engagement profiles. For example:

* **High-engagement cluster:** Higher reactions, comments, and shares.
* **Medium-engagement cluster:** Moderate engagement.
* **Low-engagement cluster:** Lower engagement values.

The cluster numbers themselves do not have a fixed meaning; they should be interpreted using their engagement profiles.

---

## 🧠 Conclusion

The Facebook Live dataset was successfully analyzed using the **K-Means Clustering algorithm**. Different engagement features were used to identify groups of Facebook posts with similar characteristics.

Log transformation and standardization were applied before clustering. The optimal number of clusters was selected using the silhouette score, and PCA was used to visualize the resulting clusters.

This project demonstrates how **unsupervised machine learning** can be used to understand social media engagement patterns.

---



### ⭐ If you find this project useful, please give the repository a star!

