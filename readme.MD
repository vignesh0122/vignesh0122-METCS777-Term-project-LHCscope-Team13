# MET CS 777 Term Project - Team 13: ColliderScope

**Advanced Machine Learning Models for Higgs Boson Detection using PySpark on AWS EMR**

## Team Members
* **Sujan Gowda**
* **Vignesh Swaminathan**

---

## 1. Project Overview
This project implements two parallel PySpark machine learning pipelines to classify Higgs boson signal events versus background noise. The solution leverages AWS EMR for distributed processing, utilizing physics-inspired feature engineering and advanced ensemble techniques to optimize signal detection.

The project is divided into two modeling approaches:
* **Approaches by Sujan Gowda:** Neural Network, GBT with Physics-Informed Loss, GBT with Bayesian Optimization.
* **Approaches by Vignesh Swaminathan:** GBT Baseline, Random Forest Baseline, Stacking Ensemble.

---

## 2. Dataset Description
**Source:** The data is a subset of the standard HIGGS dataset (originally from the UCI Machine Learning Repository), generated via Monte Carlo simulations of particle collisions at CERN's ATLAS experiment. Dataset can be found here: https://opendata.cern.ch/search?q=&f=type%3ADataset&l=list&order=desc&p=1&s=10&sort=mostrecent

### 2.1 Data Structure
The dataset consists of **11 million rows** (observations) and **29 columns**. 
* **Target Variable:** The first column is the class label (`1.0` for signal processes detecting the Higgs boson, `0.0` for background noise).
* **Features:** The remaining 28 columns are numerical features derived from particle sensors.

### 2.2 Feature Breakdown
The 28 features are divided into two distinct categories critical for the classification task:

1.  **Low-Level Kinematic Features (First 21 features):** These represent raw measurements of particle momenta and energy.
    * *Lepton pT, eta, phi*: Properties of the detected lepton.
    * *Missing Energy magnitude & phi*: Imbalance in energy indicating unseen particles (neutrinos).
    * *Jets 1-4 (pT, eta, phi, b-tag)*: Properties of up to four jets (sprays of hadrons) produced in the collision.

2.  **High-Level Derived Features (Last 7 features):** These are complex, non-linear functions of the low-level features, designed by physicists to discriminate between Higgs decay signatures and background.
    * *m_jj, m_jjj, m_lv, m_jlv, m_bb, m_wbb, m_wwbb*: Various invariant masses calculated from combinations of the jets and leptons. These are theoretically highly predictive of the Higgs boson presence.

### 2.3 Preprocessing & Storage
* **S3 Location:** `s3://colliderscope/dataset/sample_dataset.csv` (Sample) / `s3://colliderscope/dataset/HIGGS.csv` (Full)
* **Preprocessing Steps:** * Schema inference and type casting (DoubleType).
    * Handling of missing values (though the native HIGGS dataset is dense).
    * Standard scaling applied to kinematic features for Neural Network convergence.
    * Vector assembly for Spark MLlib consumption.

---

## 3. Results Summary
We evaluated six different models using **Area Under the ROC Curve (AUC)** as our primary metric. AUC is the standard performance measure in High Energy Physics (HEP) because it is robust against class imbalance and independent of the decision threshold.

### 3.1 Model Performance Comparison
The table below summarizes the AUC scores achieved by each model on the test set.

| Rank | Model Name | Developed By | AUC Score |
| :--- | :--- | :--- | :--- |
| 1 | Stacking Ensemble | Vignesh | 0.8435 |
| 2 | Neural Network | Sujan | 0.8393 |
| 3 | GBT (Bayesian Opt) | Sujan | 0.8201 |
| 4 | Random Forest | Vignesh | 0.8210 |
| 5 | GBT (Physics Loss) | Sujan | 0.7972 |
| 6 | GBT Baseline | Vignesh | 0.7893 |

### 3.2 Visualizations
The pipeline generates the following plots in `s3://colliderscope/results/` to visualize these metrics:
* **`week2_all_models_roc.png`**: ROC Curves overlaying all 6 models. The Stacking Ensemble (Red line) and Neural Network (Purple line) show the strongest convex curves, indicating superior true-positive rates at low false-positive thresholds.
* **`week2_auc_comparison.png`**: A bar chart ranking models by AUC score, highlighting the ~4% performance gap between the Baseline GBT and the Stacking Ensemble.
* **`week2_feature_importance.png`**: Identifies that derived mass features (like `m_bb` and `m_wwbb`) are the dominant predictors, confirming the value of physics-informed feature engineering.

### 3.3 Performance Analysis
* **Ensemble Dominance:** The **Stacking Ensemble** achieved the highest accuracy (**0.8300**), confirming that combining diverse weak learners (Random Forest, GBT, Logistic Regression) effectively reduces variance and generalizes better than any single model.
* **Optimization Impact:** The **GBT with Bayesian Optimization** (**0.8155**) significantly outperformed the **GBT Baseline** (**0.7893**), yielding a **~2.6% improvement**. This demonstrates that hyperparameter tuning (maxDepth, stepSize) is critical for tree-based models on this dataset.
* **Deep Learning Competitiveness:** The **Neural Network** (**0.8279**) was the top-performing single model, nearly matching the ensemble. This suggests that the dense, non-linear relationships in the kinematic data are well-captured by multi-layer perceptrons.
* **Physics-Informed Loss:** While the **Physics-Loss GBT** (**0.7945**) beat the baseline, it trailed the Bayesian Optimized model. This indicates that while weighting signal events helps, structural hyperparameter tuning provides a larger immediate gain in predictive power.

---

## 4. Requirements & Environment

### Software Requirements
* **PySpark:** 3.5.0
* **Pandas:** 2.2.0 (for result aggregation)
* **Matplotlib/Seaborn:** (for visualization in notebooks)

### AWS EMR Cluster Configuration
To replicate the results, provision a cluster with the following specs:

1.  **Release:** emr-7.0.0 (Applications: Hadoop, Spark)
2.  **Hardware:** * **Master:** 1x m5.xlarge
    * **Core:** 2x m5.xlarge (Minimum for distributed processing)
3.  **Logging:** Enabled to `s3://colliderscope/logs/`
4.  **Roles:** Default `EMR_DefaultRole` and `EMR_EC2_DefaultRole`

---

## 5. Execution Instructions

### Step 1: Upload Code & Data
Ensure the dataset and the python script (`collider_scope_code.py`) are uploaded to your S3 bucket:
* `s3://colliderscope/dataset/`
* `s3://colliderscope/code/`

### Step 2: Submit Spark Job via EMR Console
1.  Navigate to the **EMR Cluster** console.
2.  Select the running cluster and click **Steps** > **Add Step**.
3.  Configure the step:
    * **Type:** Spark application
    * **Name:** `ColliderScope_Full_Run`
    * **Deploy Mode:** Cluster
    * **Application Location:** `s3://colliderscope/code/collider_scope_code.py`
4.  Click **Add** to begin the pipeline.

### Step 3: Retrieve Results
Once the step status is **Completed**, download the results summary:
```bash
aws s3 cp s3://colliderscope/results/week2_results_summary.csv ./local_results/

### Visualizations (Plots)
* **`week2_all_models_roc.png`**: ROC Curves overlaying all 6 models to compare performance visually.
* **`week2_auc_comparison.png`**: A bar chart ranking models by AUC score.
* **`week2_feature_importance.png`**: Top 15 features driving the Gradient Boosted Tree (GBT) baseline predictions.
