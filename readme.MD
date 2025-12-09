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
Results are computed using Area Under the ROC Curve (AUC) and Accuracy metrics. The output files track the performance of both baseline and optimized models.

* **Metric:** AUC (Area Under Curve) is the primary metric due to the slight class imbalance and the physics requirement to maximize signal efficiency.
* **Output Location:** `s3://colliderscope/results/`

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