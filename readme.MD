# METCS777 Term Project - Team 13 ColliderScope 

Advanced machine learning models for Higgs boson detection using PySpark on AWS EMR.

## Team Members

- **Sujan Gowda**
- **Vignesh Swaminathan**

## Project Overview

This project implements two different PySpark machine learning pipelines for classifying Higgs boson signal vs background events using physics-inspired feature engineering and advanced ensemble techniques.

### Data
- **Source**: sample_dataset.csv (11M rows available, models use 500k sample)
- **Location**: `s3://colliderscope/dataset/sample_dataset.csv`
- **Schema**: 29 columns (1 label + 28 features)
- **Preprocessing**: Physics feature engineering, missing value handling, feature scaling

#### Sujan Gowda Models
1. **Neural Network**
2. **GBT with Physics-Informed Loss**
3. **GBT with Bayesian Optimization**

#### Vignesh Swaminathan Models
1. **GBT Baseline**
2. **Random Forest Baseline**
3. **Stacking Ensemble**

## Requirements

- pyspark==3.5.0
- pandas==2.2.0
- numpy==1.26.0
- matplotlib==3.8.0
- seaborn==0.13.0

## Environment Setup

1. **AWS Account** with EMR access
2. **S3 Bucket** `s3://colliderscope/` with:
   - Dataset: `s3://colliderscope/dataset/sample_dataset.csv`
   - Results: `s3://colliderscope/results/week2_results_summary.csv`
3. **Python Environment** with required packages

### EMR Cluster Setup (Cloud)

#### 1.1 Create EMR Cluster:
- Create an AWS EMR Cluster
- Name the cluster (Ex. colliderscope)
- Select EMR version of 7.0.0 or latest
- Select the applications "Hadoop", "Spark"
- Cluster size should "m5.xlarge"
- Core instances "2"
- Specify the cluster logs path to the "log" folder in your "S3" bucket
- Select the EC2 key pair "vockey", which is default
- Select the default "EMR_DefaultRole" and "EMR_EC2_DefaultRole" for the "Service Role" and "EC2 Instance Profile"
- Click "Create Cluster"

```
##### Upload directly from S3 Bucket:
- Go to the desired folder in the S3 bucket
- Click on "Upload"
- Click on "Add files"
- Browse and select your file and/or data and click "Upload"
- NOTE: Uploading time may vary depending upon the network speed


```

## 1.2 How to Run the Code
#### EMR Console Submission

2.1 Submit Job to EMR Cluster with following command:
- Once the cluster is created and in waiting status, go to "Add Step" option.
- In the "Add Step" do the following:
  - Application Type: Spark Application
  - Step name: colliderscope_step
  - Application location: s3://colliderscope/code/collider_scope_code.py
  - Spark-submit: Optional
  - Arguments: Optional

## Output Files

Results are saved to `s3://colliderscope/results/`