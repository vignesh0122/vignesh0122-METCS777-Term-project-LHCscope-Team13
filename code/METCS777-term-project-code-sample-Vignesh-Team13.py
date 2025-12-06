"""
METCS777 Term Project - Vignesh Swaminathan Code Sample

Models Implemented:
1. Stacking Ensemble
2. GBT Baseline Model 
3. Random Forest Baseline

"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import sqrt, col, count, sum as spark_sum, udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml import Pipeline
import pyspark.sql.functions as F

# Initialize Spark Session for EMR
spark = SparkSession.builder \
    .appName("Vignesh_Ensemble_Methods_EMR") \
    .master("yarn") \
    .config("spark.executor.instances", "4") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.yarn.keytab", "") \
    .config("spark.yarn.principal", "") \
    .getOrCreate()

print("=== Vignesh Swaminathan - ML Model ===")

# Define schema for HIGGS.csv: 29 columns, first is label (double), others double
column_names = ["label"] + [f"f{i}" for i in range(28)]
schema = StructType([StructField(col, DoubleType(), True) for col in column_names])

# Load 500k rows from S3
print("Loading 500k rows from HIGGS dataset...")
df = spark.read.csv("s3://colliderscope/dataset/sample_dataset.csv", schema=schema, header=False).limit(500000)

# Handle missing values
df = df.na.fill(0.0)
df = df.na.replace(-999.0, None).na.drop()
df = df.na.fill(df.agg(*[F.mean(c).alias(c) for c in df.columns]).first().asDict())

print(f"Dataset loaded: {df.count()} rows, {len(df.columns)} columns")

# Add physics-inspired features
print("Adding physics features...")

# Lepton energy calculation
df = df.withColumn("lepton_energy", sqrt(df["f0"]**2 + 0.105**2))

# Hadronic transverse energy (HT)
ht_cols = ["f5", "f9", "f13", "f17", "f21"]
df = df.withColumn("ht", sum(df[col] for col in ht_cols))

# Missing transverse energy calculation
df = df.withColumn("mt_lep_met", sqrt(2 * df["f0"] * df["f3"] * (1 - F.cos(df["f2"] - df["f4"]))))

# Number of b-jets
bjet_cols = ["f8", "f12", "f16", "f20"]
df = df.withColumn("n_bjets", sum((df[col] > 0.5).cast("int") for col in bjet_cols))

print(f"Physics features added. Final feature count: {len(df.columns)}")

# Prepare feature vector
feature_cols = [f"f{i}" for i in range(28)] + ["lepton_energy", "ht", "mt_lep_met", "n_bjets"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)

# Split data
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"Train set: {train_df.count()} rows, Test set: {test_df.count()} rows")

# Create preprocessing pipeline
pipeline = Pipeline(stages=[assembler, scaler])
pipeline_model = pipeline.fit(train_df)

train_df = pipeline_model.transform(train_df)
test_df = pipeline_model.transform(test_df)

# Initialize evaluator
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

print("\n1. Training GBT Baseline Model...")

# GBT Baseline with standard parameters
gbt_baseline = GBTClassifier(
    labelCol="label",
    featuresCol="scaledFeatures",
    maxIter=30,           # Standard iterations
    maxDepth=5,           # Standard depth
    stepSize=0.1,         # Default learning rate
    subsamplingRate=1.0,  # No subsampling
    seed=42
)

print("Fitting GBT baseline model...")
gbt_baseline_model = gbt_baseline.fit(train_df)

# Evaluate on test set
baseline_predictions = gbt_baseline_model.transform(test_df)
baseline_auc = evaluator.evaluate(baseline_predictions)

print(".4f")

print("\n2. Training Random Forest Baseline...")

# Random Forest baseline model
rf_baseline = RandomForestClassifier(
    labelCol="label",
    featuresCol="scaledFeatures",
    numTrees=100,         # Number of trees
    maxDepth=8,           # Maximum depth
    seed=42
)

print("Fitting Random Forest baseline model...")
rf_baseline_model = rf_baseline.fit(train_df)

# Evaluate on test set
rf_predictions = rf_baseline_model.transform(test_df)
rf_auc = evaluator.evaluate(rf_predictions)

print(".4f")

print("\n3. Training Stacking Ensemble...")

# Custom stacking implementation using Random Forest + GBT + Logistic Regression
# Meta-learner: GBT

# Train base models
print("Training base models for stacking ensemble...")

# Base model 1: Random Forest
rf = RandomForestClassifier(
    labelCol="label",
    featuresCol="scaledFeatures",
    numTrees=50,
    maxDepth=6,
    seed=42
)
rf_model = rf.fit(train_df)
rf_predictions = rf_model.transform(test_df)

# Base model 2: GBT (different params from baseline)
gbt_base = GBTClassifier(
    labelCol="label",
    featuresCol="scaledFeatures",
    maxIter=25,
    maxDepth=4,
    stepSize=0.15,
    seed=42
)
gbt_base_model = gbt_base.fit(train_df)
gbt_base_predictions = gbt_base_model.transform(test_df)

# Base model 3: Logistic Regression (simulating Neural Network simplicity)
lr = LogisticRegression(
    labelCol="label",
    featuresCol="scaledFeatures",
    maxIter=100,
    regParam=0.01,
    seed=42
)
lr_model = lr.fit(train_df)
lr_predictions = lr_model.transform(test_df)

# Extract probability predictions from base models
print("Extracting predictions from base models...")

# Convert prediction probabilities to features for meta-learner
rf_prob = udf(lambda v: float(v[1]), DoubleType())  # Extract probability for positive class
gbt_prob = udf(lambda v: float(v[1]), DoubleType())
lr_prob = udf(lambda v: float(v[1]), DoubleType())

# Create stacking dataset with base model predictions as features
train_stacking = train_df.withColumn("rf_prob", rf_prob("rawPrediction")) \
                        .withColumn("gbt_prob", gbt_prob("rawPrediction")) \
                        .withColumn("lr_prob", lr_prob("rawPrediction")) \
                        .select("label", "rf_prob", "gbt_prob", "lr_prob")

test_stacking = test_df.withColumn("rf_prob", rf_prob(rf_predictions["rawPrediction"])) \
                      .withColumn("gbt_prob", gbt_prob(gbt_base_predictions["rawPrediction"])) \
                      .withColumn("lr_prob", lr_prob(lr_predictions["rawPrediction"])) \
                      .select("label", "rf_prob", "gbt_prob", "lr_prob")

print("Creating meta-learner features...")
stacking_assembler = VectorAssembler(inputCols=["rf_prob", "gbt_prob", "lr_prob"], outputCol="metaFeatures")
stacking_scaler = StandardScaler(inputCol="metaFeatures", outputCol="scaledMetaFeatures", withStd=True, withMean=False)

# Prepare meta-learner data
stacking_pipeline = Pipeline(stages=[stacking_assembler, stacking_scaler])
stacking_pipeline_model = stacking_pipeline.fit(train_stacking)

train_meta = stacking_pipeline_model.transform(train_stacking)
test_meta = stacking_pipeline_model.transform(test_stacking)

# Meta-learner: GBT
print("Training GBT meta-learner...")
meta_gbt = GBTClassifier(
    labelCol="label",
    featuresCol="scaledMetaFeatures",
    maxIter=40,          # Higher capacity for meta-learning
    maxDepth=6,          # Moderate depth
    stepSize=0.1,        # Standard learning rate
    seed=42
)

meta_model = meta_gbt.fit(train_meta)

# Get final predictions from stacking ensemble
stacking_predictions = meta_model.transform(test_meta)
stacking_auc = evaluator.evaluate(stacking_predictions)

print(".4f")

# Create results summary
print("\n=== Model Performance Summary ===")
results_data = [
    ("GBT_Baseline", float(baseline_auc)),
    ("Random_Forest", float(rf_auc)),
    ("Stacking_Ensemble", float(stacking_auc))
]

results_df = spark.createDataFrame(results_data, ["Model", "AUC"])

print("Results:")
results_df.show()

# Save results to S3
print("\nSaving results to S3...")
results_df.write.mode("overwrite").csv("s3://colliderscope/results/week2_results_summary.csv", header=True)

print("Results saved to: s3://colliderscope/results/week2_results_summary.csv")

# Stop Spark session
spark.stop()

print("\nVignesh Swaminathan ensemble models completed successfully!")
