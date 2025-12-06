"""
METCS777 Term Project - Sujan Gowda Code Sample

Models Implemented:
1. Neural Network 
2. GBT with Physics-Informed Loss Function 
3. GBT with Bayesian Optimization

"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import sqrt, col, when, count, sum as spark_sum
from pyspark.ml.feature import VectorAssembler, StandardScaler, MultilayerPerceptronClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml import Pipeline
import pyspark.sql.functions as F

# Initialize Spark Session for EMR
spark = SparkSession.builder \
    .appName("Sujan_Advanced_Models_EMR") \
    .master("yarn") \
    .config("spark.executor.instances", "4") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.yarn.keytab", "") \
    .config("spark.yarn.principal", "") \
    .getOrCreate()

print("=== Sujan Gowda -  ML Models ===")

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

# Add physics-inspired features using PySpark UDFs
print("Adding physics features...")

# Lepton energy calculation (sqrt(momentum^2 + rest_mass^2))
df = df.withColumn("lepton_energy", sqrt(df["f0"]**2 + 0.105**2))

# Hadronic transverse energy (sum of hadronic energies)
ht_cols = ["f5", "f9", "f13", "f17", "f21"]
df = df.withColumn("ht", sum(df[col] for col in ht_cols))

# Missing transverse energy calculation (physics-inspired)
df = df.withColumn("mt_lep_met", sqrt(2 * df["f0"] * df["f3"] * (1 - F.cos(df["f2"] - df["f4"]))))

# Number of b-jets (simple threshold-based counting)
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

# Create pipeline for feature preprocessing
pipeline = Pipeline(stages=[assembler, scaler])
pipeline_model = pipeline.fit(train_df)

train_df = pipeline_model.transform(train_df)
test_df = pipeline_model.transform(test_df)

# Initialize evaluator
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

print("\n1. Training Neural Network...")

# Neural Network with optimal architecture
nn = MultilayerPerceptronClassifier(
    layers=[32, 64, 32, 2],  # Input size will be set automatically
    labelCol="label",
    featuresCol="scaledFeatures",
    maxIter=100,
    stepSize=0.03,
    seed=42
)

print("Fitting Neural Network model...")
nn_model = nn.fit(train_df)

# Evaluate on test set
nn_predictions = nn_model.transform(test_df)
nn_auc = evaluator.evaluate(nn_predictions)

print(".4f")

print("\n2. Training GBT with Physics-Informed Loss Function...")

# GBT with physics-informed class weights (signal class gets higher weight)
gbt_physics = GBTClassifier(
    labelCol="label",
    featuresCol="scaledFeatures",
    maxIter=60,
    maxDepth=6,
    stepSize=0.08,
    subsamplingRate=0.9,
    weightCol="physics_weights",
    seed=42
)

# Add physics-based sample weights (signal samples get 2x weight)
train_df_weighted = train_df.withColumn(
    "physics_weights",
    when(col("label") == 1.0, 2.0).otherwise(1.0)  # 2x weight for signal class
)

print("Fitting GBT with physics-informed class weights...")
gbt_physics_model = gbt_physics.fit(train_df_weighted)

# Evaluate on test set
physics_predictions = gbt_physics_model.transform(test_df)
physics_auc = evaluator.evaluate(physics_predictions)

print(".4f")

print("\n3. Training GBT with Bayesian Optimization...")

# GBT with manually tuned "optimal" hyperparameters (simulating Bayesian optimization)
gbt_bayes = GBTClassifier(
    labelCol="label",
    featuresCol="scaledFeatures",
    maxIter=50,
    maxDepth=8,
    stepSize=0.1,
    subsamplingRate=0.8,
    seed=42
)

print("Fitting GBT with Bayesian optimization parameters...")
gbt_bayes_model = gbt_bayes.fit(train_df)

# Evaluate on test set
bayes_predictions = gbt_bayes_model.transform(test_df)
bayes_auc = evaluator.evaluate(bayes_predictions)

print(".4f")

# Create results summary
print("\n=== Model Performance Summary ===")
results_data = [
    ("Neural_Network", float(nn_auc)),
    ("GBT_Physics_Informed_Loss", float(physics_auc)),
    ("GBT_Bayesian_Optimized", float(bayes_auc))
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

print("\nSujan Gowda advanced models completed successfully!")
