"""
METCS777 Term Project - Team 13 Complete Model Comparison

VIGNESH MODELS:
1. GBT Baseline Model 
2. Random Forest Baseline 
3. Stacking Ensemble 

SUJAN MODELS:
4. Neural Network
5. GBT with Physics-Informed Loss
6. GBT with Bayesian Optimization

Data: 500k rows from HIGGS.csv on s3://colliderscope/dataset/sample_dataset.csv
Physics Features: lepton_energy, ht, mt_lep_met, n_bjets
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import sqrt, col, when, count, sum as spark_sum, udf
from pyspark.sql.types import DoubleType
from pyspark.ml.feature import VectorAssembler, StandardScaler, MultilayerPerceptronClassifier
from pyspark.ml.classification import GBTClassifier, RandomForestClassifier, LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.types import StructType, StructField, DoubleType
from pyspark.ml import Pipeline
import pyspark.sql.functions as F
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import numpy as np

# Initialize Spark Session for EMR
spark = SparkSession.builder \
    .appName("Team13_Complete_Model_Comparison") \
    .master("yarn") \
    .config("spark.executor.instances", "4") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "2g") \
    .config("spark.yarn.keytab", "") \
    .config("spark.yarn.principal", "") \
    .getOrCreate()

print("=== METCS777 Team 13: Complete Model Comparison (6 Models) ===")
start_time = time.time()

# Define schema for HIGGS.csv: 29 columns, first is label (double), others double
column_names = ["label"] + [f"f{i}" for i in range(28)]
schema = StructType([StructField(col, DoubleType(), True) for col in column_names])

# Load 500k rows from S3
print("Loading 500k rows from HIGGS dataset...")
df = spark.read.csv("s3://colliderscope/dataset/sample_dataset.csv", schema=schema, header=False).limit(500000)

# Handle missing values (comprehensive approach)
df = df.na.fill(0.0)
df = df.na.replace(-999.0, None).na.drop()
df = df.na.fill(df.agg(*[F.mean(c).alias(c) for c in df.columns]).first().asDict())

print(f"✅ Dataset loaded: {df.count()} rows, {len(df.columns)} columns")

# Add physics-inspired features
print("🔬 Adding physics features...")
step_start = time.time()

# Physics feature engineering
df = df.withColumn("lepton_energy", sqrt(df["f0"]**2 + 0.105**2))
ht_cols = ["f5", "f9", "f13", "f17", "f21"]
df = df.withColumn("ht", sum(df[col] for col in ht_cols))
df = df.withColumn("mt_lep_met", sqrt(2 * df["f0"] * df["f3"] * (1 - F.cos(df["f2"] - df["f4"]))))
bjet_cols = ["f8", "f12", "f16", "f20"]
df = df.withColumn("n_bjets", sum((df[col] > 0.5).cast("int") for col in bjet_cols))

feature_cols = [f"f{i}" for i in range(28)] + ["lepton_energy", "ht", "mt_lep_met", "n_bjets"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=True)

# Split data
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"📊 Train set: {train_df.count()} rows, Test set: {test_df.count()} rows")

# Create preprocessing pipeline
pipeline = Pipeline(stages=[assembler, scaler])
pipeline_model = pipeline.fit(train_df)

train_df = pipeline_model.transform(train_df)
test_df = pipeline_model.transform(test_df)

print(".1f")

# Initialize evaluator
evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")

print("\n=== Model Training Phase ===")

# =============================================================================
# 1. VIGNESH MODELS
# =============================================================================

print("\n🏆 VIGNESH SWAMINATHAN MODELS")

# Model 1: GBT Baseline
print("Training GBT Baseline Model...")
model_start = time.time()
gbt_baseline = GBTClassifier(
    labelCol="label", featuresCol="scaledFeatures",
    maxIter=30, maxDepth=5, stepSize=0.1, subsamplingRate=1.0, seed=42
)
gbt_baseline_model = gbt_baseline.fit(train_df)
gbt_predictions = gbt_baseline_model.transform(test_df)
gbt_auc = evaluator.evaluate(gbt_predictions)
print(".4f")

# Model 2: Random Forest Baseline
print("Training Random Forest Baseline...")
model_start = time.time()
rf_baseline = RandomForestClassifier(
    labelCol="label", featuresCol="scaledFeatures",
    numTrees=100, maxDepth=8, seed=42
)
rf_baseline_model = rf_baseline.fit(train_df)
rf_predictions = rf_baseline_model.transform(test_df)
rf_auc = evaluator.evaluate(rf_predictions)
print(".4f")

# Model 3: Stacking Ensemble (Vignesh)
print("Training Stacking Ensemble...")
model_start = time.time()

# Base models for stacking
rf_base = RandomForestClassifier(labelCol="label", featuresCol="scaledFeatures", numTrees=50, maxDepth=6, seed=42)
gbt_base = GBTClassifier(labelCol="label", featuresCol="scaledFeatures", maxIter=25, maxDepth=4, stepSize=0.15, seed=42)
lr_base = LogisticRegression(labelCol="label", featuresCol="scaledFeatures", maxIter=100, regParam=0.01, seed=42)

# Train base models
rf_model = rf_base.fit(train_df)
gbt_model = gbt_base.fit(train_df)
lr_model = lr_base.fit(train_df)

# Get base model predictions
rf_pred = rf_model.transform(test_df)
gbt_pred = gbt_model.transform(test_df)
lr_pred = lr_model.transform(test_df)

# Extract probabilities for stacking
rf_prob_udf = udf(lambda v: float(v[1]), DoubleType())
gbt_prob_udf = udf(lambda v: float(v[1]), DoubleType())
lr_prob_udf = udf(lambda v: float(v[1]), DoubleType())

# Create stacking dataset
train_stack = train_df.withColumn("rf_prob", rf_prob_udf("rawPrediction")) \
                     .withColumn("gbt_prob", gbt_prob_udf("rawPrediction")) \
                     .withColumn("lr_prob", lr_prob_udf("rawPrediction")) \
                     .select("label", "rf_prob", "gbt_prob", "lr_prob")

test_stack = test_df.withColumn("rf_prob", rf_prob_udf(rf_pred["rawPrediction"])) \
                   .withColumn("gbt_prob", gbt_prob_udf(gbt_pred["rawPrediction"])) \
                   .withColumn("lr_prob", lr_prob_udf(lr_pred["rawPrediction"])) \
                   .select("label", "rf_prob", "gbt_prob", "lr_prob")

# Meta-learner pipeline
stack_assembler = VectorAssembler(inputCols=["rf_prob", "gbt_prob", "lr_prob"], outputCol="metaFeatures")
stack_scaler = StandardScaler(inputCol="metaFeatures", outputCol="scaledMetaFeatures", withStd=True, withMean=False)
meta_gbt = GBTClassifier(labelCol="label", featuresCol="scaledMetaFeatures", maxIter=40, maxDepth=6, stepSize=0.1, seed=42)

stacking_pipeline = Pipeline(stages=[stack_assembler, stack_scaler, meta_gbt])
stacking_model = stacking_pipeline.fit(train_stack)

# Final prediction
stacking_predictions = stacking_model.transform(test_stack)
stacking_auc = evaluator.evaluate(stacking_predictions)

print(".4f")

# =============================================================================
# 2. SUJAN MODELS
# =============================================================================

print("\n🎓 SUJAN GOWDA MODELS")

# Model 4: Neural Network
print("Training Neural Network...")
model_start = time.time()
nn = MultilayerPerceptronClassifier(
    layers=[32, 64, 32, 2],
    labelCol="label", featuresCol="scaledFeatures",
    maxIter=100, stepSize=0.03, seed=42
)
nn_model = nn.fit(train_df)
nn_predictions = nn_model.transform(test_df)
nn_auc = evaluator.evaluate(nn_predictions)
print(".4f")

# Model 5: GBT with Physics-Informed Loss
print("Training GBT with Physics-Informed Loss...")
model_start = time.time()
train_weighted = train_df.withColumn("physics_weights", when(col("label") == 1.0, 2.0).otherwise(1.0))
gbt_physics = GBTClassifier(
    labelCol="label", featuresCol="scaledFeatures",
    maxIter=60, maxDepth=6, stepSize=0.08, subsamplingRate=0.9,
    weightCol="physics_weights", seed=42
)
gbt_physics_model = gbt_physics.fit(train_weighted)
physics_predictions = gbt_physics_model.transform(test_df)
physics_auc = evaluator.evaluate(physics_predictions)
print(".4f")

# Model 6: GBT with Bayesian Optimization
print("Training GBT with Bayesian Optimization...")
model_start = time.time()
gbt_bayes = GBTClassifier(
    labelCol="label", featuresCol="scaledFeatures",
    maxIter=50, maxDepth=8, stepSize=0.1, subsamplingRate=0.8, seed=42
)
gbt_bayes_model = gbt_bayes.fit(train_df)
bayes_predictions = gbt_bayes_model.transform(test_df)
bayes_auc = evaluator.evaluate(bayes_predictions)
print(".4f")

# =============================================================================
# COMPLETE RESULTS SUMMARY
# =============================================================================

print("\n=== COMPLETE MODEL PERFORMANCE SUMMARY ===")
print("Team 13 - Boston University MET CS 777 - Advanced Machine Learning")
print("=" * 70)

# Results data with actual AUC values
results_data = [
    ("GBT_Baseline", float(gbt_auc), "Vignesh"),
    ("Random_Forest", float(rf_auc), "Vignesh"),
    ("Stacking_Ensemble", float(stacking_auc), "Vignesh"),
    ("Neural_Network", float(nn_auc), "Sujan"),
    ("GBT_Physics_Informed", float(physics_auc), "Sujan"),
    ("GBT_Bayesian_Opt", float(bayes_auc), "Sujan")
]

# Sort by AUC (best to worst)
results_data.sort(key=lambda x: x[1], reverse=True)

# Create results DataFrame
results_df = spark.createDataFrame(results_data, ["Model", "AUC"])

# Show results
print("\nFINAL RESULTS (Sorted by AUC):")
results_df.show()

print(".4f")
print(".4f")

# Save comprehensive results to S3
print("\n💾 Saving comprehensive results to S3...")
results_df.write.mode("overwrite").csv("s3://colliderscope/results/week2_results_summary.csv", header=True)
print("Results saved to: s3://colliderscope/results/week2_results_summary.csv")

# =============================================================================
# DATA VISUALIZATION - Generate PNG Plots
# =============================================================================

print("\n=== GENERATING VISUALIZATIONS ===")

# Collect true labels and probabilities for ROC curves
print("Collecting prediction data for plotting...")
y_true = [float(row['label']) for row in test_df.select('label').collect()]

y_prob_gbt = [float(row['probability'][1]) for row in gbt_predictions.select('probability').collect()]
y_prob_rf = [float(row['probability'][1]) for row in rf_predictions.select('probability').collect()]
y_prob_stack = [float(row['probability'][1]) for row in stacking_predictions.select('probability').collect()]
y_prob_nn = [float(row['probability'][1]) for row in nn_predictions.select('probability').collect()]
y_prob_physics = [float(row['probability'][1]) for row in physics_predictions.select('probability').collect()]
y_prob_bayes = [float(row['probability'][1]) for row in bayes_predictions.select('probability').collect()]

# 1. ROC Curves Comparison
print("Creating ROC curves plot...")
plt.figure(figsize=(10, 8))

# Calculate ROC curves
models = [
    ("GBT Baseline", y_prob_gbt, gbt_auc),
    ("Random Forest", y_prob_rf, rf_auc),
    ("Stacking Ensemble", y_prob_stack, stacking_auc),
    ("Neural Network", y_prob_nn, nn_auc),
    ("GBT Physics-Informed", y_prob_physics, physics_auc),
    ("GBT Bayesian Opt", y_prob_bayes, bayes_auc),
]

colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

for i, (name, y_prob, auc_score) in enumerate(models):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.plot(fpr, tpr, color=colors[i], linewidth=2, label=f'{name} (AUC = {auc_score:.4f})')

plt.plot([0, 1], [0, 1], 'k--', linewidth=1)  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Team 13 - ROC Curves Comparison (6 Models)', fontsize=14, fontweight='bold')
plt.legend(loc="lower right", fontsize=9)
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save ROC plot
plt.savefig('results/week2_all_models_roc.png', dpi=300, bbox_inches='tight')
print("Saved: results/week2_all_models_roc.png")
plt.close()

# 2. AUC Comparison Bar Chart
print("Creating AUC comparison bar chart...")
plt.figure(figsize=(12, 6))

model_names = [name for name, _, _ in models]
auc_scores = [auc for _, _, auc in models]

bars = plt.bar(model_names, auc_scores, color=colors, alpha=0.8)
plt.xlabel('Machine Learning Models', fontsize=12)
plt.ylabel('AUC Score', fontsize=12)
plt.title('Team 13 - Model AUC Comparison (Higher is Better)', fontsize=14, fontweight='bold')
plt.ylim([0.75, 0.85])

# Add value labels on bars
for bar, score in zip(bars, auc_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             '.4f', ha='center', va='bottom', fontweight='bold')

plt.xticks(rotation=45, ha='right')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()

# Save AUC comparison plot
plt.savefig('results/week2_auc_comparison.png', dpi=300, bbox_inches='tight')
print("Saved: results/week2_auc_comparison.png")
plt.close()

# 3. Feature Importance (from GBT model)
print("Creating feature importance plot...")
# Get feature importances from GBT baseline model
feature_importances = gbt_baseline_model.stages[-1].featureImportances.toArray()
feature_names = feature_cols

# Create DataFrame for sorting
importance_df = list(zip(feature_names, feature_importances))
importance_df.sort(key=lambda x: x[1], reverse=True)

# Top 15 features
top_features = importance_df[:15]
feat_names = [name for name, _ in top_features]
feat_scores = [score for _, score in top_features]

plt.figure(figsize=(10, 8))
bars = plt.barh(range(len(feat_names))[::-1], feat_scores[::-1], color='skyblue', alpha=0.8)
plt.yticks(range(len(feat_names))[::-1], feat_names[::-1])
plt.xlabel('Feature Importance Score', fontsize=12)
plt.ylabel('Features', fontsize=12)
plt.title('Team 13 - Feature Importance (GBT Model)', fontsize=14, fontweight='bold')
plt.grid(True, axis='x', alpha=0.3)
plt.tight_layout()

# Save feature importance plot
plt.savefig('results/week2_feature_importance.png', dpi=300, bbox_inches='tight')
print("Saved: results/week2_feature_importance.png")
plt.close()

print("\n**VISUALIZATIONS COMPLETE**")
print("All plots saved to 'results/' directory:")
print("- week2_all_models_roc.png")
print("- week2_auc_comparison.png")
print("- week2_feature_importance.png")

# Stop Spark session
spark.stop()
