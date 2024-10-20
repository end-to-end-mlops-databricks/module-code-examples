# Databricks notebook source
# MAGIC %md
# MAGIC Traing and log model

# COMMAND ----------

import yaml
from databricks.connect import DatabricksSession

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
from mlflow.models import infer_signature

mlflow.set_registry_uri('databricks-uc') # It must be -uc for registering models to Unity Catalog

# COMMAND ----------

with open("/Volumes/mlops_dev/house_prices/data/project_config.yml", "r") as file:
    config = yaml.safe_load(file)

# Extract configuration details
num_features = config.get('num_features')
cat_features = config.get('cat_features')
target = config.get('target')
parameters = config.get('parameters')
catalog_name = config.get("catalog_name")
schema_name = config.get("schema_name")

print(catalog_name)
print(schema_name)

# COMMAND ----------

spark = DatabricksSession.builder.getOrCreate()

# Load training and testing sets from Databricks tables
train_set_spark = spark.table(f"{catalog_name}.{schema_name}.train_set")
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set").toPandas()
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set").toPandas()

X_train = train_set[num_features + cat_features]
y_train = train_set[target]

X_test = test_set[num_features + cat_features]
y_test = test_set[target]

# Define the preprocessor for categorical features
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)], 
    remainder='passthrough'
)

# Create the pipeline with preprocessing and the LightGBM regressor
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LGBMRegressor(**parameters))
])


# COMMAND ----------

# MAGIC %md
# MAGIC Set mlflow experiment and execute mlflow run for training

# COMMAND ----------


mlflow.set_experiment(experiment_name='/Shared/house-prices')

# Start an MLflow run to track the training process
with mlflow.start_run(
    tags={"git_sha": "ffa63b430205ff7",
          "branch": "week2"},
) as run:
    run_id = run.info.run_id

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate the model performance
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R2 Score: {r2}")

    # Log parameters, metrics, and the model to MLflow
    mlflow.log_param("model_type", "LightGBM with preprocessing")
    mlflow.log_params(parameters)
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2_score", r2)
    signature = infer_signature(model_input=X_train, model_output=y_pred)

    dataset = mlflow.data.from_spark(
    train_set_spark, table_name=f"{catalog_name}.{schema_name}.train_set",
    version="0")
    mlflow.log_input(dataset, context="training")
    
    mlflow.sklearn.log_model(
        sk_model=pipeline,
        artifact_path="lightgbm-pipeline-model",
        registered_model_name=f"{catalog_name}.{schema_name}.house_prices_model",
        signature=signature
    )


# COMMAND ----------
run = mlflow.get_run(run_id)
dataset_info = run.inputs.dataset_inputs[0].dataset
dataset_source = mlflow.data.get_source(dataset_info)
dataset_source.load()

# COMMAND ----------

