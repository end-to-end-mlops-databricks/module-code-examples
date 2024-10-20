# Databricks notebook source
# MAGIC %md
# MAGIC # Create custom model using pyfunc

# COMMAND ----------

# Databricks notebook source
import mlflow
import numpy as np
import pandas as pd
import yaml
from databricks.connect import DatabricksSession
from mlflow.models import infer_signature

mlflow.set_tracking_uri("databricks")

# COMMAND ----------
with open("/Volumes/mlops_dev/house_prices/data/project_config.yml", "r") as file:
    config = yaml.safe_load(file)

num_features = config.get("num_features")
cat_features = config.get("cat_features")
target = config.get("target")
parameters = config.get("parameters")
catalog_name = config.get("catalog_name")
schema_name = config.get("schema_name")

spark = DatabricksSession.builder.getOrCreate()


# COMMAND ----------

# MAGIC %md
# MAGIC Load model from a previous experiment run, based on a tag

# COMMAND ----------

run_id = mlflow.search_runs(
    experiment_names=["/Shared/house-prices"],
    filter_string="tags.branch='week2'",
).run_id[0]

model = mlflow.sklearn.load_model(f'runs:/{run_id}/lightgbm-pipeline-model')

# COMMAND ----------

# MAGIC %md
# MAGIC Create custom model using Pyfunc

# COMMAND ----------

class HousePriceModelWrapper(mlflow.pyfunc.PythonModel):
    
    def __init__(self, model):
        self.model = model
        
    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            predictions = self.model.predict(model_input)
            return predictions
        else:
            raise ValueError("Input must be a pandas DataFrame.")

# COMMAND ----------
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

X_train = train_set[num_features + cat_features].toPandas()
y_train = train_set[[target]].toPandas()

X_test = test_set[num_features + cat_features].toPandas()
y_test = test_set[[target]].toPandas()

# COMMAND ----------
wrapped_model = HousePriceModelWrapper(model) # we pass the loaded model to the wrapper
example_input = X_test.iloc[0:1]  # Select the first row for prediction as example
example_prediction = wrapped_model.predict(context=None, model_input=example_input)
print("Example Prediction:", example_prediction)

# COMMAND ----------
mlflow.set_experiment(experiment_name="/Shared/house-prices-pyfunc")

with mlflow.start_run(tags={"branch": "week2"}) as run:
    
    run_id = run.info.run_id
    signature = infer_signature(model_input=X_train, model_output={'Prediction': example_prediction})
    dataset = mlflow.data.from_spark(
        train_set, table_name=f"{catalog_name}.{schema_name}.train_set", version="0")
    mlflow.log_input(dataset, context="training")
    
    mlflow.pyfunc.log_model(
        python_model=wrapped_model,
        artifact_path="pyfunc-house-price-model",
        signature=signature
    )

