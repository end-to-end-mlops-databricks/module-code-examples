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

from price_converter.utils import adjust_predictions

class HousePriceModelWrapper(mlflow.pyfunc.PythonModel):
    
    def __init__(self, model):
        self.model = model
        
    def predict(self, context, model_input):
        if isinstance(model_input, pd.DataFrame):
            predictions = self.model.predict(model_input)
            predictions = adjust_predictions(predictions)
            return {"Prediction": predictions}
        else:
            raise ValueError("Input must be a pandas DataFrame.")

# COMMAND ----------



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
example_prediction = example_prediction["Prediction"]

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
        signature=signature,
        pip_requirements=["/Volumes/mlops_dev/house_prices/packages/price_converter-0.0.1-py3-none-any.whl"]
    )


# COMMAND ----------

import mlflow

# Now to load the model back and run predictions on the test set
run_id = "8f9a50dfa3394887b488c67405b973bb"
model_uri = f"runs:/{run_id}/pyfunc-house-price-model"
loaded_model = mlflow.pyfunc.load_model(model_uri)

# Make predictions on the test set
X_test_sample = X_test.iloc[0:2]
y_test_sample = y_test.iloc[0:2]
predictions = loaded_model.predict(X_test_sample)

# Compare predictions with actual values
predictions_df = pd.DataFrame({
    "Actual": y_test_sample[target].values.flatten(),
    "Predicted": predictions["Prediction"].flatten()
})

# Display the predictions
print(predictions_df)

# COMMAND ----------

predictions
