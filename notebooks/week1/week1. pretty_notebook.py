# Databricks notebook source

from house_price.data_processor import DataProcessor
from house_price.price_model import PriceModel
from house_price.utils import visualize_results, plot_feature_importance
import yaml

# Load configuration
with open("../project_config.yml", "r") as file:
    config = yaml.safe_load(file)

print("Configuration loaded:")
print(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
# Initialize DataProcessor
data_processor = DataProcessor("/Volumes/mlops_dev/house_prices/data/data.csv", config)

# Preprocess the data
data_processor.preprocess_data()

# COMMAND ----------
# Split the data
X_train, X_test, y_train, y_test = data_processor.split_data()

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# COMMAND ----------
# Initialize and train the model
model = PriceModel(data_processor.preprocessor, config)
model.train(X_train, y_train)

# COMMAND ----------
# Evaluate the model
mse, r2 = model.evaluate(X_test, y_test)
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# COMMAND ----------
## Visualizing Results
y_pred = model.predict(X_test)
visualize_results(y_test, y_pred)

# COMMAND ----------
## Feature Importance
feature_importance, feature_names = model.get_feature_importance()
plot_feature_importance(feature_importance, feature_names)
