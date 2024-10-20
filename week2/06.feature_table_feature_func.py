# Databricks notebook source
import yaml
from databricks import feature_engineering
from databricks.connect import DatabricksSession
from pyspark.sql import SparkSession
from databricks.sdk import WorkspaceClient

# Initialize the Databricks session and clients
spark = SparkSession.builder.getOrCreate()
workspace = WorkspaceClient()
fe = feature_engineering.FeatureEngineeringClient()

# COMMAND ----------
# Load configuration from YAML file
with open("../project_config.yml", "r") as file:
    config = yaml.safe_load(file)

catalog_name = config.get("catalog_name")
schema_name = config.get("schema_name")

# COMMAND ----------
# Define table names and function names for house data
feature_table_name = f"{catalog_name}.{schema_name}.house_features"
function_name = f"{catalog_name}.{schema_name}.calculate_house_age"

# Load training and test sets from Databricks
train_set = spark.table(f"{catalog_name}.{schema_name}.train_set")
test_set = spark.table(f"{catalog_name}.{schema_name}.test_set")

# Extract specific features to save in the house_features table
house_features_df = train_set[["Id", "OverallQual", "GrLivArea", "GarageCars"]]
train_set = train_set.drop("OverallQual", "GrLivArea", "GarageCars")
