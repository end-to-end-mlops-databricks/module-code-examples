# Databricks notebook source
# MAGIC %md
# MAGIC # Preparing Dataset for House Price

# COMMAND ----------

# MAGIC %md
# MAGIC Get spark session

# COMMAND ----------

from datetime import datetime
import pandas as pd
import yaml
from databricks.connect import DatabricksSession
from sklearn.model_selection import train_test_split

spark = DatabricksSession.builder.getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC Load project_config file

# COMMAND ----------

with open("/Volumes/mlops_dev/house_prices/data/project_config.yml", "r") as file:
    config = yaml.safe_load(file)

catalog_name = config.get("catalog_name")
schema_name = config.get("schema_name")

print(f"Catalog: {catalog_name}")
print(f"Schema: {schema_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC Load dataset from Catalog

# COMMAND ----------

# Load the house prices dataset
df = spark.read.csv(
    "/Volumes/mlops_dev/house_prices/data/data.csv",
    header=True,
    inferSchema=True
)
display(df)

# COMMAND ----------

pandas_df = df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC Run preprocessing, split data into train and test

# COMMAND ----------
# Handle missing values and convert data types as needed
pandas_df['LotFrontage'] = pd.to_numeric(pandas_df['LotFrontage'], errors='coerce')

pandas_df['GarageYrBlt'] = pd.to_numeric(pandas_df['GarageYrBlt'], errors='coerce')
median_year = pandas_df['GarageYrBlt'].median()
pandas_df['GarageYrBlt'].fillna(median_year, inplace=True)
current_year = datetime.now().year

pandas_df['GarageAge'] = current_year - pandas_df['GarageYrBlt']
pandas_df.drop(columns=['GarageYrBlt'], inplace=True)

num_features = config.get("num_features", [])

for col in num_features:
    pandas_df[col] = pd.to_numeric(pandas_df[col], errors='coerce')


pandas_df.fillna({
    'LotFrontage': pandas_df['LotFrontage'].mean(),
    'MasVnrType': 'None',
    'MasVnrArea': 0,
}, inplace=True)

# Convert categorical features to the appropriate type
cat_features = config.get("cat_features", [])
for cat_col in cat_features:
    pandas_df[cat_col] = pandas_df[cat_col].astype('category')

# Extract target and numerical features
target = config.get("target")
num_features = config.get("num_features", [])

# Filter for relevant features
pandas_df = pandas_df[cat_features + num_features + [target] + ['Id']]

# Split the data into training and testing sets
train_set, test_set = train_test_split(pandas_df, test_size=0.2, random_state=42)

# Save the datasets into Databricks tables
spark.createDataFrame(train_set).write.saveAsTable(
    f"{catalog_name}.{schema_name}.train_set"
)
spark.createDataFrame(test_set).write.saveAsTable(
    f"{catalog_name}.{schema_name}.test_set"
)
