from databricks.connect import DatabricksSession

spark = DatabricksSession.builder.profile("dbc-45ad9c70-3532").getOrCreate()
df = spark.read.table("samples.nyctaxi.trips")
df.show(5)