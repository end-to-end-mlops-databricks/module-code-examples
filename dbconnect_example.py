# from databricks.connect import DatabricksSession
# spark = DatabricksSession.builder.profile("Patrick_Machine").getOrCreate()

df = spark.read.table("samples.nyctaxi.trips")
df.show(10)