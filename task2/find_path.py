from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pyspark.sql.functions as f


spark = SparkSession.builder.appName('naumovse_task2_df').master('yarn').getOrCreate()

n = 10  # number of partitions
edges_schema = StructType(fields=[
    StructField('out', IntegerType(), False),
    StructField('in', IntegerType(), False)
])

edges = spark.read.csv('/data/twitter/twitter_sample.txt', sep='\t', schema=edges_schema)
edges = edges.repartition(n)

start = 12
end = 34
max_path_len = 100
init_path = [[start, str(start)]]

paths_schema = StructType(fields=[
    StructField('end', IntegerType(), False),
    StructField('path', StringType(), False)
])

paths = spark.createDataFrame(init_path, schema=paths_schema)
paths = paths.repartition(n)

for _ in range(max_path_len):
    paths = paths.join(edges, paths['end'] == edges['in'], 'inner')
    paths = paths.select(f.col('out').alias('end'),
                         f.concat_ws(',', paths['path'], paths['out']).alias('path'))

    count = paths.filter(paths['end'] == end).count()
    if count > 0:
        break
        
end, path = paths.filter(paths['end'] == end).first()
print(path)
