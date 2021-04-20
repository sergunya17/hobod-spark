from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.window import Window
import pyspark.sql.functions as f
import re


def cut_words(item):
    article, text = item[0], item[1].split()
    output = [[article, re.sub('\W+', '', word.lower()), pos] for pos, word in enumerate(text)]
    return output


n = 400 # number of partitions
spark = SparkSession.builder.config('spark.sql.shuffle.partitions', n) \
                            .appName('naumovse_task3')                 \
                            .master('yarn')                            \
                            .getOrCreate()

articles_schema = StructType(fields=[
    StructField('id', IntegerType(), False),
    StructField('text', StringType(), False)
])
stop_words_schema = StructType(fields=[
    StructField('word', StringType(), False)
])

articles = spark.read.csv('/data/wiki/en_articles_part', sep='\t', schema=articles_schema)
articles = articles.repartition(n)
stop_words = spark.read.csv('/data/wiki/stop_words_en-xpo6.txt', stop_words_schema)

words = articles.rdd.flatMap(cut_words).toDF(['article', 'word', 'pos'])
words = words.filter(words['word'] != '') \
             .join(f.broadcast(stop_words), 'word', 'left_anti')

words_count = words.count()
words_proba = words.groupBy('word').count()                           \
                   .withColumn('proba', f.col('count') / words_count) \
                   .select('word', 'proba')

windowSpec  = Window.partitionBy('article').orderBy('pos')
bigrams = words.withColumn('first_word', f.lag(words['word']).over(windowSpec))
bigrams = bigrams.filter(bigrams['first_word'].isNotNull()) \
                 .selectExpr('first_word', 'word as second_word')

bigrams_count = bigrams.count()
bigrams_proba = bigrams.groupBy('first_word', 'second_word').count()              \
                       .filter(f.col('count') >= 500)                             \
                       .withColumn('joint_proba', f.col('count') / bigrams_count) \
                       .select('first_word', 'second_word', 'joint_proba')

bigrams_full_proba = bigrams_proba.join(words_proba, bigrams_proba['first_word'] == words_proba['word'], 'inner')  \
                                  .drop('word')                                                                    \
                                  .withColumnRenamed('proba', 'first_proba')                                       \
                                  .join(words_proba, bigrams_proba['second_word'] == words_proba['word'], 'inner') \
                                  .drop('word')                                                                    \
                                  .withColumnRenamed('proba', 'second_proba')

bigrams_top = bigrams_full_proba.withColumn('pmi', f.log(f.col('joint_proba') / (f.col('first_proba') * f.col('second_proba')))) \
                                .withColumn('npmi', -f.col('pmi') / f.log(f.col('joint_proba')))                                 \
                                .orderBy('npmi', ascending=False)                                                                \
                                .select(f.concat_ws('_', f.col('first_word'), f.col('second_word')).alias('bigram'))             \
                                .take(39)

for line in bigrams_top:
    print(line.bigram)

