from pyspark import SparkContext, SparkConf


def parse_edge(s):
    user, follower = s.split("\t")
    return (int(user), int(follower))

def step(item):
    prev_v, path, next_v = item[0], item[1][0], item[1][1]
    path += ',' + str(next_v)
    return (next_v, path)


config = SparkConf().setAppName("naumovse_task1_rdd").setMaster("yarn")
sc = SparkContext(conf=config)

n = 10  # number of partitions
edges = sc.textFile("/data/twitter/twitter_sample.txt").map(parse_edge)
forward_edges = edges.map(lambda e: (e[1], e[0])).partitionBy(n).persist()

start = 12
end = 34
max_path_len = 100
paths = sc.parallelize([(start, str(start))]).partitionBy(n)

for _ in range(max_path_len):
    paths = paths.join(forward_edges, n).map(step)
    count = paths.filter(lambda x: x[0] == end).count()
    if count > 0:
        break

end, path = paths.filter(lambda x: x[0] == end).first()
print(path)

