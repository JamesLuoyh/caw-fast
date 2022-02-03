import numpy as np
import psutil
a = open("processed/ml_uci.csv", "r")

id = set()
node_pair = set()
src = set()
tgt = set()

u_s = []
i_s = []
t_s = []
counter = 0
bipartite = True
max_ts = 0
min_ts = 999999999999
next(a)
for x in a:
	edge = x.strip().split(',')
	u = edge[1]
	i = edge[2]
	ts = int(float(edge[3]))
	id.add(u)
	id.add(i)
	node_pair.add((u,i))
	if u in tgt:
		if u in src:
			bipartite = False
		if i in tgt:
			bipartite = False
		else:
			src.add(i)
	elif u in src:
		if u in tgt:
			bipartite = False
		if i in src:
			bipartite = False
		else:
			tgt.add(i)
	else:
		if i in src:
			tgt.add(u)
		else:
			src.add(u)
			tgt.add(i)
	# if (u in tgt or i in src):
	# 	bipartite += 1
	if ts > max_ts:
		max_ts = ts
	if ts < min_ts:
		min_ts = ts
	counter += 1
	# 

print("all edges " + str(counter))
print("all nodes " + str(len(id)))
print("static edges " + str(len(node_pair)))
print("bipartite " + str(bipartite))
print("intensity " + str(2 * counter / ((len(id)) * (max_ts - min_ts))))
a.close()
