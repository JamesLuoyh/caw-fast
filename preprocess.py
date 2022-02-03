import numpy as np
# a = open("wiki-talk-temporal.txt", "r")
# f = open("wiki-talk-temporal.csv", "w")

# a = open("sx-superuser.txt", "r")
# f = open("sx-superuser.csv", "w")
a = open("sx-askubuntu.txt", "r")
f = open("sx-askubuntu.csv", "w")

rehash = {}

counter = 0
counter = 0
min_ts = 0
max_ts = 0
f.write("user_id,item_id,timestamp,state_label,comma_separated_list_of_features\n")
u_s = []
i_s = []
t_s = []
for x in a:
	edge = x.strip().split(' ')
	u = edge[0]
	i = edge[1]
	t = float(edge[2])
	if min_ts == 0:
		min_ts = t
		max_ts = t
	if t < min_ts:
		min_ts = t
	if t > max_ts:
		max_ts = t
	u_s.append(u)
	i_s.append(i)
	t_s.append(t)
	# 

order = np.argsort(t_s)

for o in order:
	u = u_s[o]
	i = i_s[o]
	t = t_s[o]
	# # if t < max_ts - 1 * 365*24*60*60:
	# 	continue
	# t -= min_ts
	if u not in rehash:
		rehash[u] = counter
		counter += 1
	u_new = rehash[u]
	if i not in rehash:
		rehash[i] = counter
		counter += 1
	i_new = rehash[i]
	f.write(','.join([str(u_new), str(i_new), str(t), "0", "0"]) + '\n') #  + ", 0" * 171
	# f.write(','.join([str(u), str(i), str(t), "0", "0"]) + '\n')

# for x in a:
# 	edge = x.strip().split(' ')
# 	f.write(','.join(edge) + ',0,0\n')
f.close()
