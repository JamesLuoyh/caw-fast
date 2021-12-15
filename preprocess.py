import numpy as np
a = open("sx-superuser.txt", "r")
f = open("sx-superuser.csv", "w")

u_rehash = {}
i_rehash = {}

u_counter = 0
i_counter = 0
min_ts = 0
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
	if t < min_ts:
		min_ts = t
	u_s.append(u)
	i_s.append(i)
	t_s.append(t)
	# 

order = np.argsort(t_s)

for o in order:
	u = u_s[o]
	i = i_s[o]
	t = t_s[o]
	t -= min_ts
	if u not in u_rehash:
		u_rehash[u] = u_counter
		u_counter += 1
	u_new = u_rehash[u]
	if i not in i_rehash:
		i_rehash[i] = i_counter
		i_counter += 1
	i_new = i_rehash[i]
	f.write(','.join([str(u_new), str(i_new), str(t), "0", "0"]) + '\n')

# for x in a:
# 	edge = x.strip().split(' ')
# 	f.write(','.join(edge) + ',0,0\n')
f.close()
