import pandas as pd
from log import *
from parser import *
from eval import *
from utils import *
from train import *
#import numba
from module import CAWN
from module2 import CAWN2, NeighborhoodEncoder
from graph import NeighborFinder
from neighbors import NeighborsBuilder
import resource

args, sys_argv = get_args()

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_EPOCH = args.n_epoch
ATTN_NUM_HEADS = args.attn_n_head
DROP_OUT = args.drop_out
GPU = args.gpu
USE_TIME = args.time
ATTN_AGG_METHOD = args.attn_agg_method
ATTN_MODE = args.attn_mode
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
POS_ENC = args.pos_enc
POS_DIM = args.pos_dim
WALK_POOL = args.walk_pool
WALK_N_HEAD = args.walk_n_head
WALK_MUTUAL = args.walk_mutual if WALK_POOL == 'attn' else False
TOLERANCE = args.tolerance
CPU_CORES = args.cpu_cores
NGH_CACHE = args.ngh_cache
VERBOSITY = args.verbosity
AGG = args.agg
SEED = args.seed
T_BATCH = args.t_batch
assert(CPU_CORES >= -1)
set_random_seed(SEED)
logger, get_checkpoint_path, get_ngh_store_path, best_model_path, best_model_ngh_store_path = set_up_logger(args, sys_argv)

# Load data and sanity check
if T_BATCH:
  g_df = pd.read_csv('./processed/batches_{}.txt'.format(DATA))
  tbatch_id_l = g_df.tbatch_id.values
else:
  g_df = pd.read_csv('./processed/batches_{}.txt'.format(DATA))
  tbatch_id_l = None
e_feat = np.load('./processed/ml_{}.npy'.format(DATA))
n_feat = np.load('./processed/ml_{}_node.npy'.format(DATA))
src_l = g_df.src.values.astype(int)
dst_l = g_df.tgt.values.astype(int)
e_idx_l = g_df.idx.values.astype(int)
label_l = g_df.label.values
ts_l = g_df.ts.values
src_e_l = g_df.e_from_u.values.astype(int)
tgt_e_l = g_df.e_from_i.values.astype(int)
src_start_l = g_df.u_start.values.astype(int)
tgt_start_l = g_df.i_start.values.astype(int)
src_ngh_n_l = g_df.u_ngh_n.values.astype(int)
tgt_ngh_n_l = g_df.i_ngh_n.values.astype(int)
tgt_post_n_l = g_df.i_post_n.values.astype(int)

max_idx = max(src_l.max(), dst_l.max())
max_e_idx = max(src_e_l.max(), tgt_e_l.max()) + 10
assert(np.unique(np.stack([src_l, dst_l])).shape[0] == max_idx)  # all nodes except node 0 should appear and be compactly indexed
assert(n_feat.shape[0] == max_idx + 1)  # the nodes need to map one-to-one to the node feat matrix

# split and pack the data by generating valid train/val/test mask according to the "mode"
val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))
if args.mode == 't':
  logger.info('Transductive training...')
  valid_train_flag = (ts_l <= val_time)
  valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
  valid_test_flag = ts_l > test_time

else:
  assert(args.mode == 'i')
  logger.info('Inductive training...')
  # pick some nodes to mask (i.e. reserved for testing) for inductive setting
  total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
  num_total_unique_nodes = len(total_node_set)
  mask_node_set = set(random.sample(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])), int(0.1 * num_total_unique_nodes)))
  mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
  mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
  none_mask_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)
  valid_train_flag = (ts_l <= val_time) * (none_mask_node_flag > 0.5)
  valid_val_flag = (ts_l <= test_time) * (ts_l > val_time) * (none_mask_node_flag > 0.5)  # both train and val edges can not contain any masked nodes
  valid_test_flag = (ts_l > test_time) * (none_mask_node_flag < 0.5)  # test edges must contain at least one masked node
  valid_test_new_new_flag = (ts_l > test_time) * mask_src_flag * mask_dst_flag
  valid_test_new_old_flag = (valid_test_flag.astype(int) - valid_test_new_new_flag.astype(int)).astype(bool)
  logger.info('Sampled {} nodes (10 %) which are masked in training and reserved for testing...'.format(len(mask_node_set)))

# split data according to the mask
if T_BATCH:
  train_tb_l, val_tb_l, test_tb_l = tbatch_id_l[valid_train_flag], tbatch_id_l[valid_val_flag], tbatch_id_l[valid_test_flag]
else:
  train_tb_l, val_tb_l, test_tb_l = [None]*3
train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l, train_src_e_l, train_tgt_e_l, train_src_start_l, train_tgt_start_l, train_src_ngh_n_l, train_tgt_ngh_n_l, train_tgt_post_n_l = src_l[valid_train_flag], dst_l[valid_train_flag], ts_l[valid_train_flag], e_idx_l[valid_train_flag], label_l[valid_train_flag], src_e_l[valid_train_flag], tgt_e_l[valid_train_flag], src_start_l[valid_train_flag], tgt_start_l[valid_train_flag], src_ngh_n_l[valid_train_flag], tgt_ngh_n_l[valid_train_flag], tgt_post_n_l[valid_train_flag]
val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l, val_src_e_l, val_tgt_e_l, val_src_start_l, val_tgt_start_l, val_src_ngh_n_l, val_tgt_ngh_n_l, val_tgt_post_n_l = src_l[valid_val_flag], dst_l[valid_val_flag], ts_l[valid_val_flag], e_idx_l[valid_val_flag], label_l[valid_val_flag], src_e_l[valid_val_flag], tgt_e_l[valid_val_flag], src_start_l[valid_val_flag], tgt_start_l[valid_val_flag], src_ngh_n_l[valid_val_flag], tgt_ngh_n_l[valid_val_flag], tgt_post_n_l[valid_val_flag]
test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l, test_src_e_l, test_tgt_e_l, test_src_start_l, test_tgt_start_l, test_src_ngh_n_l, test_tgt_ngh_n_l, test_tgt_post_n_l = src_l[valid_test_flag], dst_l[valid_test_flag], ts_l[valid_test_flag], e_idx_l[valid_test_flag], label_l[valid_test_flag], src_e_l[valid_test_flag], tgt_e_l[valid_test_flag], src_start_l[valid_test_flag], tgt_start_l[valid_test_flag], src_ngh_n_l[valid_test_flag], tgt_ngh_n_l[valid_test_flag], tgt_post_n_l[valid_test_flag]
if args.mode == 'i':
  if T_BATCH:
    test_tb_new_new_l, test_tb_new_old_l = tbatch_id_l[valid_test_new_new_flag], tbatch_id_l[valid_test_new_old_flag]
    test_tb_new_new_l, test_tb_new_old_l = [None]*2
  test_src_new_new_l, test_dst_new_new_l, test_ts_new_new_l, test_e_idx_new_new_l, test_label_new_new_l, test_src_e_new_new_l, test_tgt_e_new_new_l, test_src_start_new_new_l, test_tgt_start_new_new_l, test_src_ngh_n_new_new_l, test_tgt_ngh_n_new_new_l, test_tgt_post_n_new_new_l = src_l[valid_test_new_new_flag], dst_l[valid_test_new_new_flag], ts_l[valid_test_new_new_flag], e_idx_l[valid_test_new_new_flag], label_l[valid_test_new_new_flag], src_e_l[valid_test_new_new_flag], tgt_e_l[valid_test_new_new_flag], src_start_l[valid_test_new_new_flag], tgt_start_l[valid_test_new_new_flag], src_ngh_n_l[valid_test_new_new_flag], tgt_ngh_n_l[valid_test_new_new_flag], tgt_post_n_l[valid_test_new_new_flag]
  test_src_new_old_l, test_dst_new_old_l, test_ts_new_old_l, test_e_idx_new_old_l, test_label_new_old_l, test_src_e_new_old_l, test_tgt_e_new_old_l, test_src_start_new_old_l, test_tgt_start_new_old_l, test_src_ngh_n_new_old_l, test_tgt_ngh_n_new_old_l, test_tgt_post_n_new_old_l = src_l[valid_test_new_old_flag], dst_l[valid_test_new_old_flag], ts_l[valid_test_new_old_flag], e_idx_l[valid_test_new_old_flag], label_l[valid_test_new_old_flag], src_e_l[valid_test_new_old_flag], tgt_e_l[valid_test_new_old_flag], src_start_l[valid_test_new_old_flag], tgt_start_l[valid_test_new_old_flag], src_ngh_n_l[valid_test_new_old_flag], tgt_ngh_n_l[valid_test_new_old_flag], tgt_post_n_l[valid_test_new_old_flag]
train_data = train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l, train_src_e_l, train_tgt_e_l, train_src_start_l, train_tgt_start_l, train_src_ngh_n_l, train_tgt_ngh_n_l, train_tgt_post_n_l
val_data = val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l, val_src_e_l, val_tgt_e_l, val_src_start_l, val_tgt_start_l, val_src_ngh_n_l, val_tgt_ngh_n_l, val_tgt_post_n_l
train_val_data = (train_data, val_data)
train_val_tbatch = (train_tb_l, val_tb_l)
# create two neighbor finders to handle graph extraction.
# for transductive mode all phases use full_ngh_finder, for inductive node train/val phases use the partial one
# while test phase still always uses the full one

# # create random samplers to generate train/val/test instances
train_rand_sampler = RandEdgeSampler((train_src_l, ), (train_src_start_l, ), (train_src_ngh_n_l, ), (train_dst_l, ), (train_tgt_start_l, ), (train_tgt_ngh_n_l, ))
val_rand_sampler = RandEdgeSampler((train_src_l, val_src_l), (train_src_start_l, val_src_start_l), (train_src_ngh_n_l, val_src_ngh_n_l), (train_dst_l, val_dst_l), (train_tgt_start_l, val_tgt_start_l), (train_tgt_ngh_n_l, val_tgt_ngh_n_l))
test_rand_sampler = RandEdgeSampler((train_src_l, val_src_l, test_src_l), (train_src_start_l, val_src_start_l, test_src_start_l), (train_src_ngh_n_l, val_src_ngh_n_l, test_src_ngh_n_l), (train_dst_l, val_dst_l, test_dst_l), (train_tgt_start_l, val_tgt_start_l, test_tgt_start_l), (train_tgt_ngh_n_l, val_tgt_ngh_n_l, test_tgt_ngh_n_l))
rand_samplers = train_rand_sampler, val_rand_sampler

# multiprocessing memory setting
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (200*args.bs, rlimit[1]))

# model initialization
device = torch.device('cuda:{}'.format(GPU))
# cawn = CAWN(n_feat, e_feat, agg=AGG,
#       num_layers=NUM_LAYER, use_time=USE_TIME, attn_agg_method=ATTN_AGG_METHOD, attn_mode=ATTN_MODE,
#       n_head=ATTN_NUM_HEADS, drop_out=DROP_OUT, pos_dim=POS_DIM, pos_enc=POS_ENC,
#       num_neighbors=NUM_NEIGHBORS, walk_n_head=WALK_N_HEAD, walk_mutual=WALK_MUTUAL, walk_linear_out=args.walk_linear_out,
#       cpu_cores=CPU_CORES, verbosity=VERBOSITY, get_checkpoint_path=get_checkpoint_path)
cawn = CAWN2(n_feat, e_feat, pos_dim=POS_DIM, n_head=ATTN_NUM_HEADS, num_neighbors=NUM_NEIGHBORS, dropout=DROP_OUT, walk_linear_out=args.walk_linear_out, get_checkpoint_path=get_checkpoint_path, get_ngh_store_path=get_ngh_store_path, verbosity=VERBOSITY)
cawn.to(device)
feat_dim = n_feat.shape[1]
e_feat_dim = e_feat.shape[1]
time_dim = n_feat.shape[1]
model_dim = feat_dim + e_feat_dim + time_dim
# neighborhood_store.append(torch.sparse_coo_tensor(size=(num_nodes, num_nodes, model_dim),requires_grad=False).to(device)) # sparse tensor (node_idx, neighbor_idx, encoded_features)
# neighborhood_store.append(torch.sparse_coo_tensor(size=(num_nodes, num_nodes, model_dim),requires_grad=False).to(device))
# neighborhood_store = torch.sparse_coo_tensor([[0], [0]], torch.zeros(1, 2, model_dim), (num_nodes, num_nodes, 2, model_dim)).to(device)
# neighborhood_store = {}
neighborhood_store = torch.zeros(max_e_idx, e_feat_dim + time_dim + 3, device=device)
cawn.set_device(device)
cawn.set_neighborhood_store(neighborhood_store)
optimizer = torch.optim.Adam(cawn.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()
early_stopper = EarlyStopMonitor(tolerance=TOLERANCE)

# start train and val phases
# train_val(train_val_data, cawn, args.mode, BATCH_SIZE, NUM_EPOCH, criterion, optimizer, early_stopper, ngh_finders, rand_samplers, logger)

#YL optimized sampler
train_val(train_val_data, cawn, args.mode, BATCH_SIZE, NUM_EPOCH, criterion, optimizer, early_stopper, rand_samplers, logger, model_dim, t_batch = train_val_tbatch)

# final testing
# cawn.update_ngh_finder(full_ngh_finder)  # remember that testing phase should always use the full neighbor finder
print("_*"*50)
test_acc, test_ap, test_f1, test_auc = eval_one_epoch('test for {} nodes'.format(args.mode), cawn, test_rand_sampler, test_src_l, test_dst_l, test_ts_l, test_label_l, test_e_idx_l, test_src_e_l, test_tgt_e_l, test_src_start_l, test_tgt_start_l, test_src_ngh_n_l, test_tgt_ngh_n_l, test_tgt_post_n_l, tb=test_tb_l)
logger.info('Test statistics: {} all nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_acc, test_auc, test_ap))
test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc = [-1]*6
if args.mode == 'i':
  test_new_new_acc, test_new_new_ap, test_new_new_f1, test_new_new_auc = eval_one_epoch('test for {} nodes'.format(args.mode), cawn, test_rand_sampler, test_src_new_new_l, test_dst_new_new_l, test_ts_new_new_l, test_label_new_new_l, test_e_idx_new_new_l, test_src_e_new_new_l, test_tgt_e_new_new_l, test_src_start_new_new_l, test_tgt_start_new_new_l, test_src_ngh_n_new_new_l, test_tgt_ngh_n_new_new_l, test_tgt_post_n_new_new_l, tb=test_tb_new_new_l)
  logger.info('Test statistics: {} new-new nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_new_new_acc, test_new_new_ap, test_new_new_auc ))
  test_new_old_acc, test_new_old_ap, test_new_old_f1, test_new_old_auc = eval_one_epoch('test for {} nodes'.format(args.mode), cawn, test_rand_sampler, test_src_new_old_l, test_dst_new_old_l, test_ts_new_old_l, test_label_new_old_l, test_e_idx_new_old_l, test_src_e_new_old_l, test_tgt_e_new_old_l, test_src_start_new_old_l, test_tgt_start_new_old_l, test_src_ngh_n_new_old_l, test_tgt_ngh_n_new_old_l, test_tgt_post_n_new_old_l, tb=test_tb_new_old_l)
  logger.info('Test statistics: {} new-old nodes -- acc: {}, auc: {}, ap: {}'.format(args.mode, test_new_old_acc, test_new_old_ap, test_new_old_auc))

# save model
logger.info('Saving CAWN model ...')
torch.save(cawn.state_dict(), best_model_path)
logger.info('CAWN model saved')

# save one line result
save_oneline_result('log/', args, [test_acc, test_auc, test_ap, test_new_new_acc, test_new_new_ap, test_new_new_auc, test_new_old_acc, test_new_old_ap, test_new_old_auc])
# save walk_encodings_scores
# checkpoint_dir = '/'.join(cawn.get_checkpoint_path(0).split('/')[:-1])
# cawn.save_walk_encodings_scores(checkpoint_dir)
