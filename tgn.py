import os.path as osp

import torch
from torch.nn import Linear
from sklearn.metrics import average_precision_score, roc_auc_score

from torch_geometric.datasets import JODIEDataset
from torch_geometric.nn import TGNMemory, TransformerConv
from torch_geometric.nn.models.tgn import (LastNeighborLoader, IdentityMessage,
                                           LastAggregator)
from torch_geometric.data import TemporalData
from tqdm import tqdm
import pandas as pd
import time
import numpy as np
import random
import statistics
from utils import EarlyStopMonitor, mat_results
import logging

class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super(GraphAttentionEmbedding, self).__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 4, heads=4,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super(LinkPredictor, self).__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)

def train():
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    for batch in tqdm(train_data.seq_batches(batch_size=100)):
        optimizer.zero_grad()

        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        # Sample negative destination nodes.
        neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0), ),
                                dtype=torch.long, device=device)

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id], data.msg[e_id])

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        neg_out = link_pred(z[assoc[src]], z[assoc[neg_dst]])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events


@torch.no_grad()
def test(inference_data):
    memory.eval()
    gnn.eval()
    link_pred.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    aps, aucs = [], []
    for batch in tqdm(inference_data.seq_batches(batch_size=30)):
        src, pos_dst, t, msg = batch.src, batch.dst, batch.t, batch.msg

        neg_dst = torch.randint(min_dst_idx, max_dst_idx + 1, (src.size(0), ),
                                dtype=torch.long, device=device)

        n_id = torch.cat([src, pos_dst, neg_dst]).unique()
        n_id, edge_index, e_id = neighbor_loader(n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id], data.msg[e_id])

        pos_out = link_pred(z[assoc[src]], z[assoc[pos_dst]])
        neg_out = link_pred(z[assoc[src]], z[assoc[neg_dst]])

        y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)),
             torch.zeros(neg_out.size(0))], dim=0)

        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))

        memory.update_state(src, pos_dst, t, msg)
        neighbor_loader.insert(src, pos_dst)

    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean())
random.seed(2022)
get_checkpoint_path = lambda \
    checkpoint_name, epoch: f'./saved_checkpoints/tgn-{checkpoint_name}-{epoch}.pth'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
file_path = 'log/{}.log'.format('tgn') #TODO: improve log name
fh = logging.FileHandler(file_path)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info('Create log file at {}'.format(file_path))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'JODIE')
# dataset = JODIEDataset('./processed/', name='wikipedia')
# data = dataset[0].to(device)
DATA = 'wiki-talk-temporal'
g_df = pd.read_csv('./processed/ml_{}.csv'.format(DATA))
e_feat = np.load('./processed/ml_{}.npy'.format(DATA))
e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)), requires_grad=False)
edge_raw_embed = torch.nn.Embedding.from_pretrained(e_feat_th, padding_idx=0, freeze=True)
src_l = g_df.u.values
dst_l = g_df.i.values
e_idx_l = g_df.idx.values
label_l = g_df.label.values
ts_l = g_df.ts.values
max_idx = max(src_l.max(), dst_l.max())
assert(np.unique(np.stack([src_l, dst_l])).shape[0] == max_idx or ~math.isclose(1, args.data_usage))  # all nodes except node 0 should appear and be compactly indexed

# split and pack the data by generating valid train/val/test mask according to the "mode"
val_time, test_time = list(np.quantile(g_df.ts, [0.70, 0.85]))

for i in range(1):
    if i == 0:
        mode = 't'
    else:
        mode = 'i'
    total_node_set = set(np.unique(np.hstack([g_df.u.values, g_df.i.values])))
    if mode == 't':
        valid_train_flag = (ts_l <= val_time)
        valid_val_flag = (ts_l <= test_time) * (ts_l > val_time)
        valid_test_flag = ts_l > test_time

    else:
        assert(mode == 'i')
        # pick some nodes to mask (i.e. reserved for testing) for inductive setting
        
        num_total_unique_nodes = len(total_node_set)
        mask_node_set = set(random.sample(set(src_l[ts_l > val_time]).union(set(dst_l[ts_l > val_time])), int(0.1 * num_total_unique_nodes)))
        mask_src_flag = g_df.u.map(lambda x: x in mask_node_set).values
        mask_dst_flag = g_df.i.map(lambda x: x in mask_node_set).values
        none_mask_node_flag = (1 - mask_src_flag) * (1 - mask_dst_flag)
        total_valid_train_val_flag = ts_l <= test_time
        valid_train_flag = (ts_l <= val_time) * (none_mask_node_flag > 0.5)
        valid_val_flag = (ts_l <= test_time) * (ts_l > val_time) * (none_mask_node_flag > 0.5)  # both train and val edges can not contain any masked nodes
        valid_test_flag = (ts_l > test_time) * (none_mask_node_flag < 0.5)  # test edges must contain at least one masked node
        total_src_l, total_dst_l, total_ts_l, total_e_idx_l, total_label_l = src_l[total_valid_train_val_flag], dst_l[total_valid_train_val_flag], ts_l[total_valid_train_val_flag], e_idx_l[total_valid_train_val_flag], label_l[total_valid_train_val_flag]
        total_data = TemporalData(
        src=torch.from_numpy(total_src_l).to(torch.long).to(device),
        dst=torch.from_numpy(total_dst_l).to(torch.long).to(device),
        t=torch.from_numpy(total_ts_l).to(torch.long).to(device),
        msg=edge_raw_embed(torch.from_numpy(total_e_idx_l)).to(device),
        y=torch.from_numpy(total_label_l).to(torch.long).to(device)
    )
    train_src_l, train_dst_l, train_ts_l, train_e_idx_l, train_label_l = src_l[valid_train_flag], dst_l[valid_train_flag], ts_l[valid_train_flag], e_idx_l[valid_train_flag], label_l[valid_train_flag]
    val_src_l, val_dst_l, val_ts_l, val_e_idx_l, val_label_l = src_l[valid_val_flag], dst_l[valid_val_flag], ts_l[valid_val_flag], e_idx_l[valid_val_flag], label_l[valid_val_flag]
    test_src_l, test_dst_l, test_ts_l, test_e_idx_l, test_label_l = src_l[valid_test_flag], dst_l[valid_test_flag], ts_l[valid_test_flag], e_idx_l[valid_test_flag], label_l[valid_test_flag]

    data = TemporalData(src=torch.from_numpy(g_df.u.values).to(torch.long), dst=torch.from_numpy(g_df.i.values).to(torch.long), t=torch.from_numpy(g_df.ts.values).to(torch.long), msg=edge_raw_embed(torch.from_numpy(g_df.idx.values)), y=torch.from_numpy(g_df.label.values).to(torch.long))

    data = data.to(device)
    # min_dst_idx, max_dst_idx = int(dst_l.min()), int(dst_l.max())
    min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())
    train_data = TemporalData(
        src=torch.from_numpy(train_src_l).to(torch.long).to(device),
        dst=torch.from_numpy(train_dst_l).to(torch.long).to(device),
        t=torch.from_numpy(train_ts_l).to(torch.long).to(device),
        msg=edge_raw_embed(torch.from_numpy(train_e_idx_l)).to(device),
        y=torch.from_numpy(train_label_l).to(torch.long).to(device)
    )

    val_data = TemporalData(
        src=torch.from_numpy(val_src_l).to(torch.long).to(device),
        dst=torch.from_numpy(val_dst_l).to(torch.long).to(device),
        t=torch.from_numpy(val_ts_l).to(torch.long).to(device),
        msg=edge_raw_embed(torch.from_numpy(val_e_idx_l)).to(device),
        y=torch.from_numpy(val_label_l).to(torch.long).to(device)
    )

    test_data = TemporalData(
        src=torch.from_numpy(test_src_l).to(torch.long).to(device),
        dst=torch.from_numpy(test_dst_l).to(torch.long).to(device),
        t=torch.from_numpy(test_ts_l).to(torch.long).to(device),
        msg=edge_raw_embed(torch.from_numpy(test_e_idx_l)).to(device),
        y=torch.from_numpy(test_label_l).to(torch.long).to(device)
    )
    # train_data, val_data, test_data = data.train_val_test_split(
    #     val_ratio=0.15, test_ratio=0.15)
    



    memory_dim = time_dim = embedding_dim = 100

    transductive_auc = []
    transductive_ap = []
    inductive_auc = []
    inductive_ap = []
    test_times = []
    early_stoppers = []
    total_time = []
    n_run = 5
    for run in range(n_run):
        neighbor_loader = LastNeighborLoader(data.num_nodes, size=32, device=device)
        val_best = 0
        best_epoch = 0
        counter = 0
        test_ap, test_auc = 0, 0
        test_start, test_end = 0, 0
        total_start = time.time()
        memory = TGNMemory(
            data.num_nodes,
            data.msg.size(-1),
            memory_dim,
            time_dim,
            message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
            aggregator_module=LastAggregator(),
        ).to(device)

        gnn = GraphAttentionEmbedding(
            in_channels=memory_dim,
            out_channels=embedding_dim,
            msg_dim=data.msg.size(-1),
            time_enc=memory.time_enc,
        ).to(device)

        link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

        optimizer = torch.optim.Adam(
            set(memory.parameters()) | set(gnn.parameters())
            | set(link_pred.parameters()), lr=0.0001)
        criterion = torch.nn.BCEWithLogitsLoss()

        # Helper vector to map global node indices to local ones.
        assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)
        early_stopper = EarlyStopMonitor(max_round=3)
        train_time = []
        for epoch in range(1, 50):
            train_start = time.time()
            loss = train()
            train_end = time.time()
            logger.info(f'  Epoch: {epoch:02d}, Loss: {loss:.4f}, Time: {train_end - train_start}')
            train_time.append(train_end - train_start)
            mat_results(logger, train_time, "train_time")
            start = time.time()
            val_ap, val_auc = test(val_data)
            end = time.time()
            logger.info(f' Val AP: {val_ap:.4f},  Val AUC: {val_auc:.4f}, Time: {end - start}')
            if early_stopper.early_stop_check(val_ap):
                logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
                best_model_path = get_checkpoint_path('memory', early_stopper.best_epoch)
                memory.load_state_dict(torch.load(best_model_path))
                best_model_path = get_checkpoint_path('gnn', early_stopper.best_epoch)
                gnn.load_state_dict(torch.load(best_model_path))
                best_model_path = get_checkpoint_path('link_pred', early_stopper.best_epoch)
                link_pred.load_state_dict(torch.load(best_model_path))
                logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
                break
            else:
                torch.save(memory.state_dict(), get_checkpoint_path('memory', epoch))
                torch.save(gnn.state_dict(), get_checkpoint_path('gnn', epoch))
                torch.save(link_pred.state_dict(), get_checkpoint_path('link_pred', epoch))
        if mode == 'i':
            memory.reset_state()  # Start with a fresh memory.
            neighbor_loader.reset_state()  # Start with an empty graph.
            pre_ap, pre_auc = test(total_data)
        test_start = time.time()
        test_ap, test_auc = test(test_data)
        test_end = time.time()
        logger.info(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}, Time: {test_end - test_start}')
        total_end = time.time()
        logger.info("MAT experiment statistics:")
        if mode == "t":
            transductive_auc.append(test_auc)
            transductive_ap.append(test_ap)
            mat_results(logger, transductive_auc, "transductive_auc")
            mat_results(logger, transductive_ap, "transductive_ap")
        else:
            inductive_auc.append(test_auc)
            inductive_ap.append(test_ap)
            mat_results(logger, inductive_auc, "inductive_auc")
            mat_results(logger, inductive_ap, "inductive_ap")
        test_times.append(test_end - test_start)
        early_stoppers.append(early_stopper.best_epoch)
        total_time.append(total_end - total_start)
        mat_results(logger, test_times, "test_times")
        mat_results(logger, early_stoppers, "early_stoppers")
        mat_results(logger, total_time, "total_time")