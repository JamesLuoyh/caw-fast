import torch
import torch.nn as nn
import logging
import time
import numpy as np
import random
from GAT import GAT
from torch.utils.data import WeightedRandomSampler

class CAWN2(torch.nn.Module):
  def __init__(self, n_feat, e_feat, memory_dim, total_nodes, pos_dim=0, n_head=4, num_neighbors=['32'], history_agg_method='gru',
      dropout=0.1, walk_linear_out=False, get_checkpoint_path=None, get_ngh_store_path=None, verbosity=1, seed=1, agg_time_delta=False,
      backprop_n_history=1, n_layers=1):
    super(CAWN2, self).__init__()
    self.logger = logging.getLogger(__name__)
    self.dropout = dropout
    self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)), requires_grad=False)
    self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)), requires_grad=False)
    self.feat_dim = self.n_feat_th.shape[1]  # node feature dimension
    self.e_feat_dim = self.e_feat_th.shape[1]  # edge feature dimension
    self.time_dim = self.feat_dim  # default to be time feature dimension
    # embedding layers and encoders
    self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
    self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)
    self.time_encoder = self.init_time_encoder() # fourier
    self.rev_time_encoder = self.init_time_encoder() # fourier
    self.history_agg_method = history_agg_method
    self.feature_encoder = self.init_feature_encoder() # RNNCell
    self.time_aggregator = self.init_time_aggregator() # RNNCell
    self.backprop_n_history = backprop_n_history
    if history_agg_method == 'lstm':
      self.e_feat_dim *= 2
      self.time_dim *= 2
    self.pos_dim = pos_dim
    self.trainable_embedding = nn.Embedding(num_embeddings=30, embedding_dim=self.pos_dim) # position embedding
   
    # final projection layer
    self.walk_linear_out = walk_linear_out
    self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1, non_linear=not self.walk_linear_out) #torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)
    self.get_checkpoint_path = get_checkpoint_path
    self.get_ngh_store_path = get_ngh_store_path
    self.src_idx_l_prev = self.tgt_idx_l_prev = self.cut_time_l_prev = self.e_idx_l_prev = None
    self.prev_raw_data = {}
    self.num_neighbors = num_neighbors
    self.n_layers = n_layers
    self.ngh_id_idx = 0
    self.e_raw_idx = 1
    self.ts_raw_idx = 2
    self.num_raw = 3
    if backprop_n_history > 1:
      self.pr_e_raw_idx = 3
      self.pr_ts_raw_idx = 4
      self.num_raw += 2
    self.e_emb_idx = [self.num_raw, self.num_raw + self.e_feat_dim]
    self.ts_emb_idx = [self.e_emb_idx[1], self.e_emb_idx[1] + self.time_dim]
    self.parent_idx = self.ts_emb_idx[1]
    self.agg_time_delta = agg_time_delta      
    if self.agg_time_delta:
      self.ts_diff_emb_idx = [self.ts_emb_idx[1], self.ts_emb_idx[1] + self.time_dim]
      self.parent_idx = self.ts_diff_emb_idx[1]
      self.rev_time_aggregator = self.init_time_aggregator() # RNNCell
      self.hidden_dim = self.e_feat_dim + 2 * self.time_dim
    else:
      self.hidden_dim = self.e_feat_dim + self.time_dim
    self.memory_dim = memory_dim
    self.verbosity = verbosity
    self.caw_dim = self.hidden_dim + 1
    if self.n_layers > 1:
      self.attn_dim = 2*self.feat_dim + 3 * (self.hidden_dim) + 2 * self.pos_dim
    else:
      self.attn_dim = self.feat_dim + 2 * (self.hidden_dim) + 1 * self.pos_dim
    # self.attn_m = AttnModel(self.feat_dim, 0, self.attn_dim, n_head=n_head, drop_out=dropout)
    self.gat = GAT(1, [n_head], [self.attn_dim, self.feat_dim], add_skip_connection=False, bias=True,
                 dropout=dropout, log_attention_weights=False)
    self.total_nodes = total_nodes

  def set_seed(self, seed):
    self.seed = seed

  def clear_store(self):
    self.neighborhood_store = None

  def reset_store(self):
    [n.zero_() for n in self.neighborhood_store]
    ngh_l1 = len(self.neighborhood_store[0])
    self.neighborhood_store[0] = None
    num_raw = 3
    if self.backprop_n_history > 1:
      num_raw += 2
    raw_store = torch.zeros(ngh_l1, num_raw)
    hidden_store = torch.empty(ngh_l1, self.hidden_dim)
    self.neighborhood_store[0] = torch.cat((raw_store, nn.init.xavier_uniform_(hidden_store)), -1).to(self.device)
    if self.n_layers > 1:
      ngh_l2 = len(self.neighborhood_store[1])
      self.neighborhood_store[1] = None
      raw_store = torch.zeros(ngh_l2, num_raw)
      hidden_store = torch.empty(ngh_l2, self.hidden_dim) # plus 1 for storing parent id
      parent_store = torch.empty(ngh_l2, 1)
      self.neighborhood_store[1] = torch.cat((raw_store, nn.init.xavier_uniform_(hidden_store), parent_store), -1).to(self.device)
    self.num_neighbors_stored = torch.ones_like(self.num_neighbors_stored)
  
  def get_neighborhood_store(self):
    return self.neighborhood_store

  def set_neighborhood_store(self, neighborhood_store):
    self.neighborhood_store = neighborhood_store
    # self.ngh_encoder.update_neighborhood_store(neighborhood_store)
  def set_start_idx(self, start_idx):
    self.start_idx = start_idx
  def set_num_neighbors_stored(self, num_neighbors_stored):
    self.num_neighbors_stored = num_neighbors_stored

  def set_device(self, device):
    self.device = device

  def log_time(self, desc, start, end):
    if self.verbosity > 1:
      self.logger.info('{} for the minibatch, time eclipsed: {} seconds'.format(desc, str(end-start)))
  
  def contrast(self, src_l, tgt_l, bad_l, cut_time_l, e_idx_l=None, test=False):
    device = self.n_feat_th.device
    start = time.time()
    start_t = time.time()
    src_l_cut, src_e_l, src_start_l, src_ngh_n_l = src_l
    tgt_l_cut, tgt_e_l, tgt_start_l, tgt_ngh_n_l = tgt_l
    bad_l_cut, bad_e_l, bad_start_l, bad_ngh_n_l = bad_l

    batch_size = len(src_l_cut)
    start_l = np.concatenate((src_start_l, tgt_start_l, bad_start_l), 0)
    ngh_n_l = np.concatenate((src_ngh_n_l, tgt_ngh_n_l, bad_ngh_n_l), 0)
    src_th = torch.from_numpy(src_l_cut).to(dtype=torch.long, device=device)
    tgt_th = torch.from_numpy(tgt_l_cut).to(dtype=torch.long, device=device)
    bad_th = torch.from_numpy(bad_l_cut).to(dtype=torch.long, device=device)
    src_start_th = torch.from_numpy(src_start_l).to(dtype=torch.long, device=device)
    tgt_start_th = torch.from_numpy(tgt_start_l).to(dtype=torch.long, device=device)
    bad_start_th = torch.from_numpy(bad_start_l).to(dtype=torch.long, device=device)
    src_ngh_n_th = torch.from_numpy(src_ngh_n_l).to(dtype=torch.long, device=device)
    tgt_ngh_n_th = torch.from_numpy(tgt_ngh_n_l).to(dtype=torch.long, device=device)
    bad_ngh_n_th = torch.from_numpy(bad_ngh_n_l).to(dtype=torch.long, device=device)
    src_nghs = torch.sum(src_ngh_n_th)
    tgt_nghs = torch.sum(tgt_ngh_n_th)
    bad_nghs = torch.sum(bad_ngh_n_th)
    start_th = torch.cat((src_start_th, tgt_start_th, bad_start_th), 0)
    ngh_n_th = torch.cat((src_ngh_n_th, tgt_ngh_n_th, bad_ngh_n_th), 0)
    idx_th = torch.cat((src_th, tgt_th, bad_th), 0)
    if self.n_layers > 1:
      num_ngh_l2 = self.num_neighbors[1] * batch_size

    cut_time_th = torch.from_numpy(cut_time_l).to(dtype=torch.float, device=device)
    e_idx_th = torch.from_numpy(e_idx_l).to(dtype=torch.long, device=device)
    end = time.time()
    self.log_time('init 0', start, end)
    start = time.time()
    batch_idx = torch.arange(batch_size * 3, device=device)
    parent_node_feat = self.node_raw_embed(idx_th)
    ngh_data = torch.repeat_interleave(torch.cat((cut_time_th.repeat(3).unsqueeze(1), idx_th.unsqueeze(1), batch_idx.unsqueeze(1), parent_node_feat), -1), ngh_n_th, dim=0)
    ts = ngh_data[:, 0]
    ori_idx = ngh_data[:, 1].long()
    sparse_idx = ngh_data[:, 2].long()
    parent_node_feat = ngh_data[:, 3:]
    # ori_idx = torch.repeat_interleave(idx_th, ngh_n_th, dim=0)
    # end = time.time()
    # self.log_time('repeat_interleave', start, end)
    # start = time.time()
    # sparse_idx = torch.repeat_interleave(sparse_idx, ngh_n_th, dim=0)
    
    end = time.time()
    self.log_time('init 1', start, end)
    start = time.time()

    self.neighborhood_store[0][start_th,0] = idx_th.float()
    # self.neighborhood_store[0][start_th,-1] = cut_time_th.repeat(3)

    # n_id is the node idx of neighbors of query node
    # dense_idx is the position of each neighbors in the batch*nngh tensor
    # sprase_idx is a tensor of batch idx repeated with ngh_n timesfor each node
    updated_mem, n_pos_idx, pr_hidden_states = self.get_updated_memory(idx_th, ori_idx, ts, start_l, ngh_n_l, device)

    ngh_id = updated_mem[:, self.ngh_id_idx].long()
    src_msk = torch.cat((torch.zeros(src_nghs, device = device), torch.ones(tgt_nghs + bad_nghs, device = device)), 0)
    pos_raw = (ngh_id != 0) * ((ngh_id != ori_idx).long() + 2) #src_msk + 
    # pos_encoding = self.trainable_embedding(pos_raw.long())
    node_features = self.node_raw_embed(ngh_id)
    if self.n_layers > 1:
      updated_mem_l2 = self.get_updated_memory_l2(idx_th, cut_time_th.repeat(3), device)
      ngh_id_l2 = updated_mem_l2[:, self.ngh_id_idx].long()
      node_features_l2 = self.node_raw_embed(ngh_id_l2)
      parent_id_l2 = updated_mem_l2[:, self.parent_idx].long()
      parent_node_feat_l2 = self.node_raw_embed(parent_id_l2)
      src_msk = torch.cat((torch.zeros(num_ngh_l2, device = device), torch.ones(2 * num_ngh_l2, device = device)), 0) # 4, 8, 5,10
      pos_raw_l2 = (ngh_id_l2 != 0) * (ngh_id_l2 != idx_th.repeat_interleave(self.num_neighbors[1])).long() * (7) # + src_msk) # 17 18
      src_n_f_l2 = node_features_l2[0:num_ngh_l2]
      tgt_n_f_l2 = node_features_l2[num_ngh_l2:2 * num_ngh_l2]
      bad_n_f_l2 = node_features_l2[2 * num_ngh_l2:]

      # pos_encoding_l2 = self.trainable_embedding(pos_raw_l2.long())
    # node_features *= (ngh_id != 0).repeat(self.feat_dim, 1).T
    end = time.time()
    self.log_time('retrieve memory', start, end)
    start = time.time()

    src_n_f = node_features[0:src_nghs]
    tgt_n_f = node_features[src_nghs:src_nghs+tgt_nghs]
    bad_n_f = node_features[src_nghs+tgt_nghs:]
    
    if self.agg_time_delta:
      hidden_states = updated_mem[:, self.e_emb_idx[0]: self.ts_diff_emb_idx[1]]
    else:
      hidden_states = updated_mem[:, self.e_emb_idx[0]: self.ts_emb_idx[1]]
    hidden_states = torch.cat((hidden_states, pos_raw.unsqueeze(1)), -1)
    src_prev_f = hidden_states[0:src_nghs]
    tgt_prev_f = hidden_states[src_nghs:src_nghs + tgt_nghs]
    bad_prev_f = hidden_states[src_nghs + tgt_nghs:]

    src_ngh_id = ngh_id[0:src_nghs]
    tgt_ngh_id = ngh_id[src_nghs:src_nghs + tgt_nghs]
    bad_ngh_id = ngh_id[src_nghs + tgt_nghs:]
    src_sparse_idx = sparse_idx[0:src_nghs]
    src_n_sparse_idx = src_sparse_idx + batch_size
    tgt_sparse_idx = sparse_idx[src_nghs:src_nghs + tgt_nghs] - batch_size
    tgt_bad_sparse_idx = sparse_idx[src_nghs:] - batch_size
    bad_sparse_idx = sparse_idx[src_nghs + tgt_nghs:] - 2 * batch_size
    end = time.time()
    self.log_time('caw prep', start, end)
    start = time.time()
    # caw = self.get_relative_id(torch.cat((src_sparse_idx, src_n_sparse_idx), 0), tgt_bad_sparse_idx, src_n_id.repeat(2), torch.cat((tgt_n_id, bad_n_id), 0), torch.cat((src_e_f,src_ts_f), -1).repeat(2, 1), torch.cat((torch.cat((tgt_e_f,tgt_ts_f), -1), torch.cat((bad_e_f,bad_ts_f), -1)), 0))
    caw = self.get_relative_id(torch.cat((src_sparse_idx, src_n_sparse_idx), 0), tgt_bad_sparse_idx, src_ngh_id.repeat(2), torch.cat((tgt_ngh_id, bad_ngh_id), 0), src_prev_f.repeat(2, 1), torch.cat((tgt_prev_f, bad_prev_f), 0))
    if self.pos_dim == 0:
      caw = caw[:, :-1]
    else:
      pos_raw = caw[:, -1]
      pos_encoding = self.trainable_embedding(pos_raw.long())
      caw = torch.cat((caw[:, :-1], pos_encoding), -1)
    # 2nd hop neighbors
    if self.n_layers > 1:
      sparse_idx_l2 = torch.arange(num_ngh_l2 * 2, device=device)
      if self.agg_time_delta:
        hidden_states_l2 = updated_mem_l2[:, self.e_emb_idx[0]: self.ts_diff_emb_idx[1]]
      else:
        hidden_states_l2 = updated_mem_l2[:, self.e_emb_idx[0]: self.ts_emb_idx[1]]
      hidden_states_l2 = torch.cat((hidden_states_l2, pos_raw_l2.unsqueeze(1)), -1)
      src_hidden_l2 = hidden_states_l2[:num_ngh_l2]
      tgt_hidden_l2 = hidden_states_l2[num_ngh_l2 : 2 * num_ngh_l2]
      bad_hidden_l2 = hidden_states_l2[-num_ngh_l2:]
      caw_l2 = self.get_relative_id(sparse_idx_l2, sparse_idx_l2, ngh_id_l2[:num_ngh_l2].repeat(2), ngh_id_l2[num_ngh_l2:], src_hidden_l2.repeat(2, 1), torch.cat((tgt_hidden_l2, bad_hidden_l2), 0))
      pos_raw = caw_l2[:, -1]

      pos_encoding = self.trainable_embedding(pos_raw.long())
      caw_l2 = torch.cat((caw_l2[:, :-1], pos_encoding), -1)
      caw_l1l2 = self.get_relative_id(torch.cat((src_sparse_idx, src_n_sparse_idx), 0), sparse_idx_l2, src_ngh_id.repeat(2), ngh_id_l2[num_ngh_l2:], src_prev_f.repeat(2, 1), torch.cat((tgt_hidden_l2, bad_hidden_l2), 0))
      pos_raw = caw_l1l2[:, -1]

      pos_encoding = self.trainable_embedding(pos_raw.long())
      caw_l1l2 = torch.cat((caw_l1l2[:, :-1], pos_encoding), -1)
      caw_l2l1 = self.get_relative_id(sparse_idx_l2, tgt_bad_sparse_idx, ngh_id_l2[:num_ngh_l2].repeat(2), torch.cat((tgt_ngh_id, bad_ngh_id), 0), src_hidden_l2.repeat(2, 1), torch.cat((tgt_prev_f, bad_prev_f), 0))
      pos_raw = caw_l2l1[:, -1]

      pos_encoding = self.trainable_embedding(pos_raw.long())
      caw_l2l1 = torch.cat((caw_l2l1[:, :-1], pos_encoding), -1)
    end = time.time()
    self.log_time('caw actual', start, end)
    start = time.time()
    src_caw_p = caw[0:src_nghs]
    src_caw_n = caw[src_nghs:2*src_nghs]
    tgt_caw_p = caw[2*src_nghs:2*src_nghs + tgt_nghs]
    bad_caw_n = caw[2*src_nghs + tgt_nghs:]
    if self.n_layers > 1:
      src_caw_p_l2 = caw_l2[0:num_ngh_l2]
      src_caw_n_l2 = caw_l2[num_ngh_l2:2*num_ngh_l2]
      tgt_caw_p_l2 = caw_l2[2*num_ngh_l2:3*num_ngh_l2]
      bad_caw_n_l2 = caw_l2[3*num_ngh_l2:]
      src_caw_p_l1l2 = caw_l1l2[0:src_nghs]
      src_caw_n_l1l2 = caw_l1l2[src_nghs:2*src_nghs]
      tgt_caw_p_l1l2 = caw_l1l2[2*src_nghs:2*src_nghs + num_ngh_l2]
      bad_caw_n_l1l2 = caw_l1l2[2*src_nghs + num_ngh_l2:]
      src_caw_p_l2l1 = caw_l2l1[0:num_ngh_l2]
      src_caw_n_l2l1 = caw_l2l1[num_ngh_l2:2*num_ngh_l2]
      tgt_caw_p_l2l1 = caw_l2l1[2*num_ngh_l2:2*num_ngh_l2 + tgt_nghs]
      bad_caw_n_l2l1 = caw_l2l1[2*num_ngh_l2 + tgt_nghs:]
    # src_caw_p = pos_caw[0:src_nghs]
    # tgt_caw_p = pos_caw[src_nghs:src_nghs + tgt_nghs]
    # neg_caw = self.get_relative_id(src_sparse_idx, bad_sparse_idx, src_n_id, bad_n_id, src_ts_f, bad_ts_f)
    # src_caw_n = neg_caw[0:src_nghs]
    # bad_caw_n = neg_caw[src_nghs:src_nghs + bad_nghs]
    end = time.time()
    self.log_time('caw final', start, end)
    start = time.time()
    # self_msk = ori_idx == ngh_id
    # other_msk = n_id != 0 #torch.logical_and(n_id != 0, ori_idx != n_id)

    # src_self_m = self_msk[0:src_nghs].nonzero().squeeze()
    # tgt_self_m = self_msk[src_nghs:src_nghs + tgt_nghs].nonzero().squeeze()
    # bad_self_m = self_msk[src_nghs + tgt_nghs:src_nghs + tgt_nghs + bad_nghs].nonzero().squeeze()
    # src_other_m = other_msk[0:src_nghs].nonzero().squeeze()
    # tgt_other_m = other_msk[src_nghs:src_nghs + tgt_nghs].nonzero().squeeze()
    # bad_other_m = other_msk[src_nghs + tgt_nghs:src_nghs + tgt_nghs + bad_nghs].nonzero().squeeze()
    if self.n_layers > 1:
      p_src_f = torch.cat((src_n_f, src_prev_f[:, :-1], src_caw_p, src_caw_p_l1l2, parent_node_feat[:src_nghs]), -1)
      n_src_f = torch.cat((src_n_f, src_prev_f[:, :-1], src_caw_n, src_caw_n_l1l2, parent_node_feat[:src_nghs]), -1)
      tgt_f = torch.cat((tgt_n_f, tgt_prev_f[:, :-1], tgt_caw_p, tgt_caw_p_l2l1, parent_node_feat[src_nghs:src_nghs+tgt_nghs]), -1)
      bad_f = torch.cat((bad_n_f, bad_prev_f[:, :-1], bad_caw_n, bad_caw_n_l2l1, parent_node_feat[src_nghs+tgt_nghs:]), -1)
      p_src_f_l2 = torch.cat((src_n_f_l2, src_hidden_l2[:, :-1], src_caw_p_l2, src_caw_p_l2l1, parent_node_feat_l2[:num_ngh_l2]), -1)
      n_src_f_l2 = torch.cat((src_n_f_l2, src_hidden_l2[:, :-1], src_caw_n_l2, src_caw_n_l2l1, parent_node_feat_l2[:num_ngh_l2]), -1)
      tgt_f_l2 = torch.cat((tgt_n_f_l2, tgt_hidden_l2[:, :-1], tgt_caw_p_l2, tgt_caw_p_l1l2, parent_node_feat_l2[num_ngh_l2:2*num_ngh_l2]), -1)
      bad_f_l2 = torch.cat((bad_n_f_l2, bad_hidden_l2[:, :-1], bad_caw_n_l2,bad_caw_n_l1l2, parent_node_feat_l2[2*num_ngh_l2:]), -1)
    else:
      p_src_f = torch.cat((src_n_f, src_prev_f[:, :-1], src_caw_p, parent_node_feat[:src_nghs]), -1)
      n_src_f = torch.cat((src_n_f, src_prev_f[:, :-1], src_caw_n, parent_node_feat[:src_nghs]), -1)
      tgt_f = torch.cat((tgt_n_f, tgt_prev_f[:, :-1], tgt_caw_p, parent_node_feat[src_nghs:src_nghs+tgt_nghs]), -1)
      bad_f = torch.cat((bad_n_f, bad_prev_f[:, :-1], bad_caw_n, parent_node_feat[src_nghs+tgt_nghs:]), -1)
    feat_l1 = torch.cat((p_src_f, n_src_f, tgt_f, bad_f), 0)
    ngh_id = torch.cat((src_ngh_id, ngh_id), 0)
    self_msk = torch.cat((ori_idx[:src_nghs], ori_idx), 0) == ngh_id
    # other_msk = torch.logical_not(self_msk)

    self_msk = self_msk.nonzero().squeeze()
    # other_msk = other_msk.nonzero().squeeze()

    ori_feat = feat_l1.index_select(0, self_msk)
    # feat_l1 = feat_l1.index_select(0, other_msk)
    # ngh_id = ngh_id.index_select(0, other_msk)
    if self.n_layers > 1:
      feat_l2 = torch.cat((p_src_f_l2, n_src_f_l2, tgt_f_l2, bad_f_l2), 0)
      all_feat = torch.cat((feat_l1, feat_l2), 0)
    else:
      all_feat = feat_l1
    src_ori = ori_idx[:src_nghs]
    tgt_ori = ori_idx[src_nghs:src_nghs + tgt_nghs]
    bad_ori = ori_idx[src_nghs + tgt_nghs:]
    src_dense_msk, src_dense_msk, tgt_dense_msk, bad_dense_msk = [None]*4
    # p_src_ori_f, n_src_ori_f, tgt_ori_f, bad_ori_f = [None]*4
    sparse_l1 = torch.cat((src_sparse_idx,sparse_idx + batch_size), 0)
    # sparse_l1 = sparse_l1.index_select(0, other_msk)
    if self.n_layers > 1:
      ori_idx = torch.cat((sparse_l1, torch.repeat_interleave(torch.arange(batch_size * 4, device=device), self.num_neighbors[1])), 0)
    else:
      ori_idx = sparse_l1
    ori_feat = ori_feat.index_select(0, ori_idx)
    # ori_idx = torch.cat((src_sparse_idx,sparse_idx + batch_size), 0)
    # ori_idx = torch.repeat_interleave(torch.arange(batch_size * 4, device=device), self.num_neighbors[1])
    # ngh_id = torch.cat((src_ngh_id, src_ngh_id, tgt_ngh_id, bad_ngh_id, ngh_id_l2[:num_ngh_l2], ngh_id_l2), 0)
    if self.n_layers > 1:
      ngh_id = torch.cat((ngh_id, ngh_id_l2[:num_ngh_l2], ngh_id_l2), 0)
    else:
      ngh_id = torch.cat((src_ngh_id, src_ngh_id, tgt_ngh_id, bad_ngh_id), 0)

    p_score, n_score, attn_score = self.forward(ori_idx, ngh_id, ori_feat, all_feat, batch_size)
    end = time.time()
    self.log_time('attention', start, end)
    start = time.time()
    self.update_memory(src_e_l, tgt_e_l, src_th, tgt_th, src_start_th, tgt_start_th, src_ngh_n_th, tgt_ngh_n_th, src_nghs, tgt_nghs, src_ngh_id, tgt_ngh_id, e_idx_th, cut_time_th, hidden_states, pr_hidden_states, updated_mem, batch_size, device)
    return p_score.sigmoid(), n_score.sigmoid()
  
  def update_memory(self, src_e_l, tgt_e_l, src_th, tgt_th, src_start_th, tgt_start_th, src_ngh_n_th, tgt_ngh_n_th, src_nghs, tgt_nghs, src_ngh_id, tgt_ngh_id, e_idx_th, cut_time_th, hidden_states, pr_hidden_states, updated_mem, batch_size, device):
    src_e_th = torch.from_numpy(src_e_l).to(dtype=torch.long, device=device)
    tgt_e_th = torch.from_numpy(tgt_e_l).to(dtype=torch.long, device=device)
    e_pos_th = torch.cat((src_e_th, tgt_e_th), 0)
    opp_th = torch.cat((tgt_th, src_th), 0)
    e_start_th = torch.cat((src_start_th, tgt_start_th), 0)
    e_pos_th += e_start_th
    ori_idx = torch.cat((src_th, tgt_th), 0)

    self.num_neighbors_stored[ori_idx] = torch.max(self.num_neighbors_stored[ori_idx], e_pos_th - e_start_th + 1).long()
    hidden_states = hidden_states[:, :self.hidden_dim]
    reset_size = hidden_states.shape[1]
    if self.backprop_n_history > 1:
      reset_size += 2
    reset_memory = torch.zeros((batch_size * 2, reset_size), device=device)
    
    reset_memory_raw = torch.cat((opp_th.unsqueeze(1), e_idx_th.float().repeat(2).unsqueeze(1), cut_time_th.repeat(2).unsqueeze(1)), -1)
    self.neighborhood_store[0][e_start_th, :3] = torch.cat((ori_idx.unsqueeze(1), e_idx_th.float().repeat(2).unsqueeze(1), cut_time_th.repeat(2).unsqueeze(1)), -1)
    reset_memory = torch.cat((reset_memory_raw, reset_memory), -1)
    self.neighborhood_store[0][e_pos_th] = reset_memory
    ngh_n_th = torch.cat((src_ngh_n_th, tgt_ngh_n_th), 0)

    e_pos_th = torch.repeat_interleave(e_pos_th, ngh_n_th, dim=0)
    parent_th = torch.repeat_interleave(ori_idx, ngh_n_th, dim=0)
    ori_idx = torch.repeat_interleave(ori_idx, ngh_n_th, dim=0)
    e_start_th = torch.repeat_interleave(e_start_th, ngh_n_th, dim=0)
    ts_th = torch.repeat_interleave(cut_time_th.repeat(2), ngh_n_th, dim=0)
    e_idx_th = torch.repeat_interleave(e_idx_th.repeat(2), ngh_n_th, dim=0)
    opp_th = torch.repeat_interleave(opp_th, ngh_n_th, dim=0)
    ngh_id = torch.cat((src_ngh_id, tgt_ngh_id), 0).long()
    updated_mem = updated_mem.detach()
    
    # Update second hop neighbors
    if self.n_layers > 1:
      l2_id = opp_th * self.num_neighbors[1] + (hash(ngh_id) ^ self.seed) % self.num_neighbors[1]
      occupied_l2 = torch.logical_and(self.neighborhood_store[1][l2_id,self.ngh_id_idx] != 0, self.neighborhood_store[1][l2_id,self.ngh_id_idx] != updated_mem[:src_nghs + tgt_nghs, 0])
      set_new_l2 =  (occupied_l2 * torch.rand(occupied_l2.shape[0], device=device)) < 0.3
      l2_id *= set_new_l2
      l2_id *= ngh_id != parent_th
      l2_id *= ngh_id != opp_th
      l2_id *= ngh_id != 0
      l2_id *= torch.rand(l2_id.shape[0], device=device) > 0.3
      if self.backprop_n_history == 1:
        self.neighborhood_store[1][l2_id] = torch.cat((updated_mem[:src_nghs + tgt_nghs, :-1], parent_th.unsqueeze(1)), -1)
        # updated_mem_l2 = updated_mem_l2.detach()
        # existing_l2 = updated_mem_l2[:num_ngh_l2*2, self.ngh_id_idx]
        # existing_l2_parent = updated_mem_l2[:num_ngh_l2*2, self.parent_idx]
        # existing_l2_hidden = updated_mem_l2[:num_ngh_l2*2, 3:3+self.hidden_dim]
        # existing_l2_idx = torch.cat((tgt_th, src_th), 0).repeat_interleave(self.num_neighbors[1]) * self.num_neighbors[1] + (hash(existing_l2) ^ self.seed) % self.num_neighbors[1]
        # existing_l2_msk = (existing_l2 == self.neighborhood_store[1][existing_l2_idx,self.ngh_id_idx]).unsqueeze(1).repeat(1, self.hidden_dim)
        # self.neighborhood_store[1][existing_l2_idx, 3:3+self.hidden_dim] *= torch.logical_not(existing_l2_msk)
        # self.neighborhood_store[1][existing_l2_idx, 3:3+self.hidden_dim] += existing_l2_hidden * existing_l2_msk
      else:
        self.neighborhood_store[1][l2_id] = torch.cat((updated_mem[:src_nghs + tgt_nghs, :3], pr_hidden_states[:src_nghs + tgt_nghs], parent_th.unsqueeze(1)), -1)
    e_msk = (ngh_id == opp_th).nonzero().squeeze()
    self_msk = (ngh_id == ori_idx).nonzero().squeeze()
    if self.backprop_n_history == 1:
      agg_p = hidden_states[:src_nghs + tgt_nghs].detach()
    else:
      agg_p = pr_hidden_states[:src_nghs + tgt_nghs]
    # Update neighbors
    self.store_memory(opp_th.index_select(0, e_msk), e_pos_th.index_select(0, e_msk), ts_th.index_select(0, e_msk), e_idx_th.index_select(0, e_msk), agg_p.index_select(0, e_msk))
    # Update self
    self.store_memory(ori_idx.index_select(0, self_msk), e_start_th.index_select(0, self_msk), ts_th.index_select(0, self_msk), e_idx_th.index_select(0, self_msk), agg_p.index_select(0, self_msk))


  def store_memory(self, n_id, e_pos_th, ts_th, e_th, agg_p):
    prev_data = torch.cat((n_id.unsqueeze(1), e_th.unsqueeze(1), ts_th.unsqueeze(1), agg_p), -1)
    self.neighborhood_store[0][e_pos_th.long()] = prev_data

  def get_relative_id(self, src_sparse_idx, tgt_sparse_idx, src_n_id, tgt_n_id, src_ts_hidden, tgt_ts_hidden):
    sparse_idx = torch.cat((src_sparse_idx, tgt_sparse_idx), 0)
    n_id = torch.cat((src_n_id, tgt_n_id), 0)
    ts_hidden = torch.cat((src_ts_hidden, tgt_ts_hidden), 0)
    key = torch.cat((sparse_idx.unsqueeze(1), n_id.unsqueeze(1)), -1) # tuple of (idx in the current batch, n_id)
    unique, inverse_idx = key.unique(return_inverse=True, dim=0)
    # SCATTER ADD FOR TS WITH INV IDX
    relative_ts = torch.zeros(unique.shape[0], self.caw_dim, device=self.device)
    relative_ts.scatter_add_(0, inverse_idx.unsqueeze(1).repeat(1,self.caw_dim), ts_hidden)
    relative_ts = relative_ts.index_select(0, inverse_idx)
    assert(relative_ts.shape[0] == sparse_idx.shape[0] == ts_hidden.shape[0])
    return relative_ts

  def get_updated_memory_l2(self, ori_idx, cut_time_th, device):
    start = time.time()
    ngh = self.neighborhood_store[1].view(self.total_nodes, self.num_neighbors[1], self.memory_dim)[ori_idx].view(ori_idx.shape[0] * self.num_neighbors[1], self.memory_dim)
    end = time.time()
    self.log_time('get_updated_memory 2', start, end)
    start = time.time()
    cut_time_th = cut_time_th.repeat_interleave(self.num_neighbors[1])
    ngh_id = ngh[:,self.ngh_id_idx].long()
    ngh_e_raw = ngh[:,self.e_raw_idx].long()
    ngh_ts_raw = ngh[:,self.ts_raw_idx]
    ngh_e_emb = ngh[:,self.e_emb_idx[0]:self.e_emb_idx[1]]
    ngh_ts_emb = ngh[:,self.ts_emb_idx[0]:self.ts_emb_idx[1]]
    ngh_parent_id = ngh[:,self.parent_idx]
    e_feat = self.edge_raw_embed(ngh_e_raw)
    ts_feat = self.time_encoder(ngh_ts_raw)
    
    if self.backprop_n_history > 1:
      ngh_pr_e_raw = ngh[:,self.pr_e_raw_idx].long()
      ngh_pr_ts_raw = ngh[:,self.pr_ts_raw_idx]
      pr_e_feat = self.edge_raw_embed(ngh_pr_e_raw)
      pr_ts_feat = self.time_encoder(ngh_pr_ts_raw)
      # ts_diff_feat = self.time_encoder(ngh_ts_diff_raw)
      pr_e_hidden_state = self.feature_encoder(pr_e_feat, ngh_e_emb)
      pr_time_hidden_state = self.time_aggregator(pr_ts_feat, ngh_ts_emb)
      pr_e_hidden_state *= (ngh_pr_e_raw != 0).unsqueeze(1).repeat(1, self.e_feat_dim)

      pr_time_hidden_state *= (ngh_pr_ts_raw.long() != 0).unsqueeze(1).repeat(1, self.time_dim)
      e_hidden_state = self.feature_encoder(e_feat, pr_e_hidden_state)
      time_hidden_state = self.time_aggregator(ts_feat, pr_time_hidden_state)
    else:
      e_hidden_state = self.feature_encoder(e_feat, ngh_e_emb)
      time_hidden_state = self.time_aggregator(ts_feat, ngh_ts_emb)
    if self.agg_time_delta:
      ngh_ts_diff_emb = ngh[:,self.ts_diff_emb_idx[0]:self.ts_diff_emb_idx[1]]
      ngh_ts_diff_raw = cut_time_th - ngh_ts_raw
      ts_diff_feat = self.time_encoder(ngh_ts_diff_raw)
      if self.backprop_n_history > 1:
        pr_ts_diff_raw = ngh_ts_raw - ngh_pr_ts_raw
        pr_ts_diff_feat = self.time_encoder(pr_ts_diff_raw)
        pr_time_diff_hidden_state = self.rev_time_aggregator(pr_ts_diff_feat, ngh_ts_diff_emb)
        pr_time_diff_hidden_state *= (ngh_pr_ts_raw.long() != 0).unsqueeze(1).repeat(1, self.time_dim)
        time_diff_hidden_state = self.rev_time_aggregator(ts_diff_feat, pr_time_diff_hidden_state)
        pr_time_hidden_state = torch.cat((pr_time_hidden_state, time_diff_hidden_state), -1)
      else:
        time_diff_hidden_state = self.rev_time_aggregator(ts_diff_feat, ngh_ts_diff_emb)
      time_hidden_state = torch.cat((time_hidden_state, time_diff_hidden_state), -1)


    e_hidden_state *= (ngh_e_raw != 0).unsqueeze(1).repeat(1, self.e_feat_dim)
    
    ori_idx = torch.repeat_interleave(ori_idx, self.num_neighbors[1])
    msk = torch.logical_and(torch.logical_and(ngh_ts_raw < cut_time_th, ngh_ts_raw != 0), torch.logical_and(ngh_id != ori_idx, ngh_e_raw != 0))
    updated_mem = torch.cat((ngh[:, :self.num_raw], e_hidden_state, time_hidden_state, ngh_parent_id.unsqueeze(1)), -1)
    updated_mem *= msk.unsqueeze(1).repeat(1, self.memory_dim)
    return updated_mem

  def get_updated_memory(self, ori_idx, ori_idx_expand, cut_time_th, start_l, ngh_n_l, device):
    # TODO: can be optimize by customized CUDA kernel
    start = time.time()
    n_idx = torch.from_numpy(
      np.concatenate([np.arange(start_l[i], start_l[i]+ngh_n_l[i]) for i in range(len(start_l))])).long().to(device)
    end = time.time()
    self.log_time('get_updated_memory 1', start, end)
    start = time.time()
    ngh = self.neighborhood_store[0][n_idx]
    end = time.time()
    self.log_time('get_updated_memory 2', start, end)
    start = time.time()
    ngh_id = ngh[:,self.ngh_id_idx].long()
    ngh_e_raw = ngh[:,self.e_raw_idx].long()
    ngh_ts_raw = ngh[:,self.ts_raw_idx]
    ngh_e_emb = ngh[:,self.e_emb_idx[0]:self.e_emb_idx[1]]
    ngh_ts_emb = ngh[:,self.ts_emb_idx[0]:self.ts_emb_idx[1]]
      
    end = time.time()
    self.log_time('get_updated_memory 2.25', start, end)
    start = time.time()
    # self_msk = (ngh_id == ori_idx).nonzero().squeeze()
    ngh_ts_raw += cut_time_th * (ngh_id == ori_idx_expand)
    ngh_ts_diff_raw = cut_time_th - ngh_ts_raw
    end = time.time()
    self.log_time('get_updated_memory 2.5', start, end)
    start = time.time()
    e_feat = self.edge_raw_embed(ngh_e_raw)
    ts_feat = self.time_encoder(ngh_ts_raw)
    if self.backprop_n_history > 1:
      ngh_pr_e_raw = ngh[:,self.pr_e_raw_idx].long()
      ngh_pr_ts_raw = ngh[:,self.pr_ts_raw_idx]
      pr_e_feat = self.edge_raw_embed(ngh_pr_e_raw)
      pr_ts_feat = self.time_encoder(ngh_pr_ts_raw)
      # ts_diff_feat = self.time_encoder(ngh_ts_diff_raw)
      pr_e_hidden_state = self.feature_encoder(pr_e_feat, ngh_e_emb)
      pr_time_hidden_state = self.time_aggregator(pr_ts_feat, ngh_ts_emb)
      pr_e_hidden_state *= (ngh_pr_e_raw != 0).unsqueeze(1).repeat(1, self.e_feat_dim)

      pr_time_hidden_state *= (ngh_pr_ts_raw.long() != 0).unsqueeze(1).repeat(1, self.time_dim)
      e_hidden_state = self.feature_encoder(e_feat, pr_e_hidden_state)
      time_hidden_state = self.time_aggregator(ts_feat, pr_time_hidden_state)
    else:
      e_hidden_state = self.feature_encoder(e_feat, ngh_e_emb)
      time_hidden_state = self.time_aggregator(ts_feat, ngh_ts_emb)

    if self.agg_time_delta:
      ngh_ts_diff_emb = ngh[:,self.ts_diff_emb_idx[0]:self.ts_diff_emb_idx[1]]
      ngh_ts_diff_raw = cut_time_th - ngh_ts_raw
      ts_diff_feat = self.time_encoder(ngh_ts_diff_raw)
      if self.backprop_n_history > 1:
        pr_ts_diff_raw = ngh_ts_raw - ngh_pr_ts_raw
        pr_ts_diff_feat = self.time_encoder(pr_ts_diff_raw)
        pr_time_diff_hidden_state = self.rev_time_aggregator(pr_ts_diff_feat, ngh_ts_diff_emb)
        pr_time_diff_hidden_state *= (ngh_pr_ts_raw.long() != 0).unsqueeze(1).repeat(1, self.time_dim)
        time_diff_hidden_state = self.rev_time_aggregator(ts_diff_feat, pr_time_diff_hidden_state)
        pr_time_hidden_state = torch.cat((pr_time_hidden_state, time_diff_hidden_state), -1)
      else:
        time_diff_hidden_state = self.rev_time_aggregator(ts_diff_feat, ngh_ts_diff_emb)
      time_hidden_state = torch.cat((time_hidden_state, time_diff_hidden_state), -1)
    end = time.time()
    self.log_time('get_updated_memory 3', start, end)
    start = time.time()
    e_hidden_state *= (ngh_e_raw != 0).repeat(self.e_feat_dim, 1).T
    msk = torch.logical_or(torch.logical_and(ngh_ts_raw < cut_time_th, ngh_ts_raw != 0), ngh_id == ori_idx_expand)
    updated_mem = torch.cat((ngh[:, :self.num_raw], e_hidden_state, time_hidden_state, ori_idx_expand.unsqueeze(1)), -1)
    
    # time_hidden_state *= (ngh_ts_raw.long() != 0).repeat(self.time_dim, 1).T
    # time_hidden_state *= msk.repeat(self.time_dim, 1).T
    # e_hidden_state *= (ngh_e_raw != 0).repeat(self.e_feat_dim, 1).T
    # e_hidden_state *= msk.repeat(self.e_feat_dim, 1).T
    # ngh_id *= msk
    updated_mem *= msk.unsqueeze(1).repeat(1, self.memory_dim)
    end = time.time()
    self.log_time('get_updated_memory 4', start, end)
    if self.backprop_n_history > 1:
      pr_hidden_states = torch.cat((ngh_e_raw.unsqueeze(1), ngh_ts_raw.unsqueeze(1), pr_e_hidden_state, pr_time_hidden_state), -1).detach()
    else:
      pr_hidden_states = None
    return updated_mem, n_idx, pr_hidden_states

  
  def forward(self, ori_idx, ngh_id, ori_feat, feat, bs):
    edge_idx = torch.stack((ngh_id, ori_idx))
    gat_data = torch.cat((edge_idx.T, feat, ori_feat), -1)
    gat_data = gat_data[ngh_id.nonzero().squeeze()]
    edge_idx = gat_data[:, 0:2].T
    ngh_feat = gat_data[:, 2:2 + self.attn_dim]
    ori_feat = gat_data[:, 2 + self.attn_dim:]
    embed, _, attn_score = self.gat((ngh_feat, ori_feat, edge_idx.long(), 4*bs))
    p_src_embed = embed[:bs]
    n_src_embed = embed[bs:2*bs]
    tgt_embed = embed[2*bs:3*bs]
    bad_embed = embed[3*bs:]

    p_score, p_score_walk = self.affinity_score(p_src_embed, tgt_embed)
    p_score.squeeze_(dim=-1)
    n_score, n_score_walk = self.affinity_score(n_src_embed, bad_embed)
    n_score.squeeze_(dim=-1)
    return p_score, n_score, attn_score

  def init_time_encoder(self):
    return TimeEncode(expand_dim=self.time_dim)

  def init_feature_encoder(self):
    if self.history_agg_method == 'gru':
      return FeatureEncoderGRU(self.e_feat_dim, self.e_feat_dim, self.dropout)
    else:
      return FeatureEncoderLSTM(self.e_feat_dim, self.e_feat_dim, self.dropout)

  def init_time_aggregator(self):
    if self.history_agg_method == 'gru':
      return FeatureEncoderGRU(self.time_dim, self.time_dim, self.dropout)
    else:
      return FeatureEncoderLSTM(self.time_dim, self.time_dim, self.dropout)

class FeatureEncoderGRU(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, dropout_p=0.1):
    super(FeatureEncoderGRU, self).__init__()
    self.gru = nn.GRUCell(input_dim, hidden_dim)
    self.dropout = nn.Dropout(dropout_p)
    self.hidden_dim = hidden_dim

  def forward(self, input_features, hidden_state, use_dropout=False):
    encoded_features = self.gru(input_features, hidden_state)
    if use_dropout:
      encoded_features = self.dropout(encoded_features)
    return encoded_features



class FeatureEncoderLSTM(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, dropout_p=0.1):
    super(FeatureEncoderLSTM, self).__init__()
    self.lstm = nn.LSTMCell(input_dim, hidden_dim)
    self.dropout = nn.Dropout(dropout_p)
    self.hidden_dim = hidden_dim

  def forward(self, input_features, hidden_state, use_dropout=False):
    hidden_state, cell_state = self.lstm(input_features, (hidden_state[:, :self.hidden_dim], hidden_state[:, self.hidden_dim:]))
    if use_dropout:
      hidden_state = self.dropout(hidden_state)
      cell_state = self.dropout(cell_state)
    return torch.cat((hidden_state, cell_state), -1)

class TimeEncode(torch.nn.Module):
  def __init__(self, expand_dim, factor=5):
    super(TimeEncode, self).__init__()

    self.time_dim = expand_dim
    self.factor = factor
    self.basis_freq = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, self.time_dim))).float())
    self.phase = torch.nn.Parameter(torch.zeros(self.time_dim).float())


  def forward(self, ts):
    # ts: [N, 1]
    batch_size = ts.size(0)

    ts = ts.view(batch_size, 1)  # [N, 1]
    map_ts = ts * self.basis_freq.view(1, -1) # [N, time_dim]
    map_ts += self.phase.view(1, -1) # [N, time_dim]
    harmonic = torch.cos(map_ts)

    return harmonic #self.dense(harmonic)

class AttnModel(torch.nn.Module):
  """Attention based temporal layers
  """
  def __init__(self, feat_dim, pos_dim, model_dim, n_head=2, drop_out=0.1):
    """
    args:
      feat_dim: dim for the node features
      n_head: number of heads in attention
      drop_out: probability of dropping a neural.
    """
    super(AttnModel, self).__init__()

    self.feat_dim = feat_dim
    self.pos_dim = pos_dim
    self.model_dim = model_dim

    self.merger = MergeLayer(self.model_dim, model_dim, feat_dim, feat_dim)

    assert(self.model_dim % n_head == 0)
    self.logger = logging.getLogger(__name__)
    self.multi_head_target = MultiHeadAttention(n_head,
                     d_model=self.model_dim,
                     d_k=self.model_dim // n_head,
                     d_v=self.model_dim // n_head,
                     dropout=drop_out)
    self.logger.info('Using scaled prod attention')

  def forward(self, src, seq, mask, src_p=None, seq_p=None):
    """"Attention based temporal attention forward pass
    args:
      src: float Tensor of shape [B, N_src, D]
      seq: float Tensor of shape [B, N_ngh, D]
      mask: boolean Tensor of shape [B, N_ngh], where the true value indicate a null value in the sequence.

    returns:
      output, weight

      output: float Tensor of shape [B, D]
      weight: float Tensor of shape [B, N]
    """

    batch, N_src, _ = src.shape
    N_ngh = seq.shape[1]
    device = src.device
    src_p_pad, seq_p_pad = src_p, seq_p
    # if src_p is None:
    #   src_p_pad = torch.zeros((batch, N_src, self.pos_dim)).float().to(device)
    #   seq_p_pad = torch.zeros((batch, N_ngh, self.pos_dim)).float().to(device)
    q = src
    k = seq
    if src_p is not None:
      q = torch.cat([src, src_p_pad], dim=2) # [B, N_src, D + De + Dt] -> [B, N_src, D]
      k = torch.cat([seq, seq_p_pad], dim=2) # [B, N_ngh, D + De + Dt] -> [B, N_ngh, D]
    output, attn = self.multi_head_target(q=q, k=k, v=k, mask=mask) # output: [B, N_src, D + De + Dt], attn: [B, N_src, n_head, num_neighbors]
    output = self.merger(output, src)
    return output, attn

class MultiHeadAttention(nn.Module):
  ''' Multi-Head Attention module '''

  def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
    super().__init__()

    self.n_head = n_head
    self.d_k = d_k
    self.d_v = d_v

    self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
    self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
    self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
    nn.init.normal_(self.w_qs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
    nn.init.normal_(self.w_ks.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
    nn.init.normal_(self.w_vs.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

    self.attention = ScaledDotProductAttention(temperature=np.power(d_k, 0.5), attn_dropout=dropout)
    self.layer_norm = nn.LayerNorm(d_model)

    self.fc = nn.Linear(n_head * d_v, d_model)

    nn.init.xavier_normal_(self.fc.weight)

    self.dropout = nn.Dropout(dropout)


  def forward(self, q, k, v, mask=None):

    d_k, d_v, n_head = self.d_k, self.d_v, self.n_head

    B, N_src, _ = q.size() # [B, N_src, model_dim]
    B, N_ngh, _ = k.size() # [B, N_ngh, model_dim]
    B, N_ngh, _ = v.size() # [B, N_ngh, model_dim]
    assert(N_ngh % N_src == 0)
    num_neighbors = int(N_ngh / N_src)
    residual = q

    q = self.w_qs(q).view(B, N_src, 1, n_head, d_k)  # [B, N_src, 1, n_head, d_k]
    k = self.w_ks(k).view(B, N_src, num_neighbors, n_head, d_k)  # [B, N_src, num_neighbors, n_head, d_k]
    v = self.w_vs(v).view(B, N_src, num_neighbors, n_head, d_v)  # [B, N_src, num_neighbors, n_head, d_k]

    q = q.transpose(2, 3).contiguous().view(B*N_src*n_head, 1, d_k)  # [B*N_src*n_head, 1, d_k]
    k = k.transpose(2, 3).contiguous().view(B*N_src*n_head, num_neighbors, d_k)  # [B*N_src*n_head, num_neighbors, d_k]
    v = v.transpose(2, 3).contiguous().view(B*N_src*n_head, num_neighbors, d_v)  # [B*N_src*n_head, num_neighbors, d_v]
    mask = mask.view(B*N_src, 1, num_neighbors).repeat(n_head, 1, 1) # [B*N_src*n_head, 1, num_neighbors]
    output, attn_map = self.attention(q, k, v, mask=mask) # output: [B*N_src*n_head, 1, d_v], attn_map: [B*N_src*n_head, 1, num_neighbors]

    output = output.view(B, N_src, n_head*d_v)  # [B, N_src, n_head*d_v]
    output = self.dropout(self.fc(output))  # [B, N_src, model_dim]
    output = self.layer_norm(output + residual)  # [B, N_src, model_dim]
    attn_map = attn_map.view(B, N_src, n_head, num_neighbors)
    return output, attn_map

class ScaledDotProductAttention(torch.nn.Module):
  ''' Scaled Dot-Product Attention '''

  def __init__(self, temperature, attn_dropout=0.1):
    super().__init__()
    self.temperature = temperature
    self.dropout = torch.nn.Dropout(attn_dropout)
    self.softmax = torch.nn.Softmax(dim=2)

  def forward(self, q, k, v, mask=None):
    # q: [B*N_src*n_head, 1, d_k]; k: [B*N_src*n_head, num_neighbors, d_k]
    # v: [B*N_src*n_head, num_neighbors, d_v], mask: [B*N_src*n_head, 1, num_neighbors]
    attn = torch.bmm(q, k.transpose(-1, -2))  # [B*N_src*n_head, 1, num_neighbors]
    attn = attn / self.temperature

    if mask is not None:
      attn = attn.masked_fill(mask, -1e10)

    attn = self.softmax(attn) # [n * b, l_q, l_k]
    attn = self.dropout(attn) # [n * b, l_v, d]

    output = torch.bmm(attn, v)  # [B*N_src*n_head, 1, d_v]

    return output, attn

class MergeLayer(torch.nn.Module):
  def __init__(self, dim1, dim2, dim3, dim4, non_linear=True):
    super().__init__()
    #self.layer_norm = torch.nn.LayerNorm(dim1 + dim2)
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

    # special linear layer for motif explainability
    self.non_linear = non_linear
    if not non_linear:
      assert(dim1 == dim2)
      self.fc = nn.Linear(dim1, 1)
      torch.nn.init.xavier_normal_(self.fc1.weight)

  def forward(self, x1, x2):
    z_walk = None
    if self.non_linear:
      x = torch.cat([x1, x2], dim=-1)
      #x = self.layer_norm(x)
      h = self.act(self.fc1(x))
      z = self.fc2(h)
    else: # for explainability
      # x1, x2 shape: [B, M, F]
      x = torch.cat([x1, x2], dim=-2)  # x shape: [B, 2M, F]
      z_walk = self.fc(x).squeeze(-1)  # z_walk shape: [B, 2M]
      z = z_walk.sum(dim=-1, keepdim=True)  # z shape [B, 1]
    return z, z_walk


