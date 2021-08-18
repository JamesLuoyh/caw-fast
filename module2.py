import torch
import torch.nn as nn
import logging
import time
import numpy as np
import random

class CAWN2(torch.nn.Module):
  def __init__(self, n_feat, e_feat, pos_dim=0, n_head=4, num_neighbors=['32'], agg_method='gru', dropout=0.1, walk_linear_out=False, get_checkpoint_path=None, get_ngh_store_path=None, verbosity=1):
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
    self.agg_method = agg_method
    self.feature_encoder = self.init_feature_encoder() # RNNCell
    self.time_aggregator = self.init_time_aggregator() # RNNCell
    self.rev_time_aggregator = self.init_time_aggregator() # RNNCell
    self.ngh_encoder = NeighborhoodEncoder(self.e_feat_dim, self.time_dim, self.agg_method, dropout)
    # self.trainable_embedding = nn.Embedding(num_embeddings=3, embedding_dim=self.pos_dim) # position embedding
    self.attn_dim = self.feat_dim + 2 * self.e_feat_dim + 2 * self.time_dim
    self.attn_m = AttnModel(self.feat_dim, 0, self.attn_dim, n_head=n_head, drop_out=dropout)
    # final projection layer
    self.walk_linear_out = walk_linear_out
    self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1, non_linear=not self.walk_linear_out) #torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)
    self.get_checkpoint_path = get_checkpoint_path
    self.get_ngh_store_path = get_ngh_store_path
    self.src_idx_l_prev = self.tgt_idx_l_prev = self.cut_time_l_prev = self.e_idx_l_prev = None
    self.prev_raw_data = {}
    self.nngh = int(num_neighbors[0])
    self.n_id_idx = 1
    self.e_emb_idx = self.n_id_idx + self.e_feat_dim
    self.ts_emb_idx = self.e_emb_idx + self.time_dim
    self.e_raw_idx = self.ts_emb_idx + 1
    self.ts_raw_idx = self.e_raw_idx + 1
    self.verbosity = verbosity

  def reset_raw_data(self):
    self.prev_raw_data = {}

  def reset_store(self):
    self.neighborhood_store.zero_()
  
  def get_neighborhood_store(self):
    return self.neighborhood_store

  def set_neighborhood_store(self, neighborhood_store):
    self.neighborhood_store = neighborhood_store
    # self.ngh_encoder.update_neighborhood_store(neighborhood_store)

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

    cut_time_th = torch.from_numpy(cut_time_l).to(dtype=torch.float, device=device)
    e_idx_th = torch.from_numpy(e_idx_l).to(dtype=torch.long, device=device)
    end = time.time()
    self.log_time('init 0', start, end)
    start = time.time()
    sparse_idx = torch.arange(batch_size * 3, device=device)
    ngh_data = torch.repeat_interleave(torch.cat((cut_time_th.repeat(3).unsqueeze(1), idx_th.unsqueeze(1), sparse_idx.unsqueeze(1)), -1), ngh_n_th, dim=0)
    ts = ngh_data[:, 0]
    ori_idx = ngh_data[:, 1].long()
    sparse_idx = ngh_data[:, 2].long()
    # ori_idx = torch.repeat_interleave(idx_th, ngh_n_th, dim=0)
    # end = time.time()
    # self.log_time('repeat_interleave', start, end)
    # start = time.time()
    # sparse_idx = torch.repeat_interleave(sparse_idx, ngh_n_th, dim=0)
    
    end = time.time()
    self.log_time('init 1', start, end)
    start = time.time()

    self.neighborhood_store[start_th,0] = idx_th.float()
    # self.neighborhood_store[start_th,-1] = cut_time_th.repeat(3)

    # n_id is the node idx of neighbors of query node
    # dense_idx is the position of each neighbors in the batch*nngh tensor
    # sprase_idx is a tensor of batch idx repeated with ngh_n timesfor each node

    n_id, e_hidden_state, time_hidden_state, n_pos_idx = self.get_updated_memory(ori_idx, ts, start_l, ngh_n_l, device)
    node_features = self.node_raw_embed(n_id)
    # node_features *= (n_id != 0).repeat(self.feat_dim, 1).T
    end = time.time()
    self.log_time('retrieve memory', start, end)
    start = time.time()
    src_n_f = node_features[0:src_nghs]
    tgt_n_f = node_features[src_nghs:src_nghs+tgt_nghs]
    bad_n_f = node_features[src_nghs+tgt_nghs:]
    src_e_f = e_hidden_state[0:src_nghs]
    tgt_e_f = e_hidden_state[src_nghs:src_nghs + tgt_nghs]
    bad_e_f = e_hidden_state[src_nghs + tgt_nghs:]
    src_ts_f = time_hidden_state[0:src_nghs]
    tgt_ts_f = time_hidden_state[src_nghs:src_nghs + tgt_nghs]
    bad_ts_f = time_hidden_state[src_nghs + tgt_nghs:]
    # src_prev_f = hidden_states[0:src_nghs]
    # tgt_prev_f = hidden_states[src_nghs:src_nghs + tgt_nghs]
    # bad_prev_f = hidden_states[src_nghs + tgt_nghs:]
    # hidden_states
    src_n_id = n_id[0:src_nghs]
    tgt_n_id = n_id[src_nghs:src_nghs + tgt_nghs]
    bad_n_id = n_id[src_nghs + tgt_nghs:]
    src_sparse_idx = sparse_idx[0:src_nghs]
    src_n_sparse_idx = src_sparse_idx + batch_size
    tgt_sparse_idx = sparse_idx[src_nghs:src_nghs + tgt_nghs] - batch_size
    tgt_bad_sparse_idx = sparse_idx[src_nghs:] - batch_size
    bad_sparse_idx = sparse_idx[src_nghs + tgt_nghs:] - 2 * batch_size
    end = time.time()
    self.log_time('caw prep', start, end)
    start = time.time()
    caw = self.get_relative_id(torch.cat((src_sparse_idx, src_n_sparse_idx), 0), tgt_bad_sparse_idx, src_n_id.repeat(2), torch.cat((tgt_n_id, bad_n_id), 0), torch.cat((src_e_f,src_ts_f), -1).repeat(2, 1), torch.cat((torch.cat((tgt_e_f,tgt_ts_f), -1), torch.cat((bad_e_f,bad_ts_f), -1)), 0))
    end = time.time()
    self.log_time('caw actual', start, end)
    start = time.time()
    src_caw_p = caw[0:src_nghs]
    src_caw_n = caw[src_nghs:2*src_nghs]
    tgt_caw_p = caw[2*src_nghs:2*src_nghs + tgt_nghs]
    bad_caw_n = caw[2*src_nghs + tgt_nghs:]
    # src_caw_p = pos_caw[0:src_nghs]
    # tgt_caw_p = pos_caw[src_nghs:src_nghs + tgt_nghs]
    # neg_caw = self.get_relative_id(src_sparse_idx, bad_sparse_idx, src_n_id, bad_n_id, src_ts_f, bad_ts_f)
    # src_caw_n = neg_caw[0:src_nghs]
    # bad_caw_n = neg_caw[src_nghs:src_nghs + bad_nghs]
    end = time.time()
    self.log_time('caw final', start, end)
    start = time.time()
    self_msk = ori_idx == n_id
    other_msk = torch.logical_and(n_id != 0, ori_idx != n_id)

    src_self_m = self_msk[0:src_nghs].nonzero().squeeze()
    tgt_self_m = self_msk[src_nghs:src_nghs + tgt_nghs].nonzero().squeeze()
    bad_self_m = self_msk[src_nghs + tgt_nghs:src_nghs + tgt_nghs + bad_nghs].nonzero().squeeze()
    src_other_m = other_msk[0:src_nghs].nonzero().squeeze()
    tgt_other_m = other_msk[src_nghs:src_nghs + tgt_nghs].nonzero().squeeze()
    bad_other_m = other_msk[src_nghs + tgt_nghs:src_nghs + tgt_nghs + bad_nghs].nonzero().squeeze()

    p_src_agg_f = torch.cat((src_n_f, src_e_f, src_ts_f, src_caw_p), -1)
    n_src_agg_f = torch.cat((src_n_f, src_e_f, src_ts_f, src_caw_n), -1)
    tgt_agg_f = torch.cat((tgt_n_f, tgt_e_f, tgt_ts_f, tgt_caw_p), -1)
    bad_agg_f = torch.cat((bad_n_f, bad_e_f, bad_ts_f, bad_caw_n), -1)
    end = time.time()
    self.log_time('agg features (prep)', start, end)
    start = time.time()
    p_src_ori_f = p_src_agg_f.index_select(0, src_self_m)
    p_src_other_f = p_src_agg_f.index_select(0, src_other_m)
    n_src_ori_f = n_src_agg_f.index_select(0, src_self_m)
    n_src_other_f = n_src_agg_f.index_select(0, src_other_m)
    tgt_ori_f = tgt_agg_f.index_select(0, tgt_self_m)
    tgt_other_f = tgt_agg_f.index_select(0, tgt_other_m)
    bad_ori_f = bad_agg_f.index_select(0, bad_self_m)
    bad_other_f = bad_agg_f.index_select(0, bad_other_m)
    end = time.time()
    self.log_time('agg features (index_select)', start, end)
    start = time.time()
    src_dense_idx, src_dense_msk = self.get_dense_idx(batch_size, src_sparse_idx, src_other_m)
    tgt_dense_idx, tgt_dense_msk = self.get_dense_idx(batch_size, tgt_sparse_idx, tgt_other_m)
    bad_dense_idx, bad_dense_msk = self.get_dense_idx(batch_size, bad_sparse_idx, bad_other_m)
    end = time.time()
    self.log_time('agg features (get_dense_idx)', start, end)
    start = time.time()
    p_src_f = self.get_dense_agg_feature(batch_size, src_dense_idx, p_src_other_f)
    n_src_f = self.get_dense_agg_feature(batch_size, src_dense_idx, n_src_other_f)
    tgt_f = self.get_dense_agg_feature(batch_size, tgt_dense_idx, tgt_other_f)
    bad_f = self.get_dense_agg_feature(batch_size, bad_dense_idx, bad_other_f)
    end = time.time()
    self.log_time('agg features (get_dense_agg_feature)', start, end)
    start = time.time()
    p_score, n_score = self.forward(p_src_ori_f, n_src_ori_f, tgt_ori_f, bad_ori_f, p_src_f, n_src_f, tgt_f, bad_f, src_dense_msk, src_dense_msk, tgt_dense_msk, bad_dense_msk, batch_size)
    # pos_score = self.forward(p_src_ori_f, tgt_ori_f, p_src_f, tgt_f, src_dense_msk, tgt_dense_msk, batch_size)
    # neg_score1 = self.forward(n_src_ori_f, bad_ori_f, n_src_f, bad_f, src_dense_msk, bad_dense_msk, batch_size)
    end = time.time()
    self.log_time('attention', start, end)
    start = time.time()
    p_src_ori_f = p_src_ori_f.detach()
    tgt_ori_f = tgt_ori_f.detach()
    p_src_agg_f = p_src_agg_f.detach()
    tgt_agg_f = tgt_agg_f.detach()
    src_e_th = torch.from_numpy(src_e_l).to(dtype=torch.long, device=device)
    tgt_e_th = torch.from_numpy(tgt_e_l).to(dtype=torch.long, device=device)
    e_pos_th = torch.cat((src_e_th, tgt_e_th), 0)
    
    opp_th = torch.cat((tgt_th, src_th), 0)
    self.neighborhood_store[e_pos_th, 0] = opp_th.float()
    self.neighborhood_store[e_pos_th, -2] = e_idx_th.float().repeat(2)
    self.neighborhood_store[e_pos_th, -1] = cut_time_th.repeat(2)
    end = time.time()
    self.log_time('update part memory', start, end)
    start = time.time()
    #TODO reverse unique
    src_src_f = p_src_ori_f[:, self.feat_dim + self.e_feat_dim:self.feat_dim + self.e_feat_dim + self.time_dim]
    tgt_tgt_f = tgt_ori_f[:, self.feat_dim + self.e_feat_dim:self.feat_dim + self.e_feat_dim + self.time_dim]
    ngh_n_th = torch.cat((src_ngh_n_th, tgt_ngh_n_th), 0)

    agg_memory = torch.cat((opp_th.unsqueeze(1), e_pos_th.unsqueeze(1), cut_time_th.repeat(2).unsqueeze(1), e_idx_th.repeat(2).unsqueeze(1)), -1)
    agg_memory = torch.repeat_interleave(agg_memory, ngh_n_th, dim=0)
    # e_pos_th = torch.repeat_interleave(e_pos_th, ngh_n_th, dim=0)
    # opp_idx = torch.repeat_interleave(opp_th, ngh_n_th, dim=0)
    # ts_th = torch.repeat_interleave(cut_time_th.repeat(2), ngh_n_th, dim=0)
    # e_idx_th = torch.repeat_interleave(e_idx_th.repeat(2), ngh_n_th, dim=0)

    n_id = torch.cat((src_n_id, tgt_n_id), 0)
    # e_msk = (n_id == opp_idx).nonzero().squeeze()
    e_msk = (n_id == agg_memory[:,0]).nonzero().squeeze()
    agg_f = torch.cat((p_src_agg_f, tgt_agg_f), 0)
    agg_memory = torch.cat((n_id.unsqueeze(1), agg_memory, agg_f), -1)
    agg_memory = agg_memory.index_select(0, e_msk)
    n_id = agg_memory[:,0]
    e_pos_th = agg_memory[:,2]
    ts_th = agg_memory[:,3]
    e_th = agg_memory[:,4]
    agg_f = agg_memory[:,5:]
    # agg_f = agg_f.index_select(0, e_msk)
    # n_id = n_id.index_select(0, e_msk)
    # e_idx_th = e_idx_th.index_select(0, e_msk)
    # ts_th = ts_th.index_select(0, e_msk)
    # e_pos_th = e_pos_th.index_select(0, e_msk)
    prev_data = torch.cat((n_id.unsqueeze(1), agg_f[:, self.feat_dim:self.feat_dim + self.e_feat_dim + self.time_dim], e_th.unsqueeze(1), ts_th.unsqueeze(1)), -1)
    # self.neighborhood_store[n_pos_idx[0:src_nghs + tgt_nghs], self.n_id_idx:self.ts_emb_idx] = torch.cat((e_hidden_state.detach(), time_hidden_state.detach()), -1)[0:src_nghs + tgt_nghs]
    # self.neighborhood_store[e_pos_th, self.n_id_idx:self.ts_emb_idx] = agg_f[:, self.feat_dim:self.feat_dim + self.e_feat_dim + self.time_dim]
    # if not test:
    end = time.time()
    self.log_time('prep update memory', start, end)
    start = time.time()
    self.neighborhood_store[e_pos_th.long()] = prev_data
    self.neighborhood_store[src_start_th, self.e_emb_idx:self.ts_emb_idx] = src_src_f
    self.neighborhood_store[tgt_start_th, self.e_emb_idx:self.ts_emb_idx] = tgt_tgt_f
    # print(self.neighborhood_store[src_start_th])
    # self.update_memory_store(src_dense_n_id, src_th, src_e_f, src_ts_f, src_e_th, tgt_th, e_idx_th, cut_time_th, src_start_th)
    # self.update_memory_store(tgt_dense_n_id, tgt_th, tgt_e_f, tgt_ts_f, tgt_e_th, src_th, e_idx_th, cut_time_th, tgt_start_th)
    end = time.time()
    self.log_time('update memory', start, end)
    end_t = time.time()
    self.log_time('total forward', start_t, end_t)
    return p_score.sigmoid(), n_score.sigmoid()
  
  def get_dense_idx(self, batch_size, sparse_idx, other_m):
    sparse_idx = sparse_idx.index_select(0, other_m)    
    dense_msk = torch.ones((batch_size * self.nngh), dtype=torch.bool, device=self.device)
    if len(other_m.shape) == 0:
      dense_msk = dense_msk.reshape(batch_size, self.nngh)
      return None, dense_msk
    dense_idx = torch.randint(self.nngh, (len(other_m),), device=self.device) + sparse_idx * self.nngh
    dense_msk[dense_idx] = False
    dense_msk = dense_msk.reshape(batch_size, self.nngh)
    return dense_idx, dense_msk

  def get_dense_agg_feature(self, batch_size, dense_idx, n_f):
    agg_dense = torch.zeros(batch_size * self.nngh, self.attn_dim, device=self.device)
    if dense_idx is not None:
      agg_dense = agg_dense.index_copy_(0, dense_idx, n_f) # mindful of nondeterministic behavior
    agg_dense = agg_dense.reshape(batch_size, self.nngh, -1)
    return agg_dense

  def update_memory_store(self, src_dense_n_id, src_th, src_e_f, src_ts_f, src_e_th, tgt_th, e_idx_th, cut_time_th, src_start_th):
    indices = (src_dense_n_id == src_th.unsqueeze(1).repeat(1, self.nngh)).unsqueeze(2)
    e_indices = indices.repeat(1, 1, self.e_feat_dim)
    ts_indices = indices.repeat(1, 1, self.time_dim)
    e_f_updated = torch.sum(src_e_f * e_indices, 1) 
    ts_f_updated = torch.sum(src_ts_f * ts_indices, 1)

    self.neighborhood_store[src_e_th,0] = tgt_th.float()
    self.neighborhood_store[src_e_th,self.n_id_idx:self.e_emb_idx] = e_f_updated.detach()
    self.neighborhood_store[src_e_th,self.e_emb_idx:self.ts_emb_idx] = ts_f_updated.detach()
    self.neighborhood_store[src_e_th,-2] = e_idx_th.float()
    self.neighborhood_store[src_e_th,-1] = cut_time_th

    self.neighborhood_store[src_start_th,self.e_emb_idx:self.ts_emb_idx] = src_ts_f[:,0].detach()

  def get_relative_id(self, src_sparse_idx, tgt_sparse_idx, src_n_id, tgt_n_id, src_ts_hidden, tgt_ts_hidden):
    
    sparse_idx = torch.cat((src_sparse_idx, tgt_sparse_idx), 0)
    n_id = torch.cat((src_n_id, tgt_n_id), 0)
    ts_hidden = torch.cat((src_ts_hidden, tgt_ts_hidden), 0)
    key = torch.cat((sparse_idx.unsqueeze(1), n_id.unsqueeze(1)), -1) # tuple of (idx in the current batch, n_id)
    unique, inverse_idx = key.unique(return_inverse=True, dim=0)
    # SCATTER ADD FOR TS WITH INV IDX
    relative_ts = torch.zeros(unique.shape[0], self.time_dim + self.e_feat_dim, device=self.device)
    relative_ts.scatter_add_(0, inverse_idx.unsqueeze(1).repeat(1,self.time_dim + self.e_feat_dim), ts_hidden)
    relative_ts = relative_ts.index_select(0, inverse_idx)
    assert(relative_ts.shape[0] == sparse_idx.shape[0] == ts_hidden.shape[0])
    return relative_ts


  def store_prev_states(self, prev_states, device):
    if prev_states is not None:
      prev_src, prev_tgt, prev_e_hidden, prev_time_hidden = prev_states
      self.ngh_encoder.update_hidden_state(prev_src, prev_tgt, prev_e_hidden.detach(), prev_time_hidden.detach(), device)

  def update_prev_raw_data(self, src_idx_l, src_idx_th, tgt_idx_th, cut_time_th, e_idx_th, device):
    start = time.time()
    for i in range(len(src_idx_th)):
      self.prev_raw_data[src_idx_l[i]] = (src_idx_th[i], tgt_idx_th[i], cut_time_th[i], e_idx_th[i])
    end = time.time()
    # self.logger.info('encode ngh for the minibatch, time eclipsed: {} seconds'.format(str(end-start)))

  def get_updated_memory(self, ori_idx, cut_time_th, start_l, ngh_n_l, device):
    # TODO: can be optimize by customized CUDA kernel
    start = time.time()
    n_idx = torch.from_numpy(
      np.concatenate([np.arange(start_l[i], start_l[i]+ngh_n_l[i]) for i in range(len(start_l))])).long().to(device)
    end = time.time()
    self.log_time('get_updated_memory 1', start, end)
    start = time.time()
    ngh = self.neighborhood_store[n_idx]
    end = time.time()
    self.log_time('get_updated_memory 2', start, end)
    start = time.time()
    ngh_id = ngh[:,0:self.n_id_idx].squeeze(1).long()
    ngh_e_emb = ngh[:,self.n_id_idx:self.e_emb_idx]
    ngh_ts_emb = ngh[:,self.e_emb_idx:self.ts_emb_idx]
    ngh_e_raw = ngh[:,self.ts_emb_idx:self.e_raw_idx].squeeze(1).long()
    ngh_ts_raw = ngh[:,self.e_raw_idx:self.ts_raw_idx].squeeze(1)
    end = time.time()
    self.log_time('get_updated_memory 2.25', start, end)
    start = time.time()
    # self_msk = (ngh_id == ori_idx).nonzero().squeeze()
    ngh_ts_raw += cut_time_th * (ngh_id == ori_idx)
    end = time.time()
    self.log_time('get_updated_memory 2.5', start, end)
    start = time.time()
    e_feat = self.edge_raw_embed(ngh_e_raw)
    ts_feat = self.time_encoder(ngh_ts_raw)
    e_hidden_state = self.feature_encoder(e_feat, ngh_e_emb)
    time_hidden_state = self.time_aggregator(ts_feat, ngh_ts_emb)
    end = time.time()
    self.log_time('get_updated_memory 3', start, end)
    start = time.time()
    time_hidden_state *= (ngh_ts_raw.long() != 0).repeat(self.time_dim, 1).T
    e_hidden_state *= (ngh_e_raw != 0).repeat(self.e_feat_dim, 1).T
    end = time.time()
    self.log_time('get_updated_memory 4', start, end)
    return ngh_id, e_hidden_state, time_hidden_state, n_idx

  def update_neighborhood_state(self, src_idx_th, cut_time_th, device, is_edge=False):
    start = time.time()
    data = [self.prev_raw_data[i] if i in self.prev_raw_data else None for i in src_idx_th.tolist()]
    data = list(filter(lambda d: d is not None, data))
    if len(data) == 0:
      return None
    src, tgt, ts, e = list(zip(*data))
    
    # time_diff = cut_time_th - ts

    e_feat = self.edge_raw_embed(torch.stack(e))
    ts_feat = self.time_encoder(torch.stack(ts))
    src_idx_l = np.array(src)
    tgt_idx_l = np.array(tgt)
    e_hidden_state, time_hidden_state = self.ngh_encoder.retrieve_hidden_state(src_idx_l, tgt_idx_l, device)
    
    e_hidden_state = self.feature_encoder(e_feat, e_hidden_state, use_dropout=False) # TODO: pass node embedding
    time_hidden_state = self.time_aggregator(ts_feat, time_hidden_state, use_dropout=False)

    
    # hidden_state, cell_state = self.feature_encoder(hidden_embeddings, time_features, edge_features, hidden_state)
    # hidden_state = hidden_state.unsqueeze(-2)
    # cell_state = cell_state.unsqueeze(-2)
    # hidden_state = torch.cat((hidden_state, cell_state), -2)
    if self.agg_method == 'gru':
      self.ngh_encoder.update_hidden_state(src_idx_l, tgt_idx_l, e_hidden_state, time_hidden_state, device)
    else:
      self.ngh_encoder.update_hidden_state(src_idx_l, tgt_idx_l, (e_hidden_state[0].detach(), e_hidden_state[1].detach()), (time_hidden_state[0].detach(), time_hidden_state[1].detach()), device)
    end = time.time()
    # self.logger.info('encode prev for the minibatch, time eclipsed: {} seconds'.format(str(end-start)))
    return (src_idx_l, tgt_idx_l, e_hidden_state, time_hidden_state)

  def forward(self, p_src_ori_f, n_src_ori_f, tgt_ori_f, bad_ori_f, p_src_f, n_src_f, tgt_f, bad_f, p_src_msk, n_src_msk, tgt_msk, bad_msk, bs):
    ori_f = torch.cat((p_src_ori_f, n_src_ori_f, tgt_ori_f, bad_ori_f), 0).unsqueeze(1)
    ngh_f = torch.cat((p_src_f, n_src_f, tgt_f, bad_f), 0)
    msk = torch.cat((p_src_msk, n_src_msk, tgt_msk, bad_msk), 0)
    embed, _ = self.attn_m(ori_f, ngh_f, msk)
    embed = embed[0].squeeze(1)
    p_src_embed = embed[:bs]
    n_src_embed = embed[bs:2*bs]
    tgt_embed = embed[2*bs:3*bs]
    bad_embed = embed[3*bs:]
    
    # src_embed, _ = self.attn_m(src_ori_f.unsqueeze(1), src_f, src_msk)
    # tgt_embed, _ = self.attn_m(tgt_ori_f.unsqueeze(1), tgt_f, tgt_msk)
    # src_embed = src_embed[0].squeeze(1)
    # tgt_embed = tgt_embed[0].squeeze(1)

    p_score, p_score_walk = self.affinity_score(p_src_embed, tgt_embed)
    p_score.squeeze_(dim=-1)
    n_score, n_score_walk = self.affinity_score(n_src_embed, bad_embed)
    n_score.squeeze_(dim=-1)
    # end = time.time()
    # self.logger.info('self attention for the minibatch, time eclipsed: {} seconds'.format(str(end-start)))
    return p_score, n_score

  def get_pos_encoding_for_origin(self, src_idx, src_relative_id, tgt_relative_id):
    src_src_relative_id = torch.index_select(src_relative_id, 1, src_idx)
    src_src_relative_id = src_src_relative_id.to_dense()
    src_src_relative_id = torch.diagonal(src_src_relative_id).T
    src_tgt_relative_id = torch.index_select(tgt_relative_id, 1, src_idx)
    src_tgt_relative_id = src_tgt_relative_id.to_dense()
    src_tgt_relative_id = torch.diagonal(src_tgt_relative_id).T
    assert(src_src_relative_id.shape == src_tgt_relative_id.shape)
    src_relative_id_combined = torch.cat((src_src_relative_id, src_tgt_relative_id), -1) # [B,1,2]
    assert(src_relative_id_combined.shape[-1] == 2)
    src_pos_encode = self.trainable_embedding(src_relative_id_combined.long())
    return src_pos_encode.sum(dim=-2)

  def get_pos_encoding(self, src_ngh_idx, src_relative_id, tgt_relative_id):
    src_src_relative_id = torch.index_select(src_relative_id, 1, src_ngh_idx)
    src_src_relative_id = src_src_relative_id.to_dense()
    src_tgt_relative_id = torch.index_select(tgt_relative_id, 1, src_ngh_idx)
    src_tgt_relative_id = src_tgt_relative_id.to_dense()
    assert(src_src_relative_id.shape == src_tgt_relative_id.shape)
    src_relative_id_combined = torch.cat((src_src_relative_id, src_tgt_relative_id), -1) # [B,N,2]
    assert(src_relative_id_combined.shape[-1] == 2)
    src_pos_encode = self.trainable_embedding(src_relative_id_combined.long())
    mask = src_src_relative_id != 1
    return src_pos_encode.sum(dim=-2), mask

  def get_positions(self, src_idx_l, src_ngh_features, device):
    unqiue_src_l = np.unique(src_idx_l)
    src_pos = torch.sparse_coo_tensor([unqiue_src_l, unqiue_src_l], 2 * torch.ones(len(unqiue_src_l), 1).long(), (self.num_nodes, self.num_nodes, 1)).to(device).coalesce()
    indices = np.array([np.take(src_idx_l,src_ngh_features.indices().cpu()[0]), np.array(src_ngh_features.indices().cpu()[1])])
    indices = np.unique(indices, axis=1)
    src_pos += torch.sparse_coo_tensor(indices, torch.ones(len(indices[0]), 1).long().to(device), (self.num_nodes, self.num_nodes, 1))
    src_pos = src_pos.coalesce()
    src_ngh_zeros = torch.sparse_coo_tensor(src_pos.indices(), torch.zeros(len(src_pos.indices()[0]), 1).long().to(device), (self.num_nodes, self.num_nodes, 1))
    return src_pos, src_ngh_zeros.coalesce()

  def init_time_encoder(self):
    return TimeEncode(expand_dim=self.time_dim)

  def init_feature_encoder(self):
    if self.agg_method == 'gru':
      return FeatureEncoderGRU(self.e_feat_dim, self.e_feat_dim, self.dropout)
    else:
      return FeatureEncoderLSTM(self.e_feat_dim, self.e_feat_dim, self.dropout)

  def init_time_aggregator(self):
    if self.agg_method == 'gru':
      return FeatureEncoderGRU(self.time_dim, self.time_dim, self.dropout)
    else:
      return FeatureEncoderLSTM(self.time_dim, self.time_dim, self.dropout)

class NeighborhoodEncoder:
  def __init__(self, e_feat_dim, time_dim, agg_method='gru', dropout=0.1):
    self.logger = logging.getLogger(__name__)
    self.dropout = dropout
     # embedding layers and encoders
    self.e_feat_dim = e_feat_dim
    self.time_dim = time_dim
    self.dropout = dropout
    self.agg_method = agg_method


  def update_neighborhood_store(self, neighborhood_store):
    self.neighborhood_store = neighborhood_store
  

  # def get_states(self, src_l, device):
  #   self.neighborhood_store.
  def init_hidden_states(self, src_l, tgt_l, device):
    for i in src_l:
      if i not in self.neighborhood_store:
        self.neighborhood_store[i] = {}
    for i in range(len(src_l)):
        s = src_l[i]
        t = tgt_l[i]
        if t not in self.neighborhood_store[s]:
          if self.agg_method == 'gru':
            self.neighborhood_store[s][t] = (torch.zeros(self.e_feat_dim).to(device), torch.zeros(self.time_dim).to(device))
          else:
            self.neighborhood_store[s][t] = ((torch.zeros(self.e_feat_dim), torch.zeros(self.e_feat_dim)),
              (torch.zeros(self.time_dim), torch.zeros(self.time_dim)))


  def retrieve_hidden_state(self, src_l, tgt_l, device=None):
    # 1. initialize new neighbors in neighborhood_store

    # mark null node to far away: no such entry in sparse matrix so should be fine
    # lots of repeated edges in the same batch

    # if not exist, add new entry
    # start = time.time()

    self.init_hidden_states(src_l, tgt_l, device)
    # self.logger.info('hidden_state retrieve, time eclipsed: {} seconds'.format(str(end-start)))
    # self.logger.info('retrieve hidden states, time eclipsed: {} seconds'.format(str(end-start)))
    e_hidden_state = []
    e_cell_state = []
    time_hidden_state = []
    time_cell_state = []
    start = time.time()
    for i in range(len(src_l)):
      s = src_l[i]
      t = tgt_l[i]
      e, ts = self.neighborhood_store[s][t]
      if self.agg_method == 'gru':
        e_hidden_state.append(e)
        time_hidden_state.append(ts)
      else:
        e_hidden_state.append(e[0])
        e_cell_state.append(e[1])
        time_hidden_state.append(ts[0])
        time_cell_state.append(ts[1])
    end = time.time()
    # self.logger.info('encode, time eclipsed: {} seconds'.format(str(end-start)))
    if self.agg_method == 'gru':
      return (torch.stack(e_hidden_state), torch.stack(time_hidden_state))
    else:
      return ((torch.stack(e_hidden_state).to(device), torch.stack(e_cell_state).to(device)), (torch.stack(time_hidden_state).to(device), torch.stack(time_cell_state).to(device)))

    

  def update_hidden_state(self, src_l, tgt_l, e_hidden_state, time_hidden_state, device=None):
    # 4. feed back into neighborhodd_store
    start = time.time()

    # self.neighborhood_store *= torch.sparse_coo_tensor([src_l, tgt_l], torch.zeros(len(src_l), 2, self.model_dim), (self.num_nodes, self.num_nodes, 2, self.model_dim)).to(device)
    # self.neighborhood_store += torch.sparse_coo_tensor([src_l, tgt_l], hidden_state, (self.num_nodes, self.num_nodes, 2, self.model_dim))
    for i in range(len(src_l)):
      s = src_l[i]
      t = tgt_l[i]
      if self.agg_method == 'gru':
        self.neighborhood_store[s][t] = (e_hidden_state[i], time_hidden_state[i])
      else:
        self.neighborhood_store[s][t] = ((e_hidden_state[0][i],e_hidden_state[1][i]), (time_hidden_state[0][i], time_hidden_state[1][i]))
    end = time.time()
    # self.logger.info('put back, time eclipsed: {} seconds'.format(str(end-start)))
  
  def remove_hidden_state(self, src_l, tgt_l):
    for i in range(len(src_l)):
      s = src_l[i]
      t = tgt_l[i]
      self.neighborhood_store[s].pop(t, None)
  
  def get_ngh(self):
    return self.neighborhood_store

  def get(self , src_idx_l, device=None):
    # return nodes in neighborhood store
    start = time.time()
    # src_idx_th = torch.from_numpy(src_idx_l).long().to(device)
    encoded_ngh = [self.neighborhood_store[i] if i in self.neighborhood_store else {} for i in src_idx_l]
    # encoded_ngh = torch.index_select(self.neighborhood_store, 0, src_idx_th).coalesce()
    end = time.time()
    self.logger.info('get, time eclipsed: {} seconds'.format(str(end-start)))
    return encoded_ngh

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
    super(FeatureEncoderGRU, self).__init__()
    self.lstm = nn.LSTMCell(input_dim, hidden_dim)
    self.dropout = nn.Dropout(dropout_p)
    self.hidden_dim = hidden_dim

  def forward(self, edge_features, hidden_state, use_dropout=False):
    hidden_state, cell_state = self.lstm(edge_features, hidden_state)
    if use_dropout:
      hidden_state = self.dropout(hidden_state)
      cell_state = self.dropout(cell_state)
    return (hidden_state.cpu(), cell_state.cpu())

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


