import torch
import torch.nn as nn
import logging
import time
import numpy as np

class CAWN2(torch.nn.Module):
  def __init__(self, num_nodes, n_feat, e_feat, pos_dim=0, dropout=0.1):
    super(CAWN2, self).__init__()
    self.logger = logging.getLogger(__name__)
    self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)), requires_grad=False)
    self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)), requires_grad=False)
    self.pos_dim = pos_dim  # position feature dimension
    self.ngh_encoder = NeighborhoodEncoder(num_nodes, self.n_feat_th, self.e_feat_th, dropout)
    self.dropout_p = dropout


  def contrast(self, src_idx_l, tgt_idx_l, bad_idx_l, cut_time_l, e_idx_l=None):

    start = time.time()
    src_ngh_features = self.ngh_encoder.get(src_idx_l)
    tgt_ngh_features = self.ngh_encoder.get(tgt_idx_l)
    bad_ngh_features = self.ngh_encoder.get(bad_idx_l)
    assert(e_idx_l is not None)
    self.ngh_encoder(src_idx_l, tgt_idx_l, cut_time_l, e_idx_l, self.n_feat_th.device)
    end = time.time()
    self.logger.info('grab subgraph for the minibatch, time eclipsed: {} seconds'.format(str(end-start)))


class NeighborhoodEncoder(torch.nn.Module):
  def __init__(self, num_nodes, n_feat_th, e_feat_th, dropout=0.1):
    super(NeighborhoodEncoder, self).__init__()
    
    self.num_nodes = num_nodes
    self.n_feat_th = n_feat_th
    self.e_feat_th = e_feat_th
    self.feat_dim = self.n_feat_th.shape[1]  # node feature dimension
    self.e_feat_dim = self.e_feat_th.shape[1]  # edge feature dimension
    self.time_dim = self.feat_dim  # default to be time feature dimension
    self.model_dim = self.feat_dim + self.e_feat_dim + self.time_dim
    self.dropout = dropout
     # embedding layers and encoders
    self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True) # TODO: to deviceS
    # self.source_edge_embed = nn.parameter(torch.tensor()self.e_feat_dim)
    self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True) # TODO: to deviceS
    self.neighborhood_store = torch.sparse_coo_tensor(size=(self.num_nodes, self.num_nodes, self.model_dim)) # sparse tensor (node_idx, neighbor_idx, encoded_features)
    self.time_encoder = self.init_time_encoder() # fourier
    self.feature_encoder = self.init_feature_encoder() # LSTMCell
    

  def forward(self, src_idx_l, tgt_idx_l, cut_time_l, e_idx_l=None, device=None):
    # 1. initialize new neighbors in neighborhood_store

    # mark null node to far away: no such entry in sparse matrix so should be fine
    # lots of repeated edges in the same batch
    self.neighborhood_store += torch.sparse_coo_tensor([src_idx_l, tgt_idx_l], torch.zeros(len(src_idx_l), self.model_dim), (self.num_nodes, self.num_nodes, self.model_dim))
    self.neighborhood_store += torch.sparse_coo_tensor([tgt_idx_l, src_idx_l], torch.zeros(len(src_idx_l), self.model_dim), (self.num_nodes, self.num_nodes, self.model_dim))
    # print(self.neighborhood_store)
    
    # 2. encode time, node, edge features
    time_features = self.time_encoder(torch.from_numpy(cut_time_l).float().to(device))
    # print(time_features)
    src_idx_th = torch.from_numpy(src_idx_l).long().to(device)
    src_node_features = self.node_raw_embed(src_idx_th)  # shape [batch, node_dim]
    tgt_idx_th = torch.from_numpy(tgt_idx_l).long().to(device)
    tgt_node_features = self.node_raw_embed(tgt_idx_th)  # shape [batch, node_dim]
    # TODO: tbd on how to aggregate src node and tgt node features
    hidden_embeddings = src_node_features + tgt_node_features
    print(hidden_embeddings.shape)
    e_idx_th = torch.from_numpy(e_idx_l).long().to(device)
    edge_features = self.edge_raw_embed(e_idx_th)  # shape [batch, node_dim]
    print(edge_features)
    print(time_features.shape)
    agg_features = torch.cat([hidden_embeddings, time_features, edge_features], dim=-1)
    hidden_states = self.feature_encoder(agg_features, ) # LSTMCell

    # 3. retrieve hidden state from previous LSTMCell
    # 4. feed back into neighborhodd_store


  def get(self , src_idx_l):
    # return nodes in neighborhood store
    pass


  def init_time_encoder(self):
    return TimeEncode(expand_dim=self.time_dim)

  def init_feature_encoder(self):
    return FeatureEncoder(self.model_dim, self.model_dim, self.dropout)

class FeatureEncoder(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, dropout_p=0.1):
    super(FeatureEncoder, self).__init__()
    self.lstm_cell = nn.LSTMCell(input_dim, hidden_dim)
    self.dropout = nn.Dropout(dropout_p)

  def forward(self, input, cell_state, hidden_state):
    return self.lstm_cell(input, (hidden_state, cell_state))

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



