import torch
import torch.nn as nn
import logging
import time
import numpy as np

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

class CAWN2(torch.nn.Module):
  def __init__(self, num_nodes, n_feat, e_feat, pos_dim=0, n_head=4, drop_out=0.1, walk_linear_out=False, get_checkpoint_path=None):
    super(CAWN2, self).__init__()
    self.logger = logging.getLogger(__name__)
    self.drop_out = drop_out
    self.num_nodes = num_nodes
    self.n_feat_th = torch.nn.Parameter(torch.from_numpy(n_feat.astype(np.float32)), requires_grad=False)
    self.e_feat_th = torch.nn.Parameter(torch.from_numpy(e_feat.astype(np.float32)), requires_grad=False)
    self.feat_dim = self.n_feat_th.shape[1]  # node feature dimension
    self.e_feat_dim = self.e_feat_th.shape[1]  # edge feature dimension
    self.time_dim = self.feat_dim  # default to be time feature dimension
    self.feat_model_dim = self.feat_dim + self.e_feat_dim + self.time_dim
    self.pos_dim = pos_dim  # position feature dimension
    # embedding layers and encoders
    self.node_raw_embed = torch.nn.Embedding.from_pretrained(self.n_feat_th, padding_idx=0, freeze=True)
    self.edge_raw_embed = torch.nn.Embedding.from_pretrained(self.e_feat_th, padding_idx=0, freeze=True)
    self.time_encoder = self.init_time_encoder() # fourier
    self.feature_encoder = self.init_feature_encoder() # LSTMCell
    self.ngh_encoder = NeighborhoodEncoder(num_nodes, self.node_raw_embed, self.edge_raw_embed, self.time_encoder, self.feature_encoder, self.feat_model_dim, drop_out)
    self.trainable_embedding = nn.Embedding(num_embeddings=3, embedding_dim=self.pos_dim) # position embedding
    self.attn_model_dim = self.feat_model_dim + self.pos_dim
    # self.attn_m = AttnModel(self.feat_model_dim, self.pos_dim, self.attn_model_dim, n_head=n_head, drop_out=drop_out)
    self.attn_m = AttnModel(self.feat_dim, 0, self.feat_model_dim * 2, n_head=n_head, drop_out=drop_out)
    # final projection layer
    self.walk_linear_out = walk_linear_out
    # self.affinity_score = MergeLayer(self.feat_model_dim, self.feat_model_dim, self.feat_model_dim, 1, non_linear=not self.walk_linear_out) #torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)
    self.affinity_score = MergeLayer(self.feat_dim, self.feat_dim, self.feat_dim, 1, non_linear=not self.walk_linear_out) #torch.nn.Bilinear(self.feat_dim, self.feat_dim, 1, bias=True)
    self.get_checkpoint_path = get_checkpoint_path
    self.src_idx_l_prev = self.tgt_idx_l_prev = self.cut_time_l_prev = self.e_idx_l_prev = None


  def update_neighborhood_encoder(self, neighborhood_store):
    self.ngh_encoder.update_neighborhood_store(neighborhood_store)

  def contrast(self, src_idx_l, tgt_idx_l, bad_idx_l, cut_time_l, e_idx_l=None):
    device = self.n_feat_th.device
    start = time.time()
    time_features = self.time_encoder(torch.from_numpy(cut_time_l).float().to(device))
    e_idx_th = torch.from_numpy(e_idx_l).long().to(device)
    edge_features = self.edge_raw_embed(e_idx_th)  # shape [batch, node_dim]
    src_idx_th = torch.from_numpy(src_idx_l).long().to(device)
    src_node_features = self.node_raw_embed(src_idx_th)  # shape [batch, node_dim]
    tgt_idx_th = torch.from_numpy(tgt_idx_l).long().to(device)
    tgt_node_features = self.node_raw_embed(tgt_idx_th)  # shape [batch, node_dim]
    bad_idx_th = torch.from_numpy(bad_idx_l).long().to(device)
    bad_node_features = self.node_raw_embed(bad_idx_th)  # shape [batch, node_dim]
    # src_features_combined = 
    self.update_prev_raw_data(src_idx_l, tgt_idx_l, cut_time_l, e_idx_l)
    if self.src_idx_l_prev is not None:
      self.update_neighborhood_state(device)
    src_ngh_features = self.ngh_encoder.get(src_idx_l, device)
    tgt_ngh_features = self.ngh_encoder.get(tgt_idx_l, device)
    bad_ngh_features = self.ngh_encoder.get(bad_idx_l, device)
    
    end = time.time()
    # self.logger.info('grab subgraph for the minibatch, time eclipsed: {} seconds'.format(str(end-start)))
    start = time.time()
    end = time.time()
    # self.logger.info('update subgraph for the minibatch, time eclipsed: {} seconds'.format(str(end-start)))
    start = time.time()
    src_features = self.get_encoded_features(src_idx_l, src_node_features, time_features, device)
    tgt_features = self.get_encoded_features(tgt_idx_l, tgt_node_features, time_features, device)
    bad_features = self.get_encoded_features(bad_idx_l, bad_node_features, time_features, device)
    end = time.time()
    # self.logger.info('src feat for the minibatch, time eclipsed: {} seconds'.format(str(end-start)))
    pos_score = self.forward(src_idx_l, tgt_idx_l, cut_time_l, e_idx_l, (src_features, tgt_features), (src_ngh_features, tgt_ngh_features), device)
    neg_score1 = self.forward(src_idx_l, bad_idx_l, cut_time_l, e_idx_l, (src_features, bad_features), (src_ngh_features, bad_ngh_features), device)

    return pos_score.sigmoid(), neg_score1.sigmoid()
  
  def update_prev_raw_data(self, src_idx_l, tgt_idx_l, cut_time_l, e_idx_l):
    self.src_idx_l_prev, self.tgt_idx_l_prev, self.cut_time_l_prev, self.e_idx_l_prev = src_idx_l, tgt_idx_l, cut_time_l, e_idx_l

  def update_neighborhood_state(self, device):
    time_features = self.time_encoder(torch.from_numpy(self.cut_time_l_prev).float().to(device))
    e_idx_th = torch.from_numpy(self.e_idx_l_prev).long().to(device)
    edge_features = self.edge_raw_embed(e_idx_th)  # shape [batch, node_dim]
    src_idx_l_concat = np.concatenate((self.src_idx_l_prev, self.tgt_idx_l_prev), -1)
    tgt_idx_l_concat = np.concatenate((self.tgt_idx_l_prev, self.src_idx_l_prev), -1)
    src_idx_th = torch.from_numpy(self.src_idx_l_prev).long().to(device)
    src_node_features = self.node_raw_embed(src_idx_th)  # shape [batch, node_dim]
    tgt_idx_th = torch.from_numpy(self.tgt_idx_l_prev).long().to(device)
    tgt_node_features = self.node_raw_embed(tgt_idx_th)  # shape [batch, node_dim]
    hidden_state, hidden_embeddings, time_features, edge_features = self.ngh_encoder.retrieve_hidden_state(
      src_idx_l_concat, tgt_idx_l_concat, src_node_features, tgt_node_features, time_features, edge_features, device)
    hidden_state, cell_state = self.feature_encoder(hidden_embeddings, time_features, edge_features, hidden_state)
    hidden_state = hidden_state.unsqueeze(-2)
    cell_state = cell_state.unsqueeze(-2)
    hidden_state = torch.cat((hidden_state, cell_state), -2)
    self.ngh_encoder.update_hidden_state(tgt_idx_l_concat, tgt_idx_l_concat, hidden_state.detach(), device)

  def relative_node_features(self, src_idx_l, ngh_features):
    start = time.time()
    src_ngh_features, tgt_ngh_features = ngh_features
    src_ngh_idx = torch.unique(src_ngh_features.indices()[1])
    src_ngh_features_dense = torch.index_select(src_ngh_features, 1, src_ngh_idx)
    src_ngh_features_dense = src_ngh_features_dense.to_dense()
    src_ngh_features_dense = src_ngh_features_dense[:,:,0]
    tgt_ngh_features_dense = torch.index_select(tgt_ngh_features, 1, src_ngh_idx)
    tgt_ngh_features_dense = tgt_ngh_features_dense.to_dense()
    tgt_ngh_features_dense = tgt_ngh_features_dense[:,:,0]
    # print("*"*50)
    # print(src_ngh_features_dense.shape)
    # print(tgt_ngh_features_dense.shape)
    relative_node_feat = torch.cat((src_ngh_features_dense, tgt_ngh_features_dense), -1)
    # print(relative_node_feat.shape)
    mask = torch.sum(src_ngh_features_dense, -1) == 0
    end = time.time()
    # self.logger.info('relative_node_feat for the minibatch, time eclipsed: {} seconds'.format(str(end-start)))
    return relative_node_feat, mask

  def forward(self, src_idx_l, tgt_idx_l, cut_time_l, e_idx_l, origin_features, ngh_features, device):
    start = time.time()
    src_ngh_features, tgt_ngh_features = ngh_features
    src_relative_node_feat, src_mask = self.relative_node_features(src_idx_l, ngh_features)
    tgt_relative_node_feat, tgt_mask = self.relative_node_features(tgt_idx_l, (tgt_ngh_features, src_ngh_features))
    src_features, tgt_features = origin_features
    # src_ngh_features, tgt_ngh_features = ngh_features
    # 
    # src_pos, src_ngh_zeros = self.get_positions(src_idx_l, src_ngh_features, device)
    # tgt_pos, tgt_ngh_zeros = self.get_positions(tgt_idx_l, tgt_ngh_features, device)
    # src_idx_th = torch.from_numpy(src_idx_l).long().to(device)
    # tgt_idx_th = torch.from_numpy(tgt_idx_l).long().to(device)
    # src_pos_th = torch.index_select(src_pos, 0, src_idx_th)
    # src_zeros_th = torch.index_select(src_ngh_zeros, 0, src_idx_th)
    # tgt_pos_th = torch.index_select(tgt_pos, 0, tgt_idx_th)
    # tgt_zeros_th = torch.index_select(tgt_ngh_zeros, 0, tgt_idx_th)
    # src_pos_th += tgt_zeros_th
    # tgt_pos_th += src_zeros_th
    # src_relative_id = src_pos_th.coalesce()
    # tgt_relative_id = tgt_pos_th.coalesce()
    
    # src_ngh_idx = torch.unique(src_ngh_features.indices()[1])
    # src_ngh_features_dense = torch.index_select(src_ngh_features, 1, src_ngh_idx)
    # src_ngh_features_dense = src_ngh_features_dense.to_dense()
    # tgt_ngh_idx = torch.unique(tgt_ngh_features.indices()[1])
    # tgt_ngh_features_dense = torch.index_select(tgt_ngh_features, 1, tgt_ngh_idx)
    # tgt_ngh_features_dense = tgt_ngh_features_dense.to_dense()
    # src_pos_encode, src_mask = self.get_pos_encoding(src_ngh_idx, src_relative_id, tgt_relative_id)
    # tgt_pos_encode, tgt_mask = self.get_pos_encoding(tgt_ngh_idx, tgt_relative_id, src_relative_id)
    # src_origin_encode = self.get_pos_encoding_for_origin(src_idx_th, src_relative_id, tgt_relative_id)
    # tgt_origin_encode = self.get_pos_encoding_for_origin(tgt_idx_th, tgt_relative_id, src_relative_id)
    # assert(src_pos_encode.shape[1] == src_ngh_features_dense.shape[1])
    # assert(tgt_pos_encode.shape[1] == tgt_ngh_features_dense.shape[1])
    # end = time.time()
    # self.logger.info('position encode for the minibatch, time eclipsed: {} seconds'.format(str(end-start)))
    
    # encode src

    # torch.cat()
    # attention
    start = time.time()

    # concat src and tgt to replace pos_encoding
    # src_embed, _ = self.attn_m(src_features.unsqueeze(1), src_origin_encode.unsqueeze(1), src_ngh_features_dense, src_pos_encode, src_mask)
    # tgt_embed, _ = self.attn_m(tgt_features.unsqueeze(1), tgt_origin_encode.unsqueeze(1), tgt_ngh_features_dense, tgt_pos_encode, tgt_mask)
    src_embed, _ = self.attn_m(torch.cat((src_features, torch.zeros_like(src_features)), -1).unsqueeze(1), src_relative_node_feat, src_mask)
    tgt_embed, _ = self.attn_m(torch.cat((tgt_features, torch.zeros_like(tgt_features)), -1).unsqueeze(1), tgt_relative_node_feat, tgt_mask)

    # src_embedding
    # tgt_embedding
    score, score_walk = self.affinity_score(src_embed[0].squeeze(1), tgt_embed[0].squeeze(1))
    score.squeeze_(dim=-1)
    end = time.time()
    # self.logger.info('self attention for the minibatch, time eclipsed: {} seconds'.format(str(end-start)))
    return score 

  def get_encoded_features(self, src_idx_l, src_node_features, time_features, device):
    # hidden_init = self.feature_encoder.init_hidden_states(src_idx_l, src_idx_l, device)
    # src_idx_th = torch.from_numpy(src_idx_l).long().to(device)
    # hidden_state = torch.index_select(torch.index_select(hidden_init.coalesce(), 0, src_idx_th), 1, src_idx_th)
    # hidden_state = torch.diagonal(hidden_state.to_dense(),0).permute(-1, 0, 1)
    # # cell_state = torch.index_select(torch.index_select(cell_init.coalesce(), 0, src_idx_th), 1, src_idx_th)
    # # cell_state = torch.diagonal(cell_state.to_dense(),0).T
    edge_features = torch.zeros((len(src_idx_l), self.e_feat_dim)).float().to(device)
    # print(hidden_state.shape)
    hidden_state = torch.nn.init.xavier_normal_(torch.zeros(len(src_idx_l), 2, self.feat_model_dim)).to(device)
    hidden_state, cell_state = self.feature_encoder(src_node_features, time_features, edge_features, hidden_state, use_dropout=False) # LSTMCell
    return hidden_state.detach()


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
    return FeatureEncoder(self.feat_model_dim, self.feat_model_dim, self.num_nodes, self.drop_out)

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

class NeighborhoodEncoder:
  def __init__(self, num_nodes, node_raw_embed, edge_raw_embed, time_encoder, feature_encoder, model_dim, dropout=0.1):
    self.logger = logging.getLogger(__name__)
    self.num_nodes = num_nodes
    self.model_dim = model_dim
    self.dropout = dropout
     # embedding layers and encoders
    self.node_raw_embed = node_raw_embed
    self.edge_raw_embed = edge_raw_embed
    self.time_encoder = time_encoder # fourier
    self.feature_encoder = feature_encoder # LSTMCell
    
  def update_neighborhood_store(self, neighborhood_store):
    self.neighborhood_store = neighborhood_store
  
  def retrieve_hidden_state(self, src_idx_l_concat, tgt_idx_l_concat,
    src_node_features, tgt_node_features, time_features, edge_features, device=None):
    # 1. initialize new neighbors in neighborhood_store

    # mark null node to far away: no such entry in sparse matrix so should be fine
    # lots of repeated edges in the same batch

    # if not exist, add new entry
    start = time.time()
    hidden_src = self.feature_encoder.init_hidden_states(src_idx_l_concat, tgt_idx_l_concat, device)
    # print(hidden_src.shape)
    self.neighborhood_store += hidden_src
    self.neighborhood_store = self.neighborhood_store.coalesce()
    # hidden_src, cell_src = self.feature_encoder.init_hidden_states(src_idx_l, tgt_idx_l, device)
    # self.neighborhood_store[0] += hidden_src
    # self.neighborhood_store[1] += cell_src
    # hidden_tgt, cell_tgt = self.feature_encoder.init_hidden_states(tgt_idx_l, src_idx_l, device)
    # self.neighborhood_store[0] += hidden_tgt
    # self.neighborhood_store[1] += cell_tgt
    # self.neighborhood_store[0] = self.neighborhood_store[0].coalesce()
    # self.neighborhood_store[1] = self.neighborhood_store[1].coalesce()
    end = time.time()
    # self.logger.info('init, time eclipsed: {} seconds'.format(str(end-start)))
    # print(self.neighborhood_store)
    
    # 2. encode time, node, edge features
    start = time.time()
    
    time_features = torch.cat((time_features,time_features), 0)
    # print(time_features)
    # src_idx_th = torch.from_numpy(src_idx_l).long().to(device)
    # tgt_idx_th = torch.from_numpy(tgt_idx_l).long().to(device)
    src_idx_th_concat = torch.from_numpy(src_idx_l_concat).long().to(device)
    tgt_idx_th_concat = torch.from_numpy(tgt_idx_l_concat).long().to(device)
    # TODO: tbd on how to aggregate src node and tgt node features
    hidden_embeddings = torch.cat((tgt_node_features, src_node_features), 0)
    # print(hidden_embeddings.shape)
    
    edge_features = torch.cat((edge_features,edge_features), 0)
    # print(edge_features)
    # print(time_features.shape)
    end = time.time()
    # self.logger.info('feature retrieve, time eclipsed: {} seconds'.format(str(end-start)))
    # 3. retrieve hidden state from previous LSTMCell
    # print("-"*50)
    # print(self.neighborhood_store[0].coalesce()[src_idx_th,tgt_idx_th])
    # print("-"*50)
    # print(self.neighborhood_store[0].coalesce())
    # print(torch.index_select(self.neighborhood_store[0].coalesce(), 0, src_idx_th))
    # print(torch.index_select(torch.index_select(self.neighborhood_store[0].coalesce(), 0, src_idx_th), 1, tgt_idx_th))
    # start = time.time()
    # hidden_state = torch.index_select(torch.index_select(self.neighborhood_store[0], 0, src_idx_th).coalesce(), 1, tgt_idx_th).coalesce()
    # # print(hidden_state)
    # end = time.time()
    # hidden_state = torch.diagonal(hidden_state.to_dense(),0).T
    # end = time.time()
    # cell_state = torch.index_select(torch.index_select(self.neighborhood_store[1], 0, src_idx_th), 1, tgt_idx_th)
    # cell_state = torch.diagonal(cell_state.to_dense(),0).T
    # hidden_state_tgt = torch.index_select(torch.index_select(self.neighborhood_store[0], 0, tgt_idx_th), 1, src_idx_th)
    # hidden_state_tgt = torch.diagonal(hidden_state_tgt.to_dense(),0).T
    # cell_state_tgt = torch.index_select(torch.index_select(self.neighborhood_store[1], 0, tgt_idx_th), 1, src_idx_th)
    # cell_state_tgt = torch.diagonal(cell_state_tgt.to_dense(),0).T
    # hidden_state = torch.cat((hidden_state, hidden_state_tgt), 0)
    # cell_state = torch.cat((cell_state, cell_state_tgt), 0)
    # end = time.time()

    start = time.time()
    hidden_state = torch.index_select(torch.index_select(self.neighborhood_store, 0, src_idx_th_concat), 1, tgt_idx_th_concat).coalesce()
    hidden_state = torch.diagonal(hidden_state.to_dense(),0).permute(-1, 0, 1)
    end = time.time()
    # self.logger.info('hidden_state retrieve, time eclipsed: {} seconds'.format(str(end-start)))
    # self.logger.info('retrieve hidden states, time eclipsed: {} seconds'.format(str(end-start)))
    return hidden_state, hidden_embeddings, time_features, edge_features
    # print(hidden_state.T.shape)
    # hidden_state, cell_state = self.feature_encoder(hidden_embeddings, time_features, edge_features, hidden_state, cell_state) # LSTMCell

  def update_hidden_state(self, src_idx_l_concat, tgt_idx_l_concat, hidden_state, device=None):
    # 4. feed back into neighborhodd_store
    start = time.time()

    self.neighborhood_store *= torch.sparse_coo_tensor([src_idx_l_concat, tgt_idx_l_concat], torch.zeros(len(src_idx_l_concat), 2, self.model_dim), (self.num_nodes, self.num_nodes, 2, self.model_dim)).to(device)
    self.neighborhood_store += torch.sparse_coo_tensor([src_idx_l_concat, tgt_idx_l_concat], hidden_state, (self.num_nodes, self.num_nodes, 2, self.model_dim))
    # self.neighborhood_store[0] *= torch.sparse_coo_tensor([src_idx_l, tgt_idx_l], torch.zeros(len(src_idx_l), self.model_dim), (self.num_nodes, self.num_nodes, self.model_dim)).to(device)
    # self.neighborhood_store[1] *= torch.sparse_coo_tensor([src_idx_l, tgt_idx_l], torch.zeros(len(src_idx_l), self.model_dim), (self.num_nodes, self.num_nodes, self.model_dim)).to(device)
    # self.neighborhood_store[0] *= torch.sparse_coo_tensor([tgt_idx_l, src_idx_l], torch.zeros(len(src_idx_l), self.model_dim), (self.num_nodes, self.num_nodes, self.model_dim)).to(device)
    # self.neighborhood_store[1] *= torch.sparse_coo_tensor([tgt_idx_l, src_idx_l], torch.zeros(len(src_idx_l), self.model_dim), (self.num_nodes, self.num_nodes, self.model_dim)).to(device)
    
    # self.neighborhood_store[0] += torch.sparse_coo_tensor([src_idx_l, tgt_idx_l], hidden_state[:len(src_idx_l)], (self.num_nodes, self.num_nodes, self.model_dim))
    # self.neighborhood_store[1] += torch.sparse_coo_tensor([src_idx_l, tgt_idx_l], cell_state[:len(src_idx_l)], (self.num_nodes, self.num_nodes, self.model_dim))
    # self.neighborhood_store[0] += torch.sparse_coo_tensor([tgt_idx_l, src_idx_l], hidden_state[len(src_idx_l):], (self.num_nodes, self.num_nodes, self.model_dim))
    # self.neighborhood_store[1] += torch.sparse_coo_tensor([tgt_idx_l, src_idx_l], cell_state[len(src_idx_l):], (self.num_nodes, self.num_nodes, self.model_dim))
    end = time.time()
    # self.logger.info('put back, time eclipsed: {} seconds'.format(str(end-start)))
  

  def get(self , src_idx_l, device=None):
    # return nodes in neighborhood store
    start = time.time()
    src_idx_th = torch.from_numpy(src_idx_l).long().to(device)
    encoded_ngh = torch.index_select(self.neighborhood_store, 0, src_idx_th).coalesce()
    end = time.time()
    # self.logger.info('get, time eclipsed: {} seconds'.format(str(end-start)))
    return encoded_ngh

class FeatureEncoder(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, num_nodes, dropout_p=0.1):
    super(FeatureEncoder, self).__init__()
    self.lstm_cell = nn.LSTMCell(input_dim, hidden_dim)
    self.dropout = nn.Dropout(dropout_p)
    self.hidden_dim = hidden_dim
    self.num_nodes = num_nodes

  def init_hidden_states(self, src_idx_l, tgt_idx_l, device):
    hidden = torch.sparse_coo_tensor([src_idx_l, tgt_idx_l], torch.nn.init.xavier_normal_(torch.zeros(len(src_idx_l), 2, self.hidden_dim)), (self.num_nodes, self.num_nodes, 2, self.hidden_dim)).to(device)
    return hidden

  def forward(self, hidden_embeddings, time_features, edge_features, hidden_state, use_dropout=True):
    agg_features = torch.cat([hidden_embeddings, time_features, edge_features], dim=-1)
    encoded_features = self.lstm_cell(agg_features, (hidden_state[:,0], hidden_state[:,1]))
    if use_dropout:
      encoded_features = self.dropout(encoded_features[0]), self.dropout(encoded_features[1])

    return encoded_features[0], encoded_features[1]

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



