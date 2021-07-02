import numpy as np
import random
import torch
import math
from torch_scatter import scatter_max


class NeighborsBuilder:
  def __init__(self, num_nodes, bias=0, device=None):
    self.num_ngh_store = 64
    self.bias = bias
    self.num_nodes = num_nodes
    self.device = device
    self.node2edges = torch.zeros((num_nodes, self.num_ngh_store, 3), device=device) # {node -> [edges]}
    self.node2idx = np.zeros(num_nodes, dtype=int) # {node -> [edges]}

    # self
    self.ngh_lengths = []  # for data analysis
    self.ngh_time_lengths = []  # for data analysis

  def build_for_node(self, src_idx_l, tgt_idx_l, cut_time_l, e_idx_l):
    replace_idx_l = np.random.randint(0, self.num_ngh_store, len(src_idx_l)) # [len(src_idx_l)]
    replace_idx_l = torch.from_numpy(replace_idx_l)
    # group by src and replace and aggregate on max ts
    scatter_max_idx = torch.from_numpy(src_idx_l).type(torch.LongTensor) * self.num_ngh_store + replace_idx_l

    out, argmax = scatter_max(torch.from_numpy(cut_time_l), scatter_max_idx)
    
    argmax = argmax.to(self.device)[argmax < len(src_idx_l)]
    src_idx_l = torch.from_numpy(src_idx_l).to(self.device).index_select(0, argmax)
    replace_idx_l = replace_idx_l.to(self.device).index_select(0, argmax)
    other_l = torch.column_stack((torch.from_numpy(tgt_idx_l).to(self.device), torch.from_numpy(cut_time_l).to(self.device), torch.from_numpy(e_idx_l).to(self.device))).type(torch.FloatTensor).to(self.device)
    other_l = other_l.index_select(0, argmax)

    self.node2edges[src_idx_l][:,replace_idx_l] = other_l
    # print("*"*50)
    # print(self.node2edges[src_idx_l][:,replace_idx_l].shape())

  # TODO: Optimize
  def build_neigbhors(self, src_idx_l, tgt_idx_l, cut_time_l, e_idx_l):
    self.build_for_node(src_idx_l, tgt_idx_l, cut_time_l, e_idx_l)
    self.build_for_node(tgt_idx_l, src_idx_l, cut_time_l, e_idx_l)



    # print(out[0:1000])
    # print(argmax[0:1000])

  
  def get_temporal_neighbors(self, src_idx_l):
    edges = self.node2edges[torch.from_numpy(src_idx_l).to(self.device)]
    out_ngh_node_batch = edges[:, 0]
    print(out_ngh_node_batch)
    out_ngh_eidx_batch = edges[:, 1]
    out_ngh_t_batch = edges[:, 2]
    return out_ngh_node_batch.detach().cpu().numpy().astype(np.int32), out_ngh_eidx_batch.detach().cpu().numpy().astype(np.int32), out_ngh_t_batch.detach().cpu().numpy()

  def add_to_neighbors(self, src_idx, tgt_idx, cut_time, e_idx):
    self.node2edges[src_idx][0][self.node2idx[src_idx]] = tgt_idx
    self.node2edges[src_idx][1][self.node2idx[src_idx]] = cut_time
    self.node2edges[src_idx][2][self.node2idx[src_idx]] = e_idx
    self.node2idx[src_idx] += 1
    if self.node2idx[src_idx] == 100:
      self.node2idx[src_idx] = 0


  def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbor=20, e_idx_l=None, tgt_idx_l=None):
    """
    Params
    ------
    src_idx_l: List[int]
    cut_time_l: List[float],
    num_neighbors: int
    """
    assert(len(src_idx_l) == len(cut_time_l))

    out_ngh_node_batch = np.zeros((len(src_idx_l), num_neighbor)).astype(np.int32)
    out_ngh_t_batch = np.zeros((len(src_idx_l), num_neighbor)).astype(np.float32)
    out_ngh_eidx_batch = np.zeros((len(src_idx_l), num_neighbor)).astype(np.int32)


    
    for i, (src_idx, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
      # ngh_idx, ngh_eidx, ngh_ts, ngh_binomial_prob = self.find_before(src_idx, cut_time, e_idx=e_idx_l[i] if e_idx_l is not None else None,
      #                                 return_binary_prob=(self.sample_method == 'binary')) #TODO: change signature and content of the function
      # if src_idx not in self.node2edges:
      #   self.node2edges[src_idx] = [[], [], []]
      ngh = self.node2edges[src_idx]
      if tgt_idx_l is not None: # else bad samples has to add a whole batch
        self.add_to_neighbors(src_idx, tgt_idx_l[i], cut_time, e_idx_l[i])
      
      # print(len(ngh[0]))
      # if len(ngh[0]) == 0:  # no previous neighbors, return padding index
      #   continue
      # print("-"*50)
      # print(ngh.shape)
      # print(len(self.node2edges[src_idx])
      # ngh = ngh.T # [N, 3] -> [3, N]
      # print(ngh.shape)
      # ngh_idx = np.array(ngh[0])
      # ngh_ts = np.array(ngh[1])
      # ngh_eidx = np.array(ngh[2])

      ngh_idx = ngh[0]
      ngh_ts = ngh[1]
      ngh_eidx = ngh[2]

      ngh_binomial_prob = None
      self.ngh_lengths.append(len(ngh_ts))  # for data anlysis
      self.ngh_time_lengths.append(ngh_ts[-1]-ngh_ts[0])  # for data anlysis
      if ngh_binomial_prob is None:  # self.sample_method is multinomial
        if math.isclose(self.bias, 0):
          sampled_idx = np.sort(np.random.randint(0, len(ngh_idx), num_neighbor))
        else:
          time_delta = cut_time - ngh_ts
          sampling_weight = np.exp(- self.bias * time_delta)
          sampling_weight = sampling_weight / sampling_weight.sum()  # normalize
          sampled_idx = np.sort(np.random.choice(np.arange(len(ngh_idx)), num_neighbor, replace=True, p=sampling_weight))
      else:
        # get a bunch of sampled idx by using sequential binary comparison, may need to be written in C later on
        sampled_idx = seq_binary_sample(ngh_binomial_prob, num_neighbor)
      out_ngh_node_batch[i, :] = ngh_idx[sampled_idx]
      out_ngh_t_batch[i, :] = ngh_ts[sampled_idx]
      out_ngh_eidx_batch[i, :] = ngh_eidx[sampled_idx]
      
    return out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch
 
  def find_k_hop(self, k, src_idx_l, cut_time_l, num_neighbors, e_idx_l=None, tgt_idx_l=None):
    """Sampling the k-hop sub graph in tree struture
    """
    if k == 0:
      return ([], [], [])
    batch = len(src_idx_l)
    layer_i = 0
    # only add tgt_idx_l for the first layer to add new edges
    # x, y, z = self.get_temporal_neighbor(src_idx_l, cut_time_l, num_neighbors[layer_i], e_idx_l=e_idx_l, tgt_idx_l=tgt_idx_l)
    x, y, z = self.get_temporal_neighbors(src_idx_l)
    node_records = [x]
    eidx_records = [y]
    t_records = [z]
    for layer_i in range(1, k):
      ngh_node_est, ngh_e_est, ngh_t_est = node_records[-1], eidx_records[-1], t_records[-1]
      ngh_node_est = ngh_node_est.flatten()
      ngh_e_est = ngh_e_est.flatten()
      ngh_t_est = ngh_t_est.flatten()
      # out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = self.get_temporal_neighbor(ngh_node_est, ngh_t_est, num_neighbors[layer_i], e_idx_l=ngh_e_est)
      out_ngh_node_batch, out_ngh_eidx_batch, out_ngh_t_batch = self.get_temporal_neighbors(src_idx_l)
      out_ngh_node_batch = out_ngh_node_batch.reshape(batch, -1)
      out_ngh_eidx_batch = out_ngh_eidx_batch.reshape(batch, -1)
      out_ngh_t_batch = out_ngh_t_batch.reshape(batch, -1)

      node_records.append(out_ngh_node_batch)
      eidx_records.append(out_ngh_eidx_batch)
      t_records.append(out_ngh_t_batch)

    return (node_records, eidx_records, t_records)  # each of them is a list of k numpy arrays, each in shape (batch,  num_neighbors ** hop_variable)
