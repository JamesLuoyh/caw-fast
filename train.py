import torch
import numpy as np
from tqdm import tqdm
import math
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from eval import *
import logging
logging.getLogger('matplotlib.font_manager').disabled = True
logging.getLogger('matplotlib.ticker').disabled = True


def train_val(train_val_data, model, mode, bs, epochs, criterion, optimizer, early_stopper, rand_samplers, logger, model_dim, t_batch=None):
  # unpack the data, prepare for the training
  train_data, val_data = train_val_data
  train_src_l, train_tgt_l, train_ts_l, train_e_idx_l, train_label_l, train_src_e_l, train_tgt_e_l, train_src_start_l, train_tgt_start_l, train_src_ngh_n_l, train_tgt_ngh_n_l, train_tgt_post_n_l = train_data
  val_src_l, val_tgt_l, val_ts_l, val_e_idx_l, val_label_l, val_src_e_l, val_tgt_e_l, val_src_start_l, val_tgt_start_l, val_src_ngh_n_l, val_tgt_ngh_n_l, val_tgt_post_n_l = val_data
  train_rand_sampler, val_rand_sampler = rand_samplers
  # partial_ngh_finder, full_ngh_finder = ngh_finders
  # if mode == 't':  # transductive
  #     model.update_ngh_finder(full_ngh_finder)
  # elif mode == 'i':  # inductive
  #     model.update_ngh_finder(partial_ngh_finder)
  # else:
  #     raise ValueError('training mode {} not found.'.format(mode))
  device = model.n_feat_th.data.device
  num_instance = len(train_src_l)  
  train_tb_l, val_tb_l = t_batch
  tb = train_tb_l is not None    
  if tb:
    num_batch = max(train_tb_l)
  else:
    num_batch = math.ceil(num_instance / bs)
  logger.info('num of training instances: {}'.format(num_instance))
  logger.info('num of batches per epoch: {}'.format(num_batch))
  idx_list = np.arange(num_instance)
  for epoch in range(epochs):
    acc, ap, f1, auc, m_loss = [], [], [], [], []
    # np.random.shuffle(idx_list)  # shuffle the training samples for every epoch
    logger.info('start {} epoch'.format(epoch))
    for k in tqdm(range(num_batch)):
      # generate training mini-batch
      if tb:
        s_idx = np.searchsorted(train_tb_l, k + 1)
        e_idx = np.searchsorted(train_tb_l, k + 1, side='right')
      else:
        s_idx = k * bs
        e_idx = min(num_instance - 1, s_idx + bs)
      
      if s_idx == e_idx:
        continue
      batch_idx = idx_list[s_idx:e_idx] # shuffle training samples for each batch
      np.random.shuffle(batch_idx)
      src_l_cut, tgt_l_cut = train_src_l[batch_idx], train_tgt_l[batch_idx]
      ts_l_cut = train_ts_l[batch_idx]
      e_l_cut = train_e_idx_l[batch_idx]
      label_l_cut = train_label_l[batch_idx]  # currently useless since we are not predicting edge labels
      src_e_l, tgt_e_l, src_start_l, tgt_start_l, src_ngh_n_l, tgt_ngh_n_l = train_src_e_l[batch_idx], train_tgt_e_l[batch_idx], train_src_start_l[batch_idx], train_tgt_start_l[batch_idx], train_src_ngh_n_l[batch_idx], train_tgt_ngh_n_l[batch_idx]
      size = len(src_l_cut)
      bad_idx = np.random.randint(0, e_idx, size)
      # bad_l_cut, bad_start_l, bad_ngh_n_l = train_tgt_l[bad_idx], train_tgt_start_l[bad_idx], np.array([train_tgt_post_n_l[b] if b < s_idx else train_tgt_ngh_n_l[b] for b in bad_idx])
      _, _, _, bad_l_cut, bad_start_l, bad_ngh_n_l = train_rand_sampler.sample(size)

      src_l = (src_l_cut, src_e_l, src_start_l, src_ngh_n_l)
      tgt_l = (tgt_l_cut, tgt_e_l, tgt_start_l, tgt_ngh_n_l)
      bad_l = (bad_l_cut, None, bad_start_l, bad_ngh_n_l)

      # feed in the data and learn from error
      optimizer.zero_grad()
      model.train()
      pos_prob, neg_prob = model.contrast(src_l, tgt_l, bad_l, ts_l_cut, e_l_cut)   # the core training code
      pos_label = torch.ones(size, dtype=torch.float, device=device, requires_grad=False)
      neg_label = torch.zeros(size, dtype=torch.float, device=device, requires_grad=False)
      loss = criterion(pos_prob, pos_label) + criterion(neg_prob, neg_label)
      loss.backward()
      optimizer.step()

      # collect training results
      with torch.no_grad():
        model.eval()
        pred_score = np.concatenate([pos_prob.cpu().detach().numpy(), neg_prob.cpu().detach().numpy()])
        pred_label = pred_score > 0.5
        true_label = np.concatenate([np.ones(size), np.zeros(size)])
        acc.append((pred_label == true_label).mean())
        ap.append(average_precision_score(true_label, pred_score))
        # f1.append(f1_score(true_label, pred_label))
        m_loss.append(loss.item())
        auc.append(roc_auc_score(true_label, pred_score))
    
    # validation phase use all information
    val_acc, val_ap, val_f1, val_auc = eval_one_epoch('val for {} nodes'.format(mode), model, val_rand_sampler, val_src_l,
                              val_tgt_l, val_ts_l, val_label_l, val_e_idx_l, val_src_e_l, val_tgt_e_l, val_src_start_l, val_tgt_start_l, val_src_ngh_n_l, val_tgt_ngh_n_l, val_tgt_post_n_l, tb=val_tb_l)
    logger.info('epoch: {}:'.format(epoch))
    logger.info('epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info('train acc: {}, val acc: {}'.format(np.mean(acc), val_acc))
    logger.info('train auc: {}, val auc: {}'.format(np.mean(auc), val_auc))
    logger.info('train ap: {}, val ap: {}'.format(np.mean(ap), val_ap))
    if epoch == 0:
      # save things for data anaysis
      checkpoint_dir = '/'.join(model.get_checkpoint_path(0).split('/')[:-1])
      # model.ngh_finder.save_ngh_stats(checkpoint_dir)  # for data analysis
      # model.save_common_node_percentages(checkpoint_dir)

    # early stop check and checkpoint saving
    if early_stopper.early_stop_check(val_ap):
      logger.info('No improvment over {} epochs, stop training'.format(early_stopper.max_round))
      logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
      best_checkpoint_path = model.get_checkpoint_path(early_stopper.best_epoch)
      model.load_state_dict(torch.load(best_checkpoint_path))
      best_ngh_store_path = model.get_ngh_store_path(early_stopper.best_epoch)
      model.set_neighborhood_store(torch.load(best_ngh_store_path))
      logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
      model.eval()
      break
    else:
      torch.save(model.neighborhood_store, model.get_ngh_store_path(epoch))
      torch.save(model.state_dict(), model.get_checkpoint_path(epoch))
    model.reset_store()

