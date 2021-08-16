import math
import torch
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


def eval_one_epoch(hint, tgan, sampler, src, tgt, ts, label, e_id, src_e, tgt_e, src_start, tgt_start, src_ngh_n, tgt_ngh_n, tgt_post_n, tb=None):
  val_acc, val_ap, val_f1, val_auc = [], [], [], []
  with torch.no_grad():
    tgan = tgan.eval()
    TEST_BATCH_SIZE = 30
    num_test_instance = len(src)
    # 
    if tb is not None:
      b_max = max(tb) + 1
      b_min = min(tb)
    else:
      b_max = math.ceil(num_test_instance / TEST_BATCH_SIZE)
      b_min = 0
    for k in range(b_min, b_max):
      # percent = 100 * k / num_test_batch
      # if k % int(0.2 * num_test_batch) == 0:
      #     logger.info('{0} progress: {1:10.4f}'.format(hint, percent))
      if tb is None:
        s_idx = k * TEST_BATCH_SIZE
        e_idx = min(num_test_instance - 1, s_idx + TEST_BATCH_SIZE)
      else:
        s_idx = np.searchsorted(tb, k)
        e_idx = np.searchsorted(tb, k, side='right')
      if s_idx == e_idx:
        continue
      batch_idx = np.arange(s_idx, e_idx)
      np.random.shuffle(batch_idx)
      src_l_cut = src[batch_idx]
      tgt_l_cut = tgt[batch_idx]
      ts_l_cut = ts[batch_idx]
      e_l_cut = e_id[batch_idx] if (e_idx is not None) else None
      # label_l_cut = label[s_idx:e_idx]

      src_e_l, tgt_e_l, src_start_l, tgt_start_l, src_ngh_n_l, tgt_ngh_n_l = src_e[batch_idx], tgt_e[batch_idx], src_start[batch_idx], tgt_start[batch_idx], src_ngh_n[batch_idx], tgt_ngh_n[batch_idx]
      size = len(src_l_cut)
      # src_l_fake, dst_l_fake = sampler.sample(size)
      _, _, _, bad_l_cut, bad_start_l, bad_ngh_n_l = sampler.sample(size)
      # bad_idx = np.random.randint(0, e_idx, size)
      # bad_l_cut, bad_start_l, bad_ngh_n_l = tgt[bad_idx], tgt_start[bad_idx], tgt_post_n[bad_idx]
      src_l = (src_l_cut, src_e_l, src_start_l, src_ngh_n_l)
      tgt_l = (tgt_l_cut, tgt_e_l, tgt_start_l, tgt_ngh_n_l)
      bad_l = (bad_l_cut, None, bad_start_l, bad_ngh_n_l)
      pos_prob, neg_prob = tgan.contrast(src_l, tgt_l, bad_l, ts_l_cut, e_l_cut, test=True)

      pred_score = np.concatenate([(pos_prob).cpu().numpy(), (neg_prob).cpu().numpy()])
      pred_label = pred_score > 0.5
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_acc.append((pred_label == true_label).mean())
      val_ap.append(average_precision_score(true_label, pred_score))
      # val_f1.append(f1_score(true_label, pred_label))
      val_auc.append(roc_auc_score(true_label, pred_score))
  return np.mean(val_acc), np.mean(val_ap), None, np.mean(val_auc)