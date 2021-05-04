# coding: UTF-8
# @Author: Shaowei Chen,     Contact: chenshaowei0507@163.com
# @Date:   2021-5-4

import torch
from torch.nn import functional as F
import logging


def normalize_size(tensor):
    if len(tensor.size()) == 3:
        tensor = tensor.contiguous().view(-1, tensor.size(2))
    elif len(tensor.size()) == 2:
        tensor = tensor.contiguous().view(-1)

    return tensor


def calculate_entity_loss(pred_start, pred_end, gold_start, gold_end):
    pred_start = normalize_size(pred_start)
    pred_end = normalize_size(pred_end)
    gold_start = normalize_size(gold_start)
    gold_end = normalize_size(gold_end)
    
    weight = torch.tensor([1, 3]).float().cuda()

    loss_start = F.cross_entropy(pred_start, gold_start.long(), size_average=False, weight=weight, ignore_index=-1)
    loss_end = F.cross_entropy(pred_end, gold_end.long(), size_average=False, weight=weight, ignore_index=-1)

    return 0.5 * loss_start + 0.5 * loss_end


def calculate_sentiment_loss(pred_sentiment, gold_sentiment):
    return F.cross_entropy(pred_sentiment, gold_sentiment.long(), size_average=False, ignore_index=-1)


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

def filter_prob(f_asp_prob, f_opi_prob, f_opi_start_index, f_opi_end_index, beta):
    filter_start = []
    filter_end = []
    for idx in range(len(f_opi_prob)):
        if f_asp_prob*f_opi_prob[idx]>=beta:
            filter_start.append(f_opi_start_index[idx])
            filter_end.append(f_opi_end_index[idx])
    return filter_start, filter_end

def filter_unpaired(start_prob, end_prob, start, end):
    filtered_start = []
    filtered_end = []
    filtered_prob = []
    if len(start)>0 and len(end)>0:
        length = start[-1]+1 if start[-1]>=end[-1] else end[-1]+1
        temp_seq = [0]*length
        for s in start:
            temp_seq[s]+=1
        for e in end:
            temp_seq[e]+=2
        last_start = -1
        for idx in range(len(temp_seq)):
            assert temp_seq[idx]<4
            if temp_seq[idx] == 1:
                last_start = idx
            elif temp_seq[idx] == 2:
                if last_start!=-1 and idx-last_start<5:
                    filtered_start.append(last_start)
                    filtered_end.append(idx)
                    prob = start_prob[start.index(last_start)] * end_prob[end.index(idx)]
                    filtered_prob.append(prob)
                last_start = -1
            elif temp_seq[idx] == 3:
                filtered_start.append(idx)
                filtered_end.append(idx)
                prob = start_prob[start.index(idx)] * end_prob[end.index(idx)]
                filtered_prob.append(prob)
                last_start = -1
    return filtered_start, filtered_end, filtered_prob