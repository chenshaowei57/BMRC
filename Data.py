# coding: UTF-8
# @Author: Shaowei Chen,     Contact: chenshaowei0507@163.com
# @Date:   2021-5-4

from torch.utils.data import Dataset, DataLoader
import numpy as np


class OriginalDataset(Dataset):
    def __init__(self, pre_data):
        self._forward_asp_query = pre_data['_forward_asp_query']
        self._forward_opi_query = pre_data['_forward_opi_query']
        self._forward_asp_answer_start = pre_data['_forward_asp_answer_start']
        self._forward_asp_answer_end = pre_data['_forward_asp_answer_end']
        self._forward_opi_answer_start = pre_data['_forward_opi_answer_start']
        self._forward_opi_answer_end = pre_data['_forward_opi_answer_end']
        self._forward_asp_query_mask = pre_data['_forward_asp_query_mask']
        self._forward_opi_query_mask = pre_data['_forward_opi_query_mask']
        self._forward_asp_query_seg = pre_data['_forward_asp_query_seg']
        self._forward_opi_query_seg = pre_data['_forward_opi_query_seg']

        self._backward_asp_query = pre_data['_backward_asp_query']
        self._backward_opi_query = pre_data['_backward_opi_query']
        self._backward_asp_answer_start = pre_data['_backward_asp_answer_start']
        self._backward_asp_answer_end = pre_data['_backward_asp_answer_end']
        self._backward_opi_answer_start = pre_data['_backward_opi_answer_start']
        self._backward_opi_answer_end = pre_data['_backward_opi_answer_end']
        self._backward_asp_query_mask = pre_data['_backward_asp_query_mask']
        self._backward_opi_query_mask = pre_data['_backward_opi_query_mask']
        self._backward_asp_query_seg = pre_data['_backward_asp_query_seg']
        self._backward_opi_query_seg = pre_data['_backward_opi_query_seg']

        self._sentiment_query = pre_data['_sentiment_query']
        self._sentiment_answer = pre_data['_sentiment_answer']
        self._sentiment_query_mask = pre_data['_sentiment_query_mask']
        self._sentiment_query_seg = pre_data['_sentiment_query_seg']

        self._aspect_num = pre_data['_aspect_num']
        self._opinion_num = pre_data['_opinion_num']


class ReviewDataset(Dataset):
    def __init__(self, train, dev, test, set):
        '''
        评论数据集
        :param train: list, training set of 14 lap, 14 res, 15 res, 16 res
        :param dev: list, the same
        :param test: list, the same
        '''
        self._train_set = train
        self._dev_set = dev
        self._test_set = test
        if set == 'train':
            self._dataset = self._train_set
        elif set == 'dev':
            self._dataset = self._dev_set
        elif set == 'test':
            self._dataset = self._test_set

        self._forward_asp_query = self._dataset._forward_asp_query
        self._forward_opi_query = self._dataset._forward_opi_query
        self._forward_asp_answer_start = self._dataset._forward_asp_answer_start
        self._forward_asp_answer_end = self._dataset._forward_asp_answer_end
        self._forward_opi_answer_start = self._dataset._forward_opi_answer_start
        self._forward_opi_answer_end = self._dataset._forward_opi_answer_end
        self._forward_asp_query_mask = self._dataset._forward_asp_query_mask
        self._forward_opi_query_mask = self._dataset._forward_opi_query_mask
        self._forward_asp_query_seg = self._dataset._forward_asp_query_seg
        self._forward_opi_query_seg = self._dataset._forward_opi_query_seg
        self._backward_asp_query = self._dataset._backward_asp_query
        self._backward_opi_query = self._dataset._backward_opi_query
        self._backward_asp_answer_start = self._dataset._backward_asp_answer_start
        self._backward_asp_answer_end = self._dataset._backward_asp_answer_end
        self._backward_opi_answer_start = self._dataset._backward_opi_answer_start
        self._backward_opi_answer_end = self._dataset._backward_opi_answer_end
        self._backward_asp_query_mask = self._dataset._backward_asp_query_mask
        self._backward_opi_query_mask = self._dataset._backward_opi_query_mask
        self._backward_asp_query_seg = self._dataset._backward_asp_query_seg
        self._backward_opi_query_seg = self._dataset._backward_opi_query_seg
        self._sentiment_query = self._dataset._sentiment_query
        self._sentiment_answer = self._dataset._sentiment_answer
        self._sentiment_query_mask = self._dataset._sentiment_query_mask
        self._sentiment_query_seg = self._dataset._sentiment_query_seg
        self._aspect_num = self._dataset._aspect_num
        self._opinion_num = self._dataset._opinion_num

    def get_batch_num(self, batch_size):
        return len(self._forward_asp_query) // batch_size

    def __len__(self):
        return len(self._forward_asp_query)

    def __getitem__(self, item):
        forward_asp_query = self._forward_asp_query[item]
        forward_opi_query = self._forward_opi_query[item]
        forward_asp_answer_start = self._forward_asp_answer_start[item]
        forward_asp_answer_end = self._forward_asp_answer_end[item]
        forward_opi_answer_start = self._forward_opi_answer_start[item]
        forward_opi_answer_end = self._forward_opi_answer_end[item]
        forward_asp_query_mask = self._forward_asp_query_mask[item]
        forward_opi_query_mask = self._forward_opi_query_mask[item]
        forward_asp_query_seg = self._forward_asp_query_seg[item]
        forward_opi_query_seg = self._forward_opi_query_seg[item]
        backward_asp_query = self._backward_asp_query[item]
        backward_opi_query = self._backward_opi_query[item]
        backward_asp_answer_start = self._backward_asp_answer_start[item]
        backward_asp_answer_end = self._backward_asp_answer_end[item]
        backward_opi_answer_start = self._backward_opi_answer_start[item]
        backward_opi_answer_end = self._backward_opi_answer_end[item]
        backward_asp_query_mask = self._backward_asp_query_mask[item]
        backward_opi_query_mask = self._backward_opi_query_mask[item]
        backward_asp_query_seg = self._backward_asp_query_seg[item]
        backward_opi_query_seg = self._backward_opi_query_seg[item]
        sentiment_query = self._sentiment_query[item]
        sentiment_answer = self._sentiment_answer[item]
        sentiment_query_mask = self._sentiment_query_mask[item]
        sentiment_query_seg = self._sentiment_query_seg[item]
        aspect_num = self._aspect_num[item]
        opinion_num = self._opinion_num[item]

        return {"forward_asp_query": np.array(forward_asp_query),
                "forward_opi_query": np.array(forward_opi_query),
                "forward_asp_answer_start": np.array(forward_asp_answer_start),
                "forward_asp_answer_end": np.array(forward_asp_answer_end),
                "forward_opi_answer_start": np.array(forward_opi_answer_start),
                "forward_opi_answer_end": np.array(forward_opi_answer_end),
                "forward_asp_query_mask": np.array(forward_asp_query_mask),
                "forward_opi_query_mask": np.array(forward_opi_query_mask),
                "forward_asp_query_seg": np.array(forward_asp_query_seg),
                "forward_opi_query_seg": np.array(forward_opi_query_seg),
                "backward_asp_query": np.array(backward_asp_query),
                "backward_opi_query": np.array(backward_opi_query),
                "backward_asp_answer_start": np.array(backward_asp_answer_start),
                "backward_asp_answer_end": np.array(backward_asp_answer_end),
                "backward_opi_answer_start": np.array(backward_opi_answer_start),
                "backward_opi_answer_end": np.array(backward_opi_answer_end),
                "backward_asp_query_mask": np.array(backward_asp_query_mask),
                "backward_opi_query_mask": np.array(backward_opi_query_mask),
                "backward_asp_query_seg": np.array(backward_asp_query_seg),
                "backward_opi_query_seg": np.array(backward_opi_query_seg),
                "sentiment_query": np.array(sentiment_query),
                "sentiment_answer": np.array(sentiment_answer),
                "sentiment_query_mask": np.array(sentiment_query_mask),
                "sentiment_query_seg": np.array(sentiment_query_seg),
                "aspect_num": np.array(aspect_num),
                "opinion_num": np.array(opinion_num)
                }


def generate_fi_batches(dataset, batch_size, shuffle=True, drop_last=True, ifgpu=True):
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size,
                            shuffle=shuffle, drop_last=drop_last)

    for data_dict in dataloader:
        out_dict = {}
        for name, tensor in data_dict.items():
            if ifgpu:
                out_dict[name] = data_dict[name].cuda()
            else:
                out_dict[name] = data_dict[name]
        yield out_dict
