# @Author: Shaowei Chen,     Contact: chenshaowei0507@163.com
# @Date:   2021-5-4

import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
import numpy as np


class dual_sample(object):
    def __init__(self,
                 original_sample,
                 text,
                 forward_querys,
                 forward_answers,
                 backward_querys,
                 backward_answers,
                 sentiment_querys,
                 sentiment_answers):
        self.original_sample = original_sample
        self.text = text  #
        self.forward_querys = forward_querys
        self.forward_answers = forward_answers
        self.backward_querys = backward_querys
        self.backward_answers = backward_answers
        self.sentiment_querys = sentiment_querys
        self.sentiment_answers = sentiment_answers


class sample_tokenized(object):
    def __init__(self,
                 original_sample,
                 forward_querys,
                 forward_answers,
                 backward_querys,
                 backward_answers,
                 sentiment_querys,
                 sentiment_answers,
                 forward_seg,
                 backward_seg,
                 sentiment_seg):
        self.original_sample = original_sample
        self.forward_querys = forward_querys
        self.forward_answers = forward_answers
        self.backward_querys = backward_querys
        self.backward_answers = backward_answers
        self.sentiment_querys = sentiment_querys
        self.sentiment_answers = sentiment_answers
        self.forward_seg = forward_seg
        self.backward_seg = backward_seg
        self.sentiment_seg = sentiment_seg


class OriginalDataset(Dataset):
    def __init__(self, pre_data):
        self._forward_asp_query = pre_data['_forward_asp_query']
        self._forward_opi_query = pre_data['_forward_opi_query']  # [max_aspect_num, max_opinion_query_length]
        self._forward_asp_answer_start = pre_data['_forward_asp_answer_start']
        self._forward_asp_answer_end = pre_data['_forward_asp_answer_end']
        self._forward_opi_answer_start = pre_data['_forward_opi_answer_start']
        self._forward_opi_answer_end = pre_data['_forward_opi_answer_end']
        self._forward_asp_query_mask = pre_data['_forward_asp_query_mask']  # [max_aspect_num, max_opinion_query_length]
        self._forward_opi_query_mask = pre_data['_forward_opi_query_mask']  # [max_aspect_num, max_opinion_query_length]
        self._forward_asp_query_seg = pre_data['_forward_asp_query_seg']  # [max_aspect_num, max_opinion_query_length]
        self._forward_opi_query_seg = pre_data['_forward_opi_query_seg']  # [max_aspect_num, max_opinion_query_length]

        self._backward_asp_query = pre_data['_backward_asp_query']
        self._backward_opi_query = pre_data['_backward_opi_query']  # [max_aspect_num, max_opinion_query_length]
        self._backward_asp_answer_start = pre_data['_backward_asp_answer_start']
        self._backward_asp_answer_end = pre_data['_backward_asp_answer_end']
        self._backward_opi_answer_start = pre_data['_backward_opi_answer_start']
        self._backward_opi_answer_end = pre_data['_backward_opi_answer_end']
        self._backward_asp_query_mask = pre_data[
            '_backward_asp_query_mask']  # [max_aspect_num, max_opinion_query_length]
        self._backward_opi_query_mask = pre_data[
            '_backward_opi_query_mask']  # [max_aspect_num, max_opinion_query_length]
        self._backward_asp_query_seg = pre_data['_backward_asp_query_seg']  # [max_aspect_num, max_opinion_query_length]
        self._backward_opi_query_seg = pre_data['_backward_opi_query_seg']  # [max_aspect_num, max_opinion_query_length]

        self._sentiment_query = pre_data['_sentiment_query']  # [max_aspect_num, max_sentiment_query_length]
        self._sentiment_answer = pre_data['_sentiment_answer']
        self._sentiment_query_mask = pre_data['_sentiment_query_mask']  # [max_aspect_num, max_sentiment_query_length]
        self._sentiment_query_seg = pre_data['_sentiment_query_seg']  # [max_aspect_num, max_sentiment_query_length]

        self._aspect_num = pre_data['_aspect_num']
        self._opinion_num = pre_data['_opinion_num']


def pre_processing(sample_list, max_len):

    _tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    _forward_asp_query = []
    _forward_opi_query = []
    _forward_asp_answer_start = []
    _forward_asp_answer_end = []
    _forward_opi_answer_start = []
    _forward_opi_answer_end = []
    _forward_asp_query_mask = []
    _forward_opi_query_mask = []
    _forward_asp_query_seg = []
    _forward_opi_query_seg = []

    _backward_asp_query = []
    _backward_opi_query = []
    _backward_asp_answer_start = []
    _backward_asp_answer_end = []
    _backward_opi_answer_start = []
    _backward_opi_answer_end = []
    _backward_asp_query_mask = []
    _backward_opi_query_mask = []
    _backward_asp_query_seg = []
    _backward_opi_query_seg = []

    _sentiment_query = []
    _sentiment_answer = []
    _sentiment_query_mask = []
    _sentiment_query_seg = []

    _aspect_num = []
    _opinion_num = []


    for instance in sample_list:
        f_query_list = instance.forward_querys
        f_answer_list = instance.forward_answers
        f_query_seg_list = instance.forward_seg
        b_query_list = instance.backward_querys
        b_answer_list = instance.backward_answers
        b_query_seg_list = instance.backward_seg
        s_query_list = instance.sentiment_querys
        s_answer_list = instance.sentiment_answers
        s_query_seg_list = instance.sentiment_seg

        # _aspect_num: 1/2/3/...
        _aspect_num.append(int(len(f_query_list) - 1))
        _opinion_num.append(int(len(b_query_list) - 1))

        # Forward
        # Aspect
        # query
        assert len(f_query_list[0]) == len(f_answer_list[0][0]) == len(f_answer_list[0][1])
        f_asp_pad_num = max_len['mfor_asp_len'] - len(f_query_list[0])

        _forward_asp_query.append(_tokenizer.convert_tokens_to_ids(
            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in f_query_list[0]]))
        _forward_asp_query[-1].extend([0] * f_asp_pad_num)

        # query_mask
        _forward_asp_query_mask.append([1 for i in range(len(f_query_list[0]))])
        _forward_asp_query_mask[-1].extend([0] * f_asp_pad_num)

        # answer
        _forward_asp_answer_start.append(f_answer_list[0][0])
        _forward_asp_answer_start[-1].extend([-1] * f_asp_pad_num)
        _forward_asp_answer_end.append(f_answer_list[0][1])
        _forward_asp_answer_end[-1].extend([-1] * f_asp_pad_num)

        # seg
        _forward_asp_query_seg.append(f_query_seg_list[0])
        _forward_asp_query_seg[-1].extend([1] * f_asp_pad_num)

        # Opinion
        single_opinion_query = []
        single_opinion_query_mask = []
        single_opinion_query_seg = []
        single_opinion_answer_start = []
        single_opinion_answer_end = []
        for i in range(1, len(f_query_list)):
            assert len(f_query_list[i]) == len(f_answer_list[i][0]) == len(f_answer_list[i][1])
            pad_num = max_len['mfor_opi_len'] - len(f_query_list[i])
            # query
            single_opinion_query.append(_tokenizer.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in f_query_list[i]]))
            single_opinion_query[-1].extend([0] * pad_num)

            # query_mask
            single_opinion_query_mask.append([1 for i in range(len(f_query_list[i]))])
            single_opinion_query_mask[-1].extend([0] * pad_num)

            # query_seg
            single_opinion_query_seg.append(f_query_seg_list[i])
            single_opinion_query_seg[-1].extend([1] * pad_num)

            # answer
            single_opinion_answer_start.append(f_answer_list[i][0])
            single_opinion_answer_start[-1].extend([-1] * pad_num)
            single_opinion_answer_end.append(f_answer_list[i][1])
            single_opinion_answer_end[-1].extend([-1] * pad_num)

        # PAD: max_aspect_num
        _forward_opi_query.append(single_opinion_query)
        _forward_opi_query[-1].extend([[0 for i in range(max_len['mfor_opi_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _forward_opi_query_mask.append(single_opinion_query_mask)
        _forward_opi_query_mask[-1].extend([[0 for i in range(max_len['mfor_opi_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _forward_opi_query_seg.append(single_opinion_query_seg)
        _forward_opi_query_seg[-1].extend([[0 for i in range(max_len['mfor_opi_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _forward_opi_answer_start.append(single_opinion_answer_start)
        _forward_opi_answer_start[-1].extend([[-1 for i in range(max_len['mfor_opi_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))
        _forward_opi_answer_end.append(single_opinion_answer_end)
        _forward_opi_answer_end[-1].extend([[-1 for i in range(max_len['mfor_opi_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        # Backward
        # opinion
        # query
        assert len(b_query_list[0]) == len(b_answer_list[0][0]) == len(b_answer_list[0][1])
        b_opi_pad_num = max_len['mback_opi_len'] - len(b_query_list[0])

        _backward_opi_query.append(_tokenizer.convert_tokens_to_ids(
            [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in b_query_list[0]]))
        _backward_opi_query[-1].extend([0] * b_opi_pad_num)

        # mask
        _backward_opi_query_mask.append([1 for i in range(len(b_query_list[0]))])
        _backward_opi_query_mask[-1].extend([0] * b_opi_pad_num)

        # answer
        _backward_opi_answer_start.append(b_answer_list[0][0])
        _backward_opi_answer_start[-1].extend([-1] * b_opi_pad_num)
        _backward_opi_answer_end.append(b_answer_list[0][1])
        _backward_opi_answer_end[-1].extend([-1] * b_opi_pad_num)

        # seg
        _backward_opi_query_seg.append(b_query_seg_list[0])
        _backward_opi_query_seg[-1].extend([1] * b_opi_pad_num)

        # Aspect
        single_aspect_query = []
        single_aspect_query_mask = []
        single_aspect_query_seg = []
        single_aspect_answer_start = []
        single_aspect_answer_end = []
        for i in range(1, len(b_query_list)):
            assert len(b_query_list[i]) == len(b_answer_list[i][0]) == len(b_answer_list[i][1])
            pad_num = max_len['mback_asp_len'] - len(b_query_list[i])
            # query
            single_aspect_query.append(_tokenizer.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in b_query_list[i]]))
            single_aspect_query[-1].extend([0] * pad_num)

            # query_mask
            single_aspect_query_mask.append([1 for i in range(len(b_query_list[i]))])
            single_aspect_query_mask[-1].extend([0] * pad_num)

            # query_seg
            single_aspect_query_seg.append(b_query_seg_list[i])
            single_aspect_query_seg[-1].extend([1] * pad_num)

            # answer
            single_aspect_answer_start.append(b_answer_list[i][0])
            single_aspect_answer_start[-1].extend([-1] * pad_num)
            single_aspect_answer_end.append(b_answer_list[i][1])
            single_aspect_answer_end[-1].extend([-1] * pad_num)

        # PAD: max_opinion_num
        _backward_asp_query.append(single_aspect_query)
        _backward_asp_query[-1].extend([[0 for i in range(max_len['mback_asp_len'])]] * (max_len['max_opinion_num'] - _opinion_num[-1]))

        _backward_asp_query_mask.append(single_aspect_query_mask)
        _backward_asp_query_mask[-1].extend([[0 for i in range(max_len['mback_asp_len'])]] * (max_len['max_opinion_num'] - _opinion_num[-1]))

        _backward_asp_query_seg.append(single_aspect_query_seg)
        _backward_asp_query_seg[-1].extend([[0 for i in range(max_len['mback_asp_len'])]] * (max_len['max_opinion_num'] - _opinion_num[-1]))

        _backward_asp_answer_start.append(single_aspect_answer_start)
        _backward_asp_answer_start[-1].extend([[-1 for i in range(max_len['mback_asp_len'])]] * (max_len['max_opinion_num'] - _opinion_num[-1]))
        _backward_asp_answer_end.append(single_aspect_answer_end)
        _backward_asp_answer_end[-1].extend([[-1 for i in range(max_len['mback_asp_len'])]] * (max_len['max_opinion_num'] - _opinion_num[-1]))

        # Sentiment
        single_sentiment_query = []
        single_sentiment_query_mask = []
        single_sentiment_query_seg = []
        single_sentiment_answer = []
        for j in range(len(s_query_list)):
            sent_pad_num = max_len['max_sent_len'] - len(s_query_list[j])
            single_sentiment_query.append(_tokenizer.convert_tokens_to_ids(
                [word.lower() if word not in ['[CLS]', '[SEP]'] else word for word in s_query_list[j]]))
            single_sentiment_query[-1].extend([0] * sent_pad_num)

            single_sentiment_query_mask.append([1 for i in range(len(s_query_list[j]))])
            single_sentiment_query_mask[-1].extend([0] * sent_pad_num)

            # query_seg
            single_sentiment_query_seg.append(s_query_seg_list[j])
            single_sentiment_query_seg[-1].extend([1] * sent_pad_num)

            single_sentiment_answer.append(s_answer_list[j])

        _sentiment_query.append(single_sentiment_query)
        _sentiment_query[-1].extend([[0 for i in range(max_len['max_sent_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _sentiment_query_mask.append(single_sentiment_query_mask)
        _sentiment_query_mask[-1].extend([[0 for i in range(max_len['max_sent_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _sentiment_query_seg.append(single_sentiment_query_seg)
        _sentiment_query_seg[-1].extend([[0 for i in range(max_len['max_sent_len'])]] * (max_len['max_aspect_num'] - _aspect_num[-1]))

        _sentiment_answer.append(single_sentiment_answer)
        _sentiment_answer[-1].extend([-1] * (max_len['max_aspect_num'] - _aspect_num[-1]))

    result = {"_forward_asp_query":_forward_asp_query, "_forward_opi_query":_forward_opi_query,
              "_forward_asp_answer_start":_forward_asp_answer_start, "_forward_asp_answer_end":_forward_asp_answer_end,
              "_forward_opi_answer_start":_forward_opi_answer_start, "_forward_opi_answer_end":_forward_opi_answer_end,
              "_forward_asp_query_mask":_forward_asp_query_mask, "_forward_opi_query_mask":_forward_opi_query_mask,
              "_forward_asp_query_seg":_forward_asp_query_seg, "_forward_opi_query_seg":_forward_opi_query_seg,
              "_backward_asp_query":_backward_asp_query, "_backward_opi_query":_backward_opi_query,
              "_backward_asp_answer_start":_backward_asp_answer_start, "_backward_asp_answer_end":_backward_asp_answer_end,
              "_backward_opi_answer_start":_backward_opi_answer_start, "_backward_opi_answer_end":_backward_opi_answer_end,
              "_backward_asp_query_mask":_backward_asp_query_mask, "_backward_opi_query_mask":_backward_opi_query_mask,
              "_backward_asp_query_seg":_backward_asp_query_seg, "_backward_opi_query_seg":_backward_opi_query_seg,
              "_sentiment_query":_sentiment_query, "_sentiment_answer":_sentiment_answer, "_sentiment_query_mask":_sentiment_query_mask,
              "_sentiment_query_seg":_sentiment_query_seg, "_aspect_num":_aspect_num, "_opinion_num":_opinion_num}
    return OriginalDataset(result)


def tokenized_data(data):
    max_forward_asp_query_length = 0
    max_forward_opi_query_length = 0
    max_backward_asp_query_length = 0
    max_backward_opi_query_length = 0
    max_sentiment_query_length = 0
    max_aspect_num = 0
    max_opinion_num = 0
    tokenized_sample_list = []
    for sample in data:
        forward_querys = []
        forward_answers = []
        backward_querys = []
        backward_answers = []
        sentiment_querys = []
        sentiment_answers = []

        forward_querys_seg = []
        backward_querys_seg = []
        sentiment_querys_seg = []
        if int(len(sample.forward_querys) - 1) > max_aspect_num:
            max_aspect_num = int(len(sample.forward_querys) - 1)
        if int(len(sample.backward_querys) - 1) > max_opinion_num:
            max_opinion_num = int(len(sample.backward_querys) - 1)
        for idx in range(len(sample.forward_querys)):
            temp_query = sample.forward_querys[idx]
            temp_text = sample.text
            temp_answer = sample.forward_answers[idx]
            temp_query_to = ['[CLS]'] + temp_query + ['[SEP]'] + temp_text
            temp_query_seg = [0] * (len(temp_query) + 2) + [1] * len(temp_text)
            temp_answer[0] = [-1] * (len(temp_query) + 2) + temp_answer[0]
            temp_answer[1] = [-1] * (len(temp_query) + 2) + temp_answer[1]
            assert len(temp_answer[0]) == len(temp_answer[1]) == len(temp_query_to) == len(temp_query_seg)
            if idx == 0:
                if len(temp_query_to) > max_forward_asp_query_length:
                    max_forward_asp_query_length = len(temp_query_to)
            else:
                if len(temp_query_to) > max_forward_opi_query_length:
                    max_forward_opi_query_length = len(temp_query_to)
            forward_querys.append(temp_query_to)
            forward_answers.append(temp_answer)
            forward_querys_seg.append(temp_query_seg)
        for idx in range(len(sample.backward_querys)):
            temp_query = sample.backward_querys[idx]
            temp_text = sample.text
            temp_answer = sample.backward_answers[idx]
            temp_query_to = ['[CLS]'] + temp_query + ['[SEP]'] + temp_text
            temp_query_seg = [0] * (len(temp_query) + 2) + [1] * len(temp_text)
            temp_answer[0] = [-1] * (len(temp_query) + 2) + temp_answer[0]
            temp_answer[1] = [-1] * (len(temp_query) + 2) + temp_answer[1]
            assert len(temp_answer[0]) == len(temp_answer[1]) == len(temp_query_to) == len(temp_query_seg)
            if idx == 0:
                if len(temp_query_to) > max_backward_opi_query_length:
                    max_backward_opi_query_length = len(temp_query_to)
            else:
                if len(temp_query_to) > max_backward_asp_query_length:
                    max_backward_asp_query_length = len(temp_query_to)
            backward_querys.append(temp_query_to)
            backward_answers.append(temp_answer)
            backward_querys_seg.append(temp_query_seg)
        for idx in range(len(sample.sentiment_querys)):
            temp_query = sample.sentiment_querys[idx]
            temp_text = sample.text
            temp_answer = sample.sentiment_answers[idx]
            if if_tokenized:
                print(2)
            temp_query_to = ['[CLS]'] + temp_query + ['[SEP]'] + temp_text
            temp_query_seg = [0] * (len(temp_query) + 2) + [1] * len(temp_text)
            assert len(temp_query_to) == len(temp_query_seg)
            if len(temp_query_to) > max_sentiment_query_length:
                max_sentiment_query_length = len(temp_query_to)
            sentiment_querys.append(temp_query_to)
            sentiment_answers.append(temp_answer)
            sentiment_querys_seg.append(temp_query_seg)

        temp_sample = sample_tokenized(sample.original_sample, forward_querys, forward_answers, backward_querys,
                                       backward_answers, sentiment_querys, sentiment_answers, forward_querys_seg,
                                       backward_querys_seg, sentiment_querys_seg)
        tokenized_sample_list.append(temp_sample)
    return tokenized_sample_list, {'mfor_asp_len': max_forward_asp_query_length,
                                   'mfor_opi_len': max_forward_opi_query_length,
                                   'mback_asp_len': max_backward_asp_query_length,
                                   'mback_opi_len': max_backward_opi_query_length,
                                   'max_sent_len': max_sentiment_query_length,
                                   'max_aspect_num': max_aspect_num,
                                   'max_opinion_num': max_opinion_num}


if __name__ == '__main__':
    for dataset_name in ['14rest', '14lap', '15rest', '16rest']:
        output_path = './data/preprocess/' + dataset_name + '.pt'
        train_data = torch.load("./data/preprocess/" + dataset_name + "_train_dual.pt")
        dev_data = torch.load("./data/preprocess/" + dataset_name + "_dev_dual.pt")
        test_data = torch.load("./data/preprocess/" + dataset_name + "_test_dual.pt")

        train_tokenized, train_max_len = tokenized_data(train_data)
        dev_tokenized, dev_max_len = tokenized_data(dev_data)
        test_tokenized, test_max_len = tokenized_data(test_data)

        print('preprocessing_data')
        train_preprocess = pre_processing(train_tokenized, train_max_len)
        dev_preprocess = pre_processing(dev_tokenized, dev_max_len)
        test_preprocess = pre_processing(test_tokenized, test_max_len)
        print('save_data')
        torch.save({'train': train_preprocess, 'dev': dev_preprocess, 'test': test_preprocess}, output_path)
