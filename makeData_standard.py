# @Author: Shaowei Chen,     Contact: chenshaowei0507@163.com
# @Date:   2021-5-4

import torch
import pickle




def make_standard(home_path, dataset_name, dataset_type):
    # read triple
    f = open(home_path + dataset_name + "/" + dataset_name + "_pair/" + dataset_type + "_pair.pkl", "rb")
    triple_data = pickle.load(f)
    f.close()

    for triplet in triple_data:

        aspect_temp = []
        opinion_temp = []
        pair_temp = []
        triplet_temp = []
        asp_pol_temp = []
        for temp_t in triplet:
            triplet_temp.append([temp_t[0][0], temp_t[0][-1], temp_t[1][0], temp_t[1][-1], temp_t[2]])
            ap = [temp_t[0][0], temp_t[0][-1], temp_t[2]]
            if ap not in asp_pol_temp:
                asp_pol_temp.append(ap)
            a = [temp_t[0][0], temp_t[0][-1]]
            if a not in aspect_temp:
                aspect_temp.append(a)
            o = [temp_t[1][0], temp_t[1][-1]]
            if o not in opinion_temp:
                opinion_temp.append(o)
            p = [temp_t[0][0], temp_t[0][-1], temp_t[1][0], temp_t[1][-1]]
            if p not in pair_temp:
                pair_temp.append(p)

        standard_list.append({'asp_target': aspect_temp, 'opi_target': opinion_temp, 'asp_opi_target': pair_temp,
                     'asp_pol_target': asp_pol_temp, 'triplet': triplet_temp})

    return standard_list


if __name__ == '__main__':
    home_path = "./data/original/"
    dataset_name_list = ["14rest", "15rest", "16rest", "14lap"]
    for dataset_name in dataset_name_list:
        output_path = "./data/preprocess/" + dataset_name + "_standard.pt"
        dev_standard = make_standard(home_path, dataset_name, 'dev')
        test_standard = make_standard(home_path, dataset_name, 'test')
        torch.save({'dev': dev_standard, 'test': test_standard}, output_path)
