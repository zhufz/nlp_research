
class MatchUtil():
    def __init__(self):
        pass

    def get_pair_id(self, text_id_list, label_list, test_size = 1):
        #return:
        #train_list:{样本id:{1:[正样本列表],0:[负样本列表]},...}
        #test_list: (样本id, [(1/0, 正/负样本id),(1/0, 正/负样本id),...]
        mp_label = defaultdict(list)
        for idx in range(len(text_id_list)):
            mp_label[label_list[idx]].append(idx)

        label_set = set(mp_label) # all labels set
        #label d1 d2
        train_list = defaultdict(dict)
        test_list = []

        #1667+91=1758
        for label in mp_label:
            #choose positive sample
            pos_list = mp_label[label]
            for idx in range(len(pos_list)-test_size):
                #if len(pos_list)-1 == 1:pdb.set_trace()
                tmp_pos_list = self._get_pos(pos_list, idx, len(pos_list)-test_size)
                for item in tmp_pos_list:
                    #train_list.append((1, pos_list[idx], item))
                    if 1 not in train_list[pos_list[idx]]:
                        train_list[pos_list[idx]][1] = []
                    train_list[pos_list[idx]][1].append(item)
                tmp_neg_list = self._get_neg(mp_label, label, label_set)
                for item in tmp_neg_list:
                    #train_list.append((0, pos_list[idx], item))
                    if 0 not in train_list[pos_list[idx]]:
                        train_list[pos_list[idx]][0] = []
                    train_list[pos_list[idx]][0].append(item)
            #test: the last sample fot each label 
            for item in pos_list[-test_size:]:
                test_list.append((item, \
                                   self._get_pos_neg(mp_label, label,
                                                     label_set, test_size)))
        return train_list, test_list

    def _get_pos(self, pos_data, idx, length):
        #select an id not equals to the idx from range(0,length) 
        assert 1 != length, "can't select diff pos sample with max=1"
        res_idx = idx
        #pdb.set_trace()
        res_list = []
        for tmp_idx in range(length):
            if idx == tmp_idx:continue
            res_list.append(pos_data[tmp_idx])
        return res_list

    def _get_neg(self, data, label, label_set):
        #select an neg label sample from data
        res_list = []
        for tmp_label in list(label_set):
            if tmp_label == label: continue
            res_list.append(random.choice(data[tmp_label][:-1]))
        return res_list

    def _get_pos_neg(self, data, label, label_set, test_size):
        data_list = []
        for tmp_label in list(label_set):
            if label == tmp_label:
                for item in data[tmp_label][:-test_size]:
                    data_list.append((1, item))
                #data_list.append((1, random.choice(data[tmp_label][:-1])))
            else:
                for item in data[tmp_label][:-test_size]:
                    data_list.append((0, item))
                #data_list.append((0, random.choice(data[tmp_label][:-1])))
        return data_list


