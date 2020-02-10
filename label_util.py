# coding=utf8
"""
本文件主要用来解析label
主要有两个功能：
1. key 到label的映射
2. label index 到具体意义的映射。
3. label 到index 的映射
"""

import pandas as pd
import pickle as pkl
from collections import defaultdict


class LabelHelper(object):
    def __init__(self, label_path, label_pkl):
        self._tag_keys = "cont_tag,emo_tag,act_tag_face,act_tag_body,act_tag_hand,act_tag_legy,char_tag,rating".split(",")
        self.label_2_index = self._read_label_map(label_path)
        self.index_2_label = dict([(v, k) for k, v in self.label_2_index.items()])
        self.gif_2_label_index = self.read_label_file(label_path)

    def read_label_file(self, fname):
        """
        本函数主要实现读取标注文件的
        :param fname:
        :return:
        """
        df = pd.read_csv(fname)
        gif_2_label_index = dict()
        for i, row in df.iterrows():
            gif_id = row["gif_id"]
            tmp_arr = []
            if row["status"] == "删除":
                continue
            for tag_key in self._tag_keys:
                label_str = tag_key + "_" + row[tag_key].strip()
                if label_str in self.label_2_index:
                    tmp_arr.append(self.label_2_index.get(label_str))
            gif_2_label_index[gif_id] = tmp_arr
        return gif_2_label_index

    def _read_label_map(self, fname):
        df = pd.read_csv(fname)
        label_set = set()
        for i, row in df.iterrows():
            if row["status"] == "删除":
                continue
            for tag_key in self._tag_keys:
                label_str = tag_key + "_" + row[tag_key].strip()
                label_set.add(label_str)
        sort_arr = sorted(label_set)
        result = dict([(x, i) for i, x in enumerate(sort_arr)])
        return result

    def get_gif_label(self, gif_id):
        return self.gif_2_label_index.get(gif_id)

    def get_gif_label_str(self, gif_id):
        label_indices = self.gif_2_label_index.get(gif_id)
        if label_indices is not None:
            return [self.index_2_label.get(index) for index in label_indices]
        return None

def get_label_groups():
    label_idx = {'act_tag_body_体育项目': 0, 'act_tag_body_打架': 1, 'act_tag_body_摔倒': 2, 'act_tag_body_没有动作': 3, 'act_tag_body_爬': 4, 'act_tag_body_跳舞': 5, 'act_tag_body_躺/睡/起床': 6, 'act_tag_body_运动': 7, 'act_tag_body_进/出': 8, 'act_tag_body_鞠躬': 9, 'act_tag_face_亲吻': 10, 'act_tag_face_吃': 11, 'act_tag_face_吐': 12, 'act_tag_face_听': 13, 'act_tag_face_唱歌': 14, 'act_tag_face_喝': 15, 'act_tag_face_抽烟': 16, 'act_tag_face_摇头': 17, 'act_tag_face_没有动作': 18, 'act_tag_face_点头': 19, 'act_tag_face_看': 20, 'act_tag_face_翻白眼': 21, 'act_tag_face_说': 22, 'act_tag_face_说话': 23, 'act_tag_hand_做饭': 24, 'act_tag_hand_写画': 25, 'act_tag_hand_切/挖/拧/砍': 26, 'act_tag_hand_干杯': 27, 'act_tag_hand_开/关': 28, 'act_tag_hand_开车': 29, 'act_tag_hand_打架': 30, 'act_tag_hand_打电话': 31, 'act_tag_hand_抬': 32, 'act_tag_hand_抱': 33, 'act_tag_hand_拍手': 34, 'act_tag_hand_拍照': 35, 'act_tag_hand_拿/抓/拎/放': 36, 'act_tag_hand_指': 37, 'act_tag_hand_按': 38, 'act_tag_hand_推拉': 39, 'act_tag_hand_握手': 40, 'act_tag_hand_搅拌': 41, 'act_tag_hand_摇手/摆手': 42, 'act_tag_hand_摸': 43, 'act_tag_hand_撩开/拨开': 44, 'act_tag_hand_没有动作': 45, 'act_tag_hand_洗漱': 46, 'act_tag_hand_玩手机': 47, 'act_tag_hand_玩电脑': 48, 'act_tag_hand_穿': 49, 'act_tag_hand_递': 50, 'act_tag_legy_坐': 51, 'act_tag_legy_没有动作': 52, 'act_tag_legy_站': 53, 'act_tag_legy_走/跑': 54, 'act_tag_legy_跳': 55, 'act_tag_legy_踢': 56, 'act_tag_legy_蹲/跪': 57, 'act_tag_legy_骑车': 58, 'act_tag_legy_骑车/开 车': 59, 'char_tag_搞笑': 60, 'char_tag_无': 61, 'cont_tag_其他动物': 62, 'cont_tag_其他虚拟角色': 63, 'cont_tag_动漫': 64, 'cont_tag_小孩': 65, 'cont_tag_成年人': 66, 'cont_tag_狗': 67, 'cont_tag_猫': 68, 'cont_tag_食物': 69, 'emo_tag_伤心': 70, 'emo_tag_厌恶': 71, 'emo_tag_尴尬': 72, 'emo_tag_恐惧': 73, 'emo_tag_惊讶': 74, 'emo_tag_没有表情': 75, 'emo_tag_生气': 76, 'emo_tag_高兴': 77, 'rating_一般': 78, 'rating_置顶': 79}
    label_group = defaultdict(list)
    none_pos = []
    for k, v in label_idx.items():
        head = '_'.join(k.split('_')[:-1])
        tag = k.split('_')[-1]
        if not ('没有' in tag or '无' in tag):
            none_pos.append(v)
        label_group[head].append(v)
    label_group['none'] = none_pos
    for k, v in label_group.items():
        mask = [0] * len(label_idx)
        for i in v:
            mask[i] = 1
        label_group[k] = mask
    label_group = [e for e in label_group.values()]
    return label_group[:-1], label_group[-1]

group_masks, none_mask = get_label_groups()


if __name__ == "__main__":
    '''
    import time
    ss = time.time()
    t_file = "../data/all.csv"
    t_pkl = "../data/y_encoder_25.pkl"
    t_obj = LabelHelper(t_file, t_pkl)
    print(t_obj.get_gif_label_str("f80d0f052d4de4f531e967407eb3aa3e"))
    print(t_obj.get_gif_label("f80d0f052d4de4f531e967407eb3aa3e"))
    print(len(t_obj.label_2_index))
    print(len(t_obj.gif_2_label_index))
    print(f"consuming time is {time.time() - ss}")
    '''
    import ipdb; ipdb.set_trace()
