# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Provides definitions for non-regularized training or test losses."""

import tensorflow as tf


class BaseLoss(object):
  """Inherit from this class when implementing new losses."""

  def calculate_loss(self, unused_predictions, unused_labels, **unused_params):
    """Calculates the average loss of the examples in a mini-batch.

     Args:
      unused_predictions: a 2-d tensor storing the prediction scores, in which
        each row represents a sample in the mini-batch and each column
        represents a class.
      unused_labels: a 2-d tensor storing the labels, which has the same shape
        as the unused_predictions. The labels must be in the range of 0 and 1.
      unused_params: loss specific parameters.

    Returns:
      A scalar loss tensor.
    """
    raise NotImplementedError()


class CrossEntropyLoss(BaseLoss):
  """Calculate the cross entropy loss between the predictions and labels.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_xent"):
      epsilon = 1e-8
      float_labels = tf.cast(labels, tf.float32)
      cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
          1 - float_labels) * tf.log(1 - predictions + epsilon)
      cross_entropy_loss = tf.negative(cross_entropy_loss)
      return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))


class HingeLoss(BaseLoss):
  """Calculate the hinge loss between the predictions and labels.

  Note the subgradient is used in the backpropagation, and thus the optimization
  may converge slower. The predictions trained by the hinge loss are between -1
  and +1.
  """

  def calculate_loss(self, predictions, labels, b=1.0, **unused_params):
    with tf.name_scope("loss_hinge"):
      float_labels = tf.cast(labels, tf.float32)
      all_zeros = tf.zeros(tf.shape(float_labels), dtype=tf.float32)
      all_ones = tf.ones(tf.shape(float_labels), dtype=tf.float32)
      sign_labels = tf.subtract(tf.scalar_mul(2, float_labels), all_ones)
      hinge_loss = tf.maximum(
          all_zeros, tf.scalar_mul(b, all_ones) - sign_labels * predictions)
      return tf.reduce_mean(tf.reduce_sum(hinge_loss, 1))


class SoftmaxLoss(BaseLoss):
  """Calculate the softmax loss between the predictions and labels.

  The function calculates the loss in the following way: first we feed the
  predictions to the softmax activation function and then we calculate
  the minus linear dot product between the logged softmax activations and the
  normalized ground truth label.

  It is an extension to the one-hot label. It allows for more than one positive
  labels for each sample.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_softmax"):
      epsilon = 1e-8
      float_labels = tf.cast(labels, tf.float32)
      # l1 normalization (labels are no less than 0)
      label_rowsum = tf.maximum(
          tf.reduce_sum(float_labels, 1, keep_dims=True),
          epsilon)
      norm_float_labels = tf.div(float_labels, label_rowsum)
      softmax_outputs = tf.nn.softmax(predictions)
      softmax_loss = tf.negative(tf.reduce_sum(
          tf.multiply(norm_float_labels, tf.log(softmax_outputs)), 1))
    return tf.reduce_mean(softmax_loss)


class GroupSoftmaxLoss(BaseLoss):
  """Calculate the softmax loss between the predictions and labels.

  The function calculates the loss in the following way: first we feed the
  predictions to the softmax activation function and then we calculate
  the minus linear dot product between the logged softmax activations and the
  normalized ground truth label.

  It is an extension to the one-hot label. It allows for more than one positive
  labels for each sample.
  """

  def calculate_loss(self, predictions, labels, **unused_params):
    with tf.name_scope("loss_softmax"):
      epsilon = 1e-8
      float_labels = tf.cast(labels, tf.float32)
      # l1 normalization (labels are no less than 0)
      label_rowsum = tf.maximum(
          tf.reduce_sum(float_labels, 1, keep_dims=True),
          epsilon)
      norm_float_labels = tf.div(float_labels, label_rowsum)
      for group in get_label_groups():
          predictions_group = tf.gather(predictions, group, axis=1)
          softmax_outputs = tf.nn.softmax(predictions)
      softmax_loss = tf.negative(tf.reduce_sum(
          tf.multiply(norm_float_labels, tf.log(softmax_outputs)), 1))
      # tf.scatter_nd
    return tf.reduce_mean(softmax_loss)

from collections import defaultdict
def get_label_groups():
    label_idx = {'act_tag_body_体育项目': 0, 'act_tag_body_打架': 1, 'act_tag_body_摔倒': 2, 'act_tag_body_没有动作': 3, 'act_tag_body_爬': 4, 'act_tag_body_跳舞': 5, 'act_tag_body_躺/睡/起床': 6, 'act_tag_body_运动': 7, 'act_tag_body_进/出': 8, 'act_tag_body_鞠躬': 9, 'act_tag_face_亲吻': 10, 'act_tag_face_吃': 11, 'act_tag_face_吐': 12, 'act_tag_face_听': 13, 'act_tag_face_唱歌': 14, 'act_tag_face_喝': 15, 'act_tag_face_抽烟': 16, 'act_tag_face_摇头': 17, 'act_tag_face_没有动作': 18, 'act_tag_face_点头': 19, 'act_tag_face_看': 20, 'act_tag_face_翻白眼': 21, 'act_tag_face_说': 22, 'act_tag_face_说话': 23, 'act_tag_hand_做饭': 24, 'act_tag_hand_写画': 25, 'act_tag_hand_切/挖/拧/砍': 26, 'act_tag_hand_干杯': 27, 'act_tag_hand_开/关': 28, 'act_tag_hand_开车': 29, 'act_tag_hand_打架': 30, 'act_tag_hand_打电话': 31, 'act_tag_hand_抬': 32, 'act_tag_hand_抱': 33, 'act_tag_hand_拍手': 34, 'act_tag_hand_拍照': 35, 'act_tag_hand_拿/抓/拎/放': 36, 'act_tag_hand_指': 37, 'act_tag_hand_按': 38, 'act_tag_hand_推拉': 39, 'act_tag_hand_握手': 40, 'act_tag_hand_搅拌': 41, 'act_tag_hand_摇手/摆手': 42, 'act_tag_hand_摸': 43, 'act_tag_hand_撩开/拨开': 44, 'act_tag_hand_没有动作': 45, 'act_tag_hand_洗漱': 46, 'act_tag_hand_玩手机': 47, 'act_tag_hand_玩电脑': 48, 'act_tag_hand_穿': 49, 'act_tag_hand_递': 50, 'act_tag_legy_坐': 51, 'act_tag_legy_没有动作': 52, 'act_tag_legy_站': 53, 'act_tag_legy_走/跑': 54, 'act_tag_legy_跳': 55, 'act_tag_legy_踢': 56, 'act_tag_legy_蹲/跪': 57, 'act_tag_legy_骑车': 58, 'act_tag_legy_骑车/开车': 59, 'char_tag_搞笑': 60, 'char_tag_无': 61, 'cont_tag_其他动物': 62, 'cont_tag_其他虚拟角色': 63, 'cont_tag_动漫': 64, 'cont_tag_小孩': 65, 'cont_tag_成年人': 66, 'cont_tag_狗': 67, 'cont_tag_猫': 68, 'cont_tag_食物': 69, 'emo_tag_伤心': 70, 'emo_tag_厌恶': 71, 'emo_tag_尴尬': 72, 'emo_tag_恐惧': 73, 'emo_tag_惊讶': 74, 'emo_tag_没有表情': 75, 'emo_tag_生气': 76, 'emo_tag_高兴': 77, 'rating_一般': 78, 'rating_置顶': 79}
    label_group = defaultdict(list)
    none_pos = []
    for k, v in label_idx.items():
        head = '_'.join(k.split('_')[:-1])
        tag = k.split('_')[-1]
        if '没有' in tag or '无' in tag:
            none_pos.append(v)
        label_group[head].append(v)
    label_group['none'] = none_pos
    
    for k, v in label_group.items():
        mask = [0] * len(label_idx)
        for i in v:
            mask[i] = 1
        label_group[k] = mask
    result = [e for e in label_group.values()]
    return result[:-1], result[-1]


if __name__ == '__main__':
    label_group = get_label_groups()
    import ipdb; ipdb.set_trace()
    
