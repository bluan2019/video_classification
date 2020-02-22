# coding=utf8

import cv2
import numpy as np
import tensorflow as tf
from feature_extractor.feature_extractor import YouTube8MFeatureExtractor

REPEAT_TIME = 16


def _int64_list_feature(int64_list):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=int64_list))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _make_bytes(int_array):
  if bytes == str:  # Python2
    return ''.join(map(chr, int_array))
  else:
    return bytes(int_array)


class PicFeatureExtractor(object):
    def __init__(self, model_dir):
        self.yt8m_extractor = YouTube8MFeatureExtractor(model_dir)

    def extract_feature(self, vid_path):
        rgb_features = []
        rgb = self._read_img(vid_path)
        features = self.yt8m_extractor.extract_rgb_frame_features(rgb[:, :, ::-1])
        for _ in range(REPEAT_TIME):
            rgb_features.append(tf.train.Feature(float_list=tf.train.FloatList(value=features)))

        feature_list = {
            "rgb": tf.train.FeatureList(feature=rgb_features),
        }

        context_features = {
        "labels": _int64_list_feature(
            sorted([0, 1])),
        "data_path": _bytes_feature(_make_bytes(
            map(ord, vid_path))),
        }

        example = tf.train.SequenceExample(
          context=tf.train.Features(feature=context_features),
          feature_lists=tf.train.FeatureLists(feature_list=feature_list))

        return np.array([example.SerializeToString()])

    def _read_img(self, f_path):
        """Uses OpenCV to iterate over all frames of filename at a given frequency.

        Args:
        f_path: Path to image file (e.g. jpg)
        Yields:
        RGB frame with shape (image height, image width, channels)
        """
        return cv2.imread(f_path)
