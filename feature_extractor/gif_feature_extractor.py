# coding=utf8
"""
本文件主要实现从gif 提取特征到
"""

import cv2
import numpy as np
import tensorflow as tf
from feature_extractor.feature_extractor import YouTube8MFeatureExtractor

CAP_PROP_POS_MSEC = 0


def _int64_list_feature(int64_list):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=int64_list))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _make_bytes(int_array):
  if bytes == str:  # Python2
    return ''.join(map(chr, int_array))
  else:
    return bytes(int_array)


class FeatureExtractor(object):
    def __init__(self, model_dir):
        self.yt8m_extractor = YouTube8MFeatureExtractor(model_dir)

    def extract_feature(self, vid_path, fps):
        rgb_features = []
        for rgb in self.frame_iterator(vid_path, every_ms=1000.0/fps):
            features = self.yt8m_extractor.extract_rgb_frame_features(rgb[:, :, ::-1])
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

    def frame_iterator(self, filename, every_ms=1000, max_num_frames=300):
        """Uses OpenCV to iterate over all frames of filename at a given frequency.

        Args:
        filename: Path to video file (e.g. mp4)
        every_ms: The duration (in milliseconds) to skip between frames.
        max_num_frames: Maximum number of frames to process, taken from the
          beginning of the video.

        Yields:
        RGB frame with shape (image height, image width, channels)
        """
        video_capture = cv2.VideoCapture()
        if not video_capture.open(filename):
            return
        last_ts = -99999  # The timestamp of last retrieved frame.
        num_retrieved = 0

        while num_retrieved < max_num_frames:
            while video_capture.get(CAP_PROP_POS_MSEC) < every_ms + last_ts:
                if not video_capture.read()[0]:
                    return

            last_ts = video_capture.get(CAP_PROP_POS_MSEC)
            has_frames, frame = video_capture.read()
            if not has_frames:
                break
            yield frame
            num_retrieved += 1
