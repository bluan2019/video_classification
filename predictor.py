# coding=utf8

import tensorflow as tf
from feature_extractor.gif_feature_extractor import FeatureExtractor
# from readers import GifFeatureReader
import glob
import os
import numpy as np


class Predictor(object):
    def __init__(self, args):
        self.fe_extractor = FeatureExtractor(args.frame_model)
        self.fps = args.fps
        self.max_frames = args.max_frames  # 标识模型中最多用多少frame来进行预测
        # self.reader = GifFeatureReader(num_classes=80, feature_names=["rgb"], feature_sizes=[1024])
        _meta_file = self._get_latest_meta_file(args.model_dir)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) 
        saver = tf.train.import_meta_graph(_meta_file, clear_devices=False)
        ckp_file = _meta_file.rstrip(".meta")
        saver.restore(self.sess, ckp_file)
        self.input_tensor = tf.get_collection("input_batch_raw")[0]
        self.num_frames = tf.get_collection("num_frames")[0]
        self.predictions_tensor = tf.get_collection("predictions")[0]
        self.sess.run(self._set_up_init_ops(tf.get_collection_ref(tf.GraphKeys.LOCAL_VARIABLES)))

    def _set_up_init_ops(self, variables):
        init_op_list = []
        for variable in list(variables):
            if "train_input" in variable.name:
                init_op_list.append(tf.assign(variable, 1))
                variables.remove(variable)
        init_op_list.append(tf.variables_initializer(variables))
        return init_op_list

    def predict(self, vid_path):
        rgb_features = self.fe_extractor.extract_feature(vid_path, self.fps)
        num_frames = min(rgb_features.shape[0], self.max_frames)
        feature_matrix = self.resize_axis(rgb_features, 0, self.max_frames)
        prediction_vals = self.sess.run([self.predictions_tensor], feed_dict={
            self.input_tensor: np.array([feature_matrix]),
            self.num_frames: np.array([num_frames])
        })
        print(prediction_vals)
        return prediction_vals

    def transform_feature(self, features):
        pass

    def _get_latest_meta_file(self, model_dir):
        index_files = glob.glob(os.path.join(model_dir, 'model.ckpt-*.meta'))
        # No files
        if not index_files:
            return None
        # Index file path with the maximum step size.
        latest_index_file = sorted(
            [(int(os.path.basename(f).split("-")[-1].split(".")[0]), f)
             for f in index_files])[-1][1]
        # # Chop off .index suffix and return
        # return latest_index_file[:-6]
        return latest_index_file

    @staticmethod
    def resize_axis(tensor, axis, new_size, fill_value=0):
        """Truncates or pads a tensor to new_size on on a given axis.

        Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
        size increases, the padding will be performed at the end, using fill_value.

        Args:
          tensor: The tensor to be resized, numpy array
          axis: An integer representing the dimension to be sliced.
          new_size: An integer or 0d tensor representing the new value for
            tensor.shape[axis].
          fill_value: Value to use to fill any new entries in the tensor. Will be
            cast to the type of tensor.

        Returns:
          The resized tensor.
        """

        shape = tensor.shape
        # 当目标tensor的维度小于原始tensor时， 截取
        if new_size <= shape[axis]:
            return np.take(tensor, range(new_size), axis=axis)
        # 当目标tensor的维度 大于原始tensor是需要做padding
        pad_shape = list(shape) 
        pad_shape[axis] = new_size - shape[axis]
        pad_tensor = np.zeros(pad_shape, dtype=tensor.dtype)
        return np.concatenate((tensor, pad_tensor), axis=axis)


def main(args):
    try:
        predictor = Predictor(args)
        video_path = args.input
        # import ipdb; ipdb.set_trace()
        predictor.predict(video_path)
    finally:
        predictor.sess.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_model", type=str, help="the pretrained model for extracting rgb feature of the frames")
    parser.add_argument("--fps", type=float, default=10.0, help="how many frames to extract per second")
    parser.add_argument("--max_frames", type=int, default=100, help="maximum frames of features")
    parser.add_argument("--model_dir", type=str, help="path to the finally model")
    parser.add_argument("--input", type=str, help="path to video or gif file")
    _args = parser.parse_args()
    main(_args)
