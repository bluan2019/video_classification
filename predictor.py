# coding=utf8

import tensorflow as tf
import pickle
from feature_extractor.gif_feature_extractor import FeatureExtractor
from readers import GifFeatureReader
import glob
import os
import numpy as np
from tensorflow.python.framework import meta_graph


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'LabelHelper':
            from label_util import LabelHelper
            return LabelHelper
        return super().find_class(module, name)


def _load_pkl(f_path):
    pickle_data = CustomUnpickler(open(f_path, 'rb')).load()
    return pickle_data


class Predictor(object):
    def __init__(self, args):
        self.fe_extractor = FeatureExtractor(args.frame_model)
        self.fps = args.fps
        self.max_frames = args.max_frames  # 标识模型中最多用多少frame来进行预测
        self.label_helper = _load_pkl(args.label_pkl)
        # self.reader = GifFeatureReader(num_classes=80, feature_names=["rgb"], feature_sizes=[1024])
        _meta_file = self._get_latest_meta_file(args.model_dir)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) 
        saver = tf.train.import_meta_graph(_meta_file, clear_devices=False)
        model_dir = "{0}/export/step_80010".format(args.model_dir)
        meta_graph_def = tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING],
                                                    model_dir)

        # Get signature.
        signature_def = meta_graph_def.signature_def
        signature = signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
        self.input_tensor = signature.inputs["example_bytes"].name
        self.predictions_tensor = signature.outputs["predictions"].name
        self.predictions_indices = signature.outputs["class_indexes"].name

        
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
        prediction_vals, pred_indices = self.sess.run([self.predictions_tensor, self.predictions_indices], feed_dict={
            self.input_tensor: rgb_features,
        })  
        pred_indices = pred_indices[0][0]
        pred_labels = [self.label_helper.index_2_label.get(x, "None") for x in pred_indices]
        print(prediction_vals)
        print(pred_indices)
        print(pred_labels)
        return prediction_vals, pred_indices

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
        predictor.predict(video_path)
    finally:
        predictor.sess.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame_model", type=str, help="the pretrained model for extracting rgb feature of the frames")
    parser.add_argument("--fps", type=float, default=10.0, help="how many frames to extract per second")
    parser.add_argument("--max_frames", type=int, default=100, help="maximum frames of features")
    parser.add_argument("--label_pkl", type=str, default="data/label_pkl.pkl", help=" path to pkl of labels")
    parser.add_argument("--model_dir", type=str, help="path to the finally model")
    parser.add_argument("--input", type=str, help="path to video or gif file")
    _args = parser.parse_args()
    main(_args)
