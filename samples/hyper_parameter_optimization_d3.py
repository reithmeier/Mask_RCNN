import argparse
import os
import sys

import imgaug.augmenters as iaa
import numpy as np

from samples.sun.sund3 import SunD3Config, SunD3Dataset

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append('.')
sys.path.append(ROOT_DIR)  # To find local version of the library
import mrcnn.model as modellib
from mrcnn import utils

import tensorflow as tf
from tensorboard.plugins.hparams import api as tbhp

import multiprocessing

from hyperopt import hp, Trials
from hyperopt import fmin, tpe, space_eval

import joblib

from collections import OrderedDict

# hyper parameters

HP_BACKBONE = tbhp.HParam('backbone', tbhp.Discrete(["resnet50_batch_size1", "resnet50_batch_size2", "resnet101"]))
HP_TRAIN_ROIS_PER_IMAGE = tbhp.HParam('train_rois_per_image', tbhp.Discrete([50, 100, 200]))
HP_DETECTION_MIN_CONFIDENCE = tbhp.HParam('detection_min_confidence', tbhp.Discrete([0.6, 0.7, 0.8]))
HP_OPTIMIZER = tbhp.HParam('optimizer', tbhp.Discrete(['ADAM', 'SGD']))

space = OrderedDict([
    ('backbone', hp.choice('backbone', ["resnet50_batch_size1", "resnet50_batch_size2", "resnet101"])),
    ('train_rois_per_image', hp.choice('train_rois_per_image', [50, 100, 200])),
    ('detection_min_confidence', hp.choice('detection_min_confidence', [0.6, 0.7, 0.8])),
    ('optimizer', hp.choice('optimizer', ['ADAM', 'SGD']))
])

METRIC_MAP = 'mean average precision'
METRIC_F1S = 'f1 score'

with tf.summary.create_file_writer('logs/hparam_tuning_sund3').as_default():
    tbhp.hparams_config(
        hparams=[HP_BACKBONE, HP_DETECTION_MIN_CONFIDENCE, HP_TRAIN_ROIS_PER_IMAGE, HP_OPTIMIZER],
        metrics=[tbhp.Metric(METRIC_MAP, display_name='mean average precision'),
                 tbhp.Metric(METRIC_F1S, display_name='f1 score')]
    )


def inference_calculation(config, model_dir, model_path, dataset_val):
    model = modellib.MaskRCNN(mode="inference",
                              config=config,
                              model_dir=model_dir, unique_log_name=False)
    model.load_weights(model_path, by_name=True)

    mean_average_precision, f1_score = calc_mean_average_precision(dataset_val=dataset_val, inference_config=config,
                                                                   model=model)
    print("mAP: ", mean_average_precision)
    print("F1s: ", f1_score)
    return mean_average_precision, f1_score


def calc_mean_average_precision(dataset_val, inference_config, model):
    # Compute VOC-Style mAP @ IoU=0.5
    # Running on 10 images. Increase for better accuracy.
    image_ids = np.random.choice(dataset_val.image_ids, 1000)
    APs = []
    F1s = []
    for image_id in image_ids:
        # Load image and ground truth data
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_val, inference_config,
                                   image_id, use_mini_mask=False)
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # Run object detection
        results = model.detect([image], verbose=0)
        r = results[0]
        # Compute AP
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                             r["rois"], r["class_ids"], r["scores"], r['masks'])
        precision = np.mean(precisions)
        recall = np.mean(recalls)
        F1_instance = 2 * (precision * recall) / (precision + recall)
        APs.append(AP)
        F1s.append(F1_instance)
    return np.mean(APs), np.mean(F1s)


def train_test_model(hparams, data_dir, log_dir, run_name, epochs):
    print("train_test_model started")

    config = SunD3Config()
    dataset_train = SunD3Dataset()
    dataset_train.load_sun_d3(data_dir, "train13")
    dataset_train.prepare()
    dataset_val = SunD3Dataset()
    dataset_val.load_sun_d3(data_dir, "split/val13")
    dataset_val.prepare()

    # Create model in training mode
    if hparams['HP_BACKBONE'] == "resnet50_batch_size1":
        backbone = "resnet50"
        batch_size = 1
    elif hparams['HP_BACKBONE'] == "resnet50_batch_size2":
        backbone = "resnet50"
        batch_size = 2
    else:
        backbone = "resnet101"
        batch_size = 1
    config.BACKBONE = backbone
    config.BATCH_SIZE = batch_size
    config.IMAGES_PER_GPU = batch_size
    config.OPTIMIZER = hparams['HP_OPTIMIZER']
    if config.OPTIMIZER == "ADAM":
        config.LEARNING_RATE = config.LEARNING_RATE / 10
    # config.VALIDATION_STEPS = 1000  # bigger val steps
    # config.STEPS_PER_EPOCH = 10

    model = modellib.MaskRCNN(mode="training", config=config,
                              model_dir=log_dir, unique_log_name=False)
    augmentation = iaa.Sequential([
        iaa.Fliplr(0.5)
    ])

    custom_callbacks = [tbhp.KerasCallback(log_dir, hparams)]

    print("start training")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epochs,
                layers='all',
                augmentation=augmentation,
                custom_callbacks=custom_callbacks)

    print("save model")
    model_path = log_dir + run_name + ".h5"
    model.keras_model.save_weights(model_path)

    # inference calculation
    print("calculate inference")
    config.BATCH_SIZE = 1
    config.IMAGES_PER_GPU = 1
    m_ap, f1s = inference_calculation(config=config, model_path=model_path, model_dir=log_dir, dataset_val=dataset_val)

    print("train_test_model finished")
    return m_ap, f1s


def run(hparams, data_dir, log_dir, run_name, epochs, return_dict):
    print("remote process started")
    print("data_dir=" + str(data_dir))
    print("log_dir=" + str(log_dir))
    print("run_name=" + str(run_name))
    print("epochs=" + str(epochs))
    with tf.summary.create_file_writer(log_dir).as_default():
        tbhp.hparams(hparams)  # record the values used in this trial
        m_ap, f1s = train_test_model(hparams, data_dir=data_dir, log_dir=log_dir, run_name=run_name, epochs=epochs)

        tf.summary.scalar(METRIC_MAP, m_ap, step=1)
        tf.summary.scalar(METRIC_F1S, f1s, step=1)
        return_dict['m_ap'] = m_ap
        return_dict['f1s'] = f1s
        print("remote process finished")
        return m_ap


class TPESearch:
    def __init__(self, data_dir, log_dir, epochs):
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.epochs = epochs
        self.session_cnt = 0

    def objective(self, params):
        hparams = {
            'HP_BACKBONE': params['backbone'],
            'HP_TRAIN_ROIS_PER_IMAGE': params['train_rois_per_image'],
            'HP_DETECTION_MIN_CONFIDENCE': params['detection_min_confidence'],
            'HP_OPTIMIZER': params['optimizer']
        }
        run_name = "run-%d" % self.session_cnt
        print('--- Starting trial: %s' % run_name)
        print({h: hparams[h] for h in hparams})
        run_dir = self.log_dir + run_name

        # run iterative calculations in separate processes
        # because of an known tensorflow bug
        # that prevents gpu ram to be freed correctly
        manager = multiprocessing.Manager()
        return_dict = manager.dict()
        process_run = multiprocessing.Process(target=run, args=(
            hparams, self.data_dir, run_dir, run_name, self.epochs, return_dict))
        process_run.start()
        process_run.join()

        m_ap = return_dict['m_ap']
        f1s = return_dict['f1s']
        print("mAP: ", m_ap)
        print("F1s: ", f1s)
        print('--- Finished trial: %s' % run_name)
        self.session_cnt += 1
        return -f1s  # value will be minimized -> inversion needed

    def run(self):
        trials = Trials()
        for i in range(0, 10):
            best = fmin(self.objective, space, algo=tpe.suggest, max_evals=i + 1, trials=trials)
            print(best)
            print(space_eval(space, best))
            joblib.dump(trials, self.log_dir + 'hyperopt_trials_' + str(i) + '.pkl')


def grid_search(data_dir, log_dir, epochs):
    session_num = 0

    for backbone in HP_BACKBONE.domain.values:
        # for train_rois_per_image in HP_TRAIN_ROIS_PER_IMAGE.domain.values:
        for detection_min_confidence in HP_DETECTION_MIN_CONFIDENCE.domain.values:
            for optimizer in HP_OPTIMIZER.domain.values:
                hparams = {
                    HP_BACKBONE: backbone,
                    # HP_TRAIN_ROIS_PER_IMAGE: train_rois_per_image,
                    HP_DETECTION_MIN_CONFIDENCE: detection_min_confidence,
                    HP_OPTIMIZER: optimizer
                }
                print(hparams)
                run_name = "run-%d" % session_num
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})
                run_dir = log_dir + run_name
                run(hparams, data_dir=data_dir, log_dir=run_dir, run_name=run_name, epochs=epochs)
                session_num += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dir", type=str, help="Data directory",
                        default=os.path.abspath(
                            "C:\public\master_thesis_reithmeier_lukas\sunrgbd\SUN_RGBD\crop"))  # os.path.abspath("I:\Data\elevator\preprocessed"))
    parser.add_argument("-m", "--model_dir", type=str, help="Directory to store weights and results",
                        default=ROOT_DIR + "/logs/hparam_tuning_sund3/")
    parser.add_argument("-e", "--epochs", type=int, help="number of epochs to train with each configuration",
                        default=100)
    args = parser.parse_args()

    tpe_search = TPESearch(data_dir=args.data_dir, log_dir=args.model_dir, epochs=args.epochs)
    tpe_search.run()

    # grid_search(data_dir=args.data_dir, log_dir=args.model_dir, epochs=args.epochs)
