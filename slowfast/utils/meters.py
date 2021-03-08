#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Meters."""

import datetime
import numpy as np
import os
from collections import defaultdict, deque
import torch
import torch.nn.functional as F
from fvcore.common.timer import Timer
from sklearn.metrics import average_precision_score

import slowfast.utils.logging as logging
import slowfast.utils.metrics as metrics
import slowfast.utils.misc as misc

logger = logging.get_logger(__name__)


class TestMeter(object):
    """
    Perform the multi-view ensemble for testing: each audio with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the audio.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
        self,
        num_audios,
        num_clips,
        num_cls,
        overall_iters,
        multi_label=False,
        ensemble_method="sum",
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each audio, and calculate the metrics on
        num_audios audios.
        Args:
            num_audios (int): number of audios to test.
            num_clips (int): number of clips sampled from each audio for
                aggregating the final prediction for the audio.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            multi_label (bool): if True, use map as the metric.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
        """

        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.multi_label = multi_label
        self.ensemble_method = ensemble_method
        # Initialize tensors.
        self.audio_preds = torch.zeros((num_audios, num_cls))
        self.audio_preds_clips = torch.zeros((num_audios, num_clips, num_cls))
        if multi_label:
            self.audio_preds -= 1e10

        self.audio_labels = (
            torch.zeros((num_audios, num_cls))
            if multi_label
            else torch.zeros((num_audios)).long()
        )
        self.clip_count = torch.zeros((num_audios)).long()
        self.topk_accs = []
        self.stats = {}

        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.audio_preds.zero_()
        self.audio_preds_clips.zero_()
        if self.multi_label:
            self.audio_preds -= 1e10
        self.audio_labels.zero_()

    def update_stats(self, preds, labels, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(preds.shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            clip_temporal_id = int(clip_ids[ind]) % self.num_clips
            if self.audio_labels[vid_id].sum() > 0:
                assert torch.equal(
                    self.audio_labels[vid_id].type(torch.FloatTensor),
                    labels[ind].type(torch.FloatTensor),
                )
            self.audio_labels[vid_id] = labels[ind]
            if self.ensemble_method == "sum":
                self.audio_preds[vid_id] += preds[ind]
            elif self.ensemble_method == "max":
                self.audio_preds[vid_id] = torch.max(
                    self.audio_preds[vid_id], preds[ind]
                )
            else:
                raise NotImplementedError(
                    "Ensemble Method {} is not supported".format(
                        self.ensemble_method
                    )
                )
            self.audio_preds_clips[vid_id, clip_temporal_id] = preds[ind]
            self.clip_count[vid_id] += 1

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def finalize_metrics(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    ", ".join(
                        [
                            "{}: {}".format(i, k)
                            for i, k in enumerate(self.clip_count.tolist())
                        ]
                    ),
                    self.num_clips,
                )
            )

        self.stats = {"split": "test_final"}
        if self.multi_label:
            map = get_map(
                self.audio_preds.cpu().numpy(), self.audio_labels.cpu().numpy()
            )
            self.stats["map"] = map
        else:
            num_topks_correct = metrics.topks_correct(
                self.audio_preds, self.audio_labels, ks
            )
            topks = [
                (x / self.audio_preds.size(0)) * 100.0
                for x in num_topks_correct
            ]
            assert len({len(ks), len(topks)}) == 1
            for k, topk in zip(ks, topks):
                self.stats["top{}_acc".format(k)] = "{:.{prec}f}".format(
                    topk, prec=2
                )

        logging.log_json_stats(self.stats)
        return self.audio_preds.numpy().copy(), \
               self.audio_preds_clips.numpy().copy(), \
               F.one_hot(self.audio_labels, num_classes=self.audio_preds.shape[1]).numpy().copy(), \
               None


class ScalarMeter(object):
    """
    A scalar meter uses a deque to track a series of scaler values with a given
    window size. It supports calculating the median and average values of the
    window, and also supports calculating the global average.
    """

    def __init__(self, window_size):
        """
        Args:
            window_size (int): size of the max length of the deque.
        """
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0

    def reset(self):
        """
        Reset the deque.
        """
        self.deque.clear()
        self.total = 0.0
        self.count = 0

    def add_value(self, value):
        """
        Add a new scalar value to the deque.
        """
        self.deque.append(value)
        self.count += 1
        self.total += value

    def get_win_median(self):
        """
        Calculate the current median value of the deque.
        """
        return np.median(self.deque)

    def get_win_avg(self):
        """
        Calculate the current average value of the deque.
        """
        return np.mean(self.deque)

    def get_global_avg(self):
        """
        Calculate the global mean value.
        """
        return self.total / self.count


class TrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.lr = None
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.lr = None
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_err, top5_err, loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        self.loss.add_value(loss)
        self.lr = lr
        self.loss_total += loss * mb_size
        self.num_samples += mb_size

        if not self._cfg.DATA.MULTI_LABEL:
            # Current minibatch stats
            self.mb_top1_err.add_value(top1_err)
            self.mb_top5_err.add_value(top5_err)
            # Aggregate stats
            self.num_top1_mis += top1_err * mb_size
            self.num_top5_mis += top5_err * mb_size

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            stats["top1_err"] = self.mb_top1_err.get_win_median()
            stats["top5_err"] = self.mb_top5_err.get_win_median()
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            top1_err = self.num_top1_mis / self.num_samples
            top5_err = self.num_top5_mis / self.num_samples
            avg_loss = self.loss_total / self.num_samples
            stats["top1_err"] = top1_err
            stats["top5_err"] = top5_err
            stats["loss"] = avg_loss
        logging.log_json_stats(stats)


class ValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        # Current minibatch errors (smoothed over a window).
        self.mb_top1_err = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_err = ScalarMeter(cfg.LOG_PERIOD)
        # Min errors (over the full val set).
        self.min_top1_err = 100.0
        self.min_top5_err = 100.0
        # Number of misclassified examples.
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []
        self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.mb_top1_err.reset()
        self.mb_top5_err.reset()
        self.num_top1_mis = 0
        self.num_top5_mis = 0
        self.num_samples = 0
        self.all_preds = []
        self.all_labels = []

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_err, top5_err, mb_size):
        """
        Update the current stats.
        Args:
            top1_err (float): top1 error rate.
            top5_err (float): top5 error rate.
            mb_size (int): mini batch size.
        """
        self.mb_top1_err.add_value(top1_err)
        self.mb_top5_err.add_value(top5_err)
        self.num_top1_mis += top1_err * mb_size
        self.num_top5_mis += top5_err * mb_size
        self.num_samples += mb_size

    def update_predictions(self, preds, labels):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
        """
        # TODO: merge update_prediction with update_stats.
        self.all_preds.append(preds)
        self.all_labels.append(labels)

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        if not self._cfg.DATA.MULTI_LABEL:
            stats["top1_err"] = self.mb_top1_err.get_win_median()
            stats["top5_err"] = self.mb_top5_err.get_win_median()
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        if self._cfg.DATA.MULTI_LABEL:
            stats["map"] = get_map(
                torch.cat(self.all_preds).cpu().numpy(),
                torch.cat(self.all_labels).cpu().numpy(),
            )
        else:
            top1_err = self.num_top1_mis / self.num_samples
            top5_err = self.num_top5_mis / self.num_samples
            is_best_epoch = top1_err < self.min_top1_err
            self.min_top1_err = min(self.min_top1_err, top1_err)
            self.min_top5_err = min(self.min_top5_err, top5_err)

            stats["top1_err"] = top1_err
            stats["top5_err"] = top5_err
            stats["min_top1_err"] = self.min_top1_err
            stats["min_top5_err"] = self.min_top5_err

        logging.log_json_stats(stats)

        return is_best_epoch, {"top1_err": top1_err}


class EPICTrainMeter(object):
    """
    Measure training stats.
    """

    def __init__(self, epoch_iters, cfg):
        """
        Args:
            epoch_iters (int): the overall number of iterations of one epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.epoch_iters = epoch_iters
        self.MAX_EPOCH = cfg.SOLVER.MAX_EPOCH * epoch_iters
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.loss = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_total = 0.0
        self.loss_verb = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_verb_total = 0.0
        self.loss_noun = ScalarMeter(cfg.LOG_PERIOD)
        self.loss_noun_total = 0.0
        self.lr = None
        # Current minibatch accuracies (smoothed over a window).
        self.mb_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_verb_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_verb_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_noun_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_noun_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        # Number of correctly classified examples.
        self.num_top1_cor = 0
        self.num_top5_cor = 0
        self.num_verb_top1_cor = 0
        self.num_verb_top5_cor = 0
        self.num_noun_top1_cor = 0
        self.num_noun_top5_cor = 0
        self.num_samples = 0
        self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        """
        Reset the Meter.
        """
        self.loss.reset()
        self.loss_total = 0.0
        self.loss_verb.reset()
        self.loss_verb_total = 0.0
        self.loss_noun.reset()
        self.loss_noun_total = 0.0
        self.lr = None
        self.mb_top1_acc.reset()
        self.mb_top5_acc.reset()
        self.mb_verb_top1_acc.reset()
        self.mb_verb_top5_acc.reset()
        self.mb_noun_top1_acc.reset()
        self.mb_noun_top5_acc.reset()
        self.num_top1_cor = 0
        self.num_top5_cor = 0
        self.num_verb_top1_cor = 0
        self.num_verb_top5_cor = 0
        self.num_noun_top1_cor = 0
        self.num_noun_top5_cor = 0
        self.num_samples = 0

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_acc, top5_acc, loss, lr, mb_size):
        """
        Update the current stats.
        Args:
            top1_acc (float): top1 accuracy rate.
            top5_acc (float): top5 accuracy rate.
            loss (float): loss value.
            lr (float): learning rate.
            mb_size (int): mini batch size.
        """
        # Current minibatch stats
        self.mb_verb_top1_acc.add_value(top1_acc[0])
        self.mb_verb_top5_acc.add_value(top5_acc[0])
        self.mb_noun_top1_acc.add_value(top1_acc[1])
        self.mb_noun_top5_acc.add_value(top5_acc[1])
        self.mb_top1_acc.add_value(top1_acc[2])
        self.mb_top5_acc.add_value(top5_acc[2])
        self.loss_verb.add_value(loss[0])
        self.loss_noun.add_value(loss[1])
        self.loss.add_value(loss[2])
        self.lr = lr
        # Aggregate stats
        self.num_verb_top1_cor += top1_acc[0] * mb_size
        self.num_verb_top5_cor += top5_acc[0] * mb_size
        self.num_noun_top1_cor += top1_acc[1] * mb_size
        self.num_noun_top5_cor += top5_acc[1] * mb_size
        self.num_top1_cor += top1_acc[2] * mb_size
        self.num_top5_cor += top5_acc[2] * mb_size
        self.loss_verb_total += loss[0] * mb_size
        self.loss_noun_total += loss[1] * mb_size
        self.loss_total += loss[2] * mb_size
        self.num_samples += mb_size

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch * self.epoch_iters + cur_iter + 1)
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "train_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.epoch_iters),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "verb_top1_acc": self.mb_verb_top1_acc.get_win_median(),
            "verb_top5_acc": self.mb_verb_top5_acc.get_win_median(),
            "noun_top1_acc": self.mb_noun_top1_acc.get_win_median(),
            "noun_top5_acc": self.mb_noun_top5_acc.get_win_median(),
            "top1_acc": self.mb_top1_acc.get_win_median(),
            "top5_acc": self.mb_top5_acc.get_win_median(),
            "verb_loss": self.loss_verb.get_win_median(),
            "noun_loss": self.loss_noun.get_win_median(),
            "loss": self.loss.get_win_median(),
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        eta_sec = self.iter_timer.seconds() * (
            self.MAX_EPOCH - (cur_epoch + 1) * self.epoch_iters
        )
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        verb_top1_acc = self.num_verb_top1_cor / self.num_samples
        verb_top5_acc = self.num_verb_top5_cor / self.num_samples
        noun_top1_acc = self.num_noun_top1_cor / self.num_samples
        noun_top5_acc = self.num_noun_top5_cor / self.num_samples
        top1_acc = self.num_top1_cor / self.num_samples
        top5_acc = self.num_top5_cor / self.num_samples
        avg_loss_verb = self.loss_verb_total / self.num_samples
        avg_loss_noun = self.loss_noun_total / self.num_samples
        avg_loss = self.loss_total / self.num_samples
        stats = {
            "_type": "train_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "dt": self.iter_timer.seconds(),
            "dt_data": self.data_timer.seconds(),
            "dt_net": self.net_timer.seconds(),
            "eta": eta,
            "verb_top1_acc": verb_top1_acc,
            "verb_top5_acc": verb_top5_acc,
            "noun_top1_acc": noun_top1_acc,
            "noun_top5_acc": noun_top5_acc,
            "top1_acc": top1_acc,
            "top5_acc": top5_acc,
            "verb_loss": avg_loss_verb,
            "noun_loss": avg_loss_noun,
            "loss": avg_loss,
            "lr": self.lr,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        logging.log_json_stats(stats)


class EPICValMeter(object):
    """
    Measures validation stats.
    """

    def __init__(self, max_iter, cfg):
        """
        Args:
            max_iter (int): the max number of iteration of the current epoch.
            cfg (CfgNode): configs.
        """
        self._cfg = cfg
        self.max_iter = max_iter
        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        # Current minibatch accuracies (smoothed over a window).
        self.mb_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_verb_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_verb_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_noun_top1_acc = ScalarMeter(cfg.LOG_PERIOD)
        self.mb_noun_top5_acc = ScalarMeter(cfg.LOG_PERIOD)
        # Max accuracies (over the full val set).
        self.max_top1_acc = 0.0
        self.max_top5_acc = 0.0
        self.max_verb_top1_acc = 0.0
        self.max_verb_top5_acc = 0.0
        self.max_noun_top1_acc = 0.0
        self.max_noun_top5_acc = 0.0
        # Number of correctly classified examples.
        self.num_top1_cor = 0
        self.num_top5_cor = 0
        self.num_verb_top1_cor = 0
        self.num_verb_top5_cor = 0
        self.num_noun_top1_cor = 0
        self.num_noun_top5_cor = 0
        self.num_samples = 0
        self.all_verb_preds = []
        self.all_verb_labels = []
        self.all_noun_preds = []
        self.all_noun_labels = []
        self.output_dir = cfg.OUTPUT_DIR

    def reset(self):
        """
        Reset the Meter.
        """
        self.iter_timer.reset()
        self.mb_top1_acc.reset()
        self.mb_top5_acc.reset()
        self.mb_verb_top1_acc.reset()
        self.mb_verb_top5_acc.reset()
        self.mb_noun_top1_acc.reset()
        self.mb_noun_top5_acc.reset()
        self.num_top1_cor = 0
        self.num_top5_cor = 0
        self.num_verb_top1_cor = 0
        self.num_verb_top5_cor = 0
        self.num_noun_top1_cor = 0
        self.num_noun_top5_cor = 0
        self.num_samples = 0
        self.all_verb_preds = []
        self.all_verb_labels = []
        self.all_noun_preds = []
        self.all_noun_labels = []

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def update_stats(self, top1_acc, top5_acc, mb_size):
        """
        Update the current stats.
        Args:
            top1_acc (float): top1 accuracy rate.
            top5_acc (float): top5 accuracy rate.
            mb_size (int): mini batch size.
        """
        self.mb_verb_top1_acc.add_value(top1_acc[0])
        self.mb_verb_top5_acc.add_value(top5_acc[0])
        self.mb_noun_top1_acc.add_value(top1_acc[1])
        self.mb_noun_top5_acc.add_value(top5_acc[1])
        self.mb_top1_acc.add_value(top1_acc[2])
        self.mb_top5_acc.add_value(top5_acc[2])
        self.num_verb_top1_cor += top1_acc[0] * mb_size
        self.num_verb_top5_cor += top5_acc[0] * mb_size
        self.num_noun_top1_cor += top1_acc[1] * mb_size
        self.num_noun_top5_cor += top5_acc[1] * mb_size
        self.num_top1_cor += top1_acc[2] * mb_size
        self.num_top5_cor += top5_acc[2] * mb_size
        self.num_samples += mb_size

    def update_predictions(self, preds, labels):
        """
        Update predictions and labels.
        Args:
            preds (tensor): model output predictions.
            labels (tensor): labels.
        """
        # TODO: merge update_prediction with update_stats.
        self.all_verb_preds.append(preds[0])
        self.all_verb_labels.append(labels[0])
        self.all_noun_preds.append(preds[1])
        self.all_noun_labels.append(labels[1])

    def log_iter_stats(self, cur_epoch, cur_iter):
        """
        log the stats of the current iteration.
        Args:
            cur_epoch (int): the number of current epoch.
            cur_iter (int): the number of current iteration.
        """
        if (cur_iter + 1) % self._cfg.LOG_PERIOD != 0:
            return
        eta_sec = self.iter_timer.seconds() * (self.max_iter - cur_iter - 1)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "_type": "val_iter",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "iter": "{}/{}".format(cur_iter + 1, self.max_iter),
            "time_diff": self.iter_timer.seconds(),
            "eta": eta,
            "verb_top1_acc": self.mb_verb_top1_acc.get_win_median(),
            "verb_top5_acc": self.mb_verb_top5_acc.get_win_median(),
            "noun_top1_acc": self.mb_noun_top1_acc.get_win_median(),
            "noun_top5_acc": self.mb_noun_top5_acc.get_win_median(),
            "top1_acc": self.mb_top1_acc.get_win_median(),
            "top5_acc": self.mb_top5_acc.get_win_median(),
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
        }
        logging.log_json_stats(stats)

    def log_epoch_stats(self, cur_epoch):
        """
        Log the stats of the current epoch.
        Args:
            cur_epoch (int): the number of current epoch.
        """
        verb_top1_acc = self.num_verb_top1_cor / self.num_samples
        verb_top5_acc = self.num_verb_top5_cor / self.num_samples
        noun_top1_acc = self.num_noun_top1_cor / self.num_samples
        noun_top5_acc = self.num_noun_top5_cor / self.num_samples
        top1_acc = self.num_top1_cor / self.num_samples
        top5_acc = self.num_top5_cor / self.num_samples
        self.max_verb_top1_acc = max(self.max_verb_top1_acc, verb_top1_acc)
        self.max_verb_top5_acc = max(self.max_verb_top5_acc, verb_top5_acc)
        self.max_noun_top1_acc = max(self.max_noun_top1_acc, noun_top1_acc)
        self.max_noun_top5_acc = max(self.max_noun_top5_acc, noun_top5_acc)
        is_best_epoch = top1_acc > self.max_top1_acc
        self.max_top1_acc = max(self.max_top1_acc, top1_acc)
        self.max_top5_acc = max(self.max_top5_acc, top5_acc)
        stats = {
            "_type": "val_epoch",
            "epoch": "{}/{}".format(cur_epoch + 1, self._cfg.SOLVER.MAX_EPOCH),
            "time_diff": self.iter_timer.seconds(),
            "verb_top1_acc": verb_top1_acc,
            "verb_top5_acc": verb_top5_acc,
            "noun_top1_acc": noun_top1_acc,
            "noun_top5_acc": noun_top5_acc,
            "top1_acc": top1_acc,
            "top5_acc": top5_acc,
            "max_verb_top1_acc": self.max_verb_top1_acc,
            "max_verb_top5_acc": self.max_verb_top5_acc,
            "max_noun_top1_acc": self.max_noun_top1_acc,
            "max_noun_top5_acc": self.max_noun_top5_acc,
            "max_top1_acc": self.max_top1_acc,
            "max_top5_acc": self.max_top5_acc,
            "gpu_mem": "{:.2f}G".format(misc.gpu_mem_usage()),
            "RAM": "{:.2f}/{:.2f}G".format(*misc.cpu_mem_usage()),
        }
        logging.log_json_stats(stats)

        return is_best_epoch, {"top1_acc": top1_acc, "verb_top1_acc": verb_top1_acc, "noun_top1_acc": noun_top1_acc}


class EPICTestMeter(object):
    """
    Perform the multi-view ensemble for testing: each audio with an unique index
    will be sampled with multiple clips, and the predictions of the clips will
    be aggregated to produce the final prediction for the audio.
    The accuracy is calculated with the given ground truth labels.
    """

    def __init__(
            self,
            num_audios,
            num_clips,
            num_cls,
            overall_iters,
            ensemble_method="sum",
    ):
        """
        Construct tensors to store the predictions and labels. Expect to get
        num_clips predictions from each audio, and calculate the metrics on
        num_audios audios.
        Args:
            num_audios (int): number of audios to test.
            num_clips (int): number of clips sampled from each audio for
                aggregating the final prediction for the audio.
            num_cls (int): number of classes for each prediction.
            overall_iters (int): overall iterations for testing.
            ensemble_method (str): method to perform the ensemble, options
                include "sum", and "max".
        """

        self.iter_timer = Timer()
        self.data_timer = Timer()
        self.net_timer = Timer()
        self.num_clips = num_clips
        self.overall_iters = overall_iters
        self.ensemble_method = ensemble_method
        # Initialize tensors.
        self.verb_audio_preds = torch.zeros((num_audios, num_cls[0]))
        self.noun_audio_preds = torch.zeros((num_audios, num_cls[1]))
        self.verb_audio_preds_clips = torch.zeros((num_audios, num_clips, num_cls[0]))
        self.noun_audio_preds_clips = torch.zeros((num_audios, num_clips, num_cls[1]))
        self.verb_audio_labels = torch.zeros((num_audios)).long()
        self.noun_audio_labels = torch.zeros((num_audios)).long()
        self.metadata = np.zeros(num_audios, dtype=object)
        self.clip_count = torch.zeros((num_audios)).long()
        self.topk_accs = []
        self.stats = {}

        # Reset metric.
        self.reset()

    def reset(self):
        """
        Reset the metric.
        """
        self.clip_count.zero_()
        self.verb_audio_preds.zero_()
        self.verb_audio_preds_clips.zero_()
        self.verb_audio_labels.zero_()
        self.noun_audio_preds.zero_()
        self.noun_audio_preds_clips.zero_()
        self.noun_audio_labels.zero_()
        self.metadata.fill(0)

    def update_stats(self, preds, labels, metadata, clip_ids):
        """
        Collect the predictions from the current batch and perform on-the-flight
        summation as ensemble.
        Args:
            preds (tensor): predictions from the current batch. Dimension is
                N x C where N is the batch size and C is the channel size
                (num_cls).
            labels (tensor): the corresponding labels of the current batch.
                Dimension is N.
            clip_ids (tensor): clip indexes of the current batch, dimension is
                N.
        """
        for ind in range(preds[0].shape[0]):
            vid_id = int(clip_ids[ind]) // self.num_clips
            clip_temporal_id = int(clip_ids[ind]) % self.num_clips
            if self.verb_audio_labels[vid_id].sum() > 0:
                assert torch.equal(
                    self.verb_audio_labels[vid_id].type(torch.FloatTensor),
                    labels[0][ind].type(torch.FloatTensor),
                )
                assert torch.equal(
                    self.noun_audio_labels[vid_id].type(torch.FloatTensor),
                    labels[1][ind].type(torch.FloatTensor),
                )
            self.verb_audio_labels[vid_id] = labels[0][ind]
            self.noun_audio_labels[vid_id] = labels[1][ind]
            if self.ensemble_method == "sum":
                self.verb_audio_preds[vid_id] += preds[0][ind]
                self.noun_audio_preds[vid_id] += preds[1][ind]
            elif self.ensemble_method == "max":
                self.verb_audio_preds[vid_id] = torch.max(
                    self.verb_audio_preds[vid_id], preds[0][ind]
                )
                self.noun_audio_preds[vid_id] = torch.max(
                    self.noun_audio_preds[vid_id], preds[1][ind]
                )
            else:
                raise NotImplementedError(
                    "Ensemble Method {} is not supported".format(
                        self.ensemble_method
                    )
                )
            self.verb_audio_preds_clips[vid_id, clip_temporal_id] = preds[0][ind]
            self.noun_audio_preds_clips[vid_id, clip_temporal_id] = preds[1][ind]
            self.metadata[vid_id] = metadata['narration_id'][ind]
            self.clip_count[vid_id] += 1

    def log_iter_stats(self, cur_iter):
        """
        Log the stats.
        Args:
            cur_iter (int): the current iteration of testing.
        """
        eta_sec = self.iter_timer.seconds() * (self.overall_iters - cur_iter)
        eta = str(datetime.timedelta(seconds=int(eta_sec)))
        stats = {
            "split": "test_iter",
            "cur_iter": "{}".format(cur_iter + 1),
            "eta": eta,
            "time_diff": self.iter_timer.seconds(),
        }
        logging.log_json_stats(stats)

    def iter_tic(self):
        """
        Start to record time.
        """
        self.iter_timer.reset()
        self.data_timer.reset()

    def iter_toc(self):
        """
        Stop to record time.
        """
        self.iter_timer.pause()
        self.net_timer.pause()

    def data_toc(self):
        self.data_timer.pause()
        self.net_timer.reset()

    def finalize_metrics(self, ks=(1, 5)):
        """
        Calculate and log the final ensembled metrics.
        ks (tuple): list of top-k values for topk_accuracies. For example,
            ks = (1, 5) correspods to top-1 and top-5 accuracy.
        """
        if not all(self.clip_count == self.num_clips):
            logger.warning(
                "clip count {} ~= num clips {}".format(
                    ", ".join(
                        [
                            "{}: {}".format(i, k)
                            for i, k in enumerate(self.clip_count.tolist())
                        ]
                    ),
                    self.num_clips,
                )
            )

        verb_topks = metrics.topk_accuracies(self.verb_audio_preds, self.verb_audio_labels, ks)
        noun_topks = metrics.topk_accuracies(self.noun_audio_preds, self.noun_audio_labels, ks)

        assert len({len(ks), len(verb_topks)}) == 1
        assert len({len(ks), len(noun_topks)}) == 1
        self.stats = {"split": "test_final"}
        for k, verb_topk in zip(ks, verb_topks):
            self.stats["verb_top{}_acc".format(k)] = "{:.{prec}f}".format(verb_topk, prec=2)
        for k, noun_topk in zip(ks, noun_topks):
            self.stats["noun_top{}_acc".format(k)] = "{:.{prec}f}".format(noun_topk, prec=2)
        logging.log_json_stats(self.stats)
        return (self.verb_audio_preds.numpy().copy(), self.noun_audio_preds.numpy().copy()), \
               (self.verb_audio_preds_clips.numpy().copy(), self.noun_audio_preds_clips.numpy().copy()), \
               (self.verb_audio_labels.numpy().copy(), self.noun_audio_labels.numpy().copy()), \
               self.metadata.copy()


def get_map(preds, labels):
    """
    Compute mAP for multi-label case.
    Args:
        preds (numpy tensor): num_examples x num_classes.
        labels (numpy tensor): num_examples x num_classes.
    Returns:
        mean_ap (int): final mAP score.
    """

    logger.info("Getting mAP for {} examples".format(preds.shape[0]))

    preds = preds[:, ~(np.all(labels == 0, axis=0))]
    labels = labels[:, ~(np.all(labels == 0, axis=0))]
    aps = [0]
    try:
        aps = average_precision_score(labels, preds, average=None)
    except ValueError:
        print(
            "Average precision requires a sufficient number of samples \
            in a batch which are missing in this sample."
        )

    mean_ap = np.mean(aps)
    return mean_ap

