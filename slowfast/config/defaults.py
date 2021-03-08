#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Configs."""
from fvcore.common.config import CfgNode

from . import custom_config

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CfgNode()

# ---------------------------------------------------------------------------- #
# Batch norm options
# ---------------------------------------------------------------------------- #
_C.BN = CfgNode()

_C.BN.FREEZE = False

# Precise BN stats.
_C.BN.USE_PRECISE_STATS = False

# Number of samples use to compute precise bn.
_C.BN.NUM_BATCHES_PRECISE = 200

# Weight decay value that applies on BN.
_C.BN.WEIGHT_DECAY = 0.0

# Norm type, options include `batchnorm`, `sub_batchnorm`, `sync_batchnorm`
_C.BN.NORM_TYPE = "batchnorm"

# Parameter for SubBatchNorm, where it splits the batch dimension into
# NUM_SPLITS splits, and run BN on each of them separately independently.
_C.BN.NUM_SPLITS = 1

# Parameter for NaiveSyncBatchNorm2d, where the stats across `NUM_SYNC_DEVICES`
# devices will be synchronized.
_C.BN.NUM_SYNC_DEVICES = 1


# ---------------------------------------------------------------------------- #
# Training options.
# ---------------------------------------------------------------------------- #
_C.TRAIN = CfgNode()

# If True Train the model, else skip training.
_C.TRAIN.ENABLE = True

# Dataset.
_C.TRAIN.DATASET = "vggsound"

# Total mini-batch size.
_C.TRAIN.BATCH_SIZE = 64

# Evaluate model on test data every eval period epochs.
_C.TRAIN.EVAL_PERIOD = 10

# Save model checkpoint every checkpoint period epochs.
_C.TRAIN.CHECKPOINT_PERIOD = 10

# Resume training from the latest checkpoint in the output directory.
_C.TRAIN.AUTO_RESUME = True

# Path to the checkpoint to load the initial weight.
_C.TRAIN.CHECKPOINT_FILE_PATH = ""

# If True, reset epochs when loading checkpoint.
_C.TRAIN.CHECKPOINT_EPOCH_RESET = False

# If set, clear all layer names according to the pattern provided.
_C.TRAIN.CHECKPOINT_CLEAR_NAME_PATTERN = ()  # ("backbone.",)

# ---------------------------------------------------------------------------- #
# Testing options
# ---------------------------------------------------------------------------- #
_C.TEST = CfgNode()

# If True test the model, else skip the testing.
_C.TEST.ENABLE = True

# Dataset for testing.
_C.TEST.DATASET = "vggsound"

# Total mini-batch size
_C.TEST.BATCH_SIZE = 8

# Path to the checkpoint to load the initial weight.
_C.TEST.CHECKPOINT_FILE_PATH = ""

# Number of clips to sample from an audio uniformly for aggregating the
# prediction results.
_C.TEST.NUM_ENSEMBLE_VIEWS = 10

# Path to saving prediction results file.
_C.TEST.SAVE_RESULTS_PATH = ""
# -----------------------------------------------------------------------------
# ResNet options
# -----------------------------------------------------------------------------
_C.RESNET = CfgNode()

# Transformation function.
_C.RESNET.TRANS_FUNC = "bottleneck_transform"

# Number of groups. 1 for ResNet, and larger than 1 for ResNeXt).
_C.RESNET.NUM_GROUPS = 1

# Width of each group (64 -> ResNet; 4 -> ResNeXt).
_C.RESNET.WIDTH_PER_GROUP = 64

# Apply relu in a inplace manner.
_C.RESNET.INPLACE_RELU = True

# Apply stride to 1x1 conv.
_C.RESNET.STRIDE_1X1 = False

#  If true, initialize the gamma of the final BN of each block to zero.
_C.RESNET.ZERO_INIT_FINAL_BN = False

# Number of weight layers.
_C.RESNET.DEPTH = 50

# If the current block has more than NUM_BLOCK_TEMP_KERNEL blocks, use temporal
# kernel of 1 for the rest of the blocks.
_C.RESNET.NUM_BLOCK_TEMP_KERNEL = [[3], [4], [6], [3]]

# Size of stride on different res stages.
_C.RESNET.FREQUENCY_STRIDES = [[1], [2], [2], [2]]

# Size of dilation on different res stages.
_C.RESNET.FREQUENCY_DILATIONS = [[1], [1], [1], [1]]


# -----------------------------------------------------------------------------
# Model options
# -----------------------------------------------------------------------------
_C.MODEL = CfgNode()

# Model architecture.
_C.MODEL.ARCH = "slowfast"

# Model name
_C.MODEL.MODEL_NAME = "SlowFast"

# The number of classes to predict for the model.
_C.MODEL.NUM_CLASSES = [400, ]

# Loss function.
_C.MODEL.LOSS_FUNC = "cross_entropy"

# Model architectures that has one single pathway.
_C.MODEL.SINGLE_PATHWAY_ARCH = ["slow", "fast"]

# Model architectures that has multiple pathways.
_C.MODEL.MULTI_PATHWAY_ARCH = ["slowfast"]

# Dropout rate before final projection in the backbone.
_C.MODEL.DROPOUT_RATE = 0.5

# Randomly drop rate for Res-blocks, linearly increase from res2 to res5
_C.MODEL.DROPCONNECT_RATE = 0.0

# The std to initialize the fc layer(s).
_C.MODEL.FC_INIT_STD = 0.01

# Activation layer for the output head.
_C.MODEL.HEAD_ACT = "softmax"


# -----------------------------------------------------------------------------
# SlowFast options
# -----------------------------------------------------------------------------
_C.SLOWFAST = CfgNode()

# Corresponds to the inverse of the channel reduction ratio, $\beta$ between
# the Slow and Fast pathways.
_C.SLOWFAST.BETA_INV = 8

# Corresponds to the frame rate reduction ratio, $\alpha$ between the Slow and
# Fast pathways.
_C.SLOWFAST.ALPHA = 8

# Ratio of channel dimensions between the Slow and Fast pathways.
_C.SLOWFAST.FUSION_CONV_CHANNEL_RATIO = 2

# Kernel dimension used for fusing information from Fast pathway to Slow
# pathway.
_C.SLOWFAST.FUSION_KERNEL_SZ = 5


# -----------------------------------------------------------------------------
# Data options
# -----------------------------------------------------------------------------
_C.DATA = CfgNode()

# List of input spectrogram channel dimensions.
_C.DATA.INPUT_CHANNEL_NUM = [1, 1]

# If True, calculate the map as metric.
_C.DATA.MULTI_LABEL = False

# Method to perform the ensemble, options include "sum" and "max".
_C.DATA.ENSEMBLE_METHOD = "sum"


# -----------------------------------------------------------------------------
# Audio data options
# -----------------------------------------------------------------------------
_C.AUDIO_DATA = CfgNode()

# Sampling rate of audio (in kHz)
_C.AUDIO_DATA.SAMPLING_RATE = 24000

# Duration of audio clip from which to extract the spectrogram
_C.AUDIO_DATA.CLIP_SECS = 1.279

_C.AUDIO_DATA.WINDOW_LENGTH = 10

_C.AUDIO_DATA.HOP_LENGTH = 5

# Number of timesteps of the input spectrogram
_C.AUDIO_DATA.NUM_FRAMES = 256

# Number of frequencies of the input spectrogram
_C.AUDIO_DATA.NUM_FREQUENCIES = 128


# ---------------------------------------------------------------------------- #
# Optimizer options
# ---------------------------------------------------------------------------- #
_C.SOLVER = CfgNode()

# Base learning rate.
_C.SOLVER.BASE_LR = 0.1

# Learning rate policy (see utils/lr_policy.py for options and examples).
_C.SOLVER.LR_POLICY = "cosine"

# Final learning rates for 'cosine' policy.
_C.SOLVER.COSINE_END_LR = 0.0

# Exponential decay factor.
_C.SOLVER.GAMMA = 0.1

# Step size for 'exp' and 'cos' policies (in epochs).
_C.SOLVER.STEP_SIZE = 1

# Steps for 'steps_' policies (in epochs).
_C.SOLVER.STEPS = []

# Learning rates for 'steps_' policies.
_C.SOLVER.LRS = []

# Maximal number of epochs.
_C.SOLVER.MAX_EPOCH = 300

# Momentum.
_C.SOLVER.MOMENTUM = 0.9

# Momentum dampening.
_C.SOLVER.DAMPENING = 0.0

# Nesterov momentum.
_C.SOLVER.NESTEROV = True

# L2 regularization.
_C.SOLVER.WEIGHT_DECAY = 1e-4

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARMUP_FACTOR.
_C.SOLVER.WARMUP_FACTOR = 0.1

# Gradually warm up the SOLVER.BASE_LR over this number of epochs.
_C.SOLVER.WARMUP_EPOCHS = 0.0

# The start learning rate of the warm up.
_C.SOLVER.WARMUP_START_LR = 0.01

# Optimization method.
_C.SOLVER.OPTIMIZING_METHOD = "sgd"

# Base learning rate is linearly scaled with NUM_SHARDS.
_C.SOLVER.BASE_LR_SCALE_NUM_SHARDS = False

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #

# Number of GPUs to use (applies to both training and testing).
_C.NUM_GPUS = 1

# Number of machine to use for the job.
_C.NUM_SHARDS = 1

# The index of the current machine.
_C.SHARD_ID = 0

# Output basedir.
_C.OUTPUT_DIR = "./tmp"

# Note that non-determinism may still be present due to non-deterministic
# operator implementations in GPU operator libraries.
_C.RNG_SEED = 1

# Log period in iters.
_C.LOG_PERIOD = 10

# If True, log the model info.
_C.LOG_MODEL_INFO = True

# Distributed backend.
_C.DIST_BACKEND = "nccl"


# ---------------------------------------------------------------------------- #
# Common train/test data loader options
# ---------------------------------------------------------------------------- #
_C.DATA_LOADER = CfgNode()

# Number of data loader workers per training process.
_C.DATA_LOADER.NUM_WORKERS = 8

# Load data to pinned host memory.
_C.DATA_LOADER.PIN_MEMORY = True

# Enable multi thread decoding.
_C.DATA_LOADER.ENABLE_MULTI_THREAD_DECODE = False


# -----------------------------------------------------------------------------
# EPIC-KITCHENS Dataset options
# -----------------------------------------------------------------------------
_C.EPICKITCHENS = CfgNode()

_C.EPICKITCHENS.AUDIO_DATA_FILE = ""

_C.EPICKITCHENS.ANNOTATIONS_DIR = ""

_C.EPICKITCHENS.TRAIN_LIST = "EPIC_100_train.pkl"

_C.EPICKITCHENS.VAL_LIST = "EPIC_100_validation.pkl"

_C.EPICKITCHENS.TEST_LIST = "EPIC_100_validation.pkl"

_C.EPICKITCHENS.TRAIN_PLUS_VAL = False

_C.EPICKITCHENS.TEST_SPLIT = "validation"


# -----------------------------------------------------------------------------
# VGG-Sound Dataset options
# -----------------------------------------------------------------------------
_C.VGGSOUND = CfgNode()

_C.VGGSOUND.AUDIO_DATA_DIR = ""

_C.VGGSOUND.ANNOTATIONS_DIR = ""

_C.VGGSOUND.TRAIN_LIST = "train.pkl"

_C.VGGSOUND.VAL_LIST = "test.pkl"

_C.VGGSOUND.TEST_LIST = "test.pkl"


# -----------------------------------------------------------------------------
# Tensorboard Visualization Options
# -----------------------------------------------------------------------------
_C.TENSORBOARD = CfgNode()

# Log to summary writer, this will automatically.
# log loss, lr and metrics during train/eval.
_C.TENSORBOARD.ENABLE = False
# Provide path to prediction results for visualization.
# This is a pickle file of [prediction_tensor, label_tensor]
_C.TENSORBOARD.PREDICTIONS_PATH = ""
# Path to directory for tensorboard logs.
# Default to to cfg.OUTPUT_DIR/runs-{cfg.TRAIN.DATASET}.
_C.TENSORBOARD.LOG_DIR = ""
# Path to a json file providing class_name - id mapping
# in the format {"class_name1": id1, "class_name2": id2, ...}.
# This file must be provided to enable plotting confusion matrix
# by a subset or parent categories.
_C.TENSORBOARD.CLASS_NAMES_PATH = ""

# Path to a json file for categories -> classes mapping
# in the format {"parent_class": ["child_class1", "child_class2",...], ...}.
_C.TENSORBOARD.CATEGORIES_PATH = ""

# Config for confusion matrices visualization.
_C.TENSORBOARD.CONFUSION_MATRIX = CfgNode()
# Visualize confusion matrix.
_C.TENSORBOARD.CONFUSION_MATRIX.ENABLE = False
# Figure size of the confusion matrices plotted.
_C.TENSORBOARD.CONFUSION_MATRIX.FIGSIZE = [8, 8]
# Path to a subset of categories to visualize.
# File contains class names separated by newline characters.
_C.TENSORBOARD.CONFUSION_MATRIX.SUBSET_PATH = ""

# Config for histogram visualization.
_C.TENSORBOARD.HISTOGRAM = CfgNode()
# Visualize histograms.
_C.TENSORBOARD.HISTOGRAM.ENABLE = False
# Path to a subset of classes to plot histograms.
# Class names must be separated by newline characters.
_C.TENSORBOARD.HISTOGRAM.SUBSET_PATH = ""
# Visualize top-k most predicted classes on histograms for each
# chosen true label.
_C.TENSORBOARD.HISTOGRAM.TOPK = 10
# Figure size of the histograms plotted.
_C.TENSORBOARD.HISTOGRAM.FIGSIZE = [8, 8]


# -----------------------------------------------------------------------------
# WANDB Visualization Options
# -----------------------------------------------------------------------------
_C.WANDB = CfgNode()
_C.WANDB.ENABLE = False
_C.WANDB.RUN_ID = ""


# Add custom config with default values.
custom_config.add_custom_config(_C)


def _assert_and_infer_cfg(cfg):
    # BN assertions.
    if cfg.BN.USE_PRECISE_STATS:
        assert cfg.BN.NUM_BATCHES_PRECISE >= 0
    # TRAIN assertions.
    assert cfg.TRAIN.BATCH_SIZE % cfg.NUM_GPUS == 0

    # TEST assertions.
    assert cfg.TEST.BATCH_SIZE % cfg.NUM_GPUS == 0

    # RESNET assertions.
    assert cfg.RESNET.NUM_GROUPS > 0
    assert cfg.RESNET.WIDTH_PER_GROUP > 0
    assert cfg.RESNET.WIDTH_PER_GROUP % cfg.RESNET.NUM_GROUPS == 0

    # Execute LR scaling by num_shards.
    if cfg.SOLVER.BASE_LR_SCALE_NUM_SHARDS:
        cfg.SOLVER.BASE_LR *= cfg.NUM_SHARDS

    # General assertions.
    assert cfg.SHARD_ID < cfg.NUM_SHARDS
    return cfg


def get_cfg():
    """
    Get a copy of the default config.
    """
    return _assert_and_infer_cfg(_C.clone())
