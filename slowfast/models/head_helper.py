#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""ResNe(X)t Head helper."""

import torch
import torch.nn as nn


class ResNetBasicHead(nn.Module):
    """
    ResNe(X)t 2D head.
    This layer performs a fully-connected projection during training, when the
    input size is 1x1. It performs a convolutional projection during testing
    when the input size is larger than 1x1. If the inputs are from multiple
    different pathways, the inputs will be concatenated after pooling.
    """

    def __init__(
        self,
        dim_in,
        num_classes,
        pool_size,
        dropout_rate=0.0,
        act_func="softmax",
    ):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        ResNetBasicHead takes p pathways as input where p in [1, infty].

        Args:
            dim_in (list): the list of channel dimensions of the p inputs to the
                ResNetHead.
            num_classes (int): the channel dimensions of the p outputs to the
                ResNetHead.
            pool_size (list): the list of kernel sizes of p frequency temporal
                poolings, temporal pool kernel size, frequency pool kernel size in order.
            dropout_rate (float): dropout rate. If equal to 0.0, perform no
                dropout.
            act_func (string): activation function to use. 'softmax': applies
                softmax on the output. 'sigmoid': applies sigmoid on the output.
        """
        super(ResNetBasicHead, self).__init__()
        assert (
            len({len(pool_size), len(dim_in)}) == 1
        ), "pathway dimensions are not consistent."
        self.num_pathways = len(pool_size)

        for pathway in range(self.num_pathways):
            avg_pool = nn.AvgPool2d(pool_size[pathway], stride=1)
            self.add_module("pathway{}_avgpool".format(pathway), avg_pool)

        if dropout_rate > 0.0:
            self.dropout = nn.Dropout(dropout_rate)
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        if isinstance(num_classes, (list, tuple)):
            self.projection_verb = nn.Linear(sum(dim_in), num_classes[0], bias=True)
            self.projection_noun = nn.Linear(sum(dim_in), num_classes[1], bias=True)
        else:
            self.projection = nn.Linear(sum(dim_in), num_classes, bias=True)
        self.num_classes = num_classes
        # Softmax for evaluation and testing.
        if act_func == "softmax":
            self.act = nn.Softmax(dim=3)
        elif act_func == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            raise NotImplementedError(
                "{} is not supported as an activation"
                "function.".format(act_func)
            )

    def forward(self, inputs):
        assert (
            len(inputs) == self.num_pathways
        ), "Input tensor does not contain {} pathway".format(self.num_pathways)
        pool_out = []
        for pathway in range(self.num_pathways):
            m = getattr(self, "pathway{}_avgpool".format(pathway))
            pool_out.append(m(inputs[pathway]))
        x = torch.cat(pool_out, 1)
        # (N, C, T, H) -> (N, T, H, C).
        x = x.permute((0, 2, 3, 1))
        # Perform dropout.
        if hasattr(self, "dropout"):
            x = self.dropout(x)
        if isinstance(self.num_classes, (list, tuple)):
            x_v = self.projection_verb(x)
            x_n = self.projection_noun(x)

            # Performs fully convlutional inference.
            if not self.training:
                x_v = self.act(x_v)
                x_v = x_v.mean([1, 2])

            x_v = x_v.view(x_v.shape[0], -1)

            # Performs fully convlutional inference.
            if not self.training:
                x_n = self.act(x_n)
                x_n = x_n.mean([1, 2])

            x_n = x_n.view(x_n.shape[0], -1)
            return (x_v, x_n)
        else:
            x = self.projection(x)

            # Performs fully convlutional inference.
            if not self.training:
                x = self.act(x)
                x = x.mean([1, 2])

            x = x.view(x.shape[0], -1)
            return x
