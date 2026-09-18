# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Matrix Game 3.0 — real-time interactive world model. NOT YET WIRED."""

import torch
from lib.runner import Runner


class MatrixGameRunner(Runner):
    def setup(self, model_dir, device, rank, world_size):
        self.device = device

    def generate(self, **params):
        raise NotImplementedError("Matrix Game 3.0 inference not yet wired")


def create_runner():
    return MatrixGameRunner()
