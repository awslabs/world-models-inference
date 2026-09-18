# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared pytest configuration."""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def pytest_addoption(parser):
    parser.addoption("--endpoint", default=os.environ.get("LINGBOT_ENDPOINT", ""))
