# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""scripts/capacity.py: the GPU capacity wizard's offline logic."""

import importlib.util
import urllib.parse
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("capacity", REPO / "scripts" / "capacity.py")
capacity = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(capacity)

CARTRIDGES = sorted(d.name for d in (REPO / "inference" / "models").iterdir() if (d / "endpoint.yaml").is_file())


@pytest.mark.parametrize("model", CARTRIDGES)
def test_every_cartridge_declares_its_gpu_need(model):
    """The wizard covers every cartridge only if each one says what it runs on."""
    gpus = capacity.manifest_scalar(model, "gpu_count")
    mem = capacity.manifest_scalar(model, "gpu_memory_gb")
    assert gpus and gpus.isdigit() and int(gpus) >= 1, f"{model}: gpu_count must be a positive integer"
    assert mem is not None and mem.isdigit(), f"{model}: gpu_memory_gb must be an integer (0 = any)"


@pytest.mark.parametrize("model", [m for m in CARTRIDGES if not m.startswith("_")])
def test_every_cartridge_has_candidates_including_its_default(model):
    m = next(x for x in capacity.models() if x["id"] == model)
    assert m["candidates"], f"{model}: no instance type meets gpu_count/gpu_memory_gb"
    assert m["defaultInstance"] in m["candidates"]


def test_candidates_match_gpu_count_and_memory():
    assert capacity.candidates(8, 80, "p5.48xlarge") == ["p4de.24xlarge", "p5.48xlarge", "p5e.48xlarge", "p5en.48xlarge"]
    one = capacity.candidates(1, 44, None)
    assert "g6e.2xlarge" in one and "g6.2xlarge" not in one  # L4 has 24 GB
    assert all(capacity.INSTANCES[t][0] == 1 for t in one)


def test_launch_stack_types_are_known_to_the_wizard():
    assert set(capacity.LAUNCH_TYPES) <= set(capacity.INSTANCES)


def test_launch_stack_azs_are_the_first_two_alphabetically():
    assert capacity.launch_stack_azs(["us-east-2c", "us-east-2a", "us-east-2b"]) == ["us-east-2a", "us-east-2b"]


def test_launch_url_prefills_the_quick_create_form():
    url = capacity.launch_url("my-bucket/wmi/v1", "eu-north-1", "waypoint-1-5", "g6.2xlarge", "p5.4xlarge", "cr-0123456789abcdef0")
    assert url.startswith("https://eu-north-1.console.aws.amazon.com/cloudformation/home?region=eu-north-1#/stacks/create/review?")
    q = urllib.parse.parse_qs(url.split("?", 2)[2])
    assert q["templateURL"] == ["https://my-bucket-eu-north-1.s3.eu-north-1.amazonaws.com/wmi/v1/world-model-launch.template.json"]
    assert q["param_Model"] == ["waypoint-1-5"]
    assert q["param_InstanceType"] == ["g6.2xlarge"]
    assert q["param_CapacityBlockId"] == ["cr-0123456789abcdef0"]
    default = capacity.launch_url("my-bucket/wmi/v1", "eu-north-1", "waypoint-1-5", "p5.4xlarge", "p5.4xlarge")
    assert "param_InstanceType=model-default" in default and "CapacityBlockId" not in default


def test_deploy_command():
    cmd = capacity.deploy_command("eu-north-1", "waypoint-1-5", "g6.2xlarge", "p5.4xlarge", True)
    assert cmd == "aws configure set region eu-north-1\nINSTANCE_TYPE=g6.2xlarge ACCEPT_LICENSE=1 ./deploy.sh waypoint-1-5"
    assert capacity.deploy_command("us-east-1", "vjepa2", "g5.2xlarge", "g5.2xlarge", False).endswith("\n./deploy.sh vjepa2")


@pytest.mark.parametrize("code,message,expected", [
    ("InsufficientInstanceCapacity", "Insufficient capacity.", "none"),
    ("InstanceLimitExceeded", "You have requested more instances", "quota"),
    ("Unsupported", "The instance type is not supported in this AZ", "unsupported"),
    ("UnauthorizedOperation", "not authorized", "error"),
])
def test_probe_errors_are_classified(code, message, expected):
    assert capacity.classify_probe_error(code, message) == expected


class _ClientError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.response = {"Error": {"Code": code, "Message": code}}


class _StubEC2:
    """Just enough of an EC2 client for the probe and the sweep; never touches AWS."""

    def __init__(self, cancel_failures=0, reservations=()):
        self.cancel_failures = cancel_failures
        self.reservations = list(reservations)
        self.cancelled = []
        self.describe_filters = None

    def create_capacity_reservation(self, **kw):
        assert {"Key": "purpose", "Value": "world-model-capacity-probe"} in kw["TagSpecifications"][0]["Tags"]
        return {"CapacityReservation": {"CapacityReservationId": "cr-0123456789abcdef0"}}

    def cancel_capacity_reservation(self, CapacityReservationId):
        if self.cancel_failures:
            self.cancel_failures -= 1
            raise _ClientError("RequestLimitExceeded")
        self.cancelled.append(CapacityReservationId)
        return {"Return": True}

    def describe_capacity_reservations(self, Filters, NextToken=None):
        self.describe_filters = Filters
        return {"CapacityReservations": [{"CapacityReservationId": r} for r in self.reservations]}


class _StubSession:
    def __init__(self, ec2):
        self.ec2 = ec2

    def client(self, service, region_name=None):
        assert service == "ec2"
        return self.ec2


@pytest.fixture
def sleeps(monkeypatch):
    calls = []
    monkeypatch.setattr(capacity.time, "sleep", calls.append)
    return calls


def test_probe_retries_cancel_with_backoff(sleeps):
    ec2 = _StubEC2(cancel_failures=2)
    status, detail, leak = capacity.Scanner(_StubSession(ec2), None).probe("us-east-1", "us-east-1a", "p5.48xlarge")
    assert (status, detail, leak) == ("available", "", "")
    assert ec2.cancelled == ["cr-0123456789abcdef0"]
    assert len(sleeps) == 2 and 0.5 <= sleeps[0] <= 0.75 and 1 <= sleeps[1] <= 1.5


def test_probe_that_cannot_cancel_reports_the_leaked_reservation(sleeps):
    ec2 = _StubEC2(cancel_failures=99)
    status, detail, leak = capacity.Scanner(_StubSession(ec2), None).probe("us-east-1", "us-east-1a", "p5.48xlarge")
    assert status == "error" and leak == "cr-0123456789abcdef0" and leak in detail
    assert ec2.cancelled == [] and len(sleeps) == capacity.CANCEL_ATTEMPTS - 1


def test_sweep_cancels_active_tagged_probe_reservations(sleeps):
    ec2 = _StubEC2(reservations=["cr-aaaaaaaaaaaaaaaaa", "cr-bbbbbbbbbbbbbbbbb"])
    scanner = capacity.Scanner(_StubSession(ec2), None)
    assert sorted(scanner.sweep(["us-east-1"])) == ["cr-aaaaaaaaaaaaaaaaa", "cr-bbbbbbbbbbbbbbbbb"]
    assert ec2.cancelled == ["cr-aaaaaaaaaaaaaaaaa", "cr-bbbbbbbbbbbbbbbbb"]
    assert {"Name": "tag:purpose", "Values": ["world-model-capacity-probe"]} in ec2.describe_filters
    assert {"Name": "state", "Values": ["active"]} in ec2.describe_filters
