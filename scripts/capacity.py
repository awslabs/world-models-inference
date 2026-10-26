#!/usr/bin/env python3
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU capacity wizard: find where a cartridge can run before you deploy it.

Starts a local web page (127.0.0.1 only) that uses your AWS credentials to scan
Regions, Availability Zones and GPU instance types for the cartridge you pick:

  - is the type offered in each AZ, and does your vCPU quota leave room for it
  - optionally, is there capacity right now: a 1-instance On-Demand Capacity
    Reservation is created and cancelled straight away (billed for seconds at
    most; the reservation also expires on its own after an hour)
  - Capacity Block offerings for the P-series types (earliest start, upfront fee)
  - the on-demand price

It ends with a pre-filled Launch Stack link (when the template is published in
that Region) or the matching ./deploy.sh command. It never buys anything: the
Capacity Block purchase and the capacity-hold commands are shown for you to run.

Each cartridge's need comes from its endpoint.yaml (gpu_count, gpu_memory_gb).

Usage:
  python scripts/capacity.py
  python scripts/capacity.py --launch-stack <bucket-basename>/<prefix>   # publish.py's values
  python scripts/capacity.py --port 8787 --no-browser
"""

import argparse
import atexit
import datetime as dt
import html
import json
import queue
import random
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "inference" / "models"
FRONTEND_DIR = REPO_ROOT / "frontend"
DIST_DIR = FRONTEND_DIR / "dist-capacity"
LAUNCH_TYPES = json.loads((REPO_ROOT / "cdk" / "lib" / "launch" / "instance-types.json").read_text())
TEMPLATE_NAME = "world-model-launch.template.json"  # scripts/publish.py

# type: (gpus, gpu, GB per GPU, vCPUs, quota family, Capacity Block eligible)
INSTANCES = {
    "g5.2xlarge": (1, "A10G", 24, 8, "G", False),
    "g5.4xlarge": (1, "A10G", 24, 16, "G", False),
    "g5.12xlarge": (4, "A10G", 24, 48, "G", False),
    "g5.48xlarge": (8, "A10G", 24, 192, "G", False),
    "g6.2xlarge": (1, "L4", 24, 8, "G", False),
    "g6.4xlarge": (1, "L4", 24, 16, "G", False),
    "g6.12xlarge": (4, "L4", 24, 48, "G", False),
    "g6.48xlarge": (8, "L4", 24, 192, "G", False),
    "g6e.2xlarge": (1, "L40S", 48, 8, "G", False),
    "g6e.4xlarge": (1, "L40S", 48, 16, "G", False),
    "g6e.12xlarge": (4, "L40S", 48, 48, "G", False),
    "g6e.48xlarge": (8, "L40S", 48, 192, "G", False),
    "p4d.24xlarge": (8, "A100", 40, 96, "P", True),
    "p4de.24xlarge": (8, "A100", 80, 96, "P", True),
    "p5.4xlarge": (1, "H100", 80, 16, "P", True),
    "p5.48xlarge": (8, "H100", 80, 192, "P", True),
    "p5e.48xlarge": (8, "H200", 141, 192, "P", True),
    "p5en.48xlarge": (8, "H200", 141, 192, "P", True),
}
# Running On-Demand G and VT / P instances, in vCPUs.
QUOTA_CODES = {"G": "L-DB2E81BA", "P": "L-417A185B"}
DEFAULT_REGIONS = ["us-east-1", "us-east-2", "us-west-2", "eu-west-1", "eu-central-1", "eu-north-1"]
# Every probe reservation carries this tag, so a sweep can find any the probe failed to cancel.
PROBE_TAG = {"Key": "purpose", "Value": "world-model-capacity-probe"}
CANCEL_ATTEMPTS = 5

_REGION_RE = re.compile(r"^[a-z]{2}(-[a-z]+)+-[0-9]+$")
_MODEL_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_LAUNCH_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,40}[a-z0-9]/[A-Za-z0-9_-]+(/[A-Za-z0-9_-][A-Za-z0-9._-]*)*$")


# --------------------------------------------------------------------------- #
# Cartridges
# --------------------------------------------------------------------------- #

def manifest_scalar(model: str, key: str):
    """A top-level `key: value` from endpoint.yaml (same rule as build-image.py)."""
    text = (MODELS_DIR / model / "endpoint.yaml").read_text()
    m = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, re.MULTILINE)
    if not m:
        return None
    return m.group(1).split("#")[0].strip().strip("\"'") or None


def ec2_instance(model: str):
    text = (MODELS_DIR / model / "endpoint.yaml").read_text()
    m = re.search(r"^ec2:\s*\n(?:[ \t]+.*\n)*?[ \t]+instance:\s*([\w.-]+)", text, re.MULTILINE)
    return m.group(1) if m else None


def catalogue_names() -> dict:
    """id -> (name, tagline, type) from the UI catalogue, for display only."""
    src = FRONTEND_DIR / "src" / "data" / "cartridges.ts"
    out = {}
    if src.exists():
        for m in re.finditer(r"id: '([^']+)',\s*name: '([^']+)',\s*type: '([^']+)',\s*tagline: '([^']*)'", src.read_text()):
            out[m.group(1)] = (m.group(2), m.group(4), m.group(3))
    return out


def models() -> list:
    names = catalogue_names()
    out = []
    for d in sorted(MODELS_DIR.iterdir()):
        if d.name.startswith("_") or not (d / "endpoint.yaml").is_file():
            continue
        m = d.name
        gpus, mem = manifest_scalar(m, "gpu_count"), manifest_scalar(m, "gpu_memory_gb")
        if gpus is None or mem is None:
            continue
        name, tagline, kind = next((v for k, v in names.items() if k == m or k.startswith(m + "-")), (m, "", ""))
        out.append({
            "id": m, "name": name, "tagline": tagline, "kind": kind,
            "gpuCount": int(gpus), "gpuMemoryGb": int(mem),
            "defaultInstance": ec2_instance(m),
            "restricted": manifest_scalar(m, "license_restricted") == "true",
            "license": manifest_scalar(m, "license"),
            "candidates": candidates(int(gpus), int(mem), ec2_instance(m)),
        })
    return out


def candidates(gpus: int, mem: int, default) -> list:
    """Types with exactly the GPU count the cartridge uses and enough memory per GPU.

    More GPUs than the cartridge uses would sit idle, so they're left out, except
    the cartridge's own default instance, which is always offered.
    """
    keep = [t for t, (n, _, gb, *_rest) in INSTANCES.items() if n == gpus and gb >= mem]
    if default and default not in keep:
        keep.insert(0, default)
    return keep


def instance_meta(t: str) -> dict:
    n, gpu, gb, vcpu, fam, cb = INSTANCES.get(t, (None, "?", None, None, "G" if t.startswith("g") else "P", t.startswith("p")))
    return {"type": t, "gpus": n, "gpu": gpu, "gpuMemoryGb": gb, "vcpus": vcpu, "family": fam,
            "capacityBlocks": cb, "launchStack": t in LAUNCH_TYPES}


# --------------------------------------------------------------------------- #
# Links and commands
# --------------------------------------------------------------------------- #

def launch_stack_azs(azs: list) -> list:
    """The two AZs the Launch Stack VPC uses: Fn::GetAZs is alphabetical, indices 0 and 1."""
    return sorted(azs)[:2]


def template_url(launch: str, region: str) -> str:
    basename, prefix = launch.split("/", 1)
    return f"https://{basename}-{region}.s3.{region}.amazonaws.com/{prefix}/{TEMPLATE_NAME}"


def launch_url(launch: str, region: str, model: str, instance_type: str, default, block_id: str = "") -> str:
    params = {"templateURL": template_url(launch, region), "stackName": "WorldModel", "param_Model": model,
              "param_InstanceType": "model-default" if instance_type == default else instance_type}
    if block_id:
        params["param_CapacityBlockId"] = block_id
    return (f"https://{region}.console.aws.amazon.com/cloudformation/home?region={region}"
            f"#/stacks/create/review?{urllib.parse.urlencode(params)}")


def deploy_command(region: str, model: str, instance_type: str, default, restricted: bool, block_id: str = "") -> str:
    env = []
    if instance_type != default:
        env.append(f"INSTANCE_TYPE={instance_type}")
    if block_id:
        env.append(f"CAPACITY_BLOCK_ID={block_id}")
    if restricted:
        env.append("ACCEPT_LICENSE=1")
    return f"aws configure set region {region}\n{' '.join(env + ['./deploy.sh', model])}"


def block_purchase_command(region: str, offering_id: str) -> str:
    return f"aws ec2 purchase-capacity-block --region {region} --instance-platform Linux/UNIX --capacity-block-offering-id {offering_id}"


def hold_command(region: str, az: str, instance_type: str) -> str:
    return (f"aws ec2 create-capacity-reservation --region {region} --availability-zone {az} "
            f"--instance-type {instance_type} --instance-platform Linux/UNIX --instance-count 1 "
            f"--instance-match-criteria open --end-date-type unlimited")


# --------------------------------------------------------------------------- #
# Scan
# --------------------------------------------------------------------------- #

def classify_probe_error(code: str, message: str) -> str:
    if code == "InsufficientInstanceCapacity":
        return "none"
    if "Limit" in code or "quota" in message.lower():
        return "quota"
    if code in ("Unsupported", "InvalidParameterValue") or "not supported" in message.lower():
        return "unsupported"
    return "error"


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def cancel_reservation(ec2, cr_id: str, attempts: int = CANCEL_ATTEMPTS) -> bool:
    """Cancel a capacity reservation, backing off between tries. True once it is gone.

    Up to 6 Regions x 10 workers probe at once, so EC2 throttling is the usual
    failure; 0.5, 1, 2, 4 s (plus jitter) between tries rides it out.
    """
    for i in range(attempts):
        try:
            ec2.cancel_capacity_reservation(CapacityReservationId=cr_id)
            return True
        except Exception as e:  # botocore ClientError
            if getattr(e, "response", {}).get("Error", {}).get("Code") == "InvalidCapacityReservationId.NotFound":
                return True  # already gone (a sweep got there first)
            if i == attempts - 1:
                log(f"cancel {cr_id} failed after {attempts} tries: {e}")
                return False
            delay = 0.5 * 2 ** i
            time.sleep(delay + random.uniform(0, delay / 2))  # nosec B311 - backoff jitter, not crypto
    return False


def sweep_region(ec2, region: str) -> list:
    """Cancel every active probe reservation in one Region. Best-effort; returns the IDs cancelled."""
    cancelled = []
    try:
        kwargs = {"Filters": [{"Name": f"tag:{PROBE_TAG['Key']}", "Values": [PROBE_TAG["Value"]]},
                              {"Name": "state", "Values": ["active"]}]}
        while True:
            page = ec2.describe_capacity_reservations(**kwargs)
            for r in page.get("CapacityReservations", []):
                cr_id = r["CapacityReservationId"]
                if cancel_reservation(ec2, cr_id):
                    log(f"{region}: cancelled leftover probe reservation {cr_id}")
                    cancelled.append(cr_id)
                else:
                    log(f"{region}: could NOT cancel probe reservation {cr_id}; cancel it by hand: "
                        f"aws ec2 cancel-capacity-reservation --region {region} --capacity-reservation-id {cr_id}")
            if not page.get("NextToken"):
                break
            kwargs["NextToken"] = page["NextToken"]
    except Exception as e:
        log(f"{region}: probe-reservation sweep failed: {e}")
    return cancelled


class Scanner:
    def __init__(self, session, launch: str | None):
        self.session = session
        self.launch = launch
        self._prices: dict = {}
        self._lock = threading.Lock()
        self.probed: set = set()  # Regions a probe ran in, swept again at exit

    def client(self, service: str, region: str):
        return self.session.client(service, region_name=region)

    def price(self, region: str, t: str):
        key = (region, t)
        with self._lock:
            if key in self._prices:
                return self._prices[key]
        value = None
        try:
            location = self.client("ssm", region).get_parameter(
                Name=f"/aws/service/global-infrastructure/regions/{region}/longName")["Parameter"]["Value"]
            # SSM says "Europe (Stockholm)"; the Price List API still says "EU (Stockholm)".
            location = re.sub(r"^Europe \(", "EU (", location)
            filters = [{"Type": "TERM_MATCH", "Field": f, "Value": v} for f, v in (
                ("instanceType", t), ("location", location), ("operatingSystem", "Linux"), ("tenancy", "Shared"),
                ("preInstalledSw", "NA"), ("capacitystatus", "Used"))]
            for item in self.client("pricing", "us-east-1").get_products(ServiceCode="AmazonEC2", Filters=filters, MaxResults=5)["PriceList"]:
                for term in json.loads(item)["terms"].get("OnDemand", {}).values():
                    for dim in term["priceDimensions"].values():
                        usd = float(dim["pricePerUnit"].get("USD", 0))
                        if usd > 0:
                            value = usd
        except Exception:  # pricing is decoration; a missing price must not stop a scan
            value = None
        with self._lock:
            self._prices[key] = value
        return value

    def template_published(self, region: str) -> bool:
        if not self.launch:
            return False
        basename, prefix = self.launch.split("/", 1)
        try:
            self.client("s3", region).head_object(Bucket=f"{basename}-{region}", Key=f"{prefix}/{TEMPLATE_NAME}")
            return True
        except Exception:
            # A public template may still be readable anonymously.
            try:
                import urllib.request
                urllib.request.urlopen(urllib.request.Request(template_url(self.launch, region), method="HEAD"), timeout=5)  # nosec B310 - fixed https S3 URL
                return True
            except Exception:
                return False

    def quotas(self, region: str) -> dict:
        sq, ec2 = self.client("service-quotas", region), self.client("ec2", region)
        out = {}
        for fam, code in QUOTA_CODES.items():
            try:
                out[fam] = {"limit": sq.get_service_quota(ServiceCode="ec2", QuotaCode=code)["Quota"]["Value"], "used": 0}
            except Exception:
                out[fam] = {"limit": None, "used": 0}
        try:
            pages = ec2.get_paginator("describe_instances").paginate(Filters=[
                {"Name": "instance-state-name", "Values": ["pending", "running"]},
                {"Name": "instance-type", "Values": ["g*", "p*", "vt*"]}])
            for page in pages:
                for r in page["Reservations"]:
                    for i in r["Instances"]:
                        fam = "P" if i["InstanceType"].startswith("p") else "G"
                        out[fam]["used"] += i["CpuOptions"]["CoreCount"] * i["CpuOptions"]["ThreadsPerCore"]
        except Exception:
            pass
        return out

    def sweep(self, regions) -> list:
        """Cancel leftover probe reservations across `regions`, in parallel. Best-effort."""
        regions = sorted(set(regions))
        if not regions:
            return []
        with ThreadPoolExecutor(max_workers=min(len(regions), 16)) as pool:
            found = pool.map(lambda r: sweep_region(self.client("ec2", r), r), regions)
            return [cr_id for ids in found for cr_id in ids]

    def probe(self, region: str, az: str, t: str) -> tuple:
        """(status, detail, leaked reservation ID or "")."""
        ec2 = self.client("ec2", region)
        end = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=1)).replace(microsecond=0)
        with self._lock:
            self.probed.add(region)
        try:
            cr = ec2.create_capacity_reservation(
                InstanceType=t, InstancePlatform="Linux/UNIX", AvailabilityZone=az, InstanceCount=1,
                EndDateType="limited", EndDate=end, InstanceMatchCriteria="targeted",
                TagSpecifications=[{"ResourceType": "capacity-reservation", "Tags": [PROBE_TAG]}])
        except Exception as e:  # botocore ClientError
            err = getattr(e, "response", {}).get("Error", {})
            return classify_probe_error(err.get("Code", ""), err.get("Message", str(e))), err.get("Message", str(e))[:200], ""
        cr_id = cr["CapacityReservation"]["CapacityReservationId"]
        # Targeted, so nothing running can consume it; cancel at once. A reservation
        # that survives is billed until its EndDate, so never report it as plain "available".
        cancelled = False
        try:
            cancelled = cancel_reservation(ec2, cr_id)
        finally:
            if not cancelled:
                log(f"{region}/{az}: probe reservation {cr_id} ({t}) is still active and billing; cancel it: "
                    f"aws ec2 cancel-capacity-reservation --region {region} --capacity-reservation-id {cr_id}")
        if not cancelled:
            return "error", f"capacity available, but probe reservation {cr_id} could not be cancelled and is billing", cr_id
        return "available", "", ""

    def blocks(self, region: str, t: str) -> list:
        now = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
        try:
            offers = self.client("ec2", region).describe_capacity_block_offerings(
                InstanceType=t, InstanceCount=1, CapacityDurationHours=24,
                StartDateRange=now, EndDateRange=now + dt.timedelta(days=7))["CapacityBlockOfferings"]
        except Exception:
            return []
        earliest = {}
        for o in sorted(offers, key=lambda o: o["StartDate"]):
            earliest.setdefault(o["AvailabilityZone"], o)
        return [{"az": o["AvailabilityZone"], "start": o["StartDate"].isoformat(), "end": o["EndDate"].isoformat(),
                 "fee": float(o["UpfrontFee"]), "currency": o["CurrencyCode"], "offeringId": o["CapacityBlockOfferingId"]}
                for o in earliest.values()]

    def scan_region(self, region: str, model: dict, probe: bool, want_blocks: bool, emit) -> None:
        ec2 = self.client("ec2", region)
        try:
            azs = sorted(z["ZoneName"] for z in ec2.describe_availability_zones(
                Filters=[{"Name": "zone-type", "Values": ["availability-zone"]},
                         {"Name": "opt-in-status", "Values": ["opt-in-not-required"]}])["AvailabilityZones"])
        except Exception as e:
            emit({"type": "region-error", "region": region, "error": str(e)[:300]})
            return
        types = model["candidates"]
        offered = set()
        for page in ec2.get_paginator("describe_instance_type_offerings").paginate(
                LocationType="availability-zone", Filters=[{"Name": "instance-type", "Values": types}]):
            offered |= {(o["Location"], o["InstanceType"]) for o in page["InstanceTypeOfferings"]}
        stack_azs = launch_stack_azs(azs)
        published = self.template_published(region)
        emit({"type": "region", "region": region, "azs": azs, "stackAzs": stack_azs,
              "quota": self.quotas(region), "templatePublished": published,
              "prices": {t: self.price(region, t) for t in types}})
        cells = [(az, t) for az in azs for t in types]
        with ThreadPoolExecutor(max_workers=10) as pool:
            def one(az, t):
                leaked = ""
                if (az, t) not in offered:
                    status, detail = "unsupported", "not offered in this AZ"
                elif probe:
                    status, detail, leaked = self.probe(region, az, t)
                else:
                    status, detail = "offered", ""
                cell = {"type": "cell", "region": region, "az": az, "instanceType": t, "status": status, "detail": detail,
                        "inStackAzs": az in stack_azs}
                if leaked:
                    cell["leak"] = leaked
                    cell["cancel"] = f"aws ec2 cancel-capacity-reservation --region {region} --capacity-reservation-id {leaked}"
                if status == "available":
                    cell["deploy"] = deploy_command(region, model["id"], t, model["defaultInstance"], model["restricted"])
                    cell["hold"] = hold_command(region, az, t)
                    if published and t in LAUNCH_TYPES + [model["defaultInstance"]]:
                        cell["launch"] = launch_url(self.launch, region, model["id"], t, model["defaultInstance"])
                emit(cell)
            list(pool.map(lambda c: one(*c), cells))
        if want_blocks:
            for t in types:
                if not INSTANCES.get(t, (0, 0, 0, 0, 0, False))[5]:
                    continue
                for b in self.blocks(region, t):
                    b.update({"type": "block", "region": region, "instanceType": t, "inStackAzs": b["az"] in stack_azs,
                              "purchase": block_purchase_command(region, b["offeringId"])})
                    emit(b)
        emit({"type": "region-done", "region": region})

    def scan(self, model: dict, regions: list, probe: bool, want_blocks: bool, emit) -> None:
        emit({"type": "start", "model": model["id"], "regions": regions,
              "candidates": [instance_meta(t) for t in model["candidates"]]})
        with ThreadPoolExecutor(max_workers=len(regions)) as pool:
            for f in [pool.submit(self.scan_region, r, model, probe, want_blocks, emit) for r in regions]:
                try:
                    f.result()
                except Exception as e:
                    emit({"type": "region-error", "region": "?", "error": str(e)[:300]})
        emit({"type": "done"})


# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #

def make_handler(scanner: Scanner, token: str, port: int, identity: dict, regions: list):
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    origins = {f"http://{h}" for h in hosts}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):  # quiet
            pass

        def _guard(self, api: bool) -> bool:
            # Host stops DNS rebinding; Origin stops other sites; the token stops anything
            # that did not load the page this server rendered.
            if self.headers.get("Host") not in hosts:
                self.send_error(403)
                return False
            origin = self.headers.get("Origin")
            if origin is not None and origin not in origins:
                self.send_error(403)
                return False
            if api and not secrets.compare_digest(self.headers.get("X-Wizard-Token", ""), token):
                self.send_error(401)
                return False
            return True

        def _json(self, obj, status=200):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urllib.parse.urlparse(self.path).path
            if path.startswith("/api/"):
                if not self._guard(True):
                    return
                if path == "/api/models":
                    return self._json({"models": models(), "instances": {t: instance_meta(t) for t in INSTANCES},
                                       "regions": regions, "defaultRegions": [r for r in DEFAULT_REGIONS if r in regions],
                                       "identity": identity, "launchStack": bool(scanner.launch)})
                return self.send_error(404)
            if not self._guard(False):
                return
            if path in ("/", "/index.html"):
                page = (DIST_DIR / "capacity.html").read_text().replace("__WIZARD_TOKEN__", html.escape(token))
                body = page.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Security-Policy",
                                 "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; font-src 'self' data:")
                self.end_headers()
                self.wfile.write(body)
                return
            target = (DIST_DIR / path.lstrip("/")).resolve()
            if not target.is_file() or DIST_DIR.resolve() not in target.parents:
                return self.send_error(404)
            ctype = {".js": "text/javascript", ".css": "text/css", ".woff2": "font/woff2", ".woff": "font/woff",
                     ".svg": "image/svg+xml", ".png": "image/png"}.get(target.suffix, "application/octet-stream")
            body = target.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if urllib.parse.urlparse(self.path).path != "/api/scan":
                return self.send_error(404)
            if not self._guard(True):
                return
            try:
                req = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 65536)))
                model = next(m for m in models() if m["id"] == req["model"] and _MODEL_RE.fullmatch(req["model"]))
                wanted = [r for r in req.get("regions", []) if _REGION_RE.fullmatch(r) and r in regions][:16]
                if req.get("instanceTypes"):  # watch mode re-checks a shortlist
                    model = {**model, "candidates": [t for t in req["instanceTypes"] if t in model["candidates"]]}
            except Exception:
                return self._json({"error": "bad request"}, 400)
            if not wanted:
                return self._json({"error": "no regions"}, 400)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            q: queue.Queue = queue.Queue()
            worker = threading.Thread(target=scanner.scan, daemon=True,
                                      args=(model, wanted, bool(req.get("probe")), bool(req.get("blocks", True)), q.put))
            worker.start()
            while True:
                event = q.get()
                try:
                    self.wfile.write((json.dumps(event) + "\n").encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                if event["type"] == "done":
                    return

    return Handler


def ensure_built() -> None:
    sources = list((FRONTEND_DIR / "src" / "capacity").rglob("*")) + [FRONTEND_DIR / "capacity.html"]
    page = DIST_DIR / "capacity.html"
    if page.exists() and page.stat().st_mtime >= max(p.stat().st_mtime for p in sources if p.is_file()):
        return
    npm = shutil.which("npm")
    if not npm:
        sys.exit("The wizard page is not built and npm is not installed: run `npm ci && npm run build:capacity` in frontend/.")
    if not (FRONTEND_DIR / "node_modules").exists():
        subprocess.run([npm, "ci"], cwd=FRONTEND_DIR, check=True)
    print("Building the wizard page...", flush=True)
    subprocess.run([npm, "run", "build:capacity"], cwd=FRONTEND_DIR, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument("--profile", help="AWS profile (default: the usual credential chain)")
    ap.add_argument("--launch-stack", help="<bucket-basename>/<prefix> the template is published under (scripts/publish.py)")
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    if args.launch_stack and not _LAUNCH_RE.fullmatch(args.launch_stack):
        sys.exit(f"--launch-stack must look like <bucket-basename>/<prefix>, got {args.launch_stack!r}")
    try:
        import boto3
    except ImportError:
        sys.exit("boto3 is required: `uv sync` (it is a project dependency) or `pip install boto3`.")
    session = boto3.Session(profile_name=args.profile)
    try:
        ident = session.client("sts").get_caller_identity()
    except Exception as e:
        sys.exit(f"No usable AWS credentials: {e}")
    home = session.region_name or "us-east-1"
    regions = sorted(r["RegionName"] for r in session.client("ec2", region_name=home).describe_regions()["Regions"])
    ensure_built()
    token = secrets.token_urlsafe(24)
    identity = {"account": "…" + ident["Account"][-4:], "arn": re.sub(r"\d{12}", "…", ident["Arn"])}
    scanner = Scanner(session, args.launch_stack)
    # A previous run may have died between creating and cancelling a probe reservation.
    print("Checking for leftover probe reservations...", flush=True)
    scanner.sweep(regions)
    swept = threading.Event()

    def exit_sweep():
        if not swept.is_set() and scanner.probed:
            swept.set()
            print("Cancelling any probe reservations still active...", flush=True)
            scanner.sweep(set(scanner.probed))

    atexit.register(exit_sweep)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(scanner, token, args.port, identity, regions))
    url = f"http://127.0.0.1:{args.port}/"
    print(f"GPU capacity wizard: {url}  (Ctrl+C to stop)", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        exit_sweep()


if __name__ == "__main__":
    main()
