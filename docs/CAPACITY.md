# GPU capacity wizard

GPU capacity is scarce and per-Availability-Zone. A Launch Stack or `./deploy.sh`
run whose instance type has no capacity still completes; its Auto Scaling group just
keeps retrying (`InsufficientInstanceCapacity`). The wizard answers "where can this
cartridge run right now?" before you deploy.

```bash
python scripts/capacity.py                                         # opens http://127.0.0.1:8787/
python scripts/capacity.py --launch-stack <bucket-basename>/<prefix>   # also offer pre-filled Launch Stack links
```

It uses your normal AWS credentials (`--profile` to pick one) and needs boto3 (a
project dependency) and Node, to build the page the first time.

## What it checks

For the cartridge you pick, the candidate instance types are those with exactly
the GPU count it uses and enough memory per GPU (`gpu_count` and `gpu_memory_gb` in
its `endpoint.yaml`), plus its default instance. For each selected Region:

| Check | How |
|---|---|
| Offered in the AZ | `DescribeInstanceTypeOfferings` |
| vCPU quota left | Service Quotas (On-Demand G and VT / P instances) minus what's running |
| Free right now (optional) | Creates a 1-instance, targeted Capacity Reservation and cancels it at once. Billed for seconds at most; it would expire on its own after an hour. |
| Capacity Blocks | `DescribeCapacityBlockOfferings`: 24-hour blocks over the next 7 days, P-series only |
| Price | The Price List API, On-Demand Linux |

## What you get

- **Best options**, ranked: free in the Launch Stack's two AZs with the template
  published there (one click opens the pre-filled quick-create form), then free in
  those AZs (`./deploy.sh`), then free elsewhere.
- **Hold it**: the `create-capacity-reservation` command that keeps an instance's
  worth of capacity until you deploy (billed as if running until used or cancelled).
- **Capacity Blocks**: earliest offering per AZ with its upfront fee, and the
  `purchase-capacity-block` command. Pass the reservation ID as `CapacityBlockId`.
- **Watch**: when nothing is free, re-probes the offered types every 2 minutes and
  notifies you when one frees up.

The wizard never buys anything. The page is served on 127.0.0.1 only, and the API
answers only requests carrying the per-run token written into the page it served.

The Launch Stack VPC uses the Region's first two AZs (alphabetical), marked
`STACK` in the grid; capacity in any other AZ needs `./deploy.sh`.
