# Where the measurement evidence lives

Every performance claim in this repo was measured rather than estimated, and the
raw material behind those measurements — browser screenshots, session
recordings, per-second HUD logs, benchmark reports, `nvidia-smi` samples — is
deliberately **not** committed here.

Two reasons. It is tens of megabytes of media that nobody consuming this repo
needs in order to deploy a cartridge, and it is a record of how *we* checked our
own work on specific hardware on specific dates, which is a different thing from
documentation. Some of it also carries account IDs and endpoint hostnames.

It is kept alongside the repo instead, one directory per cartridge:

```
world-model-evidence/
├── lingbot-v2-fps/          screenshots + driven-session video + HUD timeseries
├── waypoint-1-5/            per-scene captures, contact sheets, perf reports
├── vjepa2/                  UI captures
└── matrix-game-3-*          8-GPU lockstep session, UI demo, audits
```

Ask the maintainers if you need it — for a performance regression, a review, or
to reproduce a figure.

The *tool* that produces the waypoint numbers is not evidence and is still
here: `scripts/benchmark.py`, driven by `./deploy.sh bench`.

## Reproducing the numbers yourself

You do not need our captures to check our claims; you need a GPU and the
cartridge:

```bash
./deploy.sh bench <model> 60      # 60 s benchmark → docs/evidence/<model>/
```

That writes a `performance-report.md` and `.json` with the same fields we
report. `docs/evidence/` is gitignored, so a benchmark run will not
accidentally commit its own output.

Per-cartridge numbers, and the reasoning behind them, stay in the cartridge
READMEs under `inference/models/<model>/` and in the topic docs beside this
file — those are documentation and they are versioned here.
