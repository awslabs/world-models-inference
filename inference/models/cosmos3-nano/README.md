# Cosmos3-Nano

NVIDIA's 16B-parameter omnimodal world model. Generates 720p video with synchronised audio, supports action-conditioned world simulation, and runs on 1-8 GPUs.

- **Upstream:** https://github.com/NVIDIA/Cosmos
- **Weights:** `nvidia/Cosmos3-Nano` on HuggingFace (~32 GB, open access)
- **License:** OpenMDW-1.1 (permissive, commercial use allowed, no output restrictions)
- **Inference time:** ~3-4 min (single L40S) / ~54s (8×H100)

## Capabilities

| Mode | Input | Output |
|------|-------|--------|
| text-to-video | prompt | 720p MP4, up to 300 frames, with audio |
| image-to-video | image + prompt | 720p MP4 |
| forward-dynamics | image + action trajectory | predicted future video |

## Deploy

```bash
./deploy --model cosmos3-nano --instance g6e.12xlarge
```

## Instance Options

| Instance | GPUs | Generation Time (189 frames, 720p) | Cost |
|----------|------|-----------------------------------|----- |
| g6e.12xlarge | 4×L40S | ~3-4 min | $10/hr |
| p5.48xlarge | 8×H100 | ~54s | $98/hr |
