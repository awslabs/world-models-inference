// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// Catalogue of world-model cartridges. Mirrors `inference/models/<id>/endpoint.yaml`
// — there is one entry here for each endpoint directory in this repo, and no more.
// `status: 'ready'` means: we've built the server code, wired the deploy path, and
// verified an end-to-end invoke. Everything else is scaffolded but unwired.

export type CartridgeType = 'real-time' | 'video-generation' | 'representation' | '3dgs';
export type CartridgeStatus = 'ready' | 'warming' | 'cold';

export interface Cartridge {
  id: string;                  // matches inference/models/<id>/
  name: string;                // human-readable
  type: CartridgeType;         // filter category
  tagline: string;             // one-liner shown on the card
  instance: string;            // `ec2.instance` from endpoint.yaml
  gpu: string;                 // e.g. 8×H100
  model: string;               // e.g. "WAN 2.2 ~14B"
  fps: number;                 // 0 → async; >0 → real-time target fps
  price: number;               // approximate $/sec (on-demand)
  status: CartridgeStatus;     // ready / cold / warming
  deploy: Array<'ec2' | 'sagemaker'>;
  hue: number;                 // 0–360; seeds the card's gradient preview
  upstream?: string;           // link to origin repo
  notes?: string;              // shown on the "not wired" detail view
  license?: string;            // weights license, e.g. 'Apache-2.0'
  licenseUrl?: string;         // link to the license / model card
  restricted?: boolean;        // true → weights license restricts deployment; UI warns, deploy.sh gates
}

export const CARTRIDGES: Cartridge[] = [
  {
    id: 'lingbot-fast',
    name: 'LingBot Fast',
    type: 'video-generation',
    tagline: 'Camera-controlled video generation — 8-GPU persistent torchrun pipeline.',
    instance: 'p5.48xlarge',
    gpu: '8× H100 80GB',
    model: 'WAN 2.2 A14B',
    fps: 0,
    price: 0.40,
    status: 'ready',
    deploy: ['ec2', 'sagemaker'],
    hue: 205,
    upstream: 'https://github.com/robbyant/lingbot-world',
    license: 'Apache-2.0',
  },
  {
    id: 'cosmos3-nano',
    name: 'Cosmos 3 Nano',
    type: 'video-generation',
    tagline: 'NVIDIA omnimodal world model — text/image-to-video, forward dynamics, policy.',
    instance: 'g6e.12xlarge',
    gpu: '4× L40S',
    model: 'Cosmos 3 Nano 16B',
    fps: 0,
    price: 0.12,
    status: 'ready',
    deploy: ['ec2', 'sagemaker'],
    hue: 120,
    upstream: 'https://github.com/NVIDIA/Cosmos',
    license: 'Apache-2.0',
  },
  {
    id: 'matrix-game-3',
    name: 'Matrix Game 3',
    type: 'real-time',
    tagline: 'Real-time interactive world model with WASD + mouse control.',
    instance: 'p5.48xlarge',
    gpu: '8× H100 80GB',
    model: '~10B params',
    fps: 8,
    price: 0.40,
    status: 'cold',
    deploy: ['ec2', 'sagemaker'],
    hue: 340,
    upstream: 'https://github.com/SkyworkAI/Matrix-Game-3.0',
    notes: 'Endpoint.yaml + real-time handler scaffold exist. Needs wiring of upstream inference code into the real-time handler base.',
  },
  {
    id: 'vjepa2-ac',
    name: 'V-JEPA 2-AC',
    type: 'representation',
    tagline: 'Action-conditioned world model for robotics — Meta FAIR JEPA family.',
    instance: 'g6e.2xlarge',
    gpu: '1× L40S',
    model: 'ViT-g/384',
    fps: 30,
    price: 0.06,
    status: 'cold',
    deploy: ['ec2', 'sagemaker'],
    hue: 220,
    upstream: 'https://github.com/facebookresearch/vjepa2',
    notes: 'Endpoint scaffold present. Needs the VJEPA2 AC inference wrapper hooked into the real-time handler.',
  },
  {
    id: 'lyra-2',
    name: 'Nvidia Lyra 2.0 — Step 1 (Video)',
    type: 'video-generation',
    tagline: 'Image → 3D-consistent camera fly-through video. Step 1 of Lyra-2; the Step-2 3D Gaussian-splat reconstruction is not exposed by this endpoint.',
    instance: 'p5.4xlarge',
    gpu: '1× H100 80GB',
    model: 'Nvidia Lyra 2.0',
    fps: 0,
    price: 0.05,
    status: 'ready',
    deploy: ['ec2'],
    hue: 275,
    upstream: 'https://github.com/nv-tlabs/lyra',
    license: 'NVIDIA Internal Scientific Research',
    licenseUrl: 'https://huggingface.co/nvidia/Lyra-2.0',
    restricted: true,
    notes: 'In-process single-GPU async job wrapping upstream Step-1 video generation (Step-2 .ply reconstruction not exposed). ⚠️ Weights are under the NVIDIA Internal Scientific Research license — internal R&D only, not for production or public deployment. You are responsible for compliant use in your own account.',
  },
  {
    id: 'echo-async',
    name: 'Echo (smoke test)',
    type: 'video-generation',
    tagline: 'Tiny synthetic-video cartridge for verifying a deployment end to end — no model weights, cheap GPU, seconds per job.',
    instance: 'g5.2xlarge',
    gpu: '1× A10G 24GB',
    model: 'DummyUNet (synthetic)',
    fps: 0,
    price: 0.0004,
    status: 'ready',
    deploy: ['ec2', 'sagemaker'],
    hue: 150,
    upstream: '',
    license: 'Apache-2.0',
    notes: 'Not a world model — it runs a tiny conv net over noise and returns a real MP4. Use it to confirm the deploy path, auth, and UI wiring work before spending on a large GPU.',
  },
];

export const CARTRIDGE_TYPES: { value: CartridgeType | 'all'; label: string }[] = [
  { value: 'all',              label: 'All' },
  { value: 'real-time',        label: 'Real-time' },
  { value: 'video-generation', label: 'Video Gen' },
  { value: 'representation',   label: 'Representation' },
  { value: '3dgs',             label: '3D GS' },
];

export const TYPE_LABEL: Record<CartridgeType, string> = {
  'real-time':        'Real-time',
  'video-generation': 'Video Gen',
  'representation':   'Representation',
  '3dgs':             '3D GS',
};
