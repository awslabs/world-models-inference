// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// Bundled Lyra-2 examples — hardcoded in the frontend so the examples row always
// renders regardless of server availability, mirroring lingbot-examples.ts. IDs
// match the directories under `inference/models/lyra-2/examples/<id>/` on the
// server; the server serves the thumbnails at `<apiUrl>/examples/<id>/image`.
//
// These are NVIDIA's own zoomgs sample scenes (Lyra-2/assets/samples/<id>.png +
// <id>.txt). Each prompt is the canonical caption shipped with that image: a
// static-world framing (a steady forward camera push through a scene that is
// "frozen in time"), which is what the zoomgs path expects.

import { BundledExample } from './lingbot-examples';

export const LYRA2_EXAMPLES: BundledExample[] = [
  {
    id: '00',
    label: 'Harbor galleons',
    has_poses: false,
    prompt:
      'A slow, steady camera push forward along the weathered wooden dock toward the two massive galleons. The scene is a frozen tableau: the turquoise water is glass-like and motionless, the canvas sails are rigid, and the distant clouds are fixed in the golden sky. Every element, from the foreground wooden barrels and coiled ropes to the intricate coastal architecture, remains perfectly still. The warm, late-afternoon sunlight and soft shadows are permanent. As the camera advances, more of the harbor\'s stone buildings and the ships\' wooden hulls are revealed, maintaining identical textures and colors. The entire world is locked in a silent, breathless moment with zero object or environmental movement.',
  },
  {
    id: '02',
    label: 'Oriental pagoda',
    has_poses: false,
    prompt:
      'A cinematic camera push forward along the central wooden bridge toward the grand, dark-tiled pagoda. The vast cityscape of intricate, multi-tiered oriental temples remains perfectly frozen in time. Every ornate roof tile, golden dragon sculpture, and weathered wooden plank is completely stationary. The soft, hazy daylight and long shadows are fixed and unchanging. This is a world in stasis; no wind, no flickering lights, and no movement of any kind from the architecture or environment. As the camera advances, the surrounding dense cluster of traditional buildings maintains consistent textures and materials, revealing more of the silent, motionless urban expanse under a pale, static sky.',
  },
  {
    id: '03',
    label: 'Steampunk fortress',
    has_poses: false,
    prompt:
      'A cinematic drone shot executes a slow, steady forward push toward the intricate steampunk fortress. The massive structure features weathered metal domes, terracotta roofs, and tall industrial chimneys. Every element is frozen in time: the thick plume of dark smoke is a solid, unmoving sculpture against the blue sky; the fluffy white cumulus clouds are stationary; and the distant rolling hills and village remain perfectly still. The camera glide reveals more of the grassy hilltop and mechanical undercarriage while maintaining the original lighting and muted color palette. All objects, shadows, and atmospheric effects are entirely motionless, creating a silent, paused-world aesthetic.',
  },
  {
    id: '04',
    label: 'Stone amphitheater',
    has_poses: false,
    prompt:
      'A cinematic drone shot glides forward through the center of a sun-drenched, ancient stone amphitheater. The weathered limestone blocks, crumbling arches, and dense green ivy are frozen in time. Every leaf, shadow, and dust particle is perfectly still. The warm, golden sunlight remains constant across the textured ruins and cobblestone path. As the camera advances, more of the circular arena and tiered seating are revealed, maintaining the same overgrown, historical aesthetic. All environmental elements, including the distant foliage and sky, are entirely motionless, creating a silent, statuesque world where only the camera’s perspective shifts.',
  },
];
