// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// Bundled LingBot-World examples — hardcoded in the frontend so the examples
// row always renders regardless of server availability. IDs match the
// directories under `inference/models/lingbot-fast/examples/<id>/` on the
// server; the server serves the thumbnails at `<apiUrl>/examples/<id>/image`.
//
// Prompts 00–02 and 05 come from each example's prompt.txt. Prompts 03 and 04
// are bundled empty on disk, so we use the canonical upstream run_fast.sh
// prompts for those images (serene lakeside scene / Great Wall of China).

export interface BundledExample {
  id: string;
  /** Short label shown on the card. */
  label: string;
  /** Full prompt loaded into the textarea on click. */
  prompt: string;
  /** Is a poses.npy + intrinsics.npy pair bundled alongside the image? */
  has_poses: boolean;
}

export const BUNDLED_EXAMPLES: BundledExample[] = [
  {
    id: '00',
    label: 'Fantasy jungle',
    has_poses: true,
    prompt:
      "The video presents a soaring journey through a fantasy jungle. The wind whips past the rider's blue hands gripping the reins, causing the leather straps to vibrate. The ancient gothic castle approaches steadily, its stone details becoming clearer against the backdrop of floating islands and distant waterfalls.",
  },
  {
    id: '01',
    label: 'Stonehenge',
    has_poses: true,
    prompt:
      'A slow panoramic sweep around Stonehenge on a misty, overcast day, capturing the ancient standing stones in serene stillness, with soft ambient wind and distant bird calls enhancing the timeless atmosphere.',
  },
  {
    id: '02',
    label: 'Urban cinematic',
    has_poses: true,
    prompt:
      'A cinematic, first-person wandering experience through a hyper-realistic urban environment rendered in a video game engine. Sun-drenched alley framed by graffiti-laden industrial walls and overhead power lines, camera pans right and tilts upward to reveal a sprawling cityscape dominated by towering skyscrapers, warm late-afternoon light casting long shadows and dramatic lens flares.',
  },
  {
    id: '03',
    label: 'Serene lake',
    has_poses: true,
    prompt:
      'A serene lakeside scene with a lone tree standing in calm water, surrounded by distant snow-capped mountains under a bright blue sky with drifting white clouds — gentle ripples reflect the tree and sky, creating a tranquil, meditative atmosphere.',
  },
  {
    id: '04',
    label: 'Great Wall',
    has_poses: true,
    prompt:
      'A sweeping cinematic journey along the Great Wall of China, winding through golden autumn hills under a brilliant blue sky — stone pathways stretch into the distance, watchtowers stand sentinel, and vibrant foliage blankets the mountainsides as the camera glides smoothly forward, capturing the grandeur and timeless majesty of this ancient wonder.',
  },
];
