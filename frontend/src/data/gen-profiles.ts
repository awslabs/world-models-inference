// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// Per-cartridge generation profiles for the async batch UI (LingbotGenerator).
//
// Different async cartridges honour different `/generate` form fields — the UI
// must only surface controls the target runner actually reads, otherwise a knob
// silently does nothing. This maps a cartridge id → which controls to show, their
// defaults, bundled examples, and a realistic progress hint. Cartridges without
// an entry fall back to DEFAULT_PROFILE (the original lingbot-fast control set).

import { BUNDLED_EXAMPLES, BundledExample } from './lingbot-examples';
import { LYRA2_EXAMPLES } from './lyra2-examples';

export interface GenProfile {
  /** Bundled example scenes shown in the picker (empty → upload-only). */
  examples: BundledExample[];
  /** Prompt pre-filled when no example is selected. */
  defaultPrompt: string;
  /** Initial values for the advanced controls. */
  defaults: {
    frameNum: string;
    size: string;
    samplingSteps: string;
    guideScale: string;
    seed: string;
  };
  /** Which controls to render — hide the ones the runner ignores. */
  show: {
    examples: boolean;
    camera: boolean;
    frames: boolean;
    size: boolean;
    guidance: boolean;
    /** DMD "fast mode" toggle (lyra-2 only). */
    dmd: boolean;
  };
  /** Hint under the Frames field. */
  framesHint: string;
  /** Elapsed-time hint shown while generating. */
  progressHint: string;
}

// lingbot-fast: the original behaviour — all controls, bundled poses examples.
export const DEFAULT_PROFILE: GenProfile = {
  examples: BUNDLED_EXAMPLES,
  defaultPrompt: BUNDLED_EXAMPLES[0].prompt,
  defaults: { frameNum: '81', size: '480*832', samplingSteps: '20', guideScale: '5.0', seed: '-1' },
  show: { examples: true, camera: true, frames: true, size: true, guidance: true, dmd: false },
  framesHint: '81 ≈ 5s',
  progressHint: '(typical: 40–60s on 8× H100)',
};

export const GEN_PROFILES: Record<string, GenProfile> = {
  'lyra-2': {
    // NVIDIA's own zoomgs sample scenes (image + canonical static-world caption),
    // bundled on the server under examples/<id>/. A user can still upload their
    // own real photo (the depth model needs real scene structure; synthetic
    // inputs degenerate).
    examples: LYRA2_EXAMPLES,
    // Fallback when no example is selected (upload path). Static-world framing:
    // Lyra reconstructs a frozen 3D scene the camera moves through, so the prompt
    // should describe a motionless world + a steady forward push.
    defaultPrompt:
      'A slow, steady camera push forward through the scene. Every element is ' +
      'frozen in time and perfectly still — no object or environmental motion. ' +
      'As the camera advances, more of the space is revealed, maintaining ' +
      'identical textures, colours, and lighting. Only the camera moves.',
    // Only sampling_steps + seed + use_dmd are honoured by the lyra-2 runner;
    // frames/size/guidance are fixed by the runner (322 frames, 480×832) so we
    // hide them rather than show dead controls.
    defaults: { frameNum: '322', size: '480*832', samplingSteps: '50', guideScale: '5.0', seed: '1' },
    show: { examples: true, camera: false, frames: false, size: false, guidance: false, dmd: true },
    framesHint: '',
    progressHint: '(~33 min at 50 steps on 1× H100 · Fast mode ≈ 3 min)',
  },

  'cosmos3-nano': {
    // Cosmos 3 Nano generates from a text prompt (image optional). It ships no
    // bundled examples on the server, so we show none — otherwise the UI would
    // try to load example thumbnails that 404. Camera control is lingbot-only.
    examples: [],
    defaultPrompt: 'A cinematic aerial shot flying low over a misty mountain range at sunrise, volumetric light.',
    // The runner honours prompt + num_frames + guidance + seed; it fixes 720p
    // output, so hide the size control. No camera path, no bundled examples.
    defaults: { frameNum: '121', size: '1280*720', samplingSteps: '35', guideScale: '7.0', seed: '-1' },
    show: { examples: false, camera: false, frames: true, size: false, guidance: true, dmd: false },
    framesHint: '121 ≈ 5s @ 24fps',
    progressHint: '(typical: 1–3 min on 4× L40S)',
  },

  'echo-async': {
    // Smoke-test cartridge: the runner ignores generation params and returns a
    // fixed-size synthetic clip, so hide every knob that would silently no-op.
    examples: [],
    defaultPrompt: 'Smoke test — echo returns a synthetic clip regardless of prompt.',
    defaults: { frameNum: '72', size: '320*240', samplingSteps: '1', guideScale: '1.0', seed: '0' },
    show: { examples: false, camera: false, frames: false, size: false, guidance: false, dmd: false },
    framesHint: '',
    progressHint: '(a few seconds — synthetic, no model weights)',
  },
};

export function profileFor(cartridgeId?: string): GenProfile {
  return (cartridgeId && GEN_PROFILES[cartridgeId]) || DEFAULT_PROFILE;
}
