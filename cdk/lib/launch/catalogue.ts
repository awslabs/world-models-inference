// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * The cartridges a Launch Stack offers, as scripts/launch_buildspecs.py
 * describes them. The buildspecs are generated in Python so that they are the
 * same commands scripts/stage-weights.py and scripts/build-image.py run.
 */
export interface LaunchWeights {
  readonly repo: string;
  readonly subdir: string;
  readonly revision: string;
  /** Artifacts-bucket key written when this revision has been staged. */
  readonly marker: string;
  /** Buildspec JSON. Refers to $ARTIFACTS_BUCKET and $AWS_REGION. */
  readonly buildspec: string;
}

export interface LaunchCartridge {
  readonly version: string;
  readonly ec2Instance: string;
  readonly ec2VolumeSize: number;
  readonly modelDataPrefix: string;
  readonly license: string;
  readonly licenseUrl: string;
  readonly licenseRestricted: boolean;
  readonly weights: LaunchWeights[];
  /** Buildspec JSON. Refers to $WM_REGISTRY and $AWS_REGION. */
  readonly image: string;
}

export interface LaunchCatalogue {
  readonly models: Record<string, LaunchCartridge>;
}

// LingBot World V2's weights are CC BY-NC-SA 4.0 (non-commercial); see
// docs/MODEL_LICENSES.md. Refused here as well as in the generator, so a
// hand-edited catalogue cannot put it in a public template either.
const NEVER_OFFERED = /^lingbot-(world-)?v2/;

const MODEL_RE = /^[a-z0-9](-?[a-z0-9])*$/;
const SHA_RE = /^[0-9a-f]{40}$/;

/** Validate a parsed catalogue; throws on anything a real one would not contain. */
export function checkCatalogue(raw: unknown): LaunchCatalogue {
  const cat = raw as LaunchCatalogue;
  const ids = Object.keys(cat?.models ?? {});
  if (ids.length === 0) throw new Error('Launch catalogue lists no cartridges');
  for (const id of ids) {
    const c = cat.models[id];
    if (NEVER_OFFERED.test(id)) throw new Error(`${id} must never be offered (non-commercial weights licence)`);
    // The ID becomes part of resource names, ARNs and the ALB name (wm-<id>-tg <= 32).
    if (!MODEL_RE.test(id) || id.length > 26) throw new Error(`Invalid cartridge id ${JSON.stringify(id)}`);
    if (!/^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$/.test(c.version)) throw new Error(`${id}: bad version`);
    if (!/^[a-z][a-z0-9-]*\.[a-z0-9]+$/.test(c.ec2Instance)) throw new Error(`${id}: bad ec2Instance`);
    if (!Number.isInteger(c.ec2VolumeSize) || c.ec2VolumeSize < 20) throw new Error(`${id}: bad ec2VolumeSize`);
    if (!/^[A-Za-z0-9_-]+(\/[A-Za-z0-9_-]+)*\/?$/.test(c.modelDataPrefix)) throw new Error(`${id}: bad modelDataPrefix`);
    if (typeof c.licenseRestricted !== 'boolean' || !c.license || !c.licenseUrl) {
      throw new Error(`${id}: every cartridge must declare license, license_url and license_restricted`);
    }
    for (const w of c.weights) {
      if (!SHA_RE.test(w.revision)) throw new Error(`${id}: ${w.repo} is not pinned to a commit`);
      if (w.marker !== `_staged/${id}/${w.revision}`) throw new Error(`${id}: unexpected marker ${w.marker}`);
      JSON.parse(w.buildspec);
    }
    JSON.parse(c.image);
  }
  return cat;
}
