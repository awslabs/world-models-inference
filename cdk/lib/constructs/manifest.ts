// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import * as fs from 'fs';
import * as path from 'path';
import * as yaml from 'js-yaml';

export interface ModelManifest {
  readonly endpointId: string;
  readonly version: string;
  readonly modelDataPrefix: string;
  readonly sagemaker: {
    readonly instance: string;
    readonly scale: { min: number; max: number };
    readonly output?: string;
    readonly maxConcurrent: number;
  };
  readonly ec2: {
    readonly instance: string;
    readonly volumeSize?: number;
  };
}

interface RawManifest {
  model_data: string;
  version?: string;
  sagemaker: {
    instance: string;
    scale: { min: number; max: number };
    output?: string;
    max_concurrent?: number;
  };
  ec2: { instance: string; volume_size?: number };
}

export function loadManifest(endpointId: string, accountId: string): ModelManifest {
  // endpointId comes from CDK context (`-c model=<id>`). Constrain it to a
  // single safe path segment so it can't traverse out of inference/models/
  // (e.g. `-c model=../../etc`) when joined into the manifest path.
  if (!/^[A-Za-z0-9_-]+$/.test(endpointId)) {
    throw new Error(`Invalid model id '${endpointId}': expected [A-Za-z0-9_-]+`);
  }
  // endpointId is validated to a single safe segment above and the resolved
  // path is bounds-checked below, so these joins cannot traverse out of the
  // models dir.
  const modelsRoot = path.resolve(__dirname, '../../../inference/models'); // nosemgrep
  const manifestPath = path.join(modelsRoot, endpointId, 'endpoint.yaml'); // nosemgrep
  if (!manifestPath.startsWith(modelsRoot + path.sep)) {
    throw new Error(`Invalid model id '${endpointId}': path escapes models dir`);
  }
  if (!fs.existsSync(manifestPath)) {
    throw new Error(`No endpoint.yaml for ${endpointId}`);
  }
  const raw = yaml.load(fs.readFileSync(manifestPath, 'utf8')) as RawManifest;

  if (!raw.sagemaker?.instance) throw new Error(`${endpointId}: sagemaker.instance required`);
  if (!raw.sagemaker?.scale) throw new Error(`${endpointId}: sagemaker.scale required`);
  if (!raw.ec2?.instance) throw new Error(`${endpointId}: ec2.instance required`);

  // Extract the prefix after the bucket: s3://bucket/prefix/ → prefix/
  const modelDataPrefix = raw.model_data.replace(/^s3:\/\/[^/]+\//, '');

  return {
    endpointId,
    version: raw.version || 'latest',
    modelDataPrefix,
    sagemaker: {
      instance: raw.sagemaker.instance,
      scale: raw.sagemaker.scale,
      output: raw.sagemaker.output,
      maxConcurrent: raw.sagemaker.max_concurrent ?? 1,
    },
    ec2: {
      instance: raw.ec2.instance,
      volumeSize: raw.ec2.volume_size,
    },
  };
}

export function inferMode(manifest: ModelManifest): 'async' | 'real-time' {
  return manifest.sagemaker.output ? 'async' : 'real-time';
}

export function imageUri(manifest: ModelManifest, accountId: string, region: string): string {
  return `${accountId}.dkr.ecr.${region}.amazonaws.com/world-model-${manifest.endpointId}:${manifest.version}`;
}
