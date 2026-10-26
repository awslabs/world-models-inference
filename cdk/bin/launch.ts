// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * World Model Inference — one-click Launch Stack entry point.
 *
 * Synthesises the WorldModelLaunch template for the CloudFormation console's
 * "Launch Stack" link. Run it through scripts/publish.py, which prepares the
 * inputs below, synthesises, and uploads the template and its assets:
 *
 *   python scripts/publish.py --bucket-basename <name> --prefix <prefix> --region <region>
 *
 * The stack is environment-agnostic (deployable into any account), and its
 * assets — nested templates, Lambda code, the inference/ source and the built
 * UI — live in the publisher's per-region bucket, <publishBucket>-<region>,
 * under <publishPrefix>/assets/. No CDK bootstrap is needed in the account
 * that launches it.
 *
 * Context (all required):
 *   publishBucket     — bucket basename; the Region is appended
 *   publishPrefix     — key prefix inside it
 *   launchBuildspecs  — JSON from scripts/launch_buildspecs.py
 *   imageSource       — directory containing inference/ (zipped as the image build source)
 *   uiDist            — the built frontend (frontend/dist)
 */
import * as fs from 'fs';
import * as cdk from 'aws-cdk-lib';
import { checkCatalogue } from '../lib/launch/catalogue';
import { LaunchStack } from '../lib/launch/launch-stack';

const app = new cdk.App();

const ctx = (key: string, re: RegExp): string => {
  const v = app.node.tryGetContext(key) as string | undefined;
  if (v === undefined || !re.test(String(v))) {
    throw new Error(`launch.ts needs -c ${key}=<value> matching ${re}, got ${JSON.stringify(v)}`);
  }
  return String(v);
};

// <basename>-<region> must stay a valid bucket name for the longest Region code.
const publishBucket = ctx('publishBucket', /^[a-z0-9][a-z0-9-]{1,40}[a-z0-9]$/);
const publishPrefix = ctx('publishPrefix', /^[A-Za-z0-9_-]+(\/[A-Za-z0-9_-][A-Za-z0-9._-]*)*$/);
const catalogue = checkCatalogue(JSON.parse(fs.readFileSync(ctx('launchBuildspecs', /./), 'utf8')));
const imageSource = ctx('imageSource', /./);
const uiDist = ctx('uiDist', /./);
for (const [what, dir, inside] of [['imageSource', imageSource, 'inference'], ['uiDist', uiDist, 'index.html']]) {
  if (!fs.existsSync(`${dir}/${inside}`)) throw new Error(`${what} ${dir} has no ${inside}`);
}

new LaunchStack(app, 'WorldModelLaunch', {
  synthesizer: new cdk.DefaultStackSynthesizer({
    fileAssetsBucketName: `${publishBucket}-\${AWS::Region}`,
    bucketPrefix: `${publishPrefix}/assets/`,
    // The launching account is not bootstrapped and needs no CDK roles: the
    // console deploys the template with the caller's own credentials.
    generateBootstrapVersionRule: false,
  }),
  catalogue,
  imageSource,
  uiDist,
});
