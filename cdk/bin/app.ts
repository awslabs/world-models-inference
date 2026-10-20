// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0
/**
 * World Model Inference — CDK Entry Point
 *
 * Usage:
 *   npx cdk deploy WorldModelFoundation
 *   npx cdk deploy WorldModel-echo-async -c model=echo-async -c target=ec2
 *   npx cdk deploy WorldModel-echo-prod -c model=echo-async -c target=sagemaker -c name=echo-prod
 *   npx cdk destroy WorldModel-echo-prod
 *
 * Context:
 *   model    — directory name under inference/models/
 *   target   — ec2 | sagemaker
 *   name     — deployment name (defaults to model)
 *   capacity — Capacity Block reservation ID (EC2 only)
 *   certificateArn, endpointDomain, hostedZoneId
 *                  — HTTPS on the EC2 ALB under a custom name (Route 53 alias)
 *   allowPlaintext — 'true' serves plain HTTP from the ALB. Without it, and
 *                    without a certificate, EC2 endpoints get HTTPS through a
 *                    CloudFront distribution (no domain needed)
 *   allowedCidr    — IPv4 CIDR of viewers CloudFront admits
 *   authMode       — 'cognito' (default in the container) or 'disabled'
 *   cognitoUserPoolId, cognitoClientIds, cognitoScope
 *                  — Cognito settings passed to the container. Not secret; the
 *                    container verifies pool-issued access tokens against them.
 *   allowedOrigins — CORS allowlist passed to the container (R1)
 *   rateLimit      — per-IP req/min on inference routes (R1)
 *   notificationEmail — email subscribed to EC2 scaling-event notifications
 *   imageDigest    — sha256:... of the image to run (deploy.sh resolves it from
 *                    the version tag); without it the version tag is used
 */
import * as cdk from 'aws-cdk-lib';
import { SharedStack } from '../lib/shared';
import { ModelStack } from '../lib/model';
import { loadManifest } from '../lib/constructs/manifest';

const app = new cdk.App();
const env = {
  account: process.env.CDK_DEFAULT_ACCOUNT || process.env.AWS_ACCOUNT_ID,
  region: process.env.CDK_DEFAULT_REGION || process.env.AWS_REGION || 'us-east-1',
};

new SharedStack(app, 'WorldModelFoundation', { env });

const model = app.node.tryGetContext('model') as string | undefined;
if (model) {
  const target = (app.node.tryGetContext('target') as string) || 'sagemaker';
  const name = (app.node.tryGetContext('name') as string) || model;
  // The deployment name becomes part of stack, endpoint, ALB, log-group and
  // S3-prefix names, and so of the IAM resource ARNs scoped to them: keep it to
  // one bounded segment that is valid in all of them — alphanumerics and single
  // inner hyphens (SageMaker/ELB forbid leading, trailing and doubled hyphens),
  // at most 26 chars so `wm-<name>-tg` still fits the 32-char ALB/target-group limit.
  if (name.length > 26 || !/^[A-Za-z0-9](-?[A-Za-z0-9])*$/.test(name)) {
    throw new Error(
      `Invalid deployment name '${name}': use letters, digits and single inner hyphens, at most 26 characters`,
    );
  }
  // Context values reach the instance's boot environment, the token verifier
  // and resource names. deploy.sh checks them too, but `npx cdk deploy` can be run
  // without it, so the app refuses anything outside each value's real shape.
  const ctx = (key: string, re: RegExp): string | undefined => {
    const v = app.node.tryGetContext(key) as string | undefined;
    if (v !== undefined && !re.test(String(v))) {
      throw new Error(`Invalid context ${key}: ${JSON.stringify(v)}`);
    }
    return v === undefined ? undefined : String(v);
  };
  const capacity = ctx('capacity', /^cr-[0-9a-f]{8,17}$/);
  const capacitySubnets = ctx('capacitySubnets', /^subnet-[0-9a-f]{8,17}(,subnet-[0-9a-f]{8,17})*$/);
  const instanceTypeOverride = ctx('instanceType', /^[a-z][a-z0-9-]*\.[a-z0-9]+$/);
  const certificateArn = ctx('certificateArn', /^arn:aws[a-z-]*:acm:[a-z0-9-]+:\d{12}:certificate\/[0-9a-f-]{36}$/);
  const endpointDomain = ctx('endpointDomain', /^([a-z0-9]([a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,}$/);
  const hostedZoneId = ctx('hostedZoneId', /^Z[A-Z0-9]{1,32}$/);
  const allowPlaintext = ctx('allowPlaintext', /^(true|false)$/) === 'true';
  const allowedCidr = ctx('allowedCidr', /^\d{1,3}(\.\d{1,3}){3}\/\d{1,2}$/);
  const authMode = ctx('authMode', /^(cognito|disabled)$/);
  const cognitoUserPoolId = ctx('cognitoUserPoolId', /^[a-z]{2}(-[a-z]+)+-\d+_[A-Za-z0-9]+$/);
  const cognitoClientIds = ctx('cognitoClientIds', /^[A-Za-z0-9]+(,[A-Za-z0-9]+)*$/);
  const cognitoScope = ctx('cognitoScope', /^[A-Za-z0-9._~:\/-]+$/);
  const allowedOrigins = ctx('allowedOrigins', /^https?:\/\/[A-Za-z0-9.-]+(:\d+)?(,https?:\/\/[A-Za-z0-9.-]+(:\d+)?)*$/);
  const rateLimit = ctx('rateLimit', /^\d{1,6}$/);
  const imageDigest = ctx('imageDigest', /^sha256:[0-9a-f]{64}$/);
  const notificationEmail = ctx('notificationEmail', /^[^\s@,;'"`$\\]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$/);

  new ModelStack(app, `WorldModel-${name}`, {
    env,
    deploymentName: name,
    endpointId: model,
    target: target as 'ec2' | 'sagemaker',
    manifest: loadManifest(model, env.account!),
    capacityReservationId: capacity,
    capacitySubnetIds: capacitySubnets,
    instanceTypeOverride,
    certificateArn,
    endpointDomain,
    hostedZoneId,
    allowPlaintext,
    allowedCidr,
    authMode,
    cognitoUserPoolId,
    cognitoClientIds,
    cognitoScope,
    allowedOrigins,
    rateLimit,
    notificationEmail,
    imageDigest,
  });
}
