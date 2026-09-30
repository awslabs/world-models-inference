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
 *   certificateArn — ACM cert ARN; enables HTTPS on the EC2 ALB (M1)
 *   authMode       — 'cognito' (default in the container) or 'disabled'
 *   cognitoUserPoolId, cognitoClientIds, cognitoScope
 *                  — Cognito settings passed to the container. Not secret; the
 *                    container verifies pool-issued access tokens against them.
 *   allowedOrigins — CORS allowlist passed to the container (R1)
 *   rateLimit      — per-IP req/min on inference routes (R1)
 *   notificationEmail — email subscribed to EC2 scaling-event notifications
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
  const capacity = app.node.tryGetContext('capacity') as string | undefined;
  const capacitySubnets = app.node.tryGetContext('capacitySubnets') as string | undefined;
  const instanceTypeOverride = app.node.tryGetContext('instanceType') as string | undefined;
  const certificateArn = app.node.tryGetContext('certificateArn') as string | undefined;
  const authMode = app.node.tryGetContext('authMode') as string | undefined;
  const cognitoUserPoolId = app.node.tryGetContext('cognitoUserPoolId') as string | undefined;
  const cognitoClientIds = app.node.tryGetContext('cognitoClientIds') as string | undefined;
  const cognitoScope = app.node.tryGetContext('cognitoScope') as string | undefined;
  const allowedOrigins = app.node.tryGetContext('allowedOrigins') as string | undefined;
  const rateLimit = app.node.tryGetContext('rateLimit') as string | undefined;
  const notificationEmail = app.node.tryGetContext('notificationEmail') as string | undefined;

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
    authMode,
    cognitoUserPoolId,
    cognitoClientIds,
    cognitoScope,
    allowedOrigins,
    rateLimit,
    notificationEmail,
  });
}
