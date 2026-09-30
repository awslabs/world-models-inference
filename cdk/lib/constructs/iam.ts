// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import { listCartridges } from './manifest';

/**
 * Least-privilege IAM building blocks shared by the foundation and model stacks.
 *
 * Every grant is scoped to this project's own resources. The few statements that
 * must use Resource "*" are the actions IAM does not support resource-level
 * permissions for; each says so, with the AWS reference, where it is defined.
 */

/**
 * The ECR repository of every cartridge in the catalogue, named exactly.
 *
 * The shared image-build role builds whichever cartridge it is asked to, so it
 * needs each cartridge's repository (`world-model-<id>`, created by
 * scripts/build-image.py on first build). They are enumerated from
 * inference/models/ at synth time rather than matched by a prefix, so the role
 * reaches no repository outside the catalogue. Adding a cartridge therefore
 * needs a foundation redeploy before its first image build.
 */
export function cartridgeRepositoryArns(stack: cdk.Stack): string[] {
  return listCartridges().map(
    (id) => `arn:aws:ecr:${stack.region}:${stack.account}:repository/world-model-${id}`,
  );
}

/**
 * ecr:GetAuthorizationToken on Resource "*".
 *
 * Why "*": GetAuthorizationToken returns a registry-wide login token and has no
 * resource type, so IAM only accepts Resource "*" for it. It grants no access to
 * any image by itself — every pull/push is separately scoped to named
 * repositories.
 * Reference: https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazonelasticcontainerregistry.html
 */
export function ecrAuthStatement(): iam.PolicyStatement {
  return new iam.PolicyStatement({
    sid: 'EcrAuthTokenNoResourceLevelSupport',
    actions: ['ecr:GetAuthorizationToken'],
    resources: ['*'],
  });
}

/** What a SageMaker endpoint's execution role is scoped to. */
export interface SageMakerRuntimeScope {
  /** The one ECR repository the endpoint's image lives in. */
  readonly repositoryName: string;
  /** The endpoint name; its logs go to /aws/sagemaker/Endpoints/<name>. */
  readonly endpointName: string;
  /** The VPC the model's network interfaces are created in. */
  readonly vpcId: string;
}

/**
 * What a SageMaker hosting container's execution role needs at run time: pull
 * its own image, write its own endpoint's logs and metrics, and manage the
 * network interfaces SageMaker attaches when the model runs in the VPC.
 */
export function sageMakerRuntimeStatements(
  stack: cdk.Stack, scope: SageMakerRuntimeScope,
): iam.PolicyStatement[] {
  const logGroup = `arn:aws:logs:${stack.region}:${stack.account}:log-group:/aws/sagemaker/Endpoints/${scope.endpointName}`;
  const vpcArn = `arn:aws:ec2:${stack.region}:${stack.account}:vpc/${scope.vpcId}`;
  return [
    ecrAuthStatement(),
    new iam.PolicyStatement({
      sid: 'EcrPullOwnRepo',
      actions: ['ecr:BatchCheckLayerAvailability', 'ecr:GetDownloadUrlForLayer', 'ecr:BatchGetImage'],
      resources: [`arn:aws:ecr:${stack.region}:${stack.account}:repository/${scope.repositoryName}`],
    }),
    // ":log-stream:*" is every stream inside this endpoint's one log group:
    // SageMaker names a stream per variant and instance at run time, so the
    // names cannot be known in advance.
    // Reference: https://docs.aws.amazon.com/sagemaker/latest/dg/logging-cloudwatch.html
    new iam.PolicyStatement({
      sid: 'CloudWatchLogsOwnEndpoint',
      actions: ['logs:CreateLogGroup', 'logs:CreateLogStream', 'logs:PutLogEvents', 'logs:DescribeLogStreams'],
      resources: [logGroup, `${logGroup}:log-stream:*`],
    }),
    // Why "*": cloudwatch:PutMetricData has no resource type. The condition limits
    // it to SageMaker's own namespaces, so the role cannot write other metrics.
    // Reference: https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazoncloudwatch.html
    new iam.PolicyStatement({
      sid: 'CloudWatchMetricsSageMakerNamespaces',
      actions: ['cloudwatch:PutMetricData'],
      resources: ['*'],
      conditions: { StringLike: { 'cloudwatch:namespace': ['/aws/sagemaker/*', 'AWS/SageMaker'] } },
    }),
    // Why "*": these EC2 describe actions have no resource type.
    // Reference: https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazonec2.html
    new iam.PolicyStatement({
      sid: 'SageMakerVpcDescribe',
      actions: [
        'ec2:DescribeNetworkInterfaces',
        'ec2:DescribeVpcs',
        'ec2:DescribeDhcpOptions',
        'ec2:DescribeSubnets',
        'ec2:DescribeSecurityGroups',
      ],
      resources: ['*'],
    }),
    // SageMaker creates and deletes the model's ENIs itself, so their IDs cannot
    // be named in advance; the ec2:Vpc condition confines every one of these
    // calls to the project VPC. "IfExists" because the key is only present on
    // the network-interface/subnet resources of the request, and a statement
    // that required it on every resource would deny the call outright.
    // Reference: https://docs.aws.amazon.com/sagemaker/latest/dg/host-vpc.html
    new iam.PolicyStatement({
      sid: 'SageMakerVpcNetworkInterfacesInProjectVpc',
      actions: [
        'ec2:CreateNetworkInterface',
        'ec2:CreateNetworkInterfacePermission',
        'ec2:DeleteNetworkInterface',
        'ec2:DeleteNetworkInterfacePermission',
      ],
      resources: ['*'],
      conditions: { ArnEqualsIfExists: { 'ec2:Vpc': vpcArn } },
    }),
  ];
}
