// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as sagemaker from 'aws-cdk-lib/aws-sagemaker';
import { Construct } from 'constructs';

export interface WorldModelEndpointProps {
  readonly endpointId: string;
  readonly image: string;
  readonly instance: string;
  readonly mode: 'async' | 'real-time';
  readonly scale: { min: number; max: number };
  readonly modelDataS3Prefix: string;
  readonly maxConcurrent: number;
  readonly asyncConfig?: { output: string };
  /** SSM SecureString name holding the shared API token (R1). */
  readonly apiTokenParam?: string;
  /** CORS allowlist forwarded to the container (R1). */
  readonly allowedOrigins?: string;
  /** Per-IP req/min on inference routes (R1). */
  readonly rateLimit?: string;
}

export class WorldModelEndpoint extends Construct {
  public readonly endpointName: string;

  constructor(scope: Construct, id: string, props: WorldModelEndpointProps) {
    super(scope, id);

    const name = `world-model-${props.endpointId}`;

    // Minimal execution role — replaces AmazonSageMakerFullAccess (M5) with the
    // ECR pull + CloudWatch permissions a hosting container actually needs. S3
    // is scoped to the model-data (and, for async, output) bucket below.
    const role = new iam.Role(this, 'Role', {
      assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
    });

    role.addToPolicy(new iam.PolicyStatement({
      sid: 'EcrPull',
      actions: [
        'ecr:GetAuthorizationToken',
        'ecr:BatchCheckLayerAvailability',
        'ecr:GetDownloadUrlForLayer',
        'ecr:BatchGetImage',
      ],
      resources: ['*'],  // GetAuthorizationToken requires *; pulls gated by repo policy
    }));
    role.addToPolicy(new iam.PolicyStatement({
      sid: 'CloudWatchLogsAndMetrics',
      actions: [
        'logs:CreateLogGroup',
        'logs:CreateLogStream',
        'logs:PutLogEvents',
        'cloudwatch:PutMetricData',
      ],
      resources: ['*'],
    }));

    const bucket = props.modelDataS3Prefix.replace(/^s3:\/\//, '').split('/')[0];
    role.addToPolicy(new iam.PolicyStatement({
      actions: ['s3:GetObject', 's3:ListBucket'],
      resources: [`arn:aws:s3:::${bucket}`, `arn:aws:s3:::${bucket}/*`],
    }));

    if (props.asyncConfig) {
      const outputBucket = props.asyncConfig.output.replace(/^s3:\/\//, '').split('/')[0];
      role.addToPolicy(new iam.PolicyStatement({
        actions: ['s3:PutObject', 's3:GetObject'],
        resources: [`arn:aws:s3:::${outputBucket}/*`],
      }));
    }

    // Security env (R1). The token is injected via a CloudFormation SecureString
    // dynamic reference, so it is resolved at deploy time and never stored in
    // the template. Origins/rate-limit are non-secret plain values.
    const containerEnv: Record<string, string> = {};
    if (props.apiTokenParam) {
      containerEnv.WORLD_MODEL_API_TOKEN = `{{resolve:ssm-secure:${props.apiTokenParam}}}`;
    }
    if (props.allowedOrigins) {
      containerEnv.WORLD_MODEL_ALLOWED_ORIGINS = props.allowedOrigins;
    }
    if (props.rateLimit) {
      containerEnv.WORLD_MODEL_RATE_LIMIT = props.rateLimit;
    }

    const model = new sagemaker.CfnModel(this, 'Model', {
      modelName: name,
      executionRoleArn: role.roleArn,
      primaryContainer: {
        image: props.image,
        environment: Object.keys(containerEnv).length ? containerEnv : undefined,
        modelDataSource: {
          s3DataSource: {
            s3Uri: props.modelDataS3Prefix,
            s3DataType: 'S3Prefix',
            compressionType: 'None',
          },
        },
      },
    });

    const endpointConfigProps: sagemaker.CfnEndpointConfigProps = {
      endpointConfigName: `${name}-config`,
      productionVariants: [{
        modelName: model.modelName!,
        variantName: 'primary',
        instanceType: props.instance,
        initialInstanceCount: Math.max(props.scale.min, 1),
      }],
    };

    if (props.asyncConfig) {
      (endpointConfigProps as any).asyncInferenceConfig = {
        outputConfig: { s3OutputPath: props.asyncConfig.output },
        clientConfig: { maxConcurrentInvocationsPerInstance: props.maxConcurrent },
      };
    }

    const endpointConfig = new sagemaker.CfnEndpointConfig(this, 'EndpointConfig', endpointConfigProps);
    endpointConfig.addDependency(model);

    const endpoint = new sagemaker.CfnEndpoint(this, 'Endpoint', {
      endpointName: name,
      endpointConfigName: endpointConfig.endpointConfigName!,
    });
    endpoint.addDependency(endpointConfig);
    endpoint.node.addDependency(role);
    this.endpointName = endpoint.endpointName!;

    new cdk.CfnOutput(this, 'EndpointNameOutput', {
      value: endpoint.endpointName!,
      description: `SageMaker endpoint: ${props.endpointId}`,
    });
  }
}
