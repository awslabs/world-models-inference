// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as sagemaker from 'aws-cdk-lib/aws-sagemaker';
import { Construct } from 'constructs';
import { sageMakerRuntimeStatements } from './iam';

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
  readonly cognitoUserPoolId?: string;
  readonly cognitoClientIds?: string;
  readonly cognitoScope?: string;
  readonly authMode?: string;
  /** CORS allowlist forwarded to the container (R1). */
  readonly allowedOrigins?: string;
  /** Per-IP req/min on inference routes (R1). */
  readonly rateLimit?: string;
  /**
   * Comma-separated private subnet IDs and the security group the model's
   * network interfaces use. The model runs inside the foundation VPC, so its
   * egress is the VPC's (NAT + S3 gateway endpoint) and can be filtered there.
   * Network isolation is not used: the container must reach Cognito's JWKS to
   * verify access tokens.
   */
  readonly subnetIds: string;
  readonly securityGroupId: string;
  readonly vpcId: string;
  /** The ECR repository holding `image`; the role may pull from it alone. */
  readonly repositoryName: string;
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

    const runtimeScope = { repositoryName: props.repositoryName, endpointName: name, vpcId: props.vpcId };
    for (const statement of sageMakerRuntimeStatements(cdk.Stack.of(this), runtimeScope)) {
      role.addToPolicy(statement);
    }

    // S3 is scoped to this endpoint's own prefixes, not whole buckets: read its
    // model data, and (async) write its outputs. The "*" after each prefix is the
    // objects under that one prefix; ListBucket is confined to it by s3:prefix.
    // Reference: https://docs.aws.amazon.com/AmazonS3/latest/userguide/security_iam_service-with-iam.html
    const [bucket, ...modelPath] = props.modelDataS3Prefix.replace(/^s3:\/\//, '').split('/');
    const modelPrefix = modelPath.join('/');
    role.addToPolicy(new iam.PolicyStatement({
      sid: 'S3ReadOwnModelData',
      actions: ['s3:GetObject'],
      resources: [`arn:aws:s3:::${bucket}/${modelPrefix}*`],
    }));
    role.addToPolicy(new iam.PolicyStatement({
      sid: 'S3ListOwnModelData',
      actions: ['s3:ListBucket'],
      resources: [`arn:aws:s3:::${bucket}`],
      conditions: { StringLike: { 's3:prefix': [`${modelPrefix}*`] } },
    }));

    if (props.asyncConfig) {
      const [outputBucket, ...outputPath] = props.asyncConfig.output.replace(/^s3:\/\//, '').split('/');
      role.addToPolicy(new iam.PolicyStatement({
        sid: 'S3ReadWriteOwnAsyncOutputs',
        actions: ['s3:PutObject', 's3:GetObject'],
        resources: [`arn:aws:s3:::${outputBucket}/${outputPath.join('/')}*`],
      }));
    }

    // Security env (R1). The token is injected via a CloudFormation SecureString
    // dynamic reference, so it is resolved at deploy time and never stored in
    // the template. Origins/rate-limit are non-secret plain values.
    const containerEnv: Record<string, string> = {};
    // SageMaker endpoints are IAM-authenticated: sagemaker:InvokeEndpoint is the
    // authorisation boundary, so the in-container check is defence in depth here
    // rather than the only control. Still wired for consistency, and because the
    // same image serves the EC2/ALB path where it IS the only control.
    if (props.authMode) {
      containerEnv.WORLD_MODEL_AUTH_MODE = props.authMode;
    }
    if (props.cognitoUserPoolId) {
      containerEnv.WORLD_MODEL_COGNITO_USER_POOL_ID = props.cognitoUserPoolId;
    }
    if (props.cognitoClientIds) {
      containerEnv.WORLD_MODEL_COGNITO_CLIENT_IDS = props.cognitoClientIds;
    }
    if (props.cognitoScope) {
      containerEnv.WORLD_MODEL_COGNITO_SCOPE = props.cognitoScope;
    }
    if (props.allowedOrigins) {
      containerEnv.WORLD_MODEL_ALLOWED_ORIGINS = props.allowedOrigins;
    }
    if (props.rateLimit) {
      containerEnv.WORLD_MODEL_RATE_LIMIT = props.rateLimit;
    }

    // The model and endpoint config are deliberately left unnamed so CloudFormation
    // can replace them: changing an immutable property (VpcConfig, the image, the
    // role) forces a replacement, which CloudFormation refuses for a custom-named
    // resource. Only the endpoint — the name clients call — keeps a fixed name,
    // and it updates in place onto the new config.
    const model = new sagemaker.CfnModel(this, 'Model', {
      executionRoleArn: role.roleArn,
      vpcConfig: {
        subnets: cdk.Fn.split(',', props.subnetIds),
        securityGroupIds: [props.securityGroupId],
      },
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
      productionVariants: [{
        modelName: model.attrModelName,
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

    // SageMaker validates the ENI permissions when the model is created, so the
    // role's policy has to exist first.
    model.node.addDependency(role);

    const endpointConfig = new sagemaker.CfnEndpointConfig(this, 'EndpointConfig', endpointConfigProps);
    endpointConfig.addDependency(model);

    const endpoint = new sagemaker.CfnEndpoint(this, 'Endpoint', {
      endpointName: name,
      endpointConfigName: endpointConfig.attrEndpointConfigName,
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
