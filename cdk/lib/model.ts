// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import * as cdk from 'aws-cdk-lib';
import * as ssm from 'aws-cdk-lib/aws-ssm';
import { Construct } from 'constructs';
import { Ec2Inference } from './constructs/ec2';
import { WorldModelEndpoint } from './constructs/sagemaker';
import { ModelManifest, inferMode, imageUri } from './constructs/manifest';
import { describeStack } from './solution';

export interface ModelStackProps extends cdk.StackProps {
  readonly deploymentName: string;
  readonly endpointId: string;
  readonly target: 'ec2' | 'sagemaker';
  readonly manifest: ModelManifest;
  readonly capacityReservationId?: string;
  readonly capacitySubnetIds?: string;
  /**
   * Override the manifest's ec2.instance for one deployment. p5.48xlarge
   * capacity is frequently unavailable, and several cartridges run (more
   * slowly) on a smaller box, so being able to retarget without editing the
   * manifest is the difference between a demo and no demo.
   */
  readonly instanceTypeOverride?: string;
  /** ACM certificate ARN for the EC2 ALB — enables HTTPS (M1). */
  readonly certificateArn?: string;
  /** SSM SecureString name holding the shared inference API token (R1). */
  readonly cognitoUserPoolId?: string;
  readonly cognitoClientIds?: string;
  readonly cognitoScope?: string;
  readonly authMode?: string;
  /** CORS allowlist forwarded to the inference container (R1). */
  readonly allowedOrigins?: string;
  /** Per-IP request/min limit on inference routes (R1). */
  readonly rateLimit?: string;
}

export class ModelStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: ModelStackProps) {
    // One dashboard entry per target, not per model — otherwise every model
    // deployed from this repo would report separately.
    super(scope, id, {
      ...props,
      description: describeStack(
        `World Model Inference — ${props.endpointId} endpoint on ${props.target}`,
        props.target,
      ),
    });

    const { deploymentName, endpointId, target, manifest } = props;

    // Read foundation params from SSM
    const sgId = ssm.StringParameter.valueForStringParameter(this, '/world-model/foundation/sg-id');
    const albSgId = ssm.StringParameter.valueForStringParameter(this, '/world-model/foundation/alb-sg-id');
    const vpcId = ssm.StringParameter.valueForStringParameter(this, '/world-model/foundation/vpc-id');
    const ec2RoleName = ssm.StringParameter.valueForStringParameter(this, '/world-model/foundation/ec2-role-name');
    const privateSubnets = ssm.StringParameter.valueForStringParameter(this, '/world-model/foundation/private-subnet-ids');
    const publicSubnets = ssm.StringParameter.valueForStringParameter(this, '/world-model/foundation/public-subnet-ids');
    const artifactsBucket = ssm.StringParameter.valueForStringParameter(this, '/world-model/foundation/artifacts-bucket');
    const outputsBucket = ssm.StringParameter.valueForStringParameter(this, '/world-model/foundation/outputs-bucket');

    // Derived paths
    const modelDataUri = `s3://${artifactsBucket}/${manifest.modelDataPrefix}`;
    const outputUri = `s3://${outputsBucket}/${deploymentName}/`;
    const image = imageUri(manifest, this.account, this.region);

    if (target === 'ec2') {
      new Ec2Inference(this, 'EC2', {
        endpointId,
        deploymentName,
        instanceType: props.instanceTypeOverride || manifest.ec2.instance,
        volumeSize: manifest.ec2.volumeSize,
        securityGroupId: sgId,
        albSecurityGroupId: albSgId,
        subnetIds: privateSubnets,
        publicSubnetIds: publicSubnets,
        vpcId,
        roleName: ec2RoleName,
        imageUri: image,
        modelDataS3Uri: modelDataUri,
        capacityReservationId: props.capacityReservationId,
        capacitySubnetIds: props.capacitySubnetIds,
        certificateArn: props.certificateArn,
        cognitoUserPoolId: props.cognitoUserPoolId,
        cognitoClientIds: props.cognitoClientIds,
        cognitoScope: props.cognitoScope,
        authMode: props.authMode,
        allowedOrigins: props.allowedOrigins,
        rateLimit: props.rateLimit,
      });

      new ssm.StringParameter(this, 'EndpointRegistry', {
        parameterName: `/world-model/endpoints/${deploymentName}`,
        stringValue: JSON.stringify({
          name: deploymentName,
          model: endpointId,
          target: 'ec2',
          mode: inferMode(manifest),
          instance_type: props.instanceTypeOverride || manifest.ec2.instance,
        }),
      });

    } else {
      const mode = inferMode(manifest);

      new WorldModelEndpoint(this, 'SM', {
        endpointId: deploymentName,
        image,
        instance: manifest.sagemaker.instance,
        mode,
        scale: manifest.sagemaker.scale,
        modelDataS3Prefix: modelDataUri,
        maxConcurrent: manifest.sagemaker.maxConcurrent,
        asyncConfig: mode === 'async' ? { output: outputUri } : undefined,
        cognitoUserPoolId: props.cognitoUserPoolId,
        cognitoClientIds: props.cognitoClientIds,
        cognitoScope: props.cognitoScope,
        authMode: props.authMode,
        allowedOrigins: props.allowedOrigins,
        rateLimit: props.rateLimit,
      });

      new ssm.StringParameter(this, 'EndpointRegistry', {
        parameterName: `/world-model/endpoints/${deploymentName}`,
        stringValue: JSON.stringify({
          name: deploymentName,
          model: endpointId,
          target: 'sagemaker',
          mode,
          instance_type: manifest.sagemaker.instance,
          endpoint_name: `world-model-${deploymentName}`,
        }),
      });
    }
  }
}
