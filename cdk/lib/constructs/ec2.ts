// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import * as cdk from 'aws-cdk-lib';
import * as autoscaling from 'aws-cdk-lib/aws-autoscaling';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import * as iam from 'aws-cdk-lib/aws-iam';
import { Construct } from 'constructs';

export interface Ec2InferenceProps {
  readonly endpointId: string;
  readonly deploymentName: string;
  readonly instanceType: string;
  readonly volumeSize?: number;
  readonly securityGroupId: string;
  readonly albSecurityGroupId: string;
  readonly subnetIds: string;
  readonly publicSubnetIds: string;
  readonly vpcId: string;
  readonly roleName: string;
  readonly imageUri: string;
  readonly modelDataS3Uri: string;
  /**
   * Capacity Block (or on-demand capacity reservation) to launch into. Set this
   * and the launch template targets the reservation explicitly and, for a
   * Capacity Block, sets the required `capacity-block` market type. Without
   * both, the ASG launches ordinary on-demand capacity and a paid-for block
   * sits idle — reserved capacity is never used implicitly.
   */
  readonly capacityReservationId?: string;
  /**
   * Subnets to launch into, overriding `subnetIds`. A Capacity Block lives in
   * one Availability Zone, so the ASG has to be pinned to the private subnet in
   * that AZ; left to its own devices it will pick either AZ and fail half the
   * time.
   */
  readonly capacitySubnetIds?: string;
  /**
   * ACM certificate ARN. When set, the ALB serves HTTPS on 443 and redirects
   * HTTP:80 → HTTPS:443 (M1). When unset, it falls back to plain HTTP:80 for
   * local/demo use.
   */
  readonly certificateArn?: string;
  /**
   * Cognito settings passed to the container as WORLD_MODEL_COGNITO_*.
   *
   * These replace the shared bearer token the instance used to fetch from SSM.
   * None of them is secret — a user pool ID, app client IDs and a scope name are
   * all public identifiers — so unlike the old token they can be baked straight
   * into user data with nothing to decrypt at boot.
   */
  readonly cognitoUserPoolId?: string;
  readonly cognitoClientIds?: string;
  readonly cognitoScope?: string;
  /**
   * Set to 'disabled' to run the endpoint unauthenticated. The container refuses
   * to start if this is unset AND no user pool is given, which is deliberate: a
   * deploy that forgets the Cognito wiring must fail, not quietly serve an open
   * GPU endpoint.
   */
  readonly authMode?: string;
  /** CORS allowlist → WORLD_MODEL_ALLOWED_ORIGINS on the container (R1). */
  readonly allowedOrigins?: string;
  /** Per-IP req/min → WORLD_MODEL_RATE_LIMIT on the container (R1). */
  readonly rateLimit?: string;
}

/**
 * EC2-based inference with ALB for public access.
 * GPU instance in private subnet, ALB in public subnet forwards to port 8080.
 */
export class Ec2Inference extends Construct {
  constructor(scope: Construct, id: string, props: Ec2InferenceProps) {
    super(scope, id);

    const volumeSize = props.volumeSize || 200;
    const region = cdk.Stack.of(this).region;
    const account = cdk.Stack.of(this).account;

    // Resolve the Deep Learning AMI from the public SSM parameter rather than
    // MachineImage.lookup(). lookup() is a CREDENTIALED API CALL AT SYNTH TIME:
    // it caches into cdk.context.json (which is gitignored), so a fresh clone or
    // any environment with stale credentials fails to synth — which also broke
    // `dsr assess`. The SSM form resolves at deploy time via CloudFormation, so
    // synth is offline and portable. Override with -c amiSsmParameter=<path>.
    const amiSsmParameter =
      (cdk.Stack.of(this).node.tryGetContext('amiSsmParameter') as string | undefined) ||
      '/aws/service/deeplearning/ami/x86_64/oss-nvidia-driver-gpu-pytorch-2.7-ubuntu-22.04/latest/ami-id';
    const ami = ec2.MachineImage.fromSsmParameter(amiSsmParameter, {
      os: ec2.OperatingSystemType.LINUX,
    });

    const userData = ec2.UserData.forLinux();
    userData.addCommands(
      '#!/bin/bash',
      'set -ex',
      'exec > /var/log/world-model-startup.log 2>&1',
      '',
      'nvidia-smi',
      '',
      `aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com`,
      `docker pull ${props.imageUri}`,
      '',
      'if [ -d /opt/dlami/nvme ]; then',
      '  mkdir -p /opt/dlami/nvme/checkpoints',
      '  ln -sf /opt/dlami/nvme/checkpoints /opt/checkpoints',
      'else',
      '  mkdir -p /opt/checkpoints',
      'fi',
      '',
      `CKPT_DIR="/opt/checkpoints/${props.endpointId}"`,
      'mkdir -p "$CKPT_DIR"',
      `aws s3 sync "${props.modelDataS3Uri}" "$CKPT_DIR/" --region ${region} || true`,
      '',
      // Security env. Cognito identifiers, CORS origins and the rate limit are
      // all non-secret, so they are baked in directly — there is no longer a
      // secret to fetch from SSM at boot.
      'SEC_ENV=()',
      ...(props.authMode
        ? [`SEC_ENV+=(-e "WORLD_MODEL_AUTH_MODE=${props.authMode}")`]
        : []),
      ...(props.cognitoUserPoolId
        ? [`SEC_ENV+=(-e "WORLD_MODEL_COGNITO_USER_POOL_ID=${props.cognitoUserPoolId}")`]
        : []),
      ...(props.cognitoClientIds
        ? [`SEC_ENV+=(-e "WORLD_MODEL_COGNITO_CLIENT_IDS=${props.cognitoClientIds}")`]
        : []),
      ...(props.cognitoScope
        ? [`SEC_ENV+=(-e "WORLD_MODEL_COGNITO_SCOPE=${props.cognitoScope}")`]
        : []),
      ...(props.allowedOrigins
        ? [`SEC_ENV+=(-e "WORLD_MODEL_ALLOWED_ORIGINS=${props.allowedOrigins}")`]
        : []),
      ...(props.rateLimit
        ? [`SEC_ENV+=(-e "WORLD_MODEL_RATE_LIMIT=${props.rateLimit}")`]
        : []),
      '',
      `docker run -d --gpus all --net host --ipc host \\`,
      `  -v "$CKPT_DIR":/opt/ml/model \\`,
      `  -e NVIDIA_VISIBLE_DEVICES=all \\`,
      `  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\`,
      `  "\${SEC_ENV[@]}" \\`,
      `  --name world-model \\`,
      `  ${props.imageUri}`,
      '',
      'echo "Container started."',
    );

    const launchTemplate = new ec2.LaunchTemplate(this, 'LaunchTemplate', {
      instanceType: new ec2.InstanceType(props.instanceType),
      machineImage: ami,
      userData,
      securityGroup: ec2.SecurityGroup.fromSecurityGroupId(this, 'SG', props.securityGroupId),
      role: iam.Role.fromRoleName(this, 'Role', props.roleName),
      blockDevices: [{
        deviceName: '/dev/sda1',
        volume: ec2.BlockDeviceVolume.ebs(volumeSize, { volumeType: ec2.EbsDeviceVolumeType.GP3, iops: 6000 }),
      }],
    });

    // Target reserved capacity explicitly. The L2 LaunchTemplate does not
    // surface either property, hence the overrides on the underlying resource.
    if (props.capacityReservationId) {
      const cfnLaunchTemplate = launchTemplate.node.defaultChild as ec2.CfnLaunchTemplate;
      cfnLaunchTemplate.addPropertyOverride(
        'LaunchTemplateData.CapacityReservationSpecification',
        {
          CapacityReservationTarget: {
            CapacityReservationId: props.capacityReservationId,
          },
        },
      );
      // P5 Capacity Blocks reject a launch that does not declare this market
      // type; it is not the same as an ordinary on-demand reservation.
      cfnLaunchTemplate.addPropertyOverride('LaunchTemplateData.InstanceMarketOptions', {
        MarketType: 'capacity-block',
      });
    }

    // minSize 0 and no desiredCapacity ON PURPOSE: CloudFormation waits for the
    // group to reach the template's min/desired before marking any ASG mutation
    // complete, so with min 1 a GPU capacity shortage (p5
    // InsufficientInstanceCapacity is routine) turns every deploy — even a pure
    // launch-template fix — into a ~40-minute hang followed by a rollback that
    // can resurrect the very launch template the deploy was replacing. With a
    // template floor of 0, deploys complete immediately; deploy.sh sets the
    // group's desired capacity to 1 out-of-band after the stack lands, and the
    // ASG keeps retrying the launch in the background until capacity appears.
    const asg = new autoscaling.CfnAutoScalingGroup(this, 'ASG', {
      autoScalingGroupName: `world-model-${props.deploymentName}`,
      minSize: '0',
      maxSize: '1',
      launchTemplate: {
        launchTemplateId: launchTemplate.launchTemplateId!,
        version: launchTemplate.latestVersionNumber,
      },
      vpcZoneIdentifier: cdk.Fn.split(',', props.capacitySubnetIds || props.subnetIds),
      targetGroupArns: [],
      tags: [
        { key: 'Name', value: `world-model-${props.deploymentName}`, propagateAtLaunch: true },
        { key: 'Model', value: props.endpointId, propagateAtLaunch: true },
        { key: 'ManagedBy', value: 'world-model-cdk', propagateAtLaunch: true },
      ],
    });

    // --- ALB (public-facing) ---

    const publicSubnetIds = cdk.Fn.split(',', props.publicSubnetIds);

    const alb = new elbv2.CfnLoadBalancer(this, 'ALB', {
      name: `wm-${props.deploymentName}`.slice(0, 32),
      scheme: 'internet-facing',
      type: 'application',
      securityGroups: [props.albSecurityGroupId],
      subnets: publicSubnetIds,
    });

    const targetGroup = new elbv2.CfnTargetGroup(this, 'TG', {
      name: `wm-${props.deploymentName}-tg`.slice(0, 32),
      port: 8080,
      protocol: 'HTTP',
      vpcId: props.vpcId,
      targetType: 'instance',
      healthCheckPath: '/health',
      healthCheckIntervalSeconds: 30,
      healthyThresholdCount: 2,
      unhealthyThresholdCount: 3,
    });

    // Listener(s). With an ACM certificate the ALB terminates TLS on 443 and
    // HTTP:80 becomes a redirect to HTTPS (M1). Without one it stays HTTP-only
    // for local/demo use — never expose that to untrusted networks.
    if (props.certificateArn) {
      new elbv2.CfnListener(this, 'HttpsListener', {
        loadBalancerArn: alb.ref,
        port: 443,
        protocol: 'HTTPS',
        certificates: [{ certificateArn: props.certificateArn }],
        sslPolicy: 'ELBSecurityPolicy-TLS13-1-2-2021-06',
        defaultActions: [{ type: 'forward', targetGroupArn: targetGroup.ref }],
      });
      new elbv2.CfnListener(this, 'HttpRedirectListener', {
        loadBalancerArn: alb.ref,
        port: 80,
        protocol: 'HTTP',
        defaultActions: [{
          type: 'redirect',
          redirectConfig: { protocol: 'HTTPS', port: '443', statusCode: 'HTTP_301' },
        }],
      });
    } else {
      new elbv2.CfnListener(this, 'HttpListener', {
        loadBalancerArn: alb.ref,
        port: 80,
        protocol: 'HTTP',
        defaultActions: [{ type: 'forward', targetGroupArn: targetGroup.ref }],
      });
    }

    // Attach the ASG to the target group
    asg.targetGroupArns = [targetGroup.ref];

    // --- Outputs ---

    new cdk.CfnOutput(this, 'ASGName', {
      value: asg.autoScalingGroupName!,
      description: `ASG for ${props.deploymentName}`,
    });

    new cdk.CfnOutput(this, 'EndpointUrl', {
      value: cdk.Fn.join('', [props.certificateArn ? 'https://' : 'http://', alb.attrDnsName]),
      description: `Public endpoint URL for ${props.deploymentName}`,
    });
  }
}
