// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import * as cdk from 'aws-cdk-lib';
import * as codebuild from 'aws-cdk-lib/aws-codebuild';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as ssm from 'aws-cdk-lib/aws-ssm';
import { Construct } from 'constructs';
import { describeStack } from './solution';

export class SharedStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, {
      ...props,
      description: describeStack(
        'World Model Inference — shared foundation: VPC, IAM roles, S3 buckets, CodeBuild',
        'foundation',
      ),
    });

    // --- S3 Buckets (import existing or create) ---
    // The buckets are RETAIN'd (see below), so they outlive `cdk destroy`. That
    // means a redeploy — or a deploy on an account that ran an earlier foundation
    // stack — would hit `BucketAlreadyExists` if we unconditionally created them,
    // while a genuinely clean account fails on `fromBucketName` because nothing
    // creates them. CDK can't detect bucket existence at synth time, so the caller
    // picks via context: `-c importExistingBuckets=true` imports (buckets already
    // exist), the default creates them. deploy.sh sets this automatically after an
    // existence probe; see its `bucketsExist` check.

    const artifactsBucketName = `world-model-artifacts-${this.account}-${this.region}`;
    const outputsBucketName = `world-model-outputs-${this.account}-${this.region}`;
    const importExistingBuckets =
      this.node.tryGetContext('importExistingBuckets') === true ||
      this.node.tryGetContext('importExistingBuckets') === 'true';

    // RETAIN so tearing down the stack never deletes build artifacts or outputs.
    // Dedicated bucket for S3 server access logs (DSR: buckets must have access
    // logging). Self-logging would recurse, so this one is the log sink and is
    // excluded from logging itself.
    const logsBucket = importExistingBuckets
      ? undefined
      : new s3.Bucket(this, 'AccessLogsBucket', {
          bucketName: `world-model-logs-${this.account}-${this.region}`,
          blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
          encryption: s3.BucketEncryption.S3_MANAGED,
          enforceSSL: true,
          removalPolicy: cdk.RemovalPolicy.RETAIN,
          lifecycleRules: [{ expiration: cdk.Duration.days(90) }],
        });

    const makeBucket = (id: string, bucketName: string): s3.IBucket =>
      importExistingBuckets
        ? s3.Bucket.fromBucketName(this, id, bucketName)
        : new s3.Bucket(this, id, {
            bucketName,
            blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
            encryption: s3.BucketEncryption.S3_MANAGED,
            enforceSSL: true,  // deny non-TLS requests (DSR least-privilege)
            removalPolicy: cdk.RemovalPolicy.RETAIN,
            // Access logging + lifecycle (DSR: buckets must log access and have
            // a lifecycle policy). Abort incomplete multipart uploads and expire
            // old object versions to bound storage growth.
            serverAccessLogsBucket: logsBucket,
            serverAccessLogsPrefix: `${id}/`,
            lifecycleRules: [{ abortIncompleteMultipartUploadAfter: cdk.Duration.days(7) }],
          });

    const artifactsBucket = makeBucket('ArtifactsBucket', artifactsBucketName);
    const outputsBucket = makeBucket('OutputsBucket', outputsBucketName);

    // --- VPC (public + private subnets, NAT for ECR pulls) ---

    const vpc = new ec2.Vpc(this, 'Vpc', {
      vpcName: 'world-model-vpc',
      maxAzs: 2,
      natGateways: 1,
      subnetConfiguration: [
        { name: 'Public', subnetType: ec2.SubnetType.PUBLIC, cidrMask: 24 },
        { name: 'Private', subnetType: ec2.SubnetType.PRIVATE_WITH_EGRESS, cidrMask: 24 },
      ],
    });

    vpc.addGatewayEndpoint('S3Endpoint', { service: ec2.GatewayVpcEndpointAwsService.S3 });

    // --- Security Groups ---

    // The ALB accepts HTTP:80 and HTTPS:443. TLS is enabled per-deployment by
    // passing an ACM certificate ARN (-c certificateArn=...): the listener then
    // serves 443 and redirects 80 → 443 (M1). Without a cert it stays HTTP-only
    // for local/demo use.
    //
    // Ingress is restricted to `allowedCidr` (context) rather than 0.0.0.0/0.
    // deploy.sh defaults this to the deployer's own public IP (/32), so the
    // one-command flow still works while the endpoint is not world-open. Pass
    // `-c allowedCidr=0.0.0.0/0` to deliberately expose it to the internet.
    const allowedCidr =
      (this.node.tryGetContext('allowedCidr') as string | undefined) || '127.0.0.1/32';
    const albSg = new ec2.SecurityGroup(this, 'AlbSG', {
      vpc,
      securityGroupName: 'world-model-alb-sg',
      description: 'ALB for World Model Inference endpoints',
      allowAllOutbound: true,
    });
    albSg.addIngressRule(ec2.Peer.ipv4(allowedCidr), ec2.Port.tcp(80), `Allow HTTP from ${allowedCidr}`);
    albSg.addIngressRule(ec2.Peer.ipv4(allowedCidr), ec2.Port.tcp(443), `Allow HTTPS from ${allowedCidr}`);

    const sg = new ec2.SecurityGroup(this, 'InferenceSG', {
      vpc,
      securityGroupName: 'world-model-inference-sg',
      description: 'World Model Inference',
      allowAllOutbound: true,
    });
    sg.addIngressRule(sg, ec2.Port.tcp(8080), 'Allow inference traffic within SG');
    sg.addIngressRule(albSg, ec2.Port.tcp(8080), 'Allow ALB to reach inference port');

    // --- IAM: EC2 Role + Instance Profile ---

    // ARNs for the two project buckets — used to scope S3 access instead of the
    // account-wide AmazonS3FullAccess managed policy (H6).
    const bucketArns = [
      `arn:aws:s3:::${artifactsBucketName}`,
      `arn:aws:s3:::${artifactsBucketName}/*`,
      `arn:aws:s3:::${outputsBucketName}`,
      `arn:aws:s3:::${outputsBucketName}/*`,
    ];

    const ec2Role = new iam.Role(this, 'EC2Role', {
      roleName: `WorldModelEC2Role-${this.region}`,
      assumedBy: new iam.ServicePrincipal('ec2.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonSSMManagedInstanceCore'),
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEC2ContainerRegistryReadOnly'),
        // Lets the instance ship container/boot logs + metrics to CloudWatch
        // (least-privilege baseline expected of an EC2 role; EC2-002).
        iam.ManagedPolicy.fromAwsManagedPolicyName('CloudWatchAgentServerPolicy'),
      ],
    });

    // Scope S3 to the project buckets only (H6). The instance syncs model
    // weights from artifacts and writes generated outputs.
    ec2Role.addToPolicy(new iam.PolicyStatement({
      actions: ['s3:GetObject', 's3:PutObject', 's3:ListBucket'],
      resources: bucketArns,
    }));

    // Read the shared inference API token at boot (R1). Scoped to the single
    // SecureString parameter; the instance injects it into the container env.
    ec2Role.addToPolicy(new iam.PolicyStatement({
      sid: 'ReadApiToken',
      actions: ['ssm:GetParameter'],
      resources: [
        `arn:aws:ssm:${this.region}:${this.account}:parameter/world-model/security/api-token`,
      ],
    }));

    const ec2Profile = new iam.CfnInstanceProfile(this, 'EC2Profile', {
      instanceProfileName: `WorldModelEC2Profile-${this.region}`,
      roles: [ec2Role.roleName],
    });

    // --- IAM: SageMaker Execution Role ---

    // Replaces AmazonSageMakerFullAccess (M5) and AmazonS3FullAccess (H6) with
    // the minimal set a SageMaker execution role needs: pull the inference
    // image from ECR, write logs/metrics, and read/write the project buckets.
    const smRole = new iam.Role(this, 'SageMakerRole', {
      roleName: `WorldModelSageMakerRole-${this.region}`,
      assumedBy: new iam.ServicePrincipal('sagemaker.amazonaws.com'),
    });

    smRole.addToPolicy(new iam.PolicyStatement({
      sid: 'S3ProjectBuckets',
      actions: ['s3:GetObject', 's3:PutObject', 's3:ListBucket'],
      resources: bucketArns,
    }));
    smRole.addToPolicy(new iam.PolicyStatement({
      sid: 'EcrPull',
      actions: [
        'ecr:GetAuthorizationToken',
        'ecr:BatchCheckLayerAvailability',
        'ecr:GetDownloadUrlForLayer',
        'ecr:BatchGetImage',
      ],
      resources: ['*'],  // GetAuthorizationToken requires *; image pulls are gated by repo policy
    }));
    smRole.addToPolicy(new iam.PolicyStatement({
      sid: 'CloudWatchLogsAndMetrics',
      actions: [
        'logs:CreateLogGroup',
        'logs:CreateLogStream',
        'logs:PutLogEvents',
        'logs:DescribeLogStreams',
        'cloudwatch:PutMetricData',
      ],
      resources: ['*'],
    }));

    // --- CodeBuild ---

    // CodeBuild stages weights to the artifacts bucket and pushes images to
    // ECR. Scope S3 to the project buckets (H6) instead of AmazonS3FullAccess;
    // keep the managed ECR-PowerUser policy for image pushes and add explicit
    // CloudWatch Logs for build output.
    const buildRole = new iam.Role(this, 'CodeBuildRole', {
      roleName: `WorldModelCodeBuildRole-${this.region}`,
      assumedBy: new iam.ServicePrincipal('codebuild.amazonaws.com'),
      managedPolicies: [
        iam.ManagedPolicy.fromAwsManagedPolicyName('AmazonEC2ContainerRegistryPowerUser'),
      ],
    });

    buildRole.addToPolicy(new iam.PolicyStatement({
      sid: 'S3ProjectBuckets',
      actions: ['s3:GetObject', 's3:PutObject', 's3:ListBucket', 's3:DeleteObject'],
      resources: bucketArns,
    }));
    buildRole.addToPolicy(new iam.PolicyStatement({
      sid: 'CloudWatchLogs',
      actions: ['logs:CreateLogGroup', 'logs:CreateLogStream', 'logs:PutLogEvents'],
      resources: ['*'],
    }));

    const defaultSpec = codebuild.BuildSpec.fromObject({
      version: '0.2',
      phases: { build: { commands: ['echo "Use --buildspec-override"'] } },
    });

    // Weight staging downloads whole model repos to local disk before syncing to
    // S3. `hf download --local-dir` keeps the blob in the HF cache AND the
    // materialised copy, so peak disk is ~2x the repo size — lingbot-fast's
    // ~60 GB base model exhausted the LARGE container ("No space left on
    // device") partway through. X2_LARGE gives a much bigger volume; the timeout
    // is generous because a 60 GB pull plus S3 sync is slow.
    const weightsProject = new codebuild.Project(this, 'WeightsProject', {
      projectName: 'world-model-weights',
      role: buildRole,
      environment: {
        buildImage: codebuild.LinuxArmBuildImage.AMAZON_LINUX_2_STANDARD_3_0,
        computeType: codebuild.ComputeType.X2_LARGE,
      },
      timeout: cdk.Duration.hours(4),
      buildSpec: defaultSpec,
    });

    const imageProject = new codebuild.Project(this, 'ImageProject', {
      projectName: 'world-model-image',
      role: buildRole,
      environment: {
        buildImage: codebuild.LinuxBuildImage.STANDARD_7_0,
        computeType: codebuild.ComputeType.LARGE,
        privileged: true,
      },
      timeout: cdk.Duration.minutes(60),
      buildSpec: defaultSpec,
    });

    // --- SSM Parameters (foundation registry) ---

    new ssm.StringParameter(this, 'ParamVpcId', {
      parameterName: '/world-model/foundation/vpc-id',
      stringValue: vpc.vpcId,
    });

    new ssm.StringParameter(this, 'ParamPrivateSubnets', {
      parameterName: '/world-model/foundation/private-subnet-ids',
      stringValue: vpc.privateSubnets.map(s => s.subnetId).join(','),
    });

    new ssm.StringParameter(this, 'ParamPublicSubnets', {
      parameterName: '/world-model/foundation/public-subnet-ids',
      stringValue: vpc.publicSubnets.map(s => s.subnetId).join(','),
    });

    new ssm.StringParameter(this, 'ParamAlbSgId', {
      parameterName: '/world-model/foundation/alb-sg-id',
      stringValue: albSg.securityGroupId,
    });

    new ssm.StringParameter(this, 'ParamSgId', {
      parameterName: '/world-model/foundation/sg-id',
      stringValue: sg.securityGroupId,
    });

    new ssm.StringParameter(this, 'ParamEC2ProfileName', {
      parameterName: '/world-model/foundation/ec2-profile-name',
      stringValue: ec2Profile.instanceProfileName!,
    });

    new ssm.StringParameter(this, 'ParamEC2RoleName', {
      parameterName: '/world-model/foundation/ec2-role-name',
      stringValue: ec2Role.roleName,
    });

    new ssm.StringParameter(this, 'ParamSMRoleArn', {
      parameterName: '/world-model/foundation/sm-role-arn',
      stringValue: smRole.roleArn,
    });

    new ssm.StringParameter(this, 'ParamArtifactsBucket', {
      parameterName: '/world-model/foundation/artifacts-bucket',
      stringValue: artifactsBucket.bucketName,
    });

    new ssm.StringParameter(this, 'ParamOutputsBucket', {
      parameterName: '/world-model/foundation/outputs-bucket',
      stringValue: outputsBucket.bucketName,
    });

    new ssm.StringParameter(this, 'ParamWeightsProject', {
      parameterName: '/world-model/foundation/weights-project',
      stringValue: weightsProject.projectName,
    });

    new ssm.StringParameter(this, 'ParamImageProject', {
      parameterName: '/world-model/foundation/image-project',
      stringValue: imageProject.projectName,
    });
  }
}
