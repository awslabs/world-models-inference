// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import * as cdk from 'aws-cdk-lib';
import * as codebuild from 'aws-cdk-lib/aws-codebuild';
import * as cognito from 'aws-cdk-lib/aws-cognito';
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
      // GPU capacity is per-AZ, so more AZs means more chances of a p5 launch
      // succeeding: we hit InsufficientInstanceCapacity in both 2a and 2b while
      // AWS reported capacity in 2c and 2d, with no subnet there to use.
      //
      // Raising this on an existing VPC does NOT work: CDK re-slices the /16
      // from the base, so the new public subnets are handed 10.0.2.0/24 and
      // 10.0.3.0/24, which the current private subnets already own, and the
      // update fails with "CIDR conflicts with another subnet". Widening the AZ
      // span needs a fresh VPC (new CIDR or a rebuilt foundation), so it is a
      // deliberate migration rather than a config tweak. Reserved capacity is
      // the better answer for p5 in any case.
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

    // No ssm:GetParameter grant here any more. The instance used to read a shared
    // SecureString API token at boot; with Cognito it receives only public
    // identifiers (pool ID, client IDs, scope) in user data and verifies tokens
    // against the pool's published JWKS. One fewer permission, and no secret on
    // the instance to read or leak.

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

    // --- Cognito (authentication) ---
    //
    // The inference API used to authenticate with a shared static bearer token
    // that deploy.sh generated into SSM and the app string-compared. That is a
    // bespoke credential scheme — "a Cognito alternative" in AWS security-review
    // terms — which makes a reusable solution ineligible for an Enhanced DSR and
    // forces a full AppSec review. It was also a single long-lived secret shared
    // by every caller, with no expiry and no way to revoke one client.
    //
    // This pool replaces it. The app verifies pool-issued access tokens
    // (inference/lib/cognito.py) and owns no credential scheme of its own.
    //
    // It lives in the foundation stack, not per-model, so every cartridge shares
    // one identity boundary and adding a cartridge does not mint a new pool.
    const userPool = new cognito.UserPool(this, 'UserPool', {
      userPoolName: 'world-model-inference',
      selfSignUpEnabled: false, // operators are invited; this is not a public app
      signInAliases: { email: true },
      passwordPolicy: {
        minLength: 12,
        requireLowercase: true,
        requireUppercase: true,
        requireDigits: true,
        requireSymbols: true,
      },
      accountRecovery: cognito.AccountRecovery.EMAIL_ONLY,
      // RETAIN: destroying the pool would delete every operator account and
      // silently break running deployments that still hold its tokens.
      removalPolicy: cdk.RemovalPolicy.RETAIN,
      // Threat protection in audit mode: record risk signals (impossible travel,
      // credential stuffing) without blocking a booth demo on a false positive.
      standardThreatProtectionMode: cognito.StandardThreatProtectionMode.AUDIT_ONLY,
    });

    // A resource server gives us a scope to authorise on, so a token proves
    // "may invoke inference" rather than merely "signed in somewhere".
    const resourceServer = userPool.addResourceServer('ResourceServer', {
      identifier: 'world-model',
      userPoolResourceServerName: 'World Model Inference API',
      scopes: [
        new cognito.ResourceServerScope({
          scopeName: 'invoke',
          scopeDescription: 'Run inference against a deployed world model',
        }),
      ],
    });
    const invokeScope = cognito.OAuthScope.resourceServer(resourceServer, {
      scopeName: 'invoke',
      scopeDescription: 'Run inference against a deployed world model',
    });

    // A Hosted UI domain is required for the authorization-code and
    // client-credentials flows below. The prefix must be globally unique, so it
    // is account/region qualified.
    const domainPrefix = `world-model-${this.account}-${this.region}`;
    const userPoolDomain = userPool.addDomain('UserPoolDomain', {
      cognitoDomain: { domainPrefix },
    });

    // Browser client: authorization code + PKCE, no client secret. A public
    // client must never hold a secret — it ships to the browser.
    const callbackUrls = (this.node.tryGetContext('callbackUrls') as string | undefined)
      ?.split(',')
      .map(u => u.trim())
      .filter(Boolean) ?? ['http://localhost:3000/'];
    const browserClient = userPool.addClient('BrowserClient', {
      userPoolClientName: 'world-model-browser',
      generateSecret: false,
      authFlows: { userSrp: true },
      oAuth: {
        flows: { authorizationCodeGrant: true },
        scopes: [cognito.OAuthScope.OPENID, invokeScope],
        callbackUrls,
        logoutUrls: callbackUrls,
      },
      // Short-lived by design. The token travels in a WebSocket query string on
      // the real-time path, which lands in ALB access logs; an hour of exposure
      // is a different proposition from the previous never-expiring secret.
      accessTokenValidity: cdk.Duration.hours(1),
      idTokenValidity: cdk.Duration.hours(1),
      refreshTokenValidity: cdk.Duration.days(1),
      preventUserExistenceErrors: true,
      enableTokenRevocation: true,
    });

    // Machine client: client credentials, no user involved. This is what
    // `deploy.sh bench` and CI use, and it is why automation no longer needs a
    // shared human-held secret.
    const machineClient = userPool.addClient('MachineClient', {
      userPoolClientName: 'world-model-machine',
      generateSecret: true,
      oAuth: {
        flows: { clientCredentials: true },
        scopes: [invokeScope],
      },
      accessTokenValidity: cdk.Duration.hours(1),
      enableTokenRevocation: true,
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

    // Cognito registry. deploy.sh reads these and passes them to the container as
    // WORLD_MODEL_COGNITO_* env vars. The machine client's SECRET is deliberately
    // absent: it is readable from the Cognito API by an authorised caller and must
    // not be copied into a plaintext SSM parameter.
    new ssm.StringParameter(this, 'ParamUserPoolId', {
      parameterName: '/world-model/foundation/cognito-user-pool-id',
      stringValue: userPool.userPoolId,
    });

    new ssm.StringParameter(this, 'ParamCognitoClientIds', {
      parameterName: '/world-model/foundation/cognito-client-ids',
      stringValue: `${browserClient.userPoolClientId},${machineClient.userPoolClientId}`,
    });

    new ssm.StringParameter(this, 'ParamCognitoBrowserClientId', {
      parameterName: '/world-model/foundation/cognito-browser-client-id',
      stringValue: browserClient.userPoolClientId,
    });

    new ssm.StringParameter(this, 'ParamCognitoMachineClientId', {
      parameterName: '/world-model/foundation/cognito-machine-client-id',
      stringValue: machineClient.userPoolClientId,
    });

    new ssm.StringParameter(this, 'ParamCognitoScope', {
      parameterName: '/world-model/foundation/cognito-scope',
      stringValue: 'world-model/invoke',
    });

    new ssm.StringParameter(this, 'ParamCognitoDomain', {
      parameterName: '/world-model/foundation/cognito-domain',
      stringValue: userPoolDomain.baseUrl(),
    });
  }
}
