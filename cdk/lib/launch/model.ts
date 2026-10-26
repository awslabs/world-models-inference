// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import * as cdk from 'aws-cdk-lib';
import * as cognito from 'aws-cdk-lib/aws-cognito';
import * as s3 from 'aws-cdk-lib/aws-s3';
import * as s3assets from 'aws-cdk-lib/aws-s3-assets';
import * as s3deploy from 'aws-cdk-lib/aws-s3-deployment';
import * as cr from 'aws-cdk-lib/custom-resources';
import { Construct } from 'constructs';
import { Ec2Inference } from '../constructs/ec2';
import { FoundationRefs } from '../shared';
import { LaunchBuild } from './build';
import { LaunchCartridge } from './catalogue';

export interface LaunchModelProps extends cdk.NestedStackProps {
  readonly model: string;
  readonly cartridge: LaunchCartridge;
  readonly foundation: FoundationRefs;
  /** 'model-default' or an EC2 instance type (an AllowedValues parameter). */
  readonly instanceType: string;
  readonly adminEmail: string;
  readonly allowedCidr: string;
  /** Capacity Block ID or ''. */
  readonly capacityBlockId: string;
  /** Directory holding inference/ — zipped as the image build's source. */
  readonly imageSource: string;
  /** frontend/dist, built by scripts/publish.py. */
  readonly uiDist: string;
}

/**
 * One cartridge, end to end: build it, run it on EC2 behind CloudFront, and
 * serve the UI from the same distribution with Cognito Hosted UI sign-in.
 *
 * The parent creates one of these per cartridge, each under a condition on the
 * Model parameter, so only the chosen one is deployed.
 */
export class LaunchModelStack extends cdk.NestedStack {
  /** The distribution's domain: the UI is https://<domain>/. */
  public readonly distributionDomain: string;
  public readonly asgName: string;
  public readonly buildStateMachineName: string;

  constructor(scope: Construct, id: string, props: LaunchModelProps) {
    super(scope, id, props);
    const { model, cartridge, foundation } = props;

    const useDefault = new cdk.CfnCondition(this, 'UseModelDefaultInstance', {
      expression: cdk.Fn.conditionEquals(props.instanceType, 'model-default'),
    });
    const instanceType = cdk.Fn.conditionIf(useDefault.logicalId, cartridge.ec2Instance, props.instanceType).toString();
    const hasCapacity = new cdk.CfnCondition(this, 'HasCapacityBlock', {
      expression: cdk.Fn.conditionNot(cdk.Fn.conditionEquals(props.capacityBlockId, '')),
    });

    // --- Build ---------------------------------------------------------------
    const imageSource = new s3assets.Asset(this, 'ImageSource', {
      path: props.imageSource,
      exclude: ['**/__pycache__', '**/*.pyc', '**/.DS_Store'],
    });
    const build = new LaunchBuild(this, 'Build', {
      model,
      cartridge,
      artifactsBucketName: foundation.artifactsBucketName,
      weightsProjectName: foundation.weightsProjectName,
      imageProjectName: foundation.imageProjectName,
      imageSource,
      capacityBlockId: props.capacityBlockId,
    });

    // --- Capacity Block: its one AZ decides the subnet ----------------------
    // Looked up rather than asked for: the subnets do not exist until the
    // foundation does, so the person launching cannot know their IDs.
    const blockAz = new cr.AwsCustomResource(this, 'CapacityBlockAz', {
      onUpdate: {
        service: 'EC2',
        action: 'describeCapacityReservations',
        parameters: { CapacityReservationIds: [props.capacityBlockId] },
        physicalResourceId: cr.PhysicalResourceId.of(`${model}-capacity-block-az`),
        outputPaths: ['CapacityReservations.0.AvailabilityZone'],
      },
      // Describe* calls have no resource-level permissions.
      // Reference: https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazonec2.html
      policy: cr.AwsCustomResourcePolicy.fromSdkCalls({ resources: cr.AwsCustomResourcePolicy.ANY_RESOURCE }),
      installLatestAwsSdk: false,
    });
    const blockSubnet = new cr.AwsCustomResource(this, 'CapacityBlockSubnet', {
      onUpdate: {
        service: 'EC2',
        action: 'describeSubnets',
        parameters: {
          Filters: [
            { Name: 'vpc-id', Values: [foundation.vpcId] },
            { Name: 'availability-zone', Values: [blockAz.getResponseField('CapacityReservations.0.AvailabilityZone')] },
            { Name: 'tag:aws-cdk:subnet-type', Values: ['Private'] },
          ],
        },
        physicalResourceId: cr.PhysicalResourceId.of(`${model}-capacity-block-subnet`),
        outputPaths: ['Subnets.0.SubnetId'],
      },
      policy: cr.AwsCustomResourcePolicy.fromSdkCalls({ resources: cr.AwsCustomResourcePolicy.ANY_RESOURCE }),
      installLatestAwsSdk: false,
    });
    for (const c of [blockAz, blockSubnet]) {
      (c.node.findChild('Resource').node.defaultChild as cdk.CfnResource).cfnOptions.condition = hasCapacity;
    }

    // --- UI bucket, served by the endpoint's own distribution ---------------
    const uiBucket = new s3.Bucket(this, 'UiBucket', {
      blockPublicAccess: s3.BlockPublicAccess.BLOCK_ALL,
      encryption: s3.BucketEncryption.S3_MANAGED,
      enforceSSL: true,
      // Only the built UI lives here, re-deployable from the template's assets.
      removalPolicy: cdk.RemovalPolicy.DESTROY,
      autoDeleteObjects: true,
    });

    // The container's allowlists name the distribution and the UI's client,
    // both created further down, so they are resolved lazily.
    let hostedClient: cognito.CfnUserPoolClient | undefined;
    let endpoint: Ec2Inference | undefined;
    const distributionUrl = () => `https://${endpoint!.distribution!.distributionDomainName}`;

    endpoint = new Ec2Inference(this, 'EC2', {
      endpointId: model,
      deploymentName: model,
      instanceType,
      volumeSize: cartridge.ec2VolumeSize,
      securityGroupId: foundation.securityGroupId,
      albSecurityGroupId: foundation.albSecurityGroupId,
      subnetIds: foundation.privateSubnetIds,
      publicSubnetIds: foundation.publicSubnetIds,
      vpcId: foundation.vpcId,
      roleName: foundation.ec2RoleName,
      imageUri: `${this.account}.dkr.ecr.${this.region}.amazonaws.com/world-model-${model}@${build.imageDigest}`,
      modelDataS3Uri: `s3://${foundation.artifactsBucketName}/${cartridge.modelDataPrefix}`,
      capacityReservationId: props.capacityBlockId,
      capacitySubnetIds: blockSubnet.getResponseField('Subnets.0.SubnetId'),
      capacityCondition: hasCapacity,
      allowedCidr: props.allowedCidr,
      authMode: 'cognito',
      cognitoUserPoolId: foundation.userPoolId,
      cognitoClientIds: cdk.Lazy.string({ produce: () => `${foundation.cognitoClientIds},${hostedClient!.ref}` }),
      cognitoScope: foundation.cognitoScope,
      allowedOrigins: cdk.Lazy.string({ produce: distributionUrl }),
      notificationEmail: props.adminEmail,
      uiBucket,
      setDesiredCapacityOnCreate: true,
    });

    // --- Hosted UI sign-in ---------------------------------------------------
    // A client per deployment, because its callback is this distribution. The
    // foundation's browser client keeps serving ./deploy.sh ui on localhost.
    hostedClient = new cognito.CfnUserPoolClient(this, 'HostedUiClient', {
      userPoolId: foundation.userPoolId,
      clientName: `world-model-${model}-hosted-ui`,
      generateSecret: false,  // a browser client must not hold a secret
      allowedOAuthFlowsUserPoolClient: true,
      allowedOAuthFlows: ['code'],
      allowedOAuthScopes: ['openid', foundation.cognitoScope],
      callbackUrLs: [`${distributionUrl()}/`],
      logoutUrLs: [`${distributionUrl()}/`],
      supportedIdentityProviders: ['COGNITO'],
      explicitAuthFlows: ['ALLOW_REFRESH_TOKEN_AUTH'],
      accessTokenValidity: 60,
      idTokenValidity: 60,
      refreshTokenValidity: 1,
      tokenValidityUnits: { accessToken: 'minutes', idToken: 'minutes', refreshToken: 'days' },
      preventUserExistenceErrors: 'ENABLED',
      enableTokenRevocation: true,
    });

    // Every value here is an AWS-generated ID or matches a parameter's
    // AllowedPattern, so none can leave its quotes.
    const config = `window.APP_CONFIG = {
  region: '${this.region}',
  userPoolId: '${foundation.userPoolId}',
  userPoolClientId: '${hostedClient.ref}',
  websocketUrl: '',
  lingbotApiUrl: '',
  environment: 'launch',
  cognitoDomain: '${foundation.cognitoDomain}',
  redirectSignIn: '${distributionUrl()}/',
  redirectSignOut: '${distributionUrl()}/',
  identityProvider: '',
  demoMode: false,
  deployedModel: '${model}',
  ui: 'lingbot',
  auth: 'hosted',
};
`;
    new s3deploy.BucketDeployment(this, 'Ui', {
      sources: [s3deploy.Source.asset(props.uiDist), s3deploy.Source.data('config.js', config)],
      destinationBucket: uiBucket,
      distribution: endpoint.distribution,
      distributionPaths: ['/index.html', '/config.js'],
      memoryLimit: 512,
    });

    this.distributionDomain = endpoint.distribution!.distributionDomainName;
    this.asgName = endpoint.autoScalingGroupName;
    this.buildStateMachineName = build.stateMachine.stateMachineName;
  }
}
