// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import * as cdk from 'aws-cdk-lib';
import * as autoscaling from 'aws-cdk-lib/aws-autoscaling';
import * as cloudfront from 'aws-cdk-lib/aws-cloudfront';
import * as origins from 'aws-cdk-lib/aws-cloudfront-origins';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as kms from 'aws-cdk-lib/aws-kms';
import * as route53 from 'aws-cdk-lib/aws-route53';
import * as secretsmanager from 'aws-cdk-lib/aws-secretsmanager';
import * as sns from 'aws-cdk-lib/aws-sns';
import * as subscriptions from 'aws-cdk-lib/aws-sns-subscriptions';
import * as cr from 'aws-cdk-lib/custom-resources';
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
   * How clients reach the endpoint over TLS (EDSR WM-002):
   *
   * - certificateArn + endpointDomain + hostedZoneId: the ALB serves HTTPS on
   *   443 under endpointDomain (a Route 53 alias), and HTTP:80 redirects.
   * - neither, the default: a CloudFront distribution serves HTTPS on its own
   *   *.cloudfront.net name, so no domain is needed. The ALB answers only
   *   requests carrying CloudFront's secret origin header, and allowedCidr is
   *   enforced at the edge, where the viewer's IP is still visible.
   * - allowPlaintext: the ALB serves plain HTTP:80 directly. Short-lived test
   *   endpoints only; deploy.sh requires ALLOW_PLAINTEXT=1 for it.
   */
  readonly certificateArn?: string;
  readonly endpointDomain?: string;
  readonly hostedZoneId?: string;
  readonly allowPlaintext?: boolean;
  /** Viewer CIDR CloudFront admits (IPv4). 0.0.0.0/0, or unset, admits everyone. */
  readonly allowedCidr?: string;
  /**
   * Cognito settings passed to the container as WORLD_MODEL_COGNITO_*.
   *
   * These replace the shared bearer token the instance used to fetch from SSM.
   * None of them is secret — a user pool ID, app client IDs and a scope name are
   * all public identifiers — so unlike the old token they can travel in user data
   * with nothing to decrypt at boot (as a base64 env file, never as shell text).
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
  /**
   * Email address subscribed to the ASG's scaling-event topic (launch,
   * terminate and their failures). Required: a topic with no subscriber alerts
   * no one. deploy.sh passes WORLD_MODEL_NOTIFICATION_EMAIL or the deployer's
   * git email.
   */
  readonly notificationEmail?: string;
}

/**
 * EC2-based inference with ALB for public access.
 * GPU instance in private subnet, ALB in public subnet forwards to port 8080.
 */
export class Ec2Inference extends Construct {
  constructor(scope: Construct, id: string, props: Ec2InferenceProps) {
    super(scope, id);

    // Fail before building anything: a scaling-event topic nobody subscribes to
    // alerts no one.
    if (!props.notificationEmail) {
      throw new Error(
        'EC2 deployments need a scaling-event subscriber: pass -c notificationEmail=<address> '
        + '(deploy.sh uses WORLD_MODEL_NOTIFICATION_EMAIL or your git email).',
      );
    }
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

    // Nothing from context, SSM or the manifest is interpolated into this root
    // script as shell text. The container's settings travel as a base64 env file
    // (base64 output is [A-Za-z0-9+/=], so it cannot leave its single quotes) and
    // docker reads --env-file values literally, never through a shell. Fn.base64
    // also encodes values CloudFormation resolves at deploy time, such as the
    // SSM-sourced bucket name, which no synth-time check can see (EDSR WM-004).
    const containerEnv: Record<string, string | undefined> = {
      WORLD_MODEL_AUTH_MODE: props.authMode,
      WORLD_MODEL_COGNITO_USER_POOL_ID: props.cognitoUserPoolId,
      WORLD_MODEL_COGNITO_CLIENT_IDS: props.cognitoClientIds,
      WORLD_MODEL_COGNITO_SCOPE: props.cognitoScope,
      WORLD_MODEL_ALLOWED_ORIGINS: props.allowedOrigins,
      WORLD_MODEL_RATE_LIMIT: props.rateLimit,
    };
    const envFile = Object.entries(containerEnv)
      .filter(([, v]) => v !== undefined && v !== '')
      .map(([k, v]) => `${k}=${v}\n`)
      .join('');

    const userData = ec2.UserData.forLinux();
    userData.addCommands(
      '#!/bin/bash',
      'set -ex',
      'exec > /var/log/world-model-startup.log 2>&1',
      '',
      'nvidia-smi',
      '',
      `aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ${account}.dkr.ecr.${region}.amazonaws.com`,
      `docker pull "${props.imageUri}"`,
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
      // The URI embeds the SSM-resolved artifacts bucket, so decode it and check
      // its shape at boot rather than trusting it as shell text.
      `MODEL_DATA_URI=$(echo '${cdk.Fn.base64(props.modelDataS3Uri)}' | base64 -d)`,
      '[[ "$MODEL_DATA_URI" =~ ^s3://[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]/[A-Za-z0-9_/-]*$ ]] || { echo "invalid model data URI" >&2; exit 1; }',
      `aws s3 sync "$MODEL_DATA_URI" "$CKPT_DIR/" --region ${region} || true`,
      '',
      `echo '${cdk.Fn.base64(envFile || '\n')}' | base64 -d > /etc/world-model.env`,
      'chmod 600 /etc/world-model.env',
      '',
      `docker run -d --gpus all --net host --ipc host \\`,
      `  -v "$CKPT_DIR":/opt/ml/model \\`,
      `  -e NVIDIA_VISIBLE_DEVICES=all \\`,
      `  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\`,
      `  --env-file /etc/world-model.env \\`,
      `  --name world-model \\`,
      `  "${props.imageUri}"`,
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

    // Scaling-event notifications. A GPU launch failing for capacity, or the
    // instance being replaced, is otherwise silent. The topic is encrypted with
    // a customer-managed key because the AWS-managed aws/sns key cannot grant
    // the Auto Scaling service-linked role, and publishes would fail silently.
    const notificationKey = new kms.Key(this, 'ScalingEventsKey', {
      description: `SNS encryption for world-model-${props.deploymentName} scaling events`,
      enableKeyRotation: true,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
    // Matched on aws:PrincipalArn rather than named as the principal: KMS rejects
    // a key policy whose principal does not exist yet, and the service-linked
    // role is only created with an account's first Auto Scaling group.
    //
    // The two actions are exactly what SNS documents for a publisher to an
    // encrypted topic: GenerateDataKey* to create the message data key, and
    // Decrypt, which SNS calls on the publisher's behalf to reuse a cached data
    // key. Without Decrypt, publishes to the topic fail. Resource "*" in a key
    // policy means this key only.
    // Reference: https://docs.aws.amazon.com/sns/latest/dg/sns-key-management.html#sns-what-permissions-for-sse
    notificationKey.addToResourcePolicy(new iam.PolicyStatement({
      sid: 'AllowAutoScalingToPublish',
      principals: [new iam.AnyPrincipal()],
      actions: ['kms:GenerateDataKey*', 'kms:Decrypt'],
      resources: ['*'],  // in a key policy, "*" means this key
      conditions: {
        ArnEquals: {
          'aws:PrincipalArn': `arn:aws:iam::${account}:role/aws-service-role/autoscaling.amazonaws.com/AWSServiceRoleForAutoScaling`,
        },
        StringEquals: { 'kms:CallerAccount': account },
      },
    }));
    const scalingEvents = new sns.Topic(this, 'ScalingEvents', {
      topicName: `world-model-${props.deploymentName}-scaling-events`,
      masterKey: notificationKey,
      enforceSSL: true,
    });
    scalingEvents.addSubscription(new subscriptions.EmailSubscription(props.notificationEmail));

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
      notificationConfigurations: [{
        topicArn: scalingEvents.topicArn,
        notificationTypes: [
          'autoscaling:EC2_INSTANCE_LAUNCH',
          'autoscaling:EC2_INSTANCE_LAUNCH_ERROR',
          'autoscaling:EC2_INSTANCE_TERMINATE',
          'autoscaling:EC2_INSTANCE_TERMINATE_ERROR',
        ],
      }],
      tags: [
        { key: 'Name', value: `world-model-${props.deploymentName}`, propagateAtLaunch: true },
        { key: 'Model', value: props.endpointId, propagateAtLaunch: true },
        { key: 'ManagedBy', value: 'world-model-cdk', propagateAtLaunch: true },
      ],
    });

    // --- ALB (public-facing) ---

    const publicSubnetIds = cdk.Fn.split(',', props.publicSubnetIds);
    const edge = !props.certificateArn && !props.allowPlaintext;
    if (props.certificateArn && (!props.endpointDomain || !props.hostedZoneId)) {
      throw new Error('certificateArn requires endpointDomain and hostedZoneId: an ACM certificate '
        + 'cannot cover the ALB\'s own AWS-assigned DNS name, so HTTPS needs a DNS name.');
    }

    // In edge mode CloudFront has to reach the ALB, and the foundation ALB group
    // only admits allowedCidr. A per-deployment group admits CloudFront's
    // origin-facing range on port 80 and nothing else.
    const albSecurityGroups = [props.albSecurityGroupId];
    if (edge) {
      // The managed prefix list's ID differs per region; look it up at deploy
      // time rather than at synth (synth stays offline, see the AMI note above).
      const prefixList = new cr.AwsCustomResource(this, 'CloudFrontOriginFacing', {
        onUpdate: {
          service: 'EC2',
          action: 'describeManagedPrefixLists',
          parameters: {
            Filters: [{ Name: 'prefix-list-name', Values: ['com.amazonaws.global.cloudfront.origin-facing'] }],
          },
          physicalResourceId: cr.PhysicalResourceId.of('cloudfront-origin-facing'),
          outputPaths: ['PrefixLists.0.PrefixListId'],
        },
        // DescribeManagedPrefixLists has no resource-level permissions, so it
        // can only be granted on "*". It is read-only.
        // Reference: https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazonec2.html
        policy: cr.AwsCustomResourcePolicy.fromSdkCalls({ resources: cr.AwsCustomResourcePolicy.ANY_RESOURCE }),
        installLatestAwsSdk: false,
      });
      const edgeSg = new ec2.CfnSecurityGroup(this, 'EdgeSG', {
        vpcId: props.vpcId,
        groupDescription: `CloudFront to the ${props.deploymentName} ALB`,
        securityGroupIngress: [{
          ipProtocol: 'tcp',
          fromPort: 80,
          toPort: 80,
          sourcePrefixListId: prefixList.getResponseField('PrefixLists.0.PrefixListId'),
          description: 'CloudFront origin-facing range',
        }],
      });
      albSecurityGroups.push(edgeSg.attrGroupId);
    }

    const alb = new elbv2.CfnLoadBalancer(this, 'ALB', {
      name: `wm-${props.deploymentName}`.slice(0, 32),
      scheme: 'internet-facing',
      type: 'application',
      securityGroups: albSecurityGroups,
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

    let endpointUrl: string;
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
      new route53.CfnRecordSet(this, 'EndpointAlias', {
        hostedZoneId: props.hostedZoneId!,
        name: props.endpointDomain!,
        type: 'A',
        aliasTarget: { dnsName: alb.attrDnsName, hostedZoneId: alb.attrCanonicalHostedZoneId },
      });
      endpointUrl = `https://${props.endpointDomain}`;
    } else if (edge) {
      // CloudFront proves itself to the ALB with a header only it knows. The ALB
      // admits CloudFront's range on 80 (above) and the deployer's CIDR through
      // the foundation group; either way a request without the header gets 403,
      // so the plaintext ALB is not a way around the edge.
      const originVerify = new secretsmanager.Secret(this, 'OriginVerify', {
        description: `CloudFront-to-ALB origin header for ${props.deploymentName}`,
        generateSecretString: { passwordLength: 48, excludePunctuation: true },
      });
      const verify = originVerify.secretValue.unsafeUnwrap();  // dynamic reference, resolved by CloudFormation
      const listener = new elbv2.CfnListener(this, 'HttpListener', {
        loadBalancerArn: alb.ref,
        port: 80,
        protocol: 'HTTP',
        defaultActions: [{
          type: 'fixed-response',
          fixedResponseConfig: { statusCode: '403', contentType: 'text/plain', messageBody: 'Forbidden' },
        }],
      });
      new elbv2.CfnListenerRule(this, 'FromEdge', {
        listenerArn: listener.ref,
        priority: 1,
        conditions: [{ field: 'http-header', httpHeaderConfig: { httpHeaderName: 'X-Origin-Verify', values: [verify] } }],
        actions: [{ type: 'forward', targetGroupArn: targetGroup.ref }],
      });

      // Behind CloudFront the ALB sees CloudFront's addresses, not the client's,
      // so the deployer-IP allowlist moves to the edge. IPv6 is off so every
      // viewer address is IPv4 and the check has no gap.
      const viewerAllowlist = edgeAllowlistFunction(this, props.allowedCidr);
      const distribution = new cloudfront.Distribution(this, 'Edge', {
        comment: `world-model-${props.deploymentName}`,
        enableIpv6: false,
        priceClass: cloudfront.PriceClass.PRICE_CLASS_100,
        defaultBehavior: {
          origin: new origins.HttpOrigin(alb.attrDnsName, {
            protocolPolicy: cloudfront.OriginProtocolPolicy.HTTP_ONLY,
            customHeaders: { 'X-Origin-Verify': verify },
            readTimeout: cdk.Duration.seconds(60),
          }),
          // HTTPS only, not redirect: a redirect would already have carried the
          // bearer token across the network in plaintext.
          viewerProtocolPolicy: cloudfront.ViewerProtocolPolicy.HTTPS_ONLY,
          allowedMethods: cloudfront.AllowedMethods.ALLOW_ALL,
          // Every response is per-caller; nothing may be cached. ALL_VIEWER
          // forwards Authorization and the WebSocket upgrade headers.
          cachePolicy: cloudfront.CachePolicy.CACHING_DISABLED,
          originRequestPolicy: cloudfront.OriginRequestPolicy.ALL_VIEWER,
          functionAssociations: viewerAllowlist
            ? [{ function: viewerAllowlist, eventType: cloudfront.FunctionEventType.VIEWER_REQUEST }]
            : undefined,
        },
      });
      endpointUrl = `https://${distribution.distributionDomainName}`;
    } else {
      new elbv2.CfnListener(this, 'HttpListener', {
        loadBalancerArn: alb.ref,
        port: 80,
        protocol: 'HTTP',
        defaultActions: [{ type: 'forward', targetGroupArn: targetGroup.ref }],
      });
      endpointUrl = cdk.Fn.join('', ['http://', alb.attrDnsName]);
    }

    // Attach the ASG to the target group
    asg.targetGroupArns = [targetGroup.ref];

    // --- Outputs ---

    new cdk.CfnOutput(this, 'ASGName', {
      value: asg.autoScalingGroupName!,
      description: `ASG for ${props.deploymentName}`,
    });

    new cdk.CfnOutput(this, 'EndpointUrl', {
      value: endpointUrl,
      description: `Public endpoint URL for ${props.deploymentName}`,
    });
  }
}

/**
 * A viewer-request function admitting only viewers inside `cidr`, or undefined
 * when everyone is admitted. The CIDR is validated by the CDK app and reduced to
 * two integers here, so no context text reaches the function's source.
 */
function edgeAllowlistFunction(scope: Construct, cidr?: string): cloudfront.Function | undefined {
  if (!cidr || cidr === '0.0.0.0/0') return undefined;
  const m = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\/(\d{1,2})$/.exec(cidr);
  if (!m || m.slice(1, 5).some((o) => Number(o) > 255) || Number(m[5]) > 32) {
    throw new Error(`allowedCidr must be an IPv4 CIDR, got ${JSON.stringify(cidr)}`);
  }
  const bits = Number(m[5]);
  const mask = bits === 0 ? 0 : (0xffffffff << (32 - bits)) >>> 0;
  const base = ((((Number(m[1]) << 24) | (Number(m[2]) << 16) | (Number(m[3]) << 8) | Number(m[4])) >>> 0) & mask) >>> 0;
  return new cloudfront.Function(scope, 'ViewerAllowlist', {
    runtime: cloudfront.FunctionRuntime.JS_2_0,
    comment: `Admit viewers in ${cidr}`,
    code: cloudfront.FunctionCode.fromInline(edgeAllowlistSource(base, mask)),
  });
}

/** The viewer-request handler: IPv4 `viewer.ip` masked and compared to BASE. */
function edgeAllowlistSource(base: number, mask: number): string {
  return `var BASE = ${base}, MASK = ${mask};
function handler(event) {
  var p = event.viewer.ip.split('.');
  if (p.length === 4) {
    var n = ((p[0] << 24) | (p[1] << 16) | (p[2] << 8) | p[3]) >>> 0;
    if (((n & MASK) >>> 0) === BASE) return event.request;
  }
  return { statusCode: 403, statusDescription: 'Forbidden' };
}`;
}
