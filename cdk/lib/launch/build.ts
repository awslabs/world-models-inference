// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import * as crypto from 'crypto';
import * as cdk from 'aws-cdk-lib';
import * as iam from 'aws-cdk-lib/aws-iam';
import * as lambda from 'aws-cdk-lib/aws-lambda';
import * as logs from 'aws-cdk-lib/aws-logs';
import * as s3assets from 'aws-cdk-lib/aws-s3-assets';
import * as sfn from 'aws-cdk-lib/aws-stepfunctions';
import * as tasks from 'aws-cdk-lib/aws-stepfunctions-tasks';
import * as cr from 'aws-cdk-lib/custom-resources';
import { Construct } from 'constructs';
import { LaunchCartridge } from './catalogue';

export interface LaunchBuildProps {
  readonly model: string;
  readonly cartridge: LaunchCartridge;
  readonly artifactsBucketName: string;
  readonly weightsProjectName: string;
  readonly imageProjectName: string;
  /** Zip of the repository's inference/ tree (paths start with inference/). */
  readonly imageSource: s3assets.Asset;
  /** Capacity Block ID, or '' for none. Checked before anything is built. */
  readonly capacityBlockId: string;
}

/**
 * Stage the weights and build the image inside the account, then hand the
 * image digest to CloudFormation.
 *
 * What deploy.sh does from a laptop happens here in a Step Functions state
 * machine: the same CodeBuild projects, the same buildspecs, the same
 * immutable-tag check. CloudFormation waits on a WaitCondition (up to 12 hours;
 * a custom resource gives up after one, and a large model's weights take longer)
 * which the state machine signals with success or with the failure. The digest
 * is then read from ECR, not from the signal: whoever holds the presigned
 * signal URL can choose what a signal says, but not what the repository holds.
 */
export class LaunchBuild extends Construct {
  /** sha256:... of the built (or already present) image. */
  public readonly imageDigest: string;
  public readonly stateMachine: sfn.StateMachine;

  constructor(scope: Construct, id: string, props: LaunchBuildProps) {
    super(scope, id);
    const stack = cdk.Stack.of(this);
    const { model, cartridge } = props;
    const repoName = `world-model-${model}`;
    const repoArn = `arn:${cdk.Aws.PARTITION}:ecr:${stack.region}:${stack.account}:repository/${repoName}`;
    const artifactsArn = `arn:${cdk.Aws.PARTITION}:s3:::${props.artifactsBucketName}`;
    const sourceKey = `_builds/${model}/source.zip`;
    const codebuildArn = (project: string) =>
      `arn:${cdk.Aws.PARTITION}:codebuild:${stack.region}:${stack.account}:project/${project}`;

    // Signals the WaitCondition. The handle is a presigned S3 URL; the function
    // will PUT only to an AWS endpoint over HTTPS.
    const signal = new lambda.Function(this, 'Signal', {
      runtime: lambda.Runtime.PYTHON_3_12,
      handler: 'index.handler',
      timeout: cdk.Duration.seconds(60),
      description: `Reports the ${model} launch build to CloudFormation`,
      code: lambda.Code.fromInline(SIGNAL_SOURCE),
    });

    // --- Capacity Block: fail before hours of building, not after ---------
    // `scheduled` is accepted: a block bought for later may well have started
    // by the time a large model's weights are staged.
    const capacityOk = new sfn.Pass(this, 'CapacityOk');
    const describeBlock = new tasks.CallAwsService(this, 'DescribeCapacityBlock', {
      service: 'ec2',
      action: 'describeCapacityReservations',
      parameters: { CapacityReservationIds: sfn.JsonPath.array(sfn.JsonPath.stringAt('$.capacityBlockId')) },
      // DescribeCapacityReservations has no resource-level permissions.
      // Reference: https://docs.aws.amazon.com/service-authorization/latest/reference/list_amazonec2.html
      iamResources: ['*'],
      iamAction: 'ec2:DescribeCapacityReservations',
      resultSelector: { 'state.$': '$.CapacityReservations[0].State' },
      resultPath: '$.capacity',
    });
    const checkCapacity = new sfn.Choice(this, 'CapacityBlockGiven')
      .when(sfn.Condition.stringEquals('$.capacityBlockId', ''), capacityOk)
      .otherwise(describeBlock.next(new sfn.Choice(this, 'CapacityBlockUsable')
        .when(sfn.Condition.or(
          sfn.Condition.stringEquals('$.capacity.state', 'active'),
          sfn.Condition.stringEquals('$.capacity.state', 'scheduled'),
        ), capacityOk)
        .otherwise(new sfn.Fail(this, 'CapacityBlockNotUsable', {
          error: 'CapacityBlockNotUsable',
          cause: 'The Capacity Block is expired, cancelled or not found: every instance launch would fail.',
        }))));

    // --- Weights: one CodeBuild run per pinned revision, skipped once staged -
    const startBuild = (stateId: string, parameters: Record<string, unknown>) =>
      new sfn.CustomState(this, stateId, {
        stateJson: {
          Type: 'Task',
          Resource: `arn:${cdk.Aws.PARTITION}:states:::codebuild:startBuild.sync`,
          Parameters: parameters,
          ResultPath: null,
        },
      });

    let weights: sfn.Chain = sfn.Chain.start(new sfn.Pass(this, 'WeightsStart'));
    cartridge.weights.forEach((w, i) => {
      const staged = new tasks.CallAwsService(this, `WeightsStaged${i}`, {
        service: 's3',
        action: 'listObjectsV2',
        parameters: { Bucket: props.artifactsBucketName, Prefix: w.marker, MaxKeys: 1 },
        iamAction: 's3:ListBucket',
        iamResources: [artifactsArn],
        resultSelector: { 'count.$': '$.KeyCount' },
        resultPath: '$.staged',
      });
      const done = new sfn.Pass(this, `WeightsReady${i}`);
      const stage = startBuild(`StageWeights${i}`, {
        ProjectName: props.weightsProjectName,
        BuildspecOverride: w.buildspec,
        EnvironmentVariablesOverride: [
          { Name: 'ARTIFACTS_BUCKET', Type: 'PLAINTEXT', Value: props.artifactsBucketName },
        ],
      });
      stage.next(done);
      staged.next(new sfn.Choice(this, `WeightsAlreadyStaged${i}`)
        .when(sfn.Condition.numberGreaterThan('$.staged.count', 0), done)
        .otherwise(stage));
      weights = weights.next(sfn.Chain.custom(staged, [done], done));
    });

    // --- Image: reuse the immutable tag if it exists, else build it ----------
    const createRepo = new tasks.CallAwsService(this, 'CreateRepository', {
      service: 'ecr',
      action: 'createRepository',
      // IMMUTABLE rather than build-image.py's IMMUTABLE_WITH_EXCLUSION: the
      // exclusion filter is newer than this integration's SDK may be. The only
      // cost is the :buildcache tag, whose push the buildspec already tolerates.
      parameters: {
        RepositoryName: repoName,
        ImageScanningConfiguration: { ScanOnPush: true },
        ImageTagMutability: 'IMMUTABLE',
      },
      iamAction: 'ecr:CreateRepository',
      iamResources: [repoArn],
      resultPath: sfn.JsonPath.DISCARD,
    });
    const imageReady = new sfn.Pass(this, 'ImageReady');
    const findImage = (stateId: string) => new tasks.CallAwsService(this, stateId, {
      service: 'ecr',
      action: 'describeImages',
      parameters: { RepositoryName: repoName, ImageIds: [{ ImageTag: cartridge.version }] },
      iamAction: 'ecr:DescribeImages',
      iamResources: [repoArn],
      resultSelector: { 'digest.$': '$.ImageDetails[0].ImageDigest' },
    });
    const copySource = new tasks.CallAwsService(this, 'CopyBuildSource', {
      service: 's3',
      action: 'copyObject',
      parameters: {
        Bucket: props.artifactsBucketName,
        Key: sourceKey,
        CopySource: `${props.imageSource.s3BucketName}/${props.imageSource.s3ObjectKey}`,
      },
      iamAction: 's3:PutObject',
      iamResources: [`${artifactsArn}/${sourceKey}`],
      additionalIamStatements: [new iam.PolicyStatement({
        actions: ['s3:GetObject'],
        resources: [`arn:${cdk.Aws.PARTITION}:s3:::${props.imageSource.s3BucketName}/${props.imageSource.s3ObjectKey}`],
      })],
      resultPath: sfn.JsonPath.DISCARD,
    });
    const buildImage = startBuild('BuildImage', {
      ProjectName: props.imageProjectName,
      BuildspecOverride: cartridge.image,
      SourceTypeOverride: 'S3',
      SourceLocationOverride: `${props.artifactsBucketName}/${sourceKey}`,
      EnvironmentVariablesOverride: [{
        Name: 'WM_REGISTRY',
        Type: 'PLAINTEXT',
        Value: `${stack.account}.dkr.ecr.${stack.region}.${cdk.Aws.URL_SUFFIX}`,
      }],
    });
    const existing = findImage('FindImage');
    existing.addCatch(copySource, { errors: ['Ecr.ImageNotFoundException'], resultPath: sfn.JsonPath.DISCARD });
    existing.next(imageReady);
    copySource.next(buildImage).next(findImage('FindBuiltImage')).next(imageReady);
    createRepo.addCatch(existing, { errors: ['Ecr.RepositoryAlreadyExistsException'], resultPath: sfn.JsonPath.DISCARD });
    createRepo.next(existing);

    // --- Report -------------------------------------------------------------
    const reportFailure = new tasks.LambdaInvoke(this, 'ReportFailure', {
      lambdaFunction: signal,
      payload: sfn.TaskInput.fromObject({
        status: 'FAILURE',
        url: sfn.JsonPath.stringAt('$$.Execution.Input.handleUrl'),
        error: sfn.JsonPath.stringAt('$.Error'),
        cause: sfn.JsonPath.stringAt('$.Cause'),
      }),
      resultPath: sfn.JsonPath.DISCARD,
    }).next(new sfn.Fail(this, 'BuildFailed'));
    const build = new sfn.Parallel(this, 'StageAndBuild', { resultPath: sfn.JsonPath.DISCARD })
      .branch(weights)
      .branch(createRepo);
    const reportSuccess = new tasks.LambdaInvoke(this, 'ReportSuccess', {
      lambdaFunction: signal,
      payload: sfn.TaskInput.fromObject({
        status: 'SUCCESS',
        url: sfn.JsonPath.stringAt('$$.Execution.Input.handleUrl'),
      }),
      resultPath: sfn.JsonPath.DISCARD,
    });
    capacityOk.next(build).next(reportSuccess);

    // Any failure — a dead Capacity Block, a failed build — must reach
    // CloudFormation, not leave it waiting out the full timeout.
    const definition = new sfn.Parallel(this, 'Launch')
      .branch(checkCapacity)
      .addCatch(reportFailure);

    const logGroup = new logs.LogGroup(this, 'Logs', {
      retention: logs.RetentionDays.ONE_MONTH,
      removalPolicy: cdk.RemovalPolicy.DESTROY,
    });
    this.stateMachine = new sfn.StateMachine(this, 'StateMachine', {
      stateMachineName: `world-model-${model}-launch-build`,
      definitionBody: sfn.DefinitionBody.fromChainable(definition),
      timeout: cdk.Duration.hours(12),
      // ERROR only, without execution data: the input carries the presigned
      // WaitCondition URL.
      logs: { destination: logGroup, level: sfn.LogLevel.ERROR, includeExecutionData: false },
      tracingEnabled: true,
    });

    // startBuild.sync: start, poll and stop the two projects, and the managed
    // EventBridge rule Step Functions uses to learn a build has finished.
    // Reference: https://docs.aws.amazon.com/step-functions/latest/dg/codebuild-iam.html
    this.stateMachine.addToRolePolicy(new iam.PolicyStatement({
      actions: ['codebuild:StartBuild', 'codebuild:StopBuild', 'codebuild:BatchGetBuilds'],
      resources: [codebuildArn(props.weightsProjectName), codebuildArn(props.imageProjectName)],
    }));
    this.stateMachine.addToRolePolicy(new iam.PolicyStatement({
      actions: ['events:PutTargets', 'events:PutRule', 'events:DescribeRule'],
      resources: [`arn:${cdk.Aws.PARTITION}:events:${stack.region}:${stack.account}:rule/StepFunctionsGetEventForCodeBuildStartBuildRule`],
    }));

    // A new cartridge version or a changed recipe gets a new WaitCondition (and
    // so a new build); anything else leaves the built image alone on update.
    const recipe = crypto.createHash('sha256')
      .update(JSON.stringify([cartridge.version, cartridge.image, cartridge.weights]))
      .digest('hex').slice(0, 10);
    const handle = new cdk.CfnWaitConditionHandle(this, `Handle${recipe}`);
    const start = new cr.AwsCustomResource(this, `Start${recipe}`, {
      onCreate: {
        service: 'SFN',
        action: 'startExecution',
        parameters: {
          stateMachineArn: this.stateMachine.stateMachineArn,
          input: stack.toJsonString({ handleUrl: handle.ref, capacityBlockId: props.capacityBlockId }),
        },
        physicalResourceId: cr.PhysicalResourceId.fromResponse('executionArn'),
      },
      policy: cr.AwsCustomResourcePolicy.fromStatements([new iam.PolicyStatement({
        actions: ['states:StartExecution'],
        resources: [this.stateMachine.stateMachineArn],
      })]),
      installLatestAwsSdk: false,
    });
    const wait = new cdk.CfnWaitCondition(this, `Built${recipe}`, {
      handle: handle.ref,
      count: 1,
      timeout: String(12 * 3600),
    });
    wait.node.addDependency(start);

    // The tag is immutable, so this is the image the build pushed (or found).
    const image = new cr.AwsCustomResource(this, `Image${recipe}`, {
      onUpdate: {
        service: 'ECR',
        action: 'describeImages',
        parameters: { repositoryName: repoName, imageIds: [{ imageTag: cartridge.version }] },
        physicalResourceId: cr.PhysicalResourceId.of(`${repoName}:${cartridge.version}`),
        outputPaths: ['imageDetails.0.imageDigest'],
      },
      policy: cr.AwsCustomResourcePolicy.fromStatements([new iam.PolicyStatement({
        actions: ['ecr:DescribeImages'],
        resources: [repoArn],
      })]),
      installLatestAwsSdk: false,
    });
    image.node.addDependency(wait);
    this.imageDigest = image.getResponseField('imageDetails.0.imageDigest');
  }
}

const SIGNAL_SOURCE = `
import json
import re
import urllib.request

# A presigned S3 URL on an AWS endpoint, and nothing else.
URL_RE = re.compile(r"^https://[a-z0-9.-]+\\.amazonaws\\.com(\\.cn)?/")


def handler(event, _context):
    url = event["url"]
    if not URL_RE.match(url):
        raise ValueError("not a CloudFormation wait-condition URL")
    status = event["status"]
    if status == "SUCCESS":
        reason = "Weights staged and image built"
    else:
        reason = (str(event.get("error", "")) + ": " + str(event.get("cause", "")))[:1000]
    body = json.dumps({"Status": status, "Reason": reason, "UniqueId": "build", "Data": reason}).encode()
    req = urllib.request.Request(url, data=body, method="PUT", headers={"Content-Type": ""})
    with urllib.request.urlopen(req, timeout=30) as resp:  # nosec B310 - https AWS URL, checked above
        return {"status": resp.status}
`;
