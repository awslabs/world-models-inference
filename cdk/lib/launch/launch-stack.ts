// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import * as fs from 'fs';
import * as path from 'path';
import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
import { SharedStack, FoundationRefs } from '../shared';
import { describeStack } from '../solution';
import { LaunchCatalogue } from './catalogue';
import { LaunchModelStack } from './model';

/**
 * GPU instance types offered in the console, besides the cartridge's own default.
 * Shared with scripts/capacity.py, which only offers a Launch Stack link for these.
 */
const INSTANCE_TYPES: string[] = JSON.parse(fs.readFileSync(path.join(__dirname, 'instance-types.json'), 'utf8'));

// The same shapes deploy.sh and cdk/bin/app.ts check, as AllowedPatterns:
// CloudFormation refuses the stack before any resource sees the value.
const EMAIL_PATTERN = "^[^\\s@,;'\"`$\\\\]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$";
const OCTET = '(25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9]?[0-9])';
const CIDR_PATTERN = `^(${OCTET}\\.){3}${OCTET}/(3[0-2]|[12]?[0-9])$`;

export interface LaunchStackProps extends cdk.StackProps {
  readonly catalogue: LaunchCatalogue;
  /** Directory holding inference/, zipped as the image build's source. */
  readonly imageSource: string;
  /** frontend/dist. */
  readonly uiDist: string;
}

/** The foundation, as a nested stack of the Launch Stack. */
class LaunchFoundationStack extends cdk.NestedStack {
  public readonly refs: FoundationRefs;

  constructor(scope: Construct, id: string, props: { allowedCidr: string; adminEmail: string; reuseBuckets: string }) {
    super(scope, id, { description: 'World Model Inference: foundation (VPC, IAM roles, S3, CodeBuild, Cognito)' });
    const createBuckets = new cdk.CfnCondition(this, 'CreateBuckets', {
      expression: cdk.Fn.conditionEquals(props.reuseBuckets, 'no'),
    });
    this.refs = SharedStack.define(this, {
      allowedCidr: props.allowedCidr,
      importExistingBuckets: false,
      createBucketsCondition: createBuckets,
      callbackUrls: ['http://localhost:3000/'],
      adminEmail: props.adminEmail,
    });
  }
}

/**
 * The one-click Launch Stack: everything `./deploy.sh <model>` does, from the
 * CloudFormation console, with no local tooling.
 *
 * Foundation (nested) -> chosen cartridge (nested, one per cartridge under a
 * condition): stage weights + build image in the account, wait for both, run
 * the endpoint behind CloudFront with the UI and Cognito Hosted UI on the same
 * distribution, ask for one instance.
 */
export class LaunchStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props: LaunchStackProps) {
    super(scope, id, {
      ...props,
      description: describeStack('World Model Inference - one-click launch: foundation, build, endpoint and UI', 'launch'),
    });
    const models = Object.keys(props.catalogue.models).sort();
    const restricted = models.filter((m) => props.catalogue.models[m].licenseRestricted);

    const model = new cdk.CfnParameter(this, 'Model', {
      type: 'String',
      allowedValues: models,
      default: models.includes('lingbot-fast') ? 'lingbot-fast' : models[0],
      description: 'Cartridge to deploy. Each model is licensed separately by its authors: see docs/MODEL_LICENSES.md.',
    });
    const acceptLicense = new cdk.CfnParameter(this, 'AcceptModelLicense', {
      type: 'String',
      allowedValues: ['no', 'yes'],
      default: 'no',
      description: restricted.length
        // The manifest's `license` is the weights licence; the restriction can come
        // from another layer (waypoint-1-5's image installs GPL-3.0 world_engine).
        ? `Required "yes" for licence-restricted cartridges: ${restricted
          .map((m) => `${m} (weights ${props.catalogue.models[m].license}, ${props.catalogue.models[m].licenseUrl})`)
          .join('; ')}. Why each is restricted: docs/MODEL_LICENSES.md. You are responsible for compliant use in your own account.`
        : 'No cartridge in this template is licence-restricted.',
    });
    const instanceType = new cdk.CfnParameter(this, 'InstanceType', {
      type: 'String',
      allowedValues: ['model-default', ...INSTANCE_TYPES],
      default: 'model-default',
      description: 'GPU instance. model-default uses the cartridge\'s endpoint.yaml (e.g. p5.48xlarge for lingbot-fast).',
    });
    const adminEmail = new cdk.CfnParameter(this, 'AdminEmail', {
      type: 'String',
      allowedPattern: EMAIL_PATTERN,
      constraintDescription: 'must be an email address',
      description: 'Invited as the first user (Cognito emails a temporary password) and subscribed to scaling-event alerts (confirm the AWS email).',
    });
    const allowedCidr = new cdk.CfnParameter(this, 'AllowedCidr', {
      type: 'String',
      allowedPattern: CIDR_PATTERN,
      constraintDescription: 'must be an IPv4 CIDR such as 203.0.113.7/32',
      description: 'Viewers admitted at the CloudFront edge. Your IP as /32 (https://checkip.amazonaws.com), an office range, or 0.0.0.0/0 to rely on Cognito sign-in alone.',
    });
    const capacityBlock = new cdk.CfnParameter(this, 'CapacityBlockId', {
      type: 'String',
      default: '',
      allowedPattern: '^(cr-[0-9a-f]{8,17})?$',
      constraintDescription: 'must be empty or a capacity reservation ID (cr-...)',
      description: 'Optional EC2 Capacity Block (cr-...). Empty launches on-demand capacity. The instance is pinned to the block\'s Availability Zone.',
    });
    const reuseBuckets = new cdk.CfnParameter(this, 'ReuseRetainedBuckets', {
      type: 'String',
      allowedValues: ['no', 'yes'],
      default: 'no',
      description: 'yes if an earlier launch in this account and Region left its world-model-* buckets behind (they are retained on delete). Reusing them keeps already-staged weights.',
    });

    if (restricted.length) {
      new cdk.CfnRule(this, 'LicenceAcknowledged', {
        ruleCondition: cdk.Fn.conditionContains(restricted, model.valueAsString),
        assertions: [{
          assert: cdk.Fn.conditionEquals(acceptLicense.valueAsString, 'yes'),
          assertDescription: 'This cartridge is licence-restricted: read its licence and set AcceptModelLicense to yes.',
        }],
      });
    }

    const foundation = new LaunchFoundationStack(this, 'Foundation', {
      allowedCidr: allowedCidr.valueAsString,
      adminEmail: adminEmail.valueAsString,
      reuseBuckets: reuseBuckets.valueAsString,
    });

    // One nested stack per cartridge; only the chosen one is created.
    const chosen: { cond: cdk.CfnCondition; stack: LaunchModelStack }[] = [];
    for (const m of models) {
      const cond = new cdk.CfnCondition(this, `Is${pascal(m)}`, {
        expression: cdk.Fn.conditionEquals(model.valueAsString, m),
      });
      const stack = new LaunchModelStack(this, `Model${pascal(m)}`, {
        description: `World Model Inference: ${m} build, endpoint and UI`,
        model: m,
        cartridge: props.catalogue.models[m],
        foundation: foundation.refs,
        instanceType: instanceType.valueAsString,
        adminEmail: adminEmail.valueAsString,
        allowedCidr: allowedCidr.valueAsString,
        capacityBlockId: capacityBlock.valueAsString,
        imageSource: props.imageSource,
        uiDist: props.uiDist,
      });
      stack.nestedStackResource!.cfnOptions.condition = cond;
      chosen.push({ cond, stack });
    }
    const pick = (f: (s: LaunchModelStack) => string): string =>
      chosen.reduceRight<string>(
        (rest, { cond, stack }) => cdk.Fn.conditionIf(cond.logicalId, f(stack), rest).toString(),
        '',
      );

    new cdk.CfnOutput(this, 'UiUrl', {
      value: `https://${pick((s) => s.distributionDomain)}/`,
      description: 'Open this and sign in with the emailed temporary password',
    });
    new cdk.CfnOutput(this, 'AutoScalingGroup', {
      value: pick((s) => s.asgName),
      description: 'Scale to 0 to stop paying for the GPU; to 1 to start it again',
    });
    new cdk.CfnOutput(this, 'BuildStateMachine', {
      value: pick((s) => s.buildStateMachineName),
      description: 'Step Functions state machine that staged the weights and built the image',
    });

    this.templateOptions.metadata = {
      'AWS::CloudFormation::Interface': {
        ParameterGroups: [
          { Label: { default: 'Model' }, Parameters: ['Model', 'AcceptModelLicense', 'InstanceType', 'CapacityBlockId'] },
          { Label: { default: 'Access' }, Parameters: ['AdminEmail', 'AllowedCidr'] },
          { Label: { default: 'Redeploying' }, Parameters: ['ReuseRetainedBuckets'] },
        ],
        ParameterLabels: {
          Model: { default: 'Cartridge' },
          AcceptModelLicense: { default: 'Accept the model licence' },
          InstanceType: { default: 'GPU instance type' },
          CapacityBlockId: { default: 'Capacity Block ID (optional)' },
          AdminEmail: { default: 'Admin email' },
          AllowedCidr: { default: 'Allowed viewer CIDR' },
          ReuseRetainedBuckets: { default: 'Reuse retained buckets' },
        },
      },
    };
  }
}

function pascal(id: string): string {
  return id.split('-').map((p) => p.charAt(0).toUpperCase() + p.slice(1)).join('');
}
