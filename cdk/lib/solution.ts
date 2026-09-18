// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * AWS solution tracking code. Embedding it in a stack's CloudFormation
 * Description is what registers a deployment against this project.
 *
 * This repo synthesises more than one template, so each stack also carries a
 * `tag:` marker — every distinct tag shows up as its own dashboard entry,
 * grouped under the project.
 */
export const SOLUTION_ID = 'uksb-fj93uijid5';

/**
 * Builds a stack Description carrying the tracking code and a per-stack tag.
 *
 * @param summary human-readable description of what the stack deploys
 * @param tag     dashboard entry this stack reports under (e.g. 'foundation')
 */
export function describeStack(summary: string, tag: string): string {
  return `${summary} (${SOLUTION_ID})(tag:${tag}).`;
}
