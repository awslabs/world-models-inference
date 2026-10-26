// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { createContext, useContext } from 'react';
import { getConfig } from './config';

/**
 * The Cognito ACCESS token the UI presents to the inference API, when it holds
 * one itself.
 *
 * Behind the `./deploy.sh ui` dev proxy this is empty: the proxy injects the
 * Authorization header server-side. The Launch Stack serves the UI from the
 * endpoint's own CloudFront distribution instead, signs the user in through the
 * Cognito Hosted UI, and provides the user's token here (HostedSignIn).
 */
export const ApiTokenContext = createContext<string | undefined>(undefined);

export const useApiToken = (): string | undefined => {
  const fromSignIn = useContext(ApiTokenContext);
  if (fromSignIn) return fromSignIn;
  try { return getConfig().apiToken || undefined; } catch { return undefined; }
};
