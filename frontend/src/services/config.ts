// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { AppConfig } from '../types';

// This will be injected by CDK during deployment
declare global {
  interface Window {
    APP_CONFIG?: AppConfig;
  }
}

export const getConfig = (): AppConfig => {
  if (!window.APP_CONFIG) {
    throw new Error('APP_CONFIG not loaded. Make sure config.js is loaded before the app.');
  }
  return window.APP_CONFIG;
};
