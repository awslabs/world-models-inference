// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

export type GameMode = 'universal' | 'gta_drive' | 'templerun';

export interface AppConfig {
  region: string;
  userPoolId: string;
  userPoolClientId: string;
  websocketUrl: string;
  lingbotApiUrl: string;
  environment: string;
  /** Cognito hosted UI domain (e.g. https://your-app.auth.us-east-1.amazoncognito.com) */
  cognitoDomain: string;
  /** Where Cognito redirects after sign-in (your CloudFront URL) */
  redirectSignIn: string;
  /** Where Cognito redirects after sign-out */
  redirectSignOut: string;
  /** API Gateway base URL for REST API (e.g. https://<api-id>.execute-api.<region>.amazonaws.com/prod) */
  apiUrl?: string;
  /** When true, skip Cognito auth and show the UI in demo/showcase mode */
  demoMode?: boolean;
  /** Shared bearer token for the inference endpoint (R1). Sent as
   *  `Authorization: Bearer <token>` on REST calls and `?token=` on WebSockets.
   *  Written by `./deploy.sh ui` from the SSM SecureString. Empty in demo mode. */
  apiToken?: string;
  /** Which UI to render. Default 'worlds' renders the LobbyScreen (real-time worlds).
   *  'lingbot' renders the LingbotGenerator (async video gen against lingbot-fast endpoint).
   *  The ./deploy ui subcommand writes 'lingbot' automatically. */
  ui?: 'worlds' | 'lingbot';
}

export interface GameState {
  isConnected: boolean;
  isPlaying: boolean;
  fps: number;
  latency: number;
}

export interface KeyboardAction {
  w: boolean;
  a: boolean;
  s: boolean;
  d: boolean;
}

export interface GameMessage {
  type: 'frame' | 'action' | 'control' | 'connect' | 'disconnect' | 'ping' | 'pong' | 'start' | 'stop' | 'started' | 'stopped' | 'connected' | 'error';
  data?: any;
  timestamp?: number;
  keyboard?: number[];
  mouse?: number[];
  buttons?: string[];
  mouse_dx?: number;
  mouse_dy?: number;
  session_id?: string;
  message?: string;
  prompt?: string;
  image_path?: string;
  mode?: GameMode;
}

export interface LingbotExample {
  id: string;
  prompt: string;
  /** Bundled per-frame camera poses + intrinsics .npy files */
  has_poses: boolean;
  /** Bundled per-frame WASD/IJKL action .npy (for Act2Cam mode) */
  has_action?: boolean;
}
