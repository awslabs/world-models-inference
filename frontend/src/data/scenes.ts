// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

/** Pre-defined scenes for the carousel. */
export interface Scene {
  id: string;
  name: string;
  /** Path relative to public/ — served via CloudFront */
  imageUrl: string;
}

export const DEFAULT_SCENES: Scene[] = [
  {
    id: 'mountain-highway',
    name: 'Mountain Highway',
    imageUrl: '/scenes/mountain-highway.png',
  },
  {
    id: 'industrial-yard',
    name: 'Industrial Yard',
    imageUrl: '/scenes/industrial-yard.png',
  },
  {
    id: 'futuristic-base',
    name: 'Futuristic Base',
    imageUrl: '/scenes/futuristic-base.png',
  },
];
