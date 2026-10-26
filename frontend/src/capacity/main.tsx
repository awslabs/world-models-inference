// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import React from 'react';
import ReactDOM from 'react-dom/client';
import { CapacityWizard } from './CapacityWizard';
import './capacity.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <CapacityWizard />
  </React.StrictMode>,
);
