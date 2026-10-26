// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

// Client for scripts/capacity.py. Every call carries the per-run token the server
// wrote into the page; the server answers only to the page it rendered.

export interface Model {
  id: string;
  name: string;
  tagline: string;
  kind: string;
  gpuCount: number;
  gpuMemoryGb: number;
  defaultInstance: string | null;
  restricted: boolean;
  license: string | null;
  candidates: string[];
}

export interface Instance {
  type: string;
  gpus: number | null;
  gpu: string;
  gpuMemoryGb: number | null;
  vcpus: number | null;
  family: 'G' | 'P';
  capacityBlocks: boolean;
  launchStack: boolean;
}

export interface Bootstrap {
  models: Model[];
  instances: Record<string, Instance>;
  regions: string[];
  defaultRegions: string[];
  identity: { account: string; arn: string };
  launchStack: boolean;
}

export type Status = 'available' | 'offered' | 'none' | 'quota' | 'unsupported' | 'error';

export interface RegionInfo {
  type: 'region';
  region: string;
  azs: string[];
  stackAzs: string[];
  quota: Record<'G' | 'P', { limit: number | null; used: number }>;
  templatePublished: boolean;
  prices: Record<string, number | null>;
}

export interface Cell {
  type: 'cell';
  region: string;
  az: string;
  instanceType: string;
  status: Status;
  detail: string;
  inStackAzs: boolean;
  deploy?: string;
  hold?: string;
  launch?: string;
  /** A probe reservation the server could not cancel: it is still billing. */
  leak?: string;
  cancel?: string;
}

export interface Block {
  type: 'block';
  region: string;
  az: string;
  instanceType: string;
  start: string;
  end: string;
  fee: number;
  currency: string;
  offeringId: string;
  inStackAzs: boolean;
  purchase: string;
}

export type ScanEvent =
  | { type: 'start'; model: string; regions: string[]; candidates: Instance[] }
  | RegionInfo
  | Cell
  | Block
  | { type: 'region-error'; region: string; error: string }
  | { type: 'region-done'; region: string }
  | { type: 'done' };

const token = document.querySelector<HTMLMetaElement>('meta[name="wizard-token"]')?.content ?? '';
const headers = { 'X-Wizard-Token': token };

export async function bootstrap(): Promise<Bootstrap> {
  const r = await fetch('/api/models', { headers });
  if (!r.ok) throw new Error(`The wizard server answered ${r.status}. Start it with python scripts/capacity.py.`);
  return r.json();
}

export interface ScanRequest {
  model: string;
  regions: string[];
  probe: boolean;
  blocks: boolean;
  instanceTypes?: string[];
}

/** Streams NDJSON scan events until the server sends `done` (or the request is aborted). */
export async function scan(req: ScanRequest, onEvent: (e: ScanEvent) => void, signal?: AbortSignal): Promise<void> {
  const r = await fetch('/api/scan', {
    method: 'POST',
    headers: { ...headers, 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
    signal,
  });
  if (!r.ok || !r.body) throw new Error(`Scan failed: ${r.status}`);
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (line) onEvent(JSON.parse(line) as ScanEvent);
    }
  }
}
