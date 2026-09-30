// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { ImgHTMLAttributes, useEffect, useState } from 'react';

/**
 * An <img> for an authenticated API URL.
 *
 * <img src> cannot send an Authorization header, and the inference server does
 * not accept a ?token= query param on HTTP routes (a query-string token lands in
 * access logs and browser history). With a token, this fetches the image with
 * the header and renders it from a blob URL. Without one — the usual case, where
 * the ./deploy.sh ui proxy injects the header server-side — and for data:/blob:
 * sources, it is a plain <img>.
 */
export function AuthedImg({
  src, token, ...rest
}: Omit<ImgHTMLAttributes<HTMLImageElement>, 'src'> & { src: string; token?: string }) {
  const direct = !token || src.startsWith('data:') || src.startsWith('blob:');
  const [blobUrl, setBlobUrl] = useState<string | null>(null);

  useEffect(() => {
    if (direct) return;
    let objectUrl: string | null = null;
    let cancelled = false;
    setBlobUrl(null);
    fetch(src, { headers: { Authorization: `Bearer ${token}` } })
      .then(resp => (resp.ok ? resp.blob() : Promise.reject(new Error(`HTTP ${resp.status}`))))
      .then(blob => {
        if (cancelled) return;
        objectUrl = URL.createObjectURL(blob);
        setBlobUrl(objectUrl);
      })
      .catch(() => { /* leave the image empty, as a broken <img> would be */ });
    return () => {
      cancelled = true;
      if (objectUrl) URL.revokeObjectURL(objectUrl);
    };
  }, [src, token, direct]);

  if (direct) return <img src={src} {...rest} />;
  return blobUrl ? <img src={blobUrl} {...rest} /> : null;
}
