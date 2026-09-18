// Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
// SPDX-License-Identifier: Apache-2.0

import { useState, useEffect } from 'react';
import { Sparkles, Loader2 } from 'lucide-react';
import { fetchAuthSession } from 'aws-amplify/auth';
import { getConfig } from '@/services/config';
import { Button } from '@/components/ui/button';
import {
  Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogFooter,
} from '@/components/ui/dialog';

interface GeneratedImage {
  s3Key: string;
  url: string;
}

interface SceneGeneratorProps {
  visible: boolean;
  onDismiss: () => void;
  onSelectImage: (s3Key: string, url: string, title: string) => void;
}

export const SceneGenerator = ({ visible, onDismiss, onSelectImage }: SceneGeneratorProps) => {
  const [prompt, setPrompt] = useState('');
  const [images, setImages] = useState<GeneratedImage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [suggestions, setSuggestions] = useState<string[]>([]);
  const [suggestionsLoading, setSuggestionsLoading] = useState(false);

  const cfg = getConfig();
  const apiBase = (cfg.apiUrl || '').endsWith('/') ? cfg.apiUrl || '' : `${cfg.apiUrl || ''}/`;

  // Reset suggestions when dialog closes so we get fresh ones each time
  useEffect(() => {
    if (!visible) { setSuggestions([]); return; }
    const fetchSuggestions = async () => {
      setSuggestionsLoading(true);
      try {
        const session = await fetchAuthSession();
        const token = session.tokens?.idToken?.toString();
        if (!token) return;
        const res = await fetch(`${apiBase}api/suggest-prompts`, {
          headers: { 'Authorization': token },
        });
        if (res.ok) {
          const data = await res.json();
          setSuggestions(data.prompts || []);
        }
      } catch (e) {
        console.warn('Failed to load suggestions:', e);
      } finally {
        setSuggestionsLoading(false);
      }
    };
    fetchSuggestions();
  }, [visible]);

  const handleGenerate = async () => {
    if (!apiBase) { setError('API URL not configured'); return; }
    setLoading(true); setError(null); setImages([]); setSelectedIdx(null);
    try {
      const session = await fetchAuthSession();
      const token = session.tokens?.idToken?.toString();
      if (!token) throw new Error('Not authenticated');
      const res = await fetch(`${apiBase}api/generate-scene`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': token },
        body: JSON.stringify({ prompt, count: 3 }),
      });
      if (!res.ok) throw new Error(`Generation failed (${res.status}): ${await res.text()}`);
      const data = await res.json();
      setImages(data.images || []);
    } catch (e: any) {
      setError(e.message || 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const [titling, setTitling] = useState(false);

  const handleSelect = async () => {
    if (selectedIdx === null || !images[selectedIdx]) return;
    const img = images[selectedIdx];

    // Generate title with Claude Sonnet before starting
    setTitling(true);
    let title = 'Custom Scene';
    try {
      const session = await fetchAuthSession();
      const token = session.tokens?.idToken?.toString();
      if (token) {
        const res = await fetch(`${apiBase}api/generate-title`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', 'Authorization': token },
          body: JSON.stringify({ s3Key: img.s3Key }),
        });
        if (res.ok) {
          const data = await res.json();
          if (data.title) title = data.title;
        }
      }
    } catch (e) {
      console.warn('Title generation failed:', e);
    }
    setTitling(false);

    onSelectImage(img.s3Key, img.url, title);
    setImages([]); setSelectedIdx(null);
  };

  return (
    <Dialog open={visible} onOpenChange={(open) => { if (!open) onDismiss(); }}>
      <DialogContent>
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Sparkles className="h-5 w-5 text-game-purple" />
            Generate Custom Scene
          </DialogTitle>
          <DialogDescription>
            Describe a scene and Nova Canvas will generate variations to explore
          </DialogDescription>
        </DialogHeader>

        {/* Suggested prompts — fade in when loaded */}
        {suggestionsLoading && (
          <div className="flex items-center gap-2 mt-3 text-xs text-white/30">
            <Loader2 className="h-3 w-3 animate-spin" /> Loading suggestions…
          </div>
        )}
        {suggestions.length > 0 && !loading && images.length === 0 && (
          <div className="flex flex-col gap-1.5 mt-3 animate-[fadeIn_0.5s_ease-in]">
            {suggestions.map((s, i) => (
              <button
                key={i}
                onClick={() => setPrompt(s)}
                className="text-left text-xs px-3 py-2 rounded-lg bg-white/5 border border-white/10 text-white/60 hover:bg-game-purple/15 hover:border-game-purple/30 hover:text-white transition-all cursor-pointer"
              >
                {s}
              </button>
            ))}
          </div>
        )}

        {/* Prompt input */}
        <div className="flex gap-3 mt-3">
          <input
            className="flex-1 bg-white/5 border border-white/10 rounded-lg px-4 py-2.5 text-sm text-white placeholder:text-white/30 focus:outline-none focus:border-game-accent/50 transition-colors"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Describe a game scene..."
            disabled={loading}
          />
          <Button variant="primary" onClick={handleGenerate} disabled={loading || !prompt.trim()}>
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Sparkles className="h-4 w-4" />}
            Generate
          </Button>
        </div>

        {/* Error */}
        {error && (
          <div className="mt-3 rounded-lg bg-red-500/10 border border-red-500/20 px-4 py-3 text-sm text-red-400">
            {error}
            <button className="ml-2 text-red-400/60 hover:text-red-400 cursor-pointer" onClick={() => setError(null)}>✕</button>
          </div>
        )}

        {/* Loading */}
        {loading && (
          <div className="flex flex-col items-center py-12 gap-3">
            <Loader2 className="h-8 w-8 animate-spin text-game-purple" />
            <p className="text-sm text-white/40">Generating 3 scene variations with Nova Canvas…</p>
          </div>
        )}

        {/* Image grid */}
        {images.length > 0 && (
          <div className="grid grid-cols-3 gap-3 mt-4">
            {images.map((img, idx) => (
              <button
                key={idx}
                onClick={() => setSelectedIdx(idx)}
                className={`relative rounded-xl overflow-hidden border-3 transition-all cursor-pointer ${
                  selectedIdx === idx
                    ? 'border-game-accent shadow-[0_0_0_2px_rgba(83,159,229,0.3)]'
                    : 'border-transparent hover:border-game-accent/50 hover:scale-[1.02]'
                }`}
              >
                <img src={img.url} alt={`Variation ${idx + 1}`} className="w-full block rounded-lg" />
                <div className="absolute bottom-0 inset-x-0 py-1.5 text-center bg-gradient-to-t from-black/70 to-transparent text-white text-xs font-medium">
                  Variation {idx + 1}
                </div>
                {selectedIdx === idx && (
                  <div className="absolute top-2 right-2 w-7 h-7 rounded-full bg-game-accent text-white flex items-center justify-center text-sm font-bold shadow-lg">
                    ✓
                  </div>
                )}
              </button>
            ))}
          </div>
        )}

        {/* Footer */}
        {images.length > 0 && (
          <DialogFooter>
            <Button variant="ghost" onClick={onDismiss}>Cancel</Button>
            <Button variant="primary" onClick={handleSelect} disabled={selectedIdx === null || titling}>
              {titling ? <><Loader2 className="h-4 w-4 animate-spin" /> Naming scene…</> : 'Use Selected Scene'}
            </Button>
          </DialogFooter>
        )}
      </DialogContent>
    </Dialog>
  );
};
