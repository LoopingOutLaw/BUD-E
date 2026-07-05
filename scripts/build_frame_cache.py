"""Build a sampled random-access stacked-frame cache for pick_v12 training.

The cache stores already history-stacked dual-camera observations. This avoids
MP4 decoding inside the training loop even when n_history_frames > 1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from pyarrow import parquet as pq


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--data-root', default='data/pick_v12')
    ap.add_argument('--out-dir', default='data/pick_v12/cache_224_h4_phase32k')
    ap.add_argument('--max-frames', type=int, default=32_000)
    ap.add_argument('--n-history-frames', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--early-prob', type=float, default=0.75)
    ap.add_argument('--early-max-frac', type=float, default=0.22)
    ap.add_argument('--phase-bins', type=int, default=0,
                    help='If >0, ignore --early-prob and sample equally from this many episode phase bins.')
    args = ap.parse_args()

    root = Path(args.data_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ep_files = sorted((root / 'meta' / 'episodes_index').glob('*.json'))
    episodes = []
    total = 0
    for ep_meta_path in ep_files:
        ep_meta = json.loads(ep_meta_path.read_text())
        ep_idx = int(ep_meta['episode_index'])
        chunk_idx = ep_idx // 1000
        pq_path = root / 'data' / f'chunk-{chunk_idx:03d}' / f'episode_{ep_idx:06d}.parquet'
        table = pq.read_table(str(pq_path), columns=['action'])
        T = table.num_rows
        episodes.append({'ep_idx': ep_idx, 'chunk_idx': chunk_idx, 'start': total, 'length': T})
        total += T

    rng = np.random.default_rng(args.seed)
    ep_lengths = np.asarray([e['length'] for e in episodes], dtype=np.int64)
    weights = ep_lengths.astype(np.float64); weights /= weights.sum()

    selected_by_ep: dict[int, set[int]] = {e['ep_idx']: set() for e in episodes}
    n_eps = len(episodes)
    if args.phase_bins > 0:
        per_bin = int(np.ceil(args.max_frames / args.phase_bins))
        for bin_i in range(args.phase_bins):
            lo = bin_i / args.phase_bins
            hi = (bin_i + 1) / args.phase_bins
            target = min(args.max_frames, (bin_i + 1) * per_bin)
            while sum(len(v) for v in selected_by_ep.values()) < target:
                ep_i = int(rng.choice(n_eps, p=weights))
                ep = episodes[ep_i]
                phase = float(rng.uniform(lo, hi))
                local = int(round(phase * max(0, ep['length'] - 1))) + int(rng.integers(-4, 5))
                local = min(max(local, 0), ep['length'] - 1)
                selected_by_ep[ep['ep_idx']].add(local)
    else:
        while sum(len(v) for v in selected_by_ep.values()) < args.max_frames:
            ep_i = int(rng.choice(n_eps, p=weights))
            ep = episodes[ep_i]
            if rng.random() < args.early_prob:
                phase = float(rng.uniform(0.0, args.early_max_frac))
            else:
                phase = float(rng.uniform(0.0, 1.0))
            local = int(round(phase * max(0, ep['length'] - 1))) + int(rng.integers(-4, 5))
            local = min(max(local, 0), ep['length'] - 1)
            selected_by_ep[ep['ep_idx']].add(local)

    total_sel = sum(len(v) for v in selected_by_ep.values())
    first_ep = episodes[0]
    first_vid = root / 'videos' / f"chunk-{first_ep['chunk_idx']:03d}" / 'observation.images.top' / f"episode_{first_ep['ep_idx']:06d}.mp4"
    sample = iio.imread(str(first_vid), index=0, plugin='pyav')
    H, W = sample.shape[:2]
    channels = args.n_history_frames * 6
    images_path = out / 'images.uint8.npy'
    images = np.lib.format.open_memmap(str(images_path), mode='w+', dtype=np.uint8, shape=(total_sel, H, W, channels))
    global_indices = np.empty((total_sel,), dtype=np.int64)

    row = 0
    for k, ep in enumerate(episodes):
        locals_sorted = sorted(selected_by_ep[ep['ep_idx']])
        if not locals_sorted:
            continue
        vid_top = root / 'videos' / f"chunk-{ep['chunk_idx']:03d}" / 'observation.images.top' / f"episode_{ep['ep_idx']:06d}.mp4"
        vid_wrist = root / 'videos' / f"chunk-{ep['chunk_idx']:03d}" / 'observation.images.wrist' / f"episode_{ep['ep_idx']:06d}.mp4"
        top = iio.imread(str(vid_top), plugin='pyav')
        wrist = iio.imread(str(vid_wrist), plugin='pyav') if vid_wrist.exists() else top
        frames = np.concatenate([top, wrist], axis=-1)
        for local in locals_sorted:
            start = max(0, local - (args.n_history_frames - 1))
            sel = frames[start:local + 1]
            if sel.shape[0] < args.n_history_frames:
                pad = np.repeat(sel[:1], args.n_history_frames - sel.shape[0], axis=0)
                sel = np.concatenate([pad, sel], axis=0)
            stacked = np.transpose(np.ascontiguousarray(sel), (1, 2, 0, 3)).reshape(H, W, channels)
            images[row] = stacked
            global_indices[row] = ep['start'] + local
            row += 1
        if k % 100 == 0:
            print(f'episode {k}/{len(episodes)} rows={row}/{total_sel}', flush=True)

    images.flush()
    np.save(str(out / 'global_indices.npy'), global_indices)
    meta = {
        'data_root': str(root),
        'num_frames': int(total_sel),
        'height': int(H),
        'width': int(W),
        'channels': int(channels),
        'n_history_frames': int(args.n_history_frames),
        'seed': args.seed,
        'early_prob': args.early_prob,
        'early_max_frac': args.early_max_frac,
        'phase_bins': int(args.phase_bins),
    }
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(f'done: {out} frames={total_sel} shape={(total_sel, H, W, channels)}')


if __name__ == '__main__':
    main()
