"""Build a sampled random-access stacked-frame cache for pick_v12 training.

The cache stores already history-stacked dual-camera observations. This avoids
MP4 decoding inside the training loop even when n_history_frames > 1.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from pyarrow import parquet as pq


@dataclass(frozen=True)
class EpisodeInfo:
    ep_idx: int
    chunk_idx: int
    start: int
    length: int
    contact_locals: list[int]


def parse_weighted_phase_ranges(spec: str | None) -> list[tuple[float, float, float]]:
    if spec is None or not str(spec).strip():
        return []
    ranges: list[tuple[float, float, float]] = []
    for raw_part in str(spec).split(','):
        part = raw_part.strip()
        fields = part.split(':')
        if len(fields) != 3:
            raise ValueError("phase range entries must use lo:hi:weight")
        try:
            lo, hi, weight = (float(v) for v in fields)
        except ValueError as exc:
            raise ValueError("phase range entries must use numeric lo:hi:weight") from exc
        if not (0.0 <= lo < hi <= 1.0):
            raise ValueError("phase ranges must satisfy 0 <= lo < hi <= 1")
        if weight <= 0.0:
            raise ValueError("phase range weights must be > 0")
        ranges.append((lo, hi, weight))
    return ranges


def sample_phase_from_weighted_ranges(
    rng: np.random.Generator,
    ranges: list[tuple[float, float, float]],
) -> float:
    weights = np.asarray([r[2] for r in ranges], dtype=np.float64)
    weights /= weights.sum()
    i = int(rng.choice(len(ranges), p=weights))
    lo, hi, _weight = ranges[i]
    return float(rng.uniform(lo, hi))


def _sample_non_contact_local(
    rng: np.random.Generator,
    ep: EpisodeInfo,
    phase_ranges: list[tuple[float, float, float]],
    early_prob: float,
    early_max_frac: float,
) -> int:
    if phase_ranges:
        phase = sample_phase_from_weighted_ranges(rng, phase_ranges)
    elif rng.random() < early_prob:
        phase = float(rng.uniform(0.0, early_max_frac))
    else:
        phase = float(rng.uniform(0.0, 1.0))
    local = int(round(phase * max(0, ep.length - 1))) + int(rng.integers(-4, 5))
    return min(max(local, 0), ep.length - 1)


def select_cache_frame_indices(
    episodes: list[EpisodeInfo],
    max_frames: int,
    rng: np.random.Generator,
    early_prob: float,
    early_max_frac: float,
    phase_bins: int,
    phase_ranges: list[tuple[float, float, float]],
    contact_prob: float,
    contact_jitter: int,
    min_frames_per_episode: int = 0,
) -> dict[int, set[int]]:
    selected_by_ep: dict[int, set[int]] = {e.ep_idx: set() for e in episodes}
    if not episodes or max_frames <= 0:
        return selected_by_ep

    ep_lengths = np.asarray([e.length for e in episodes], dtype=np.int64)
    weights = ep_lengths.astype(np.float64)
    weights /= weights.sum()
    n_eps = len(episodes)

    contact_refs: list[tuple[int, int]] = []
    for ep_i, ep in enumerate(episodes):
        for local in ep.contact_locals:
            contact_refs.append((ep_i, local))

    def add_local(ep_i: int, local: int) -> None:
        ep = episodes[ep_i]
        local = min(max(int(local), 0), ep.length - 1)
        selected_by_ep[ep.ep_idx].add(local)

    coverage_per_ep = min(
        max(0, int(min_frames_per_episode)),
        max_frames // n_eps,
    )
    if coverage_per_ep > 0:
        for ep_i, ep in enumerate(episodes):
            for bin_i in range(coverage_per_ep):
                phase = (bin_i + float(rng.random())) / coverage_per_ep
                local = int(round(phase * max(0, ep.length - 1)))
                add_local(ep_i, local)

    if phase_bins > 0 and not phase_ranges and not contact_refs:
        per_bin = int(np.ceil(max_frames / phase_bins))
        for bin_i in range(phase_bins):
            lo = bin_i / phase_bins
            hi = (bin_i + 1) / phase_bins
            target = min(max_frames, (bin_i + 1) * per_bin)
            while sum(len(v) for v in selected_by_ep.values()) < target:
                ep_i = int(rng.choice(n_eps, p=weights))
                ep = episodes[ep_i]
                phase = float(rng.uniform(lo, hi))
                local = int(round(phase * max(0, ep.length - 1))) + int(rng.integers(-4, 5))
                add_local(ep_i, local)
        return selected_by_ep

    attempts = 0
    max_attempts = max_frames * 100
    while sum(len(v) for v in selected_by_ep.values()) < max_frames and attempts < max_attempts:
        attempts += 1
        use_contact = contact_refs and rng.random() < contact_prob
        if use_contact:
            ep_i, center = contact_refs[int(rng.integers(0, len(contact_refs)))]
            jitter = int(rng.integers(-contact_jitter, contact_jitter + 1)) if contact_jitter > 0 else 0
            add_local(ep_i, center + jitter)
        else:
            ep_i = int(rng.choice(n_eps, p=weights))
            local = _sample_non_contact_local(
                rng, episodes[ep_i], phase_ranges, early_prob, early_max_frac)
            add_local(ep_i, local)

    return selected_by_ep


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
    ap.add_argument('--phase-ranges', default=None,
                    help='Weighted phase ranges as lo:hi:weight,...; overrides --phase-bins and --early-prob. Useful for contact/descent-focused caches.')
    ap.add_argument('--contact-prob', type=float, default=0.0,
                    help='Probability of sampling from frames where observation.state[8] any_contact is true, if present.')
    ap.add_argument('--contact-jitter', type=int, default=6,
                    help='Local frame jitter around sampled contact frames.')
    ap.add_argument('--min-frames-per-episode', type=int, default=0,
                    help='Guarantee this many phase-stratified cache rows per episode before random sampling.')
    args = ap.parse_args()

    root = Path(args.data_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ep_files = sorted((root / 'meta' / 'episodes_index').glob('*.json'))
    episodes: list[EpisodeInfo] = []
    total = 0
    n_contact_frames = 0
    for ep_meta_path in ep_files:
        ep_meta = json.loads(ep_meta_path.read_text())
        ep_idx = int(ep_meta['episode_index'])
        chunk_idx = ep_idx // 1000
        pq_path = root / 'data' / f'chunk-{chunk_idx:03d}' / f'episode_{ep_idx:06d}.parquet'
        table = pq.read_table(str(pq_path), columns=['action', 'observation.state'])
        T = table.num_rows
        contact_locals: list[int] = []
        states = table.column('observation.state').to_pylist()
        if states and len(states[0]) >= 10:
            contact_locals = [i for i, state in enumerate(states) if float(state[8]) > 0.5]
        n_contact_frames += len(contact_locals)
        episodes.append(EpisodeInfo(
            ep_idx=ep_idx, chunk_idx=chunk_idx, start=total, length=T,
            contact_locals=contact_locals,
        ))
        total += T

    rng = np.random.default_rng(args.seed)
    phase_ranges = parse_weighted_phase_ranges(args.phase_ranges)
    selected_by_ep = select_cache_frame_indices(
        episodes=episodes,
        max_frames=args.max_frames,
        rng=rng,
        early_prob=args.early_prob,
        early_max_frac=args.early_max_frac,
        phase_bins=args.phase_bins,
        phase_ranges=phase_ranges,
        contact_prob=max(0.0, min(1.0, args.contact_prob)),
        contact_jitter=max(0, args.contact_jitter),
        min_frames_per_episode=max(0, args.min_frames_per_episode),
    )
    if args.contact_prob > 0.0:
        print(f'contact-aware sampling: contact_frames={n_contact_frames} contact_prob={args.contact_prob:.2f}', flush=True)

    total_sel = sum(len(v) for v in selected_by_ep.values())
    first_ep = episodes[0]
    first_vid = root / 'videos' / f"chunk-{first_ep.chunk_idx:03d}" / 'observation.images.top' / f"episode_{first_ep.ep_idx:06d}.mp4"
    sample = iio.imread(str(first_vid), index=0, plugin='pyav')
    H, W = sample.shape[:2]
    channels = args.n_history_frames * 6
    images_path = out / 'images.uint8.npy'
    images = np.lib.format.open_memmap(str(images_path), mode='w+', dtype=np.uint8, shape=(total_sel, H, W, channels))
    global_indices = np.empty((total_sel,), dtype=np.int64)

    row = 0
    for k, ep in enumerate(episodes):
        locals_sorted = sorted(selected_by_ep[ep.ep_idx])
        if not locals_sorted:
            continue
        vid_top = root / 'videos' / f"chunk-{ep.chunk_idx:03d}" / 'observation.images.top' / f"episode_{ep.ep_idx:06d}.mp4"
        vid_wrist = root / 'videos' / f"chunk-{ep.chunk_idx:03d}" / 'observation.images.wrist' / f"episode_{ep.ep_idx:06d}.mp4"
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
            global_indices[row] = ep.start + local
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
        'phase_ranges': args.phase_ranges,
        'contact_prob': float(args.contact_prob),
        'contact_jitter': int(args.contact_jitter),
        'contact_frames_available': int(n_contact_frames),
        'min_frames_per_episode': int(args.min_frames_per_episode),
    }
    (out / 'meta.json').write_text(json.dumps(meta, indent=2))
    print(f'done: {out} frames={total_sel} shape={(total_sel, H, W, channels)}')


if __name__ == '__main__':
    main()
