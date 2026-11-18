"""Offline trajectory rendering; never advances or modifies an environment."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import imageio.v2 as imageio


def broken_xy(points, size, torus):
    points = np.asarray(points).copy()
    if torus and len(points)>1:
        cuts = np.where((np.abs(np.diff(points, axis=0)) > size/2).any(axis=1))[0] + 1
        points = np.insert(points, cuts, np.nan, axis=0)
    return points


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--evaluation', required=True)
    p.add_argument('--index', type=int, default=0, help='Predeclared scene index, not best-of selection')
    p.add_argument('--stride', type=int, default=4)
    p.add_argument('--fps', type=int, default=15)
    p.add_argument('--video', action='store_true')
    args = p.parse_args()
    if args.stride < 1 or args.fps < 1:
        p.error('stride and fps must be positive')
    root = Path(args.evaluation)
    results = json.loads((root/'results.json').read_text())
    row = results['episodes'][args.index]
    cfg = results['config']['environment']
    trace = np.load(root/f'trace-{args.index:04d}.npz')
    pursuers, prey, distances = (trace[k] for k in ('pursuers','prey','distances'))
    size, torus = cfg['world_size'], cfg['torus']
    fig, ax = plt.subplots(figsize=(7,6))
    for i in range(pursuers.shape[1]):
        xy = broken_xy(pursuers[:,i,:2], size, torus)
        line, = ax.plot(xy[:,0], xy[:,1], alpha=.6)
        ax.scatter(*pursuers[0,i,:2], marker='o', s=18, color=line.get_color())
        ax.scatter(*pursuers[-1,i,:2], marker='x', s=24, color=line.get_color())
    xy = broken_xy(prey[:,0], size, torus)
    ax.plot(xy[:,0], xy[:,1], 'r--', label='scripted prey')
    ax.set(xlim=(0,size), ylim=(0,size), aspect='equal', xlabel='x (start: o, end: x)', ylabel='y',
        title=f"{root.name} | {row['scene_id']} | success={row['success']} | torus={torus}")
    ax.legend()
    fig.savefig(root/f'trajectory-{args.index:04d}.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(distances, alpha=.35)
    ax.plot(distances.min(axis=1), color='black', label='nearest pursuer')
    ax.axhline(1, color='red', linestyle='--', label='capture: distance < 1')
    ax.set(xlabel='Environment step', ylabel='Periodic distance' if torus else 'Distance')
    ax.legend()
    fig.tight_layout()
    fig.savefig(root/f'distance-{args.index:04d}.png', dpi=160)
    plt.close(fig)
    if args.video:
        with imageio.get_writer(root/f'episode-{args.index:04d}.mp4', fps=args.fps) as writer:
            frames = sorted(set([*range(0,len(pursuers),args.stride),len(pursuers)-1]))
            for t in frames:
                fig, ax = plt.subplots(figsize=(6.4,6.4))
                for i, agent in enumerate(pursuers[t]):
                    xy = broken_xy(pursuers[max(0,t-40):t+1,i,:2],size,torus)
                    ax.plot(xy[:,0],xy[:,1],alpha=.4)
                    ax.scatter(*agent[:2],c='tab:blue')
                    ax.text(*agent[:2],str(i),fontsize=8)
                    ax.arrow(*agent[:2],2*np.cos(agent[2]),2*np.sin(agent[2]),width=.15)
                ax.scatter(*prey[t,0],c='tab:red',marker='*',s=100)
                status = ('CAPTURED' if row['success'] else 'TIMEOUT') if t==len(pursuers)-1 else 'pursuit'
                ax.set(xlim=(0,size),ylim=(0,size),aspect='equal',
                    title=f"{root.name} | {row['scene_id']}\nstep={t} | d={distances[t].min():.3f} | {status}",
                    xlabel=f"torus={torus}; stride={args.stride}, playback={args.fps} fps")
                fig.canvas.draw()
                writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:,:,:3])
                plt.close(fig)
    save = dict(scene_id=row['scene_id'], index=args.index, suite_hash=results['suite_hash'],
                checkpoint_sha256=results['checkpoint_sha256'], stride=args.stride, fps=args.fps)
    (root/f'render-{args.index:04d}.json').write_text(json.dumps(save,indent=2))


if __name__ == '__main__':
    main()
