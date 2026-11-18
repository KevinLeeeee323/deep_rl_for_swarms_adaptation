"""Reproducible side-by-side render using the bundled FFmpeg encoder."""
import argparse
import json
import subprocess
import sys
from pathlib import Path
import imageio_ffmpeg


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--before',required=True)
    p.add_argument('--after',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--index',type=int,default=0)
    p.add_argument('--stride',type=int,default=8)
    p.add_argument('--fps',type=int,default=15)
    a=p.parse_args()
    if a.stride < 1 or a.fps < 1: p.error('stride and fps must be positive')
    before,after=Path(a.before),Path(a.after)
    left=json.loads((before/'results.json').read_text())
    right=json.loads((after/'results.json').read_text())
    if left['suite_hash']!=right['suite_hash'] or left['config']['environment']!=right['config']['environment']:
        p.error('Comparison requires the same initial-state suite and environment')
    if left['episodes'][a.index]['scene_id']!=right['episodes'][a.index]['scene_id']:
        p.error('Different scenes selected')
    for root in (before,after):
        subprocess.run([sys.executable,'-m','deep_rl_for_swarms.experiments.pursuit.render',
            '--evaluation',str(root),'--index',str(a.index),'--stride',str(a.stride),
            '--fps',str(a.fps),'--video'],check=True)
    output=Path(a.output); output.parent.mkdir(parents=True,exist_ok=True)
    duration=max((r['episodes'][a.index]['steps']+a.stride-1)//a.stride+1 for r in (left,right))/a.fps
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-loglevel','error','-y',
        '-i',str(before/f'episode-{a.index:04d}.mp4'),'-i',str(after/f'episode-{a.index:04d}.mp4'),
        '-filter_complex',f'[0:v]tpad=stop_mode=clone:stop_duration={duration}[l];[1:v]tpad=stop_mode=clone:stop_duration={duration}[r];[l][r]hstack=inputs=2',
        '-t',str(duration),
        '-c:v','libx264','-pix_fmt','yuv420p',str(output)],check=True)
    output.with_suffix('.json').write_text(json.dumps(dict(before=str(before),after=str(after),
        scene_id=left['episodes'][a.index]['scene_id'],suite_hash=left['suite_hash'],
        before_checkpoint=left['checkpoint_sha256'],after_checkpoint=right['checkpoint_sha256'],
        stride=a.stride,fps=a.fps,shorter_video='hold final frame'),indent=2))


if __name__=='__main__': main()
