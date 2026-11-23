"""Headless MP4 and distance plots for MAPPO test trajectories."""

import shutil

import numpy as np
from matplotlib import animation, rc_context
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure


def ffmpeg_executable():
    executable = shutil.which('ffmpeg')
    if executable:
        return executable
    try:
        import imageio_ffmpeg
    except ImportError as error:
        raise RuntimeError(
            'MP4 testing requires ffmpeg or imageio-ffmpeg. '
            'Install imageio-ffmpeg, or use --no-test.') from error
    return imageio_ffmpeg.get_ffmpeg_exe()


def save_animation(path, positions, world_size, fps=20, frame_stride=4,
                   prey_positions=None):
    figure = Figure(figsize=(6, 6))
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    axis.set(xlim=(0, world_size), ylim=(0, world_size),
             xlabel='x', ylabel='y', aspect='equal')
    agents = axis.scatter([], [], color='royalblue', s=24,
                          label='Pursuers' if prey_positions is not None else 'Agents')
    if prey_positions is not None:
        prey = axis.scatter([], [], color='crimson', marker='*', s=110, label='Prey')
    axis.legend(loc='upper right')
    title = axis.set_title('Step 0')
    frames = list(range(0, len(positions), frame_stride))
    if frames[-1] != len(positions) - 1:
        frames.append(len(positions) - 1)
    with rc_context({'animation.ffmpeg_path': ffmpeg_executable()}):
        writer = animation.FFMpegWriter(fps=fps, codec='libx264',
                                       extra_args=['-pix_fmt', 'yuv420p'])
        with writer.saving(figure, str(path), dpi=100):
            for step in frames:
                agents.set_offsets(positions[step])
                if prey_positions is not None:
                    prey.set_offsets(prey_positions[step])
                title.set_text('Step {}'.format(step))
                writer.grab_frame()
    figure.clear()


def save_distance_plot(path, series, ylabel):
    figure = Figure(figsize=(9, 4.5))
    FigureCanvasAgg(figure)
    axis = figure.add_subplot(111)
    for label, values in series.items():
        style = {}
        if label == 'Nearest pursuer':
            style = {'color': 'black', 'linewidth': 2.2}
        elif label == 'Capture threshold':
            style = {'color': 'crimson', 'linestyle': '--'}
        axis.plot(np.arange(len(values)), values, label=label, **style)
    axis.set(xlabel='Test step', ylabel=ylabel)
    axis.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(str(path), dpi=150)
    figure.clear()
