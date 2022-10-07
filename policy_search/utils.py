import pathlib
from os import makedirs
from os.path import exists, join

import imageio_ffmpeg
import yaml


def write_args(args, dir):
    if dir:
        with open(dir / "args.yml", "w") as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def make_filepath(base_dir, dir, filename, force=False):
    if dir is not None:
        print(dir)
        dir_ = base_dir / dir
        print(dir_)
        if not exists(dir_):
            makedirs(dir_)
            assert exists(dir_)
        if filename:
            filepath = dir_ / filename
            if exists(f"{filepath}.npz") and not force:
                print("File exists!")
                exit()
            return filepath
        else:
            return dir_


class VideoRenderStream:
    """A stream for writing images to a video."""

    def __init__(self, video_name, output_root_path, fps=20, frame_size=(1024, 784)):
        """Initialize and open the video file to stream videos to.
        The caller is responsible for closing the stream. This object
        is best used in a python `with` statement.
        """
        # construct output path
        self.output_root_path = output_root_path
        self.video_path = self.output_root_path / video_name

        # video parameters
        self.fps = fps
        self.frame_size = frame_size  # (width, height)

        # initialize the video rendering pipeline
        self.video_path.parent.mkdir(exist_ok=True)
        self.video_writer = imageio_ffmpeg.write_frames(
            str(self.video_path), fps=self.fps, size=self.frame_size
        )
        self.video_writer.send(None)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Closes the video stream to no longer allow writing video data."""
        if self.video_writer is not None:
            self.video_writer.close()
            self.video_writer = None

    def write(self, image):
        """Write an image to the video."""
        self.video_writer.send(image)


class NullContext:
    """Dummy context manager."""

    def __init__(self):
        pass

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
