"""
arm_frame_grabber.py
--------------------
Threaded frame reader for a raw-video stream piped from ffmpeg.

Used to capture the arm-mounted (wrist) camera feed — an RTSP stream
served by the Kinova Gen3 — in a background thread so that the main
control loop can call get_latest_frame() without blocking.

The FrameGrabber is not specific to Kinova; it works with any ffmpeg
subprocess that writes BGR24 raw video to stdout.
"""

import threading
import numpy as np


class FrameGrabber:
    """Continuously reads BGR frames from an ffmpeg stdout pipe.

    Args:
        proc:   A subprocess.Popen object whose stdout is a raw BGR24 stream.
        width:  Frame width in pixels.
        height: Frame height in pixels.
    """

    def __init__(self, proc, width: int, height: int):
        self.proc = proc
        self.width = width
        self.height = height
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()

    def _read_frames(self):
        """Background thread: read raw BGR frames from the ffmpeg pipe."""
        frame_bytes = self.width * self.height * 3
        while self.running:
            raw = self.proc.stdout.read(frame_bytes)
            if len(raw) != frame_bytes:
                # Stream ended or was truncated.
                break
            frame = np.frombuffer(raw, np.uint8).reshape((self.height, self.width, 3))
            with self.lock:
                self.frame = frame

    def get_latest_frame(self):
        """Return a copy of the most recently decoded frame, or None."""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def stop(self):
        """Signal the reader thread to stop and wait for it to exit."""
        self.running = False
        self.thread.join()
