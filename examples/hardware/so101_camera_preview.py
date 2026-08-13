#!/usr/bin/env python

# Copyright 2026 LV Robotics Lab. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Serve a low-latency browser preview for an SO-101 collection rig.

Example:

    python examples/hardware/so101_camera_preview.py \
      --camera front=/dev/video2 \
      --camera side_45=/dev/video0

The server binds to localhost by default. Use an SSH tunnel to view it remotely.
"""

from __future__ import annotations

import argparse
import html
import sys
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

DEFAULT_CAMERAS = ("front=/dev/video2", "side_45=/dev/video0")


def parse_camera(value: str) -> tuple[str, str]:
    try:
        label, device = value.split("=", 1)
    except ValueError as error:
        raise argparse.ArgumentTypeError("camera must use NAME=DEVICE format") from error
    label = label.strip()
    device = device.strip()
    if not label or not device or "/" in label:
        raise argparse.ArgumentTypeError("camera label and device must be non-empty; label cannot contain '/'")
    return label, device


def open_camera(device: str, width: int, height: int, fps: int) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(device, cv2.CAP_V4L2)
    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    camera.set(cv2.CAP_PROP_FPS, fps)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"could not open camera at {device}")
    return camera


class CameraStore:
    def __init__(self, captures: dict[str, cv2.VideoCapture]) -> None:
        self.captures = captures
        self.frames: dict[str, bytes] = {}
        self.lock = threading.Lock()
        self.running = True

    def capture_loop(self, label: str, camera: cv2.VideoCapture) -> None:
        while self.running:
            ok, frame = camera.read()
            if not ok:
                continue
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                with self.lock:
                    self.frames[label] = encoded.tobytes()

    def get_frame(self, label: str) -> bytes | None:
        with self.lock:
            return self.frames.get(label)

    def close(self) -> None:
        self.running = False
        for camera in self.captures.values():
            camera.release()


def index_page(labels: list[str]) -> bytes:
    panels = "".join(
        f"<section><h2>{html.escape(label)}</h2><img id='{html.escape(label)}' "
        f"src='/{html.escape(label)}.jpg' alt='{html.escape(label)}'></section>"
        for label in labels
    )
    refresh = ",".join(repr(label) for label in labels)
    page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SO-101 cameras</title>
<style>
body{{background:#111;color:#eee;font-family:sans-serif;margin:20px}}
main{{display:flex;gap:16px;flex-wrap:wrap}}section{{width:min(48vw,720px)}}
img{{width:100%;background:#222}}h2{{margin:0 0 8px}}
</style></head><body><h1>SO-101 live cameras</h1><main>{panels}</main>
<script>
for(const id of [{refresh}]){{const image=document.getElementById(id);
setInterval(()=>image.src='/'+id+'.jpg?t='+Date.now(),50)}}
</script></body></html>"""
    return page.encode()


def make_handler(store: CameraStore) -> type[BaseHTTPRequestHandler]:
    labels = list(store.captures)
    page = index_page(labels)

    class PreviewHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/":
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(page)))
                self.end_headers()
                self.wfile.write(page)
                return

            label = self.path.split("?", 1)[0].removeprefix("/").removesuffix(".jpg")
            frame = store.get_frame(label) if label in store.captures else None
            if frame is None:
                self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "camera frame is not ready")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(frame)))
            self.end_headers()
            self.wfile.write(frame)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    return PreviewHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Serve SO-101 cameras in a local browser")
    parser.add_argument(
        "--camera",
        action="append",
        type=parse_camera,
        metavar="NAME=DEVICE",
        help="repeat for each camera; defaults to the Jingxiang front and side cameras",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    camera_specs = args.camera or [parse_camera(value) for value in DEFAULT_CAMERAS]
    cameras = dict(camera_specs)
    if len(cameras) != len(camera_specs):
        raise SystemExit("camera labels must be unique")

    captures: dict[str, cv2.VideoCapture] = {}
    store: CameraStore | None = None
    try:
        for label, device in cameras.items():
            captures[label] = open_camera(device, args.width, args.height, args.fps)
        store = CameraStore(captures)
        for label, camera in captures.items():
            threading.Thread(target=store.capture_loop, args=(label, camera), daemon=True).start()

        server = ThreadingHTTPServer((args.host, args.port), make_handler(store))
        print(f"Camera preview: http://{args.host}:{args.port}")
        server.serve_forever()
    except (OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nStopping camera preview.")
    finally:
        if store is not None:
            store.close()
        else:
            for camera in captures.values():
                camera.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
