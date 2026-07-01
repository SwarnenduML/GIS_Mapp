from io import BytesIO

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


def init_state():
    defaults = {
        "map_image": None,
        "map_filename": None,
        "world_corners": None,
        "samples": [],
        "overlay_image": None,
        "mask_image": None,
        "summary_df": None,
        "canvas_version": 0,
    }

    import streamlit as st

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def resize_to_max_side(img, max_side=1200):
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))

    if scale == 1.0:
        return img

    new_w = int(w * scale)
    new_h = int(h * scale)

    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.array(pil_img)


def parse_corner_text(text):
    """
    Expected format:
        longitude,latitude

    Example:
        7.12345,50.98765
    """
    lon, lat = text.split(",")
    return float(lon.strip()), float(lat.strip())


def extract_canvas_points(canvas_json, img_w, img_h):
    points = []

    if not canvas_json:
        return points

    objects = canvas_json.get("objects", [])

    for obj in objects:
        if "left" not in obj or "top" not in obj:
            continue

        x = float(obj["left"])
        y = float(obj["top"])

        if obj.get("type") == "circle":
            if obj.get("originX") != "center":
                x += float(obj.get("radius", 0))
            if obj.get("originY") != "center":
                y += float(obj.get("radius", 0))

        x = int(round(x))
        y = int(round(y))

        if 0 <= x < img_w and 0 <= y < img_h:
            points.append((x, y))

    return points


def draw_samples_on_image(img, samples_df):
    pil = Image.fromarray(img.copy())
    draw = ImageDraw.Draw(pil)

    for _, row in samples_df.iterrows():
        x = int(row["x"])
        y = int(row["y"])
        label = str(row["label"])

        r = 5
        draw.ellipse(
            (x - r, y - r, x + r, y + r),
            outline="red",
            width=2,
        )
        draw.text((x + 7, y - 7), label, fill="red")

    return np.array(pil)


def build_features(img):
    """
    Creates per-pixel features:
    - RGB
    - HSV
    - LAB
    - normalized x/y position
    """

    h, w = img.shape[:2]

    rgb = img.astype(np.float32) / 255.0

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] /= 179.0
    hsv[..., 1] /= 255.0
    hsv[..., 2] /= 255.0

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
    lab /= 255.0

    yy, xx = np.mgrid[0:h, 0:w]

    xx = xx.astype(np.float32) / max(w - 1, 1)
    yy = yy.astype(np.float32) / max(h - 1, 1)

    xx = xx[..., None]
    yy = yy[..., None]

    features = np.concatenate([rgb, hsv, lab, xx, yy], axis=-1)
    return features


def image_to_png_bytes(img):
    pil = Image.fromarray(img)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def add_world_coordinates(samples_df, world_corners, img_w, img_h):
    """
    Converts pixel x/y points to approximate real-world coordinates.

    world_corners order:
        top-left, top-right, bottom-right, bottom-left

    Each corner:
        longitude, latitude
    """

    src = np.float32(
        [
            [0, 0],
            [img_w - 1, 0],
            [img_w - 1, img_h - 1],
            [0, img_h - 1],
        ]
    )

    dst = np.float32(world_corners)

    H = cv2.getPerspectiveTransform(src, dst)

    pts = samples_df[["x", "y"]].to_numpy(np.float32).reshape(-1, 1, 2)
    world_pts = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    out = samples_df.copy()
    out["longitude"] = world_pts[:, 0]
    out["latitude"] = world_pts[:, 1]

    return out