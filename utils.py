from io import BytesIO
import zipfile

import cv2
import numpy as np
import pandas as pd
import shapefile
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
        "classification_mask": None,
        "classification_confidence": None,
        "class_names": None,
        "shapefile_zip": None,
        "shapefile_summary_df": None,
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

WGS84_ESRI_WKT = (
    'GEOGCS["GCS_WGS_1984",DATUM["D_WGS_1984",'
    'SPHEROID["WGS_1984",6378137.0,298.257223563]],'
    'PRIMEM["Greenwich",0.0],UNIT["Degree",0.0174532925199433]]'
)


def pixel_points_to_world(points_xy, world_corners, img_w, img_h):
    """Transform Nx2 pixel coordinates to longitude/latitude coordinates."""
    points_xy = np.asarray(points_xy, dtype=np.float32)
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy must have shape (N, 2)")

    src = np.float32(
        [
            [0, 0],
            [img_w - 1, 0],
            [img_w - 1, img_h - 1],
            [0, img_h - 1],
        ]
    )
    dst = np.float32(world_corners)
    transform = cv2.getPerspectiveTransform(src, dst)

    return cv2.perspectiveTransform(
        points_xy.reshape(-1, 1, 2), transform
    ).reshape(-1, 2)


def _signed_ring_area(ring):
    """Signed polygon area; positive means counter-clockwise."""
    arr = np.asarray(ring, dtype=np.float64)
    if len(arr) < 3:
        return 0.0
    x = arr[:, 0]
    y = arr[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _prepare_ring(points, clockwise):
    """Close and orient a ring for an ESRI polygon shapefile."""
    ring = [[float(x), float(y)] for x, y in points]
    if len(ring) < 3:
        return None

    if ring[0] != ring[-1]:
        ring.append(ring[0])

    area = _signed_ring_area(ring[:-1])
    should_reverse = (clockwise and area > 0) or ((not clockwise) and area < 0)
    if should_reverse:
        open_ring = list(reversed(ring[:-1]))
        ring = open_ring + [open_ring[0]]

    return ring


def classification_to_shapefile_zip(
    pred_mask,
    classes,
    world_corners,
    confidence_mask=None,
    min_region_pixels=25,
    simplify_tolerance_pixels=1.5,
    include_unknown=False,
    basename="classified_map",
):
    """
    Polygonize a classified raster mask and return a zipped ESRI Shapefile.

    Parameters
    ----------
    pred_mask:
        HxW integer array. Known classes use indices 0..len(classes)-1;
        low-confidence/unknown pixels use -1.
    world_corners:
        Four (longitude, latitude) pairs in this order:
        top-left, top-right, bottom-right, bottom-left.
    min_region_pixels:
        Connected components smaller than this are omitted to reduce noise.
    simplify_tolerance_pixels:
        Douglas-Peucker contour simplification tolerance in image pixels.
    """
    pred_mask = np.asarray(pred_mask)
    if pred_mask.ndim != 2:
        raise ValueError("pred_mask must be a 2D array")
    if world_corners is None or len(world_corners) != 4:
        raise ValueError("Four real-world corner coordinates are required")
    if min_region_pixels < 1:
        raise ValueError("min_region_pixels must be at least 1")
    if simplify_tolerance_pixels < 0:
        raise ValueError("simplify_tolerance_pixels cannot be negative")

    classes = [str(value) for value in classes]
    h, w = pred_mask.shape
    total_pixels = int(h * w)

    if confidence_mask is not None:
        confidence_mask = np.asarray(confidence_mask, dtype=np.float32)
        if confidence_mask.shape != pred_mask.shape:
            raise ValueError("confidence_mask must have the same shape as pred_mask")

    shp_io = BytesIO()
    shx_io = BytesIO()
    dbf_io = BytesIO()

    writer = shapefile.Writer(
        shp=shp_io,
        shx=shx_io,
        dbf=dbf_io,
        shapeType=shapefile.POLYGON,
        encoding="utf-8",
    )
    writer.autoBalance = 1
    writer.field("CLASS_ID", "N", size=6, decimal=0)
    writer.field("LABEL", "C", size=100)
    writer.field("COMPONENT", "N", size=10, decimal=0)
    writer.field("PIXELS", "N", size=14, decimal=0)
    writer.field("PCT_TOTAL", "F", size=14, decimal=6)
    writer.field("AVG_CONF", "F", size=12, decimal=6)

    class_items = [(idx, label) for idx, label in enumerate(classes)]
    if include_unknown:
        class_items.append((-1, "unknown / low confidence"))

    feature_rows = []
    feature_id = 0

    for class_id, label in class_items:
        binary = (pred_mask == class_id).astype(np.uint8)
        if not np.any(binary):
            continue

        n_components, component_labels, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )

        for component_id in range(1, n_components):
            pixel_count = int(stats[component_id, cv2.CC_STAT_AREA])
            if pixel_count < int(min_region_pixels):
                continue

            component_mask = (component_labels == component_id).astype(np.uint8) * 255
            contours, hierarchy = cv2.findContours(
                component_mask,
                cv2.RETR_CCOMP,
                cv2.CHAIN_APPROX_SIMPLE,
            )

            if not contours or hierarchy is None:
                continue

            hierarchy = hierarchy[0]
            mean_confidence = -1.0
            if confidence_mask is not None:
                mean_confidence = float(
                    np.mean(confidence_mask[component_labels == component_id])
                )

            for contour_index, contour in enumerate(contours):
                # Only top-level contours are outer rings. Their children are holes.
                if hierarchy[contour_index][3] != -1:
                    continue

                epsilon = float(simplify_tolerance_pixels)
                outer = cv2.approxPolyDP(contour, epsilon, closed=True)
                if len(outer) < 3:
                    continue

                outer_world = pixel_points_to_world(
                    outer.reshape(-1, 2),
                    world_corners,
                    w,
                    h,
                )
                outer_ring = _prepare_ring(outer_world, clockwise=True)
                if outer_ring is None:
                    continue

                parts = [outer_ring]
                child_index = int(hierarchy[contour_index][2])

                while child_index != -1:
                    hole = cv2.approxPolyDP(
                        contours[child_index],
                        epsilon,
                        closed=True,
                    )
                    if len(hole) >= 3:
                        hole_world = pixel_points_to_world(
                            hole.reshape(-1, 2),
                            world_corners,
                            w,
                            h,
                        )
                        hole_ring = _prepare_ring(hole_world, clockwise=False)
                        if hole_ring is not None:
                            parts.append(hole_ring)

                    child_index = int(hierarchy[child_index][0])

                feature_id += 1
                writer.poly(parts)
                writer.record(
                    int(class_id),
                    label,
                    int(component_id),
                    pixel_count,
                    100.0 * pixel_count / total_pixels,
                    mean_confidence,
                )
                feature_rows.append(
                    {
                        "feature_id": feature_id,
                        "class_id": int(class_id),
                        "label": label,
                        "component": int(component_id),
                        "pixels": pixel_count,
                        "percent_total": 100.0 * pixel_count / total_pixels,
                        "average_confidence": mean_confidence,
                    }
                )

    writer.close()

    if feature_id == 0:
        raise ValueError(
            "No polygons remained after filtering. Lower the minimum region size."
        )

    readme = (
        "Classified map polygon export\n"
        "=============================\n"
        "Coordinate reference system: WGS 84 longitude/latitude (EPSG:4326).\n"
        "Each record is one connected classified region.\n"
        f"Minimum connected-region size: {int(min_region_pixels)} pixels.\n"
        f"Contour simplification tolerance: {float(simplify_tolerance_pixels):.3f} pixels.\n"
        f"Unknown/low-confidence polygons included: {bool(include_unknown)}.\n"
        "The ZIP contains .shp, .shx, .dbf, .prj and .cpg files.\n"
    )

    zip_io = BytesIO()
    with zipfile.ZipFile(zip_io, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(f"{basename}.shp", shp_io.getvalue())
        zf.writestr(f"{basename}.shx", shx_io.getvalue())
        zf.writestr(f"{basename}.dbf", dbf_io.getvalue())
        zf.writestr(f"{basename}.prj", WGS84_ESRI_WKT)
        zf.writestr(f"{basename}.cpg", "UTF-8")
        zf.writestr("README.txt", readme)

    return zip_io.getvalue(), pd.DataFrame(feature_rows)