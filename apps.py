import hashlib
from io import BytesIO

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from streamlit_drawable_canvas import st_canvas

from utils import classification_to_shapefile_zip


st.set_page_config(page_title="Interactive Map Mapper", layout="wide")

st.title("Interactive Map Pixel Classifier")


# -----------------------------
# Utility functions
# -----------------------------

def file_hash(uploaded_file):
    return hashlib.md5(uploaded_file.getvalue()).hexdigest()


def resize_to_max_side(img, max_side=1000):
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))

    if scale == 1.0:
        return img

    new_w = int(w * scale)
    new_h = int(h * scale)

    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.array(pil_img)


def build_features(img):
    """
    Build per-pixel features using:
    - RGB
    - HSV
    - LAB
    - normalized x/y position

    Shape:
        H x W x feature_dim
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
        draw.ellipse((x - r, y - r, x + r, y + r), outline="red", width=2)
        draw.text((x + 7, y - 7), label, fill="red")

    return np.array(pil)


def image_to_png_bytes(img):
    pil = Image.fromarray(img)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def parse_corner_text(text):
    """
    Expects:
        lon,lat

    Example:
        7.12345,50.98765
    """

    lon, lat = text.split(",")
    return float(lon.strip()), float(lat.strip())


def add_world_coordinates(samples_df, world_corners, img_w, img_h):
    """
    Convert clicked pixel points to real-world coordinates.

    world_corners order:
        top-left, top-right, bottom-right, bottom-left

    Each corner should be:
        lon, lat
    """

    src = np.float32([
        [0, 0],
        [img_w - 1, 0],
        [img_w - 1, img_h - 1],
        [0, img_h - 1],
    ])

    dst = np.float32(world_corners)

    H = cv2.getPerspectiveTransform(src, dst)

    pts = samples_df[["x", "y"]].to_numpy(np.float32).reshape(-1, 1, 2)
    world_pts = cv2.perspectiveTransform(pts, H).reshape(-1, 2)

    out = samples_df.copy()
    out["longitude"] = world_pts[:, 0]
    out["latitude"] = world_pts[:, 1]

    return out


# -----------------------------
# Session state
# -----------------------------

if "samples" not in st.session_state:
    st.session_state.samples = []

if "canvas_version" not in st.session_state:
    st.session_state.canvas_version = 0

if "current_image_hash" not in st.session_state:
    st.session_state.current_image_hash = None

if "classification_mask" not in st.session_state:
    st.session_state.classification_mask = None

if "classification_confidence" not in st.session_state:
    st.session_state.classification_confidence = None

if "class_names" not in st.session_state:
    st.session_state.class_names = None

if "shapefile_zip" not in st.session_state:
    st.session_state.shapefile_zip = None

if "shapefile_summary_df" not in st.session_state:
    st.session_state.shapefile_summary_df = None


# -----------------------------
# Sidebar controls
# -----------------------------

st.sidebar.header("1. Map input")

input_mode = st.sidebar.radio(
    "Choose input type",
    [
        "Upload map image",
        "Upload image with real-world corner coordinates",
    ],
)

uploaded_file = st.sidebar.file_uploader(
    "Upload map image",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
)

max_side = st.sidebar.slider(
    "Working image size",
    min_value=400,
    max_value=1800,
    value=1000,
    step=100,
)

use_georef = input_mode == "Upload image with real-world corner coordinates"

world_corners = None

if use_georef:
    st.sidebar.header("2. Real-world coordinates")

    st.sidebar.caption(
        "Enter coordinates as longitude,latitude. "
        "Order should match image corners."
    )

    tl = st.sidebar.text_input("Top-left lon,lat", "")
    tr = st.sidebar.text_input("Top-right lon,lat", "")
    br = st.sidebar.text_input("Bottom-right lon,lat", "")
    bl = st.sidebar.text_input("Bottom-left lon,lat", "")

    try:
        if tl and tr and br and bl:
            world_corners = [
                parse_corner_text(tl),
                parse_corner_text(tr),
                parse_corner_text(br),
                parse_corner_text(bl),
            ]
    except Exception:
        st.sidebar.error("Coordinate format should be: longitude,latitude")


# -----------------------------
# Load image
# -----------------------------

if uploaded_file is None:
    st.info("Upload a map image to begin.")
    st.stop()

new_hash = file_hash(uploaded_file)

if new_hash != st.session_state.current_image_hash:
    st.session_state.samples = []
    st.session_state.classification_mask = None
    st.session_state.classification_confidence = None
    st.session_state.class_names = None
    st.session_state.shapefile_zip = None
    st.session_state.shapefile_summary_df = None
    st.session_state.canvas_version += 1
    st.session_state.current_image_hash = new_hash

img = Image.open(uploaded_file).convert("RGB")
img = np.array(img)
img = resize_to_max_side(img, max_side=max_side)

h, w = img.shape[:2]

st.write(f"Working image size: **{w} × {h} pixels**")

if use_georef and world_corners is None:
    st.warning(
        "You selected real-world coordinate mode, but the 4 corner coordinates are not complete yet."
    )


# -----------------------------
# Layout
# -----------------------------

left, right = st.columns([2, 1])

with right:
    st.header("Label samples")

    current_label = st.text_input(
        "Current label",
        placeholder="Example: water, highway, forest, building",
    )

    st.caption(
        "Draw/click points on the image for the selected label, then press "
        "'Add drawn points'."
    )

    clear_all = st.button("Clear all samples")

    if clear_all:
        st.session_state.samples = []
        st.session_state.classification_mask = None
        st.session_state.classification_confidence = None
        st.session_state.class_names = None
        st.session_state.shapefile_zip = None
        st.session_state.shapefile_summary_df = None
        st.session_state.canvas_version += 1
        st.rerun()


with left:
    st.header("Click sample points")

    canvas_result = st_canvas(
        background_image=Image.fromarray(img),
        height=h,
        width=w,
        drawing_mode="point",
        stroke_width=8,
        update_streamlit=True,
        key=f"canvas_{st.session_state.canvas_version}",
    )


# -----------------------------
# Add points
# -----------------------------

with right:
    add_points = st.button("Add drawn points to this label")

    if add_points:
        if not current_label.strip():
            st.error("Please enter a label first.")
        else:
            points = extract_canvas_points(canvas_result.json_data, w, h)

            if len(points) == 0:
                st.error("No points found. Click some points on the image first.")
            else:
                for x, y in points:
                    r, g, b = img[y, x].tolist()

                    st.session_state.samples.append(
                        {
                            "label": current_label.strip(),
                            "x": x,
                            "y": y,
                            "r": r,
                            "g": g,
                            "b": b,
                        }
                    )

                st.session_state.classification_mask = None
                st.session_state.classification_confidence = None
                st.session_state.class_names = None
                st.session_state.shapefile_zip = None
                st.session_state.shapefile_summary_df = None

                st.success(f"Added {len(points)} points for label: {current_label}")
                st.session_state.canvas_version += 1
                st.rerun()


# -----------------------------
# Show samples
# -----------------------------

samples_df = pd.DataFrame(st.session_state.samples)

if len(samples_df) > 0:
    st.header("Collected training samples")

    preview_img = draw_samples_on_image(img, samples_df)

    c1, c2 = st.columns([2, 1])

    with c1:
        st.image(preview_img, caption="Sample points", use_container_width=True)

    with c2:
        st.dataframe(samples_df, use_container_width=True)

        csv_data = samples_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download samples CSV",
            data=csv_data,
            file_name="map_label_samples.csv",
            mime="text/csv",
        )

    if use_georef and world_corners is not None:
        geo_samples = add_world_coordinates(samples_df, world_corners, w, h)

        st.subheader("Samples with real-world coordinates")
        st.dataframe(geo_samples, use_container_width=True)

        st.download_button(
            "Download georeferenced samples CSV",
            data=geo_samples.to_csv(index=False).encode("utf-8"),
            file_name="map_label_samples_georeferenced.csv",
            mime="text/csv",
        )


# -----------------------------
# Train and classify
# -----------------------------

st.header("Automatic map classification")

if len(samples_df) == 0:
    st.info("Add labelled sample points first.")
    st.stop()

label_counts = samples_df["label"].value_counts()

st.write("Samples per label:")
st.dataframe(label_counts.rename("count"), use_container_width=True)

if samples_df["label"].nunique() < 2:
    st.warning("Add at least two different labels before classification.")
    st.stop()

min_samples_per_label = label_counts.min()

if min_samples_per_label < 3:
    st.warning(
        "Some labels have fewer than 3 samples. "
        "The result may be unstable. Add more points per label for better results."
    )

confidence_threshold = st.slider(
    "Minimum confidence threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.50,
    step=0.05,
)

alpha = st.slider(
    "Overlay strength",
    min_value=0.1,
    max_value=0.9,
    value=0.45,
    step=0.05,
)

run_classification = st.button("Run automatic mapping")

if run_classification:
    with st.spinner("Classifying map..."):
        features_img = build_features(img)
        feature_dim = features_img.shape[-1]

        sample_x = samples_df["x"].to_numpy()
        sample_y = samples_df["y"].to_numpy()

        X_train = features_img[sample_y, sample_x]
        y_train = samples_df["label"].to_numpy()

        clf = make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,
            ),
        )

        clf.fit(X_train, y_train)

        X_all = features_img.reshape(-1, feature_dim)

        proba = clf.predict_proba(X_all)
        pred_idx = np.argmax(proba, axis=1)
        confidence = np.max(proba, axis=1)

        classes = list(clf.classes_)

        pred_mask = pred_idx.reshape(h, w)
        conf_mask = confidence.reshape(h, w)

        pred_mask[conf_mask < confidence_threshold] = -1

        palette = np.array(
            [
                [230, 25, 75],
                [60, 180, 75],
                [255, 225, 25],
                [0, 130, 200],
                [245, 130, 48],
                [145, 30, 180],
                [70, 240, 240],
                [240, 50, 230],
                [210, 245, 60],
                [250, 190, 190],
                [0, 128, 128],
                [230, 190, 255],
            ],
            dtype=np.uint8,
        )

        seg_rgb = np.full_like(img, 230)

        for i, label in enumerate(classes):
            seg_rgb[pred_mask == i] = palette[i % len(palette)]

        overlay = img.copy()
        known_pixels = pred_mask >= 0

        overlay[known_pixels] = (
            (1 - alpha) * img[known_pixels] + alpha * seg_rgb[known_pixels]
        ).astype(np.uint8)

        unknown_pixels = pred_mask < 0
        overlay[unknown_pixels] = img[unknown_pixels]

        st.subheader("Classification overlay")
        st.image(overlay, use_container_width=True)

        st.subheader("Pure classified mask")
        st.image(seg_rgb, use_container_width=True)

        # Summary table
        total_pixels = h * w
        rows = []

        for i, label in enumerate(classes):
            pixels = int((pred_mask == i).sum())
            rows.append(
                {
                    "label": label,
                    "pixels": pixels,
                    "percent": round(100 * pixels / total_pixels, 2),
                }
            )

        unknown_count = int((pred_mask == -1).sum())
        rows.append(
            {
                "label": "unknown / low confidence",
                "pixels": unknown_count,
                "percent": round(100 * unknown_count / total_pixels, 2),
            }
        )

        summary_df = pd.DataFrame(rows)

        st.session_state.classification_mask = pred_mask.astype(np.int16)
        st.session_state.classification_confidence = conf_mask.astype(np.float32)
        st.session_state.class_names = [str(label) for label in classes]
        st.session_state.shapefile_zip = None
        st.session_state.shapefile_summary_df = None

        st.subheader("Area summary")
        st.dataframe(summary_df, use_container_width=True)

        st.download_button(
            "Download classification overlay PNG",
            data=image_to_png_bytes(overlay),
            file_name="classified_map_overlay.png",
            mime="image/png",
        )

        st.download_button(
            "Download classified mask PNG",
            data=image_to_png_bytes(seg_rgb),
            file_name="classified_map_mask.png",
            mime="image/png",
        )

        st.download_button(
            "Download area summary CSV",
            data=summary_df.to_csv(index=False).encode("utf-8"),
            file_name="classification_area_summary.csv",
            mime="text/csv",
        )

# -----------------------------
# Vector/Shapefile export
# -----------------------------

if st.session_state.classification_mask is not None:
    st.divider()
    st.subheader("Download classified polygons as an ESRI Shapefile")

    if world_corners is None:
        st.warning(
            "Enter all four real-world corner coordinates to enable the "
            "georeferenced Shapefile export."
        )
    else:
        st.caption(
            "The classified raster is polygonized in WGS 84 longitude/latitude "
            "(EPSG:4326). The Shapefile components are bundled in one ZIP file."
        )

        c1, c2 = st.columns(2)
        with c1:
            shp_min_pixels = st.number_input(
                "Minimum connected-region size (pixels)",
                min_value=1,
                max_value=1_000_000,
                value=25,
                step=10,
                key="single_page_shp_min_pixels",
            )
        with c2:
            shp_simplify = st.number_input(
                "Polygon simplification tolerance (pixels)",
                min_value=0.0,
                max_value=25.0,
                value=1.5,
                step=0.5,
                key="single_page_shp_simplify",
            )

        shp_include_unknown = st.checkbox(
            "Include unknown / low-confidence areas",
            value=False,
            key="single_page_shp_unknown",
        )

        if st.button("Prepare Shapefile ZIP", key="single_page_prepare_shp"):
            try:
                with st.spinner("Converting classified pixels to polygons..."):
                    shp_bytes, shp_summary = classification_to_shapefile_zip(
                        pred_mask=st.session_state.classification_mask,
                        classes=st.session_state.class_names,
                        world_corners=world_corners,
                        confidence_mask=st.session_state.classification_confidence,
                        min_region_pixels=int(shp_min_pixels),
                        simplify_tolerance_pixels=float(shp_simplify),
                        include_unknown=shp_include_unknown,
                        basename="classified_map",
                    )
                st.session_state.shapefile_zip = shp_bytes
                st.session_state.shapefile_summary_df = shp_summary
                st.success(
                    f"Shapefile prepared with {len(shp_summary):,} polygon features."
                )
            except Exception as exc:
                st.session_state.shapefile_zip = None
                st.session_state.shapefile_summary_df = None
                st.error(f"Could not create the Shapefile: {exc}")

        if st.session_state.shapefile_summary_df is not None:
            st.dataframe(
                st.session_state.shapefile_summary_df,
                use_container_width=True,
            )

        if st.session_state.shapefile_zip is not None:
            st.download_button(
                "Download classified map Shapefile ZIP",
                data=st.session_state.shapefile_zip,
                file_name="classified_map_shapefile.zip",
                mime="application/zip",
                key="single_page_download_shp",
            )