import hashlib
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw
from streamlit_image_coordinates import streamlit_image_coordinates


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------

DEFAULT_STATE = {
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
    "active_sample_label": "",
    "label_candidate": "",
    "_label_message": None,
    "_last_added_click_id": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


st.title("2. Label Sample Points")

if st.session_state.map_image is None:
    st.warning("Please upload or fetch a map first.")
    st.stop()

img = np.asarray(st.session_state.map_image, dtype=np.uint8)
h, w = img.shape[:2]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def invalidate_classification_outputs() -> None:
    """Remove outputs that no longer match the current training samples."""
    for key in (
        "overlay_image",
        "mask_image",
        "summary_df",
        "classification_mask",
        "classification_confidence",
        "class_names",
        "shapefile_zip",
        "shapefile_summary_df",
    ):
        st.session_state[key] = None


def draw_samples_on_image(
    image: np.ndarray,
    samples_df: pd.DataFrame,
) -> np.ndarray:
    pil_image = Image.fromarray(image.copy())
    draw = ImageDraw.Draw(pil_image)

    for _, row in samples_df.iterrows():
        x = int(row["x"])
        y = int(row["y"])
        label = str(row["label"])
        radius = 6

        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline="red",
            width=3,
        )
        draw.text((x + 8, y - 8), label, fill="red")

    return np.asarray(pil_image)


def solve_homography(
    source_points: np.ndarray,
    destination_points: np.ndarray,
) -> np.ndarray:
    """Solve a projective transform from four point correspondences."""
    rows = []
    values = []

    for (x, y), (u, v) in zip(source_points, destination_points):
        rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        values.extend([u, v])

    coefficients = np.linalg.solve(
        np.asarray(rows, dtype=np.float64),
        np.asarray(values, dtype=np.float64),
    )

    return np.asarray(
        [
            [coefficients[0], coefficients[1], coefficients[2]],
            [coefficients[3], coefficients[4], coefficients[5]],
            [coefficients[6], coefficients[7], 1.0],
        ],
        dtype=np.float64,
    )


def add_world_coordinates(
    samples_df: pd.DataFrame,
    world_corners: Iterable[tuple[float, float]],
    image_width: int,
    image_height: int,
) -> pd.DataFrame:
    """Convert sample pixel positions to longitude/latitude without OpenCV."""
    source = np.asarray(
        [
            [0.0, 0.0],
            [image_width - 1.0, 0.0],
            [image_width - 1.0, image_height - 1.0],
            [0.0, image_height - 1.0],
        ],
        dtype=np.float64,
    )
    destination = np.asarray(list(world_corners), dtype=np.float64)

    if destination.shape != (4, 2):
        raise ValueError("world_corners must contain four longitude/latitude pairs")

    transform = solve_homography(source, destination)

    points = samples_df[["x", "y"]].to_numpy(dtype=np.float64)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    transformed = homogeneous @ transform.T
    transformed = transformed[:, :2] / transformed[:, 2:3]

    output = samples_df.copy()
    output["longitude"] = transformed[:, 0]
    output["latitude"] = transformed[:, 1]
    return output


# The original image supplied to the component never changes.
map_hash = hashlib.md5(img.tobytes()).hexdigest()

if st.session_state.get("_label_page_map_hash") != map_hash:
    st.session_state["_label_page_map_hash"] = map_hash
    st.session_state["_last_added_click_id"] = None
    st.session_state["_label_message"] = None


# ---------------------------------------------------------------------
# Active label
#
# The input is inside a form. Typing a new label therefore does not rerun
# the image component for every edit. The page reruns only when Set label
# is pressed.
# ---------------------------------------------------------------------

st.subheader("1. Set the active label")

existing_labels = sorted(
    {
        str(sample.get("label", "")).strip()
        for sample in st.session_state.samples
        if str(sample.get("label", "")).strip()
    }
)

with st.form("set_active_label_form", clear_on_submit=False):
    st.text_input(
        "Label name",
        key="label_candidate",
        placeholder="Example: water, road, forest, building",
    )

    set_label = st.form_submit_button(
        "Set active label",
        type="primary",
    )

if set_label:
    candidate = st.session_state.label_candidate.strip()

    if not candidate:
        st.session_state["_label_message"] = (
            "error",
            "Enter a non-empty label.",
        )
    else:
        st.session_state.active_sample_label = candidate
        st.session_state["_label_message"] = (
            "success",
            f"Active label changed to '{candidate}'.",
        )

active_label = st.session_state.active_sample_label.strip()

if active_label:
    st.success(f"Active label: **{active_label}**")
else:
    st.info("Set an active label before adding points.")

if existing_labels:
    st.caption("Existing labels: " + ", ".join(existing_labels))


# ---------------------------------------------------------------------
# Undo / clear
# ---------------------------------------------------------------------

undo_col, clear_col = st.columns(2)

with undo_col:
    if st.button("Undo last point", width="stretch"):
        if st.session_state.samples:
            removed = st.session_state.samples.pop()
            invalidate_classification_outputs()
            st.session_state["_label_message"] = (
                "success",
                f"Removed the last point from '{removed['label']}'.",
            )
        else:
            st.session_state["_label_message"] = (
                "warning",
                "There are no points to remove.",
            )

with clear_col:
    if st.button("Clear all points", width="stretch"):
        st.session_state.samples = []
        invalidate_classification_outputs()
        st.session_state["_last_added_click_id"] = None
        st.session_state["_label_message"] = (
            "success",
            "All sample points were cleared.",
        )


message = st.session_state.get("_label_message")
if message:
    level, text = message
    if level == "error":
        st.error(text)
    elif level == "warning":
        st.warning(text)


# ---------------------------------------------------------------------
# Stable image coordinate component
#
# Important:
# - one fixed key for the whole map;
# - no key counter;
# - no explicit st.rerun();
# - no annotated image passed back into the component.
# ---------------------------------------------------------------------

st.subheader("2. Select and add a point")

display_width = min(w, 900)
display_height = max(1, int(round(h * display_width / w)))

display_image = Image.fromarray(img).resize(
    (display_width, display_height),
    Image.Resampling.LANCZOS,
)

click = streamlit_image_coordinates(
    display_image,
    width=display_width,
    key=f"stable_map_click_{map_hash[:16]}",
    cursor="crosshair",
    image_format="PNG",
    png_compression_level=6,
)

if click is None:
    st.info("Click a location on the map.")
else:
    rendered_width = max(float(click.get("width", display_width)), 1.0)
    rendered_height = max(float(click.get("height", display_height)), 1.0)

    x = int(round(float(click["x"]) * w / rendered_width))
    y = int(round(float(click["y"]) * h / rendered_height))

    x = min(max(x, 0), w - 1)
    y = min(max(y, 0), h - 1)

    click_time = click.get("unix_time")
    click_id = (
        f"{click_time}:{x}:{y}"
        if click_time is not None
        else f"{x}:{y}"
    )

    st.write(f"Selected pixel: **x={x}, y={y}**")

    already_added = (
        click_id == st.session_state.get("_last_added_click_id")
    )

    if already_added:
        st.info("This click has already been added. Click a new location.")
    elif st.button(
        "Add selected point",
        type="primary",
        disabled=not bool(active_label),
    ):
        r, g, b = img[y, x].tolist()

        st.session_state.samples.append(
            {
                "label": active_label,
                "x": x,
                "y": y,
                "r": int(r),
                "g": int(g),
                "b": int(b),
            }
        )

        st.session_state["_last_added_click_id"] = click_id
        st.session_state["_label_message"] = (
            "success",
            f"Added point ({x}, {y}) to '{active_label}'.",
        )
        invalidate_classification_outputs()

        # No st.rerun() here. The current button-triggered run continues and
        # renders the updated table and preview below.
        st.success(
            f"Added point ({x}, {y}) to **{active_label}**."
        )


# ---------------------------------------------------------------------
# Samples and downloads
# ---------------------------------------------------------------------

samples_df = pd.DataFrame(st.session_state.samples)

st.subheader("3. Collected samples")

if samples_df.empty:
    st.info("No samples added yet.")
else:
    preview_col, table_col = st.columns([2, 1])

    with preview_col:
        preview = draw_samples_on_image(img, samples_df)
        st.image(
            preview,
            caption="Labelled sample points",
            width="stretch",
        )

    with table_col:
        st.dataframe(samples_df, width="stretch")

        st.download_button(
            "Download samples CSV",
            data=samples_df.to_csv(index=False).encode("utf-8"),
            file_name="map_label_samples.csv",
            mime="text/csv",
            width="stretch",
        )

    if st.session_state.world_corners is not None:
        try:
            geo_samples = add_world_coordinates(
                samples_df,
                st.session_state.world_corners,
                w,
                h,
            )

            st.subheader("Samples with real-world coordinates")
            st.dataframe(geo_samples, width="stretch")

            st.download_button(
                "Download georeferenced samples CSV",
                data=geo_samples.to_csv(index=False).encode("utf-8"),
                file_name="map_label_samples_georeferenced.csv",
                mime="text/csv",
            )
        except Exception as exc:
            st.warning(
                "Samples were saved, but geographic coordinates could not "
                f"be calculated: {exc}"
            )