import base64
import hashlib
from io import BytesIO

import folium
import numpy as np
import pandas as pd
import streamlit as st
from folium.raster_layers import ImageOverlay
from PIL import Image, ImageDraw
from streamlit_folium import st_folium


# ---------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------

DEFAULTS = {
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
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


st.title("2. Label Sample Points")

if st.session_state.map_image is None:
    st.warning("Please upload or fetch a map first.")
    st.stop()

img = np.asarray(st.session_state.map_image, dtype=np.uint8)
h, w = img.shape[:2]


# ---------------------------------------------------------------------
# Local helpers
#
# This page intentionally does not import utils.py. utils.py imports cv2
# globally, so keeping this page independent avoids loading OpenCV while
# the interactive map component is running.
# ---------------------------------------------------------------------

def invalidate_classification_outputs() -> None:
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


def image_to_data_url(array: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray(array).save(buffer, format="PNG", optimize=False)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def draw_samples_on_image(array: np.ndarray, samples: pd.DataFrame) -> np.ndarray:
    pil_image = Image.fromarray(array.copy())
    draw = ImageDraw.Draw(pil_image)

    for _, row in samples.iterrows():
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
    """Solve a 3x3 projective transform using four point correspondences."""
    matrix_rows = []
    target_values = []

    for (x, y), (u, v) in zip(source_points, destination_points):
        matrix_rows.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix_rows.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        target_values.extend([u, v])

    coefficients = np.linalg.solve(
        np.asarray(matrix_rows, dtype=np.float64),
        np.asarray(target_values, dtype=np.float64),
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
    samples: pd.DataFrame,
    world_corners,
    image_width: int,
    image_height: int,
) -> pd.DataFrame:
    source = np.asarray(
        [
            [0.0, 0.0],
            [image_width - 1.0, 0.0],
            [image_width - 1.0, image_height - 1.0],
            [0.0, image_height - 1.0],
        ],
        dtype=np.float64,
    )
    destination = np.asarray(world_corners, dtype=np.float64)

    transform = solve_homography(source, destination)

    points = samples[["x", "y"]].to_numpy(dtype=np.float64)
    homogeneous = np.column_stack([points, np.ones(len(points))])
    transformed = homogeneous @ transform.T
    transformed = transformed[:, :2] / transformed[:, 2:3]

    output = samples.copy()
    output["longitude"] = transformed[:, 0]
    output["latitude"] = transformed[:, 1]
    return output


# ---------------------------------------------------------------------
# Map identity and controls
# ---------------------------------------------------------------------

map_hash = hashlib.md5(img.tobytes()).hexdigest()

if st.session_state.get("_folium_label_map_hash") != map_hash:
    st.session_state["_folium_label_map_hash"] = map_hash
    st.session_state["_folium_label_version"] = 0
    st.session_state["_last_saved_click"] = None
    st.session_state["_label_message"] = None

if "_folium_label_version" not in st.session_state:
    st.session_state["_folium_label_version"] = 0

if "sample_label" not in st.session_state:
    st.session_state.sample_label = ""


label_col, undo_col, clear_col = st.columns([3, 1, 1])

with label_col:
    st.text_input(
        "Current label",
        key="sample_label",
        placeholder="Example: water, forest, building, road",
    )

with undo_col:
    st.write("")
    st.write("")
    if st.button("Undo last", width="stretch"):
        if st.session_state.samples:
            removed = st.session_state.samples.pop()
            invalidate_classification_outputs()
            st.session_state["_folium_label_version"] += 1
            st.session_state["_last_saved_click"] = None
            st.session_state["_label_message"] = (
                "success",
                f"Removed the last point from '{removed['label']}'.",
            )
            st.rerun()
        else:
            st.session_state["_label_message"] = (
                "warning",
                "There are no points to remove.",
            )

with clear_col:
    st.write("")
    st.write("")
    if st.button("Clear all", width="stretch"):
        st.session_state.samples = []
        invalidate_classification_outputs()
        st.session_state["_folium_label_version"] += 1
        st.session_state["_last_saved_click"] = None
        st.session_state["_label_message"] = (
            "success",
            "All sample points were cleared.",
        )
        st.rerun()


message = st.session_state.get("_label_message")
if message:
    level, text = message
    if level == "error":
        st.error(text)
    elif level == "warning":
        st.warning(text)
    else:
        st.success(text)


# ---------------------------------------------------------------------
# Folium map in pixel coordinates
#
# Leaflet's Simple CRS is used:
#   longitude-like coordinate -> image x
#   latitude-like coordinate  -> image height - image y
# ---------------------------------------------------------------------

st.subheader("Click a point on the map")

st.caption(
    "Click the map once, check the selected pixel coordinate, and then "
    "press 'Add selected point'."
)

map_height = int(np.clip(700 * h / max(w, 1), 450, 800))

leaflet_map = folium.Map(
    location=[h / 2.0, w / 2.0],
    zoom_start=0,
    crs="Simple",
    tiles=None,
    min_zoom=-5,
    max_zoom=6,
    zoom_control=True,
    attribution_control=False,
    prefer_canvas=True,
)

ImageOverlay(
    image=image_to_data_url(img),
    bounds=[[0, 0], [h, w]],
    opacity=1.0,
    interactive=False,
    cross_origin=False,
    zindex=1,
).add_to(leaflet_map)

for sample in st.session_state.samples:
    folium.CircleMarker(
        location=[h - float(sample["y"]), float(sample["x"])],
        radius=5,
        color="red",
        weight=2,
        fill=True,
        fill_color="red",
        fill_opacity=0.35,
        tooltip=str(sample["label"]),
    ).add_to(leaflet_map)

leaflet_map.fit_bounds([[0, 0], [h, w]])

map_result = st_folium(
    leaflet_map,
    key=(
        f"pixel_label_map_{map_hash[:12]}_"
        f"{st.session_state['_folium_label_version']}"
    ),
    width=None,
    height=map_height,
    returned_objects=["last_clicked"],
    pixelated=False,
)


# ---------------------------------------------------------------------
# Save selected click
# ---------------------------------------------------------------------

clicked = map_result.get("last_clicked") if map_result else None

if clicked is None:
    st.info("Click a location on the map.")
else:
    x = int(round(float(clicked["lng"])))
    y = int(round(h - float(clicked["lat"])))

    x = min(max(x, 0), w - 1)
    y = min(max(y, 0), h - 1)

    click_signature = f"{x}:{y}"

    st.write(f"Selected pixel: **x={x}, y={y}**")

    label = st.session_state.sample_label.strip()

    if not label:
        st.warning("Enter a label before adding this point.")

    already_saved = click_signature == st.session_state.get("_last_saved_click")

    if already_saved:
        st.info("That click was already saved. Click another map position.")
    elif st.button(
        "Add selected point",
        type="primary",
        disabled=not bool(label),
    ):
        r, g, b = img[y, x].tolist()

        st.session_state.samples.append(
            {
                "label": label,
                "x": x,
                "y": y,
                "r": int(r),
                "g": int(g),
                "b": int(b),
            }
        )

        invalidate_classification_outputs()
        st.session_state["_last_saved_click"] = click_signature
        st.session_state["_folium_label_version"] += 1
        st.session_state["_label_message"] = (
            "success",
            f"Added point ({x}, {y}) to '{label}'.",
        )
        st.rerun()


# ---------------------------------------------------------------------
# Samples and downloads
# ---------------------------------------------------------------------

samples_df = pd.DataFrame(st.session_state.samples)

st.subheader("Collected samples")

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
                "The sample points were saved, but real-world coordinates "
                f"could not be calculated: {exc}"
            )