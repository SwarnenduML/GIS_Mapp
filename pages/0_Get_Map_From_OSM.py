import math
import time
from io import BytesIO

import numpy as np
import requests
import streamlit as st
from PIL import Image

from utils import init_state


init_state()

st.title("0. Get Map From OpenStreetMap Coordinates")

st.write(
    """
    Enter a real-world bounding box. The app will download the required 
    OpenStreetMap raster tiles, stitch them into one image, crop the exact 
    selected area, and pass it to the labelling/classification workflow.
    """
)

st.warning(
    """
    This uses the public OpenStreetMap tile server for small interactive tests only.
    Do not use this for bulk downloading, large areas, high-volume apps, or offline archives.
    For production, use your own tile server or a commercial OSM tile provider.
    """
)

st.caption("Map data © OpenStreetMap contributors")


# -----------------------------
# OSM tile helpers
# -----------------------------

TILE_SIZE = 256


def clamp_lat(lat):
    """
    Web Mercator cannot represent the poles.
    """
    return max(min(lat, 85.05112878), -85.05112878)


def latlon_to_tile_fraction(lat, lon, zoom):
    """
    Converts lat/lon to fractional OSM tile coordinates.
    """
    lat = clamp_lat(lat)

    lat_rad = math.radians(lat)
    n = 2 ** zoom

    x = (lon + 180.0) / 360.0 * n

    y = (
        1.0
        - math.log(
            math.tan(lat_rad) + 1.0 / math.cos(lat_rad)
        )
        / math.pi
    ) / 2.0 * n

    return x, y


def tile_fraction_to_latlon(x, y, zoom):
    """
    Converts fractional OSM tile coordinates to lat/lon.
    """
    n = 2 ** zoom

    lon = x / n * 360.0 - 180.0

    lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
    lat = math.degrees(lat_rad)

    return lat, lon


def bbox_to_tile_range(north, south, east, west, zoom):
    """
    Returns integer tile range covering a bounding box.
    """
    x_min_f, y_min_f = latlon_to_tile_fraction(north, west, zoom)
    x_max_f, y_max_f = latlon_to_tile_fraction(south, east, zoom)

    x_min = math.floor(min(x_min_f, x_max_f))
    x_max = math.floor(max(x_min_f, x_max_f))

    y_min = math.floor(min(y_min_f, y_max_f))
    y_max = math.floor(max(y_min_f, y_max_f))

    return x_min, x_max, y_min, y_max


def fetch_osm_tile(z, x, y, user_agent):
    """
    Downloads one tile from OpenStreetMap.
    """

    url = f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"

    headers = {
        "User-Agent": user_agent,
        "Accept": "image/png,image/*;q=0.8,*/*;q=0.5",
        "Referer": "http://localhost:8501/",
    }

    response = requests.get(url, headers=headers, timeout=30)

    if response.status_code != 200:
        raise RuntimeError(
            f"Failed to fetch tile {z}/{x}/{y}: "
            f"{response.status_code} - {response.text[:300]}"
        )

    return Image.open(BytesIO(response.content)).convert("RGB")


def stitch_and_crop_osm_bbox(north, south, east, west, zoom, user_agent):
    """
    Downloads all tiles covering the bbox, stitches them,
    and crops exactly to the selected bbox.
    """
    x_min, x_max, y_min, y_max = bbox_to_tile_range(
        north=north,
        south=south,
        east=east,
        west=west,
        zoom=zoom,
    )

    num_x = x_max - x_min + 1
    num_y = y_max - y_min + 1
    total_tiles = num_x * num_y

    if total_tiles > 25:
        raise RuntimeError(
            f"This request needs {total_tiles} tiles. "
            f"Please reduce the area or lower the zoom. "
            f"For public OSM tiles, keep this small."
        )

    stitched = Image.new(
        "RGB",
        (num_x * TILE_SIZE, num_y * TILE_SIZE),
    )

    progress = st.progress(0)
    tile_count = 0

    for ty in range(y_min, y_max + 1):
        for tx in range(x_min, x_max + 1):
            tile = fetch_osm_tile(
                z=zoom,
                x=tx,
                y=ty,
                user_agent=user_agent,
            )

            paste_x = (tx - x_min) * TILE_SIZE
            paste_y = (ty - y_min) * TILE_SIZE

            stitched.paste(tile, (paste_x, paste_y))

            tile_count += 1
            progress.progress(tile_count / total_tiles)

            # Be polite to the public tile server.
            time.sleep(0.15)

    # Convert bbox edges to pixel positions inside stitched image.
    west_x_f, north_y_f = latlon_to_tile_fraction(north, west, zoom)
    east_x_f, south_y_f = latlon_to_tile_fraction(south, east, zoom)

    left_px = int(round((west_x_f - x_min) * TILE_SIZE))
    right_px = int(round((east_x_f - x_min) * TILE_SIZE))
    top_px = int(round((north_y_f - y_min) * TILE_SIZE))
    bottom_px = int(round((south_y_f - y_min) * TILE_SIZE))

    left_px, right_px = sorted([left_px, right_px])
    top_px, bottom_px = sorted([top_px, bottom_px])

    cropped = stitched.crop((left_px, top_px, right_px, bottom_px))

    # Calculate the actual georeferenced corners of the cropped image.
    cropped_w, cropped_h = cropped.size

    actual_left_tile_f = x_min + left_px / TILE_SIZE
    actual_right_tile_f = x_min + right_px / TILE_SIZE
    actual_top_tile_f = y_min + top_px / TILE_SIZE
    actual_bottom_tile_f = y_min + bottom_px / TILE_SIZE

    tl_lat, tl_lon = tile_fraction_to_latlon(
        actual_left_tile_f,
        actual_top_tile_f,
        zoom,
    )
    tr_lat, tr_lon = tile_fraction_to_latlon(
        actual_right_tile_f,
        actual_top_tile_f,
        zoom,
    )
    br_lat, br_lon = tile_fraction_to_latlon(
        actual_right_tile_f,
        actual_bottom_tile_f,
        zoom,
    )
    bl_lat, bl_lon = tile_fraction_to_latlon(
        actual_left_tile_f,
        actual_bottom_tile_f,
        zoom,
    )

    # Existing app expects corners as lon,lat.
    world_corners = [
        (tl_lon, tl_lat),
        (tr_lon, tr_lat),
        (br_lon, br_lat),
        (bl_lon, bl_lat),
    ]

    return np.array(cropped), world_corners, total_tiles


# -----------------------------
# Inputs
# -----------------------------

st.subheader("Bounding box coordinates")

st.caption("Use decimal degrees.")

col1, col2 = st.columns(2)

with col1:
    north = st.number_input("North latitude", value=50.9400, format="%.8f")
    south = st.number_input("South latitude", value=50.9300, format="%.8f")

with col2:
    west = st.number_input("West longitude", value=6.9500, format="%.8f")
    east = st.number_input("East longitude", value=6.9700, format="%.8f")

st.subheader("Tile settings")

zoom = st.slider(
    "Zoom level",
    min_value=1,
    max_value=19,
    value=15,
)

user_agent = st.text_input(
    "User-Agent",
    value="MapMapperPrototype/0.1 swarnendusengupta29@gmail.com",
    help=(
        "Use a stable app name and contact. "
        "Do not leave this as generic python-requests."
    ),
)

if south >= north:
    st.error("South latitude must be smaller than north latitude.")
    st.stop()

if west >= east:
    st.error("West longitude must be smaller than east longitude.")
    st.stop()

x_min, x_max, y_min, y_max = bbox_to_tile_range(
    north=north,
    south=south,
    east=east,
    west=west,
    zoom=zoom,
)

num_x = x_max - x_min + 1
num_y = y_max - y_min + 1
total_tiles = num_x * num_y

st.write(
    f"Tiles needed at zoom {zoom}: "
    f"**{num_x} × {num_y} = {total_tiles} tiles**"
)

if total_tiles > 25:
    st.error(
        "This is too many tiles for the public OSM server in this prototype. "
        "Reduce the area or lower the zoom."
    )

st.caption(
    "For small tests, try zoom 14–16. For a bigger area, use a lower zoom."
)


# -----------------------------
# Fetch map
# -----------------------------

if st.button("Fetch map from OpenStreetMap"):
    try:
        img, world_corners, n_tiles = stitch_and_crop_osm_bbox(
            north=north,
            south=south,
            east=east,
            west=west,
            zoom=zoom,
            user_agent=user_agent,
        )

        h, w = img.shape[:2]

        st.session_state.map_image = img
        st.session_state.map_filename = "openstreetmap_bbox.png"
        st.session_state.world_corners = world_corners

        st.session_state.samples = []
        st.session_state.overlay_image = None
        st.session_state.mask_image = None
        st.session_state.summary_df = None
        st.session_state.canvas_version += 1

        st.success("OpenStreetMap image fetched and loaded into the app.")

        st.write(f"Fetched image size: **{w} × {h} pixels**")
        st.write(f"Downloaded tiles: **{n_tiles}**")

        st.image(
            img,
            caption="OpenStreetMap crop — © OpenStreetMap contributors",
            use_container_width=True,
        )

        st.subheader("Stored image corner coordinates")

        st.write(
            {
                "top_left_lon_lat": world_corners[0],
                "top_right_lon_lat": world_corners[1],
                "bottom_right_lon_lat": world_corners[2],
                "bottom_left_lon_lat": world_corners[3],
            }
        )

        st.info(
            "Now go to page 2: Label Samples. "
            "The fetched map is already stored in the app session."
        )

    except Exception as e:
        st.error(str(e))