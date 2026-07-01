import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import init_state, build_features

init_state()

st.title("3. Run Automatic Classification")

if st.session_state.map_image is None:
    st.warning("Please upload a map first.")
    st.stop()

if len(st.session_state.samples) == 0:
    st.warning("Please add labelled sample points first.")
    st.stop()

img = st.session_state.map_image
h, w = img.shape[:2]

samples_df = pd.DataFrame(st.session_state.samples)

st.subheader("Training samples")
st.dataframe(samples_df, use_container_width=True)

label_counts = samples_df["label"].value_counts()

st.write("Samples per label:")
st.dataframe(label_counts.rename("count"), use_container_width=True)

if samples_df["label"].nunique() < 2:
    st.warning("Add at least two different labels before classification.")
    st.stop()

if label_counts.min() < 3:
    st.warning(
        "Some labels have fewer than 3 samples. "
        "The classifier may be unstable. Add more points for better results."
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

run = st.button("Run classification")

if run:
    with st.spinner("Classifying full map..."):
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
            (1 - alpha) * img[known_pixels]
            + alpha * seg_rgb[known_pixels]
        ).astype(np.uint8)

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

        unknown_pixels = int((pred_mask == -1).sum())

        rows.append(
            {
                "label": "unknown / low confidence",
                "pixels": unknown_pixels,
                "percent": round(100 * unknown_pixels / total_pixels, 2),
            }
        )

        summary_df = pd.DataFrame(rows)

        st.session_state.overlay_image = overlay
        st.session_state.mask_image = seg_rgb
        st.session_state.summary_df = summary_df

    st.success("Classification complete.")

if st.session_state.overlay_image is not None:
    st.subheader("Classification overlay")
    st.image(st.session_state.overlay_image, use_container_width=True)

if st.session_state.mask_image is not None:
    st.subheader("Pure classified mask")
    st.image(st.session_state.mask_image, use_container_width=True)

if st.session_state.summary_df is not None:
    st.subheader("Area summary")
    st.dataframe(st.session_state.summary_df, use_container_width=True)