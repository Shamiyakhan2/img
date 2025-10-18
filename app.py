import cv2
import numpy as np
import streamlit as st

# Page setup
st.set_page_config(page_title="Live Edge Detection", layout="wide")
st.title("⚡ Live Edge Detection using OpenCV")

# Load image
image_path = "sss.jpg"
img = cv2.imread(image_path)
if img is None:
    st.error(f"Image '{image_path}' not found!")
    st.stop()

# Convert BGR to RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Edge detection function
def detect_edges(img, low_thresh, high_thresh, overlay=True):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blur, low_thresh, high_thresh)
    if overlay:
        edge_colored = np.zeros_like(img)
        edge_colored[edges != 0] = [255, 0, 0]
        result = cv2.addWeighted(img, 0.7, edge_colored, 1.3, 0)
    else:
        result = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    return result

# Sidebar controls
st.sidebar.header("🎚 Edge Detection Controls")
low = st.sidebar.slider("Low Threshold", 0, 255, 50)
high = st.sidebar.slider("High Threshold", 0, 255, 150)
overlay = st.sidebar.checkbox("Show colored overlay edges", value=True)

# --- Create placeholders for images ---
col1, col2 = st.columns(2)
placeholder_original = col1.empty()
placeholder_edges = col2.empty()

# --- Display original image once ---
placeholder_original.image(img_rgb, caption="Original Image", use_container_width=True)

# --- Process image dynamically and display ---
edge_result = detect_edges(img_rgb, low, high, overlay)
placeholder_edges.image(edge_result, caption="Edge Detection Result", use_container_width=True)