import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
import pandas as pd
import random

# --- CONFIG ---
VIDEO_PATH = 'assets/Traffic_Flow_In_The_Highway.mp4'  # <-- Replace with your video path
#VIDEO_PATH = 'Traffic_Lights.mp4'  # <-- Replace with your video path
YOLO_WEIGHTS = 'yolov8n.pt'  # <-- Replace with your custom weights if needed

# --- VEHICLE & TRAFFIC LIGHT CLASSES (COCO) ---
VEHICLE_CLASSES = {'car', 'truck', 'bus', 'motorcycle'}
TRAFFIC_LIGHT_CLASS = 'traffic light'

# --- LOAD YOLO MODEL ---
model = YOLO(YOLO_WEIGHTS)

# --- SIMPLE CENTROID TRACKER ---
class CentroidTracker:
    def __init__(self):
        self.next_object_id = 0
        self.objects = dict()  # object_id: centroid
        self.disappeared = dict()  # object_id: frames disappeared
        self.max_disappeared = 10
        self.tracks = defaultdict(list)  # object_id: list of centroids

    def update(self, detections):
        # detections: list of (x1, y1, x2, y2, class_name)
        input_centroids = np.array([
            [(x1 + x2) // 2, (y1 + y2) // 2] for x1, y1, x2, y2, _ in detections
        ]) if detections else np.empty((0, 2))
        input_classes = [cls for *_, cls in detections]

        if len(self.objects) == 0:
            for i, centroid in enumerate(input_centroids):
                self.objects[self.next_object_id] = centroid
                self.tracks[self.next_object_id].append(centroid)
                self.next_object_id += 1
            return self.objects, self.tracks

        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        if len(input_centroids) == 0:
            for object_id in object_ids:
                self.disappeared[object_id] = self.disappeared.get(object_id, 0) + 1
                if self.disappeared[object_id] > self.max_disappeared:
                    del self.objects[object_id]
                    del self.tracks[object_id]
            return self.objects, self.tracks

        D = np.linalg.norm(np.array(object_centroids)[:, None] - input_centroids[None, :], axis=2)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.tracks[object_id].append(input_centroids[col])
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(len(object_centroids))) - used_rows
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] = self.disappeared.get(object_id, 0) + 1
            if self.disappeared[object_id] > self.max_disappeared:
                del self.objects[object_id]
                del self.tracks[object_id]

        unused_cols = set(range(len(input_centroids))) - used_cols
        for col in unused_cols:
            self.objects[self.next_object_id] = input_centroids[col]
            self.tracks[self.next_object_id].append(input_centroids[col])
            self.next_object_id += 1

        return self.objects, self.tracks

# --- STREAMLIT UI ---
st.set_page_config(layout="wide", page_title="🚦 Vehicle & Traffic Light Detection", page_icon="🚗")

# Custom CSS for modern look and mobile tweaks
st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fa;
    }
    .stApp {
        font-family: 'Segoe UI', 'Roboto', sans-serif;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 1.5rem;
    }
    .metric-label, .metric-value {
        font-size: 1.1rem !important;
    }
    @media (max-width: 600px) {
        .block-container {
            padding-left: 0.5rem;
            padding-right: 0.5rem;
        }
        .stColumn {
            flex-direction: column !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style='background: linear-gradient(90deg, #4f8bf9 0%, #1fc8db 100%); padding: 1.2rem 0.5rem; border-radius: 12px; margin-bottom: 1.5rem;'>
        <h1 style='color: white; text-align: center; margin-bottom: 0;'>🚦 Vehicle & Traffic Light Detection</h1>
        <p style='color: #e0e0e0; text-align: center; margin-top: 0.5rem;'>Analyze traffic videos for vehicles and traffic lights, with live detection and speed estimation.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    meters_per_pixel = st.number_input(
        "Meters per pixel (scene scale, estimate/calibrate for your video)",
        min_value=0.0001, max_value=1.0, value=0.40, step=0.001,
        help="Estimate: 1 pixel = how many meters? Calibrate using a known distance in your video."
    )
    st.markdown("---")
    st.info("Upload your own video by replacing the file in the code or extend this app to support file uploads.")

col1, col2 = st.columns([3, 1])

with col1:
    video_placeholder = st.empty()

with col2:
    st.markdown("### 📊 Live Detection Info")
    vehicle_count_placeholder = st.empty()
    vehicle_types_placeholder = st.empty()
    speed_px_placeholder = st.empty()
    speed_kmh_placeholder = st.empty()
    traffic_light_placeholder = st.empty()
    st.markdown("---")
    st.markdown("<span style='color: #888;'>Detection updates in real time as the video plays.</span>", unsafe_allow_html=True)

# Data storage for time-series charts
frame_history = []

# --- VIDEO PROCESSING ---
cap = cv2.VideoCapture(VIDEO_PATH)
tracker = CentroidTracker()
fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30

frame_num = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1
    results = model(frame)
    detections = []
    vehicle_types = set()
    traffic_light_state = None
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            class_name = model.model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if class_name in VEHICLE_CLASSES and conf > 0.3:
                detections.append((x1, y1, x2, y2, class_name))
                vehicle_types.add(class_name)
                color = (0, 255, 0)
                label = f"{class_name} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            elif class_name == TRAFFIC_LIGHT_CLASS and conf > 0.3:
                detections.append((x1, y1, x2, y2, class_name))
                # Try to infer color by region color average
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    avg_color = np.mean(roi, axis=(0, 1))
                    if avg_color[2] > 150:
                        traffic_light_state = 'Red'
                    elif avg_color[1] > 150:
                        traffic_light_state = 'Green'
                    else:
                        traffic_light_state = 'Yellow'
                color = (0, 0, 255) if traffic_light_state == 'Red' else (0, 255, 0) if traffic_light_state == 'Green' else (0, 255, 255)
                label = f"Traffic Light: {traffic_light_state or '?'}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    objects, tracks = tracker.update(detections)

    # Speed estimation (pixels/frame)
    speeds = {}
    for object_id, track in tracks.items():
        if len(track) > 1:
            speed = np.linalg.norm(track[-1] - track[-2])
            speeds[object_id] = speed
    avg_speed = np.mean(list(speeds.values())) if speeds else 0

    # Convert to km/h
    avg_speed_m_per_s = avg_speed * meters_per_pixel * fps
    avg_speed_km_per_h = avg_speed_m_per_s * 3.6

    # --- Store metrics for charts ---
    frame_history.append({
        'frame': frame_num,
        'vehicle_count': len([d for d in detections if d[4] in VEHICLE_CLASSES]),
        'avg_speed_kmh': avg_speed_km_per_h,
        'traffic_light': traffic_light_state or 'None',
    })

    # --- Update UI ---
    video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
    vehicle_count_placeholder.metric("Vehicle Count", len([d for d in detections if d[4] in VEHICLE_CLASSES]))
    vehicle_types_placeholder.write(f"Types: {', '.join(vehicle_types) if vehicle_types else 'None'}")
    speed_px_placeholder.write(f"Avg Speed: {avg_speed:.2f} px/frame")
    speed_kmh_placeholder.write(f"Avg Speed: {avg_speed_km_per_h:.2f} km/h")
    traffic_light_placeholder.write(f"Traffic Light: {traffic_light_state or 'Not detected'}")

    # Streamlit needs a small sleep to update UI
    if frame_num % 2 == 0:
        import time; time.sleep(0.01)

cap.release()

# --- Show Time-Series Charts ---
if frame_history:
    df = pd.DataFrame(frame_history)
    st.markdown("## 📈 Detection Analytics")
    chart_cols = st.columns(3)
    with chart_cols[0]:
        st.line_chart(df.set_index('frame')['vehicle_count'], height=200, use_container_width=True)
        st.caption('Vehicle Count per Frame')
    with chart_cols[1]:
        st.line_chart(df.set_index('frame')['avg_speed_kmh'], height=200, use_container_width=True)
        st.caption('Average Speed (km/h) per Frame')
    with chart_cols[2]:
        # Encode traffic light state as numbers for plotting
        color_map = {'Red': 2, 'Yellow': 1, 'Green': 0, 'None': -1}
        df['traffic_light_num'] = df['traffic_light'].map(color_map)
        st.line_chart(df.set_index('frame')['traffic_light_num'], height=200, use_container_width=True)
        st.caption('Traffic Light State (2=Red, 1=Yellow, 0=Green, -1=None)')

st.success("Video processing complete.") 