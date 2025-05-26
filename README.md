# Traffic Video Vehicle Detection

This application uses YOLOv8 and Streamlit to detect vehicles and traffic lights in a recorded video, display bounding boxes, and show live analytics in a side panel. It also estimates vehicle speed in both pixels/frame and km/h (with user calibration).

## Features
- **Vehicle Detection**: Detects cars, trucks, buses, and motorcycles using YOLOv8.
- **Traffic Light Detection**: Detects traffic lights and infers their state (red, yellow, green).
- **Live Video Display**: Shows the processed video with bounding boxes and labels.
- **Live Analytics Panel**: Displays vehicle count, types, average speed (px/frame and km/h), and traffic light state.
- **Speed Estimation**: Estimates vehicle speed in km/h using a user-provided meters-per-pixel value and the video frame rate.

## Requirements
- Python 3.8+
- A GPU-powered machine is recommended for real-time YOLO inference.
- Your own MP4 video file (replace the placeholder in the code).

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage
1. **Add your video**
   - Place your MP4 video in the project `assets` directory.
   - Update the `VIDEO_PATH` variable in `app.py` to your video filename.

2. **(Optional) Use custom YOLO weights**
   - Replace `'yolov8n.pt'` in `app.py` with your own YOLOv8 weights if you have a custom-trained model.

3. **Run the app**
```bash
streamlit run app.py
```

4. **Set meters-per-pixel**
   - In the sidebar, use the "Meters per pixel" input to calibrate speed estimation. Measure a known distance in your video (e.g., a lane width) and calculate how many meters one pixel represents.
   - The app will display both average speed in pixels/frame and km/h.

## Notes on Speed Calibration
- The accuracy of speed in km/h depends on the correct value for meters-per-pixel. For best results, use a reference object or distance in your video to calibrate.
- The app uses the video's frame rate (FPS) for conversion. If your video has an unusual FPS, check that it is read correctly.

## Example
- Vehicle count, types, and speed are shown live as the video plays.
- Traffic light state is inferred by color (red, yellow, green) if detected by YOLO.

## Troubleshooting
- If you see no detections, check that your YOLO weights are compatible and that your video path is correct.
- For best performance, use a machine with a CUDA-capable GPU.

---

**Developed with [Ultralytics YOLOv8](https://docs.ultralytics.com/) and [Streamlit](https://streamlit.io/).** 