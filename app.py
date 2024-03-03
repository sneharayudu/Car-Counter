import streamlit as st
from ultralytics import YOLO
from ultralytics.solutions import object_counter
import cv2

def main():
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture("/content/drive/MyDrive/car_detection/training_video.mp4")
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    line_points = [(20, 400), (1080, 400)]  # line or region points
    classes_to_count = [2]  # car class for count

    st.title("Car Detection and Counting")
    st.write("Input Video:")

    # Display the input video
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            st.write("Video frame is empty or video processing has been successfully completed.")
            break
        st.video(frame)

        tracks = model.track(frame, persist=True, show=False, classes=classes_to_count)
        counter = object_counter.ObjectCounter()
        counter.set_args(view_img=True, reg_pts=line_points, classes_names=model.names, draw_tracks=True)

        frame_with_count = counter.start_counting(frame, tracks)
        st.write("Output Video:")
        st.video(frame_with_count)

if __name__ == "__main__":
    main()
