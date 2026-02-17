import cv2
import os

def extract_frames(video_path, output_dir, label, frame_skip=5):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            frame_name = f"{label}_{saved_count}.jpg"
            frame_path = os.path.join(output_dir, frame_name)
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames from {video_path}")

if __name__ == "__main__":
    extract_frames(
        "dataset/real_videos/sample_real.mp4",
        "dataset/frames/real",
        label="real"
    )

    extract_frames(
        "dataset/fake_videos/sample_fake.mp4",
        "dataset/frames/fake",
        label="fake"
    )