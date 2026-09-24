from ultralytics import YOLO

MODEL_PATH = r"C:\Users\Billy\Documents\vehicle detector\runs\detect\train-6\weights\best.pt"

# Use 0 for your webcam.
# Or replace with a video path, e.g. r"C:\Users\Billy\Videos\traffic.mp4"
VIDEO_SOURCE = r"C:\Users\Billy\Documents\vehicle detector\854671-hd_1920_1080_25fps.mp4"

def main():
    model = YOLO(MODEL_PATH)

    model.track(
        source=VIDEO_SOURCE,
        conf=0.264,
        tracker="bytetrack.yaml",
        persist=True,
        show=True,
        save=True
    )

if __name__ == "__main__":
    main()
