import cv2
import yaml
from ultralytics import YOLO
 
 
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)
 
 
def get_status(count, green_max, yellow_max):
    if count <= green_max:
        return "GREEN", (0, 255, 0)
    elif count <= yellow_max:
        return "YELLOW", (0, 255, 255)
    else:
        return "RED", (0, 0, 255)
 
 
def draw_info(frame, count, status, color, green_max, alert_msg, show_boxes, boxes):
    cv2.rectangle(frame, (10, 10), (320, 150), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (320, 150), color, 3)
 
    cv2.putText(frame, f"Count: {count}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Status: {status}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Green: 0-{green_max}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
 
    if status == "RED":
        h = frame.shape[0]
        cv2.rectangle(frame, (10, h-60), (frame.shape[1]-10, h-10), (0, 0, 255), -1)
        cv2.putText(frame, alert_msg, (20, h-25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
 
    if show_boxes:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
 
    return frame
 
 
def detect_people(model, frame, confidence):
    results = model(frame, conf=confidence, verbose=False)
    people = []
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) == 0:
                people.append(box)
    return people
 
 
def main():
    config = load_config()
    model = YOLO(config["model"]["name"])
 
    confidence = config["model"]["confidence"]
    green_max = config["density"]["green_max"]
    yellow_max = config["density"]["yellow_max"]
    alert_msg = config["alerts"]["message"]
    show_boxes = config["video"]["show_boxes"]
    window_name = config["display"]["window_name"]
    camera_source = config["video"]["source"]
 
    cap = cv2.VideoCapture(camera_source, cv2.CAP_V4L2)
    cap.set(3, 640)
    cap.set(4, 480)
 
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
 
    print("Starting... Press 'q' to quit")
 
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
 
        people = detect_people(model, frame, confidence)
        count = len(people)
        status, color = get_status(count, green_max, yellow_max)
 
        frame = draw_info(frame, count, status, color, green_max, alert_msg, show_boxes, people)
 
        cv2.imshow(window_name, frame)
 
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()
