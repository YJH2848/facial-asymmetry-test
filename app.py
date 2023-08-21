import cv2
import time
import mediapipe as mp
from flask import Flask, render_template
from check_your_face import check_your_face

app = Flask(__name__)

# Mediapipe 초기화
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/capture')
def capture():
    cap = cv2.VideoCapture(0)
    start_time = time.time()

    while True:
        ret, frame = cap.read()

        if not ret:
            break 

        elapsed_time = time.time() - start_time
        remaining_time = max(0, 6 - elapsed_time)

        timer_text = "남은 시간 :{:.1f}".format(remaining_time)
        cv2.putText(frame, timer_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow('Camera', frame)

        if elapsed_time >= 6:
            cv2.imwrite('captured_face.jpg', frame)
            break

        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # 이미지 불러오기
    image_path = "captured_face.jpg"
    image = cv2.imread(image_path)
    ih, iw, _ = image.shape

    face = []

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = face_detection.process(rgb_image)

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

            # 얼굴 랜드마크 표시
            landmarks = detection.location_data.relative_keypoints
            landmark_coords = [(int(landmark.x * iw), int(landmark.y * ih)) for landmark in landmarks]

            face.append(landmark_coords) 

            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            for landmark in landmark_coords:
                lx, ly = landmark
                cv2.circle(image, (lx, ly), 5, (255, 0, 0), -1)

    cv2.imwrite('face_landmarks.jpg', image)

    left_eye, right_eye, _, _, left_cheek, right_cheek = face[0]
    
    eye = left_eye[1] - right_eye[1]
    cheek = left_cheek[1] - right_cheek[1]

    return render_template('result.html', eye=eye, cheek=cheek)

if __name__ == '__main__':
    app.run(debug=True, port=8000)
