# -AbhushanskandaSV-face-matchers
import threading
import cv2
import face_recognition
import numpy as np

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()
#cam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#camres
counter = 0
face_match = False
    
reference_img = cv2.imread("/Users/abhushanaskandasvishwamithra/Desktop/str.jpg")
if reference_img is None:
    print("Error: Could not load reference image")
    exit()
#refimg
reference_rgb = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
reference_encoding = face_recognition.face_encodings(reference_rgb)
if not reference_encoding:
    print("Error: No face found in reference image")
    exit()
reference_encoding = reference_encoding[0]
#converts refimg to rgb

def check_face(frame):
    global face_match
    try:

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame)
        if face_locations:

            face_encoding = face_recognition.face_encodings(rgb_frame, face_locations)[0]

            matches = face_recognition.compare_faces([reference_encoding], face_encoding, tolerance=0.6)
            face_match = matches[0]
        else:
            face_match = False
    except Exception:
        face_match = False

#check if refimg == to camimg
while True:
    ret, frame = cap.read()

    if ret:
        if counter % 30 == 0:
            try:
                thread = threading.Thread(target=check_face, args=(frame.copy(),))
                thread.daemon = True
                thread.start()
            except Exception:
                pass
        counter += 1

        if face_match:
            cv2.putText(frame, "MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "NO MATCH!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord(" "):
        break
#conti-loop for matching face
cap.release()
cv2.destroyAllWindows()
#closes the cam
