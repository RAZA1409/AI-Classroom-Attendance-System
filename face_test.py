from deepface import DeepFace
import cv2

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    try:
        result = DeepFace.find(
            img_path=frame,
            db_path="students",
            enforce_detection=False
        )

        if len(result[0]) > 0:
            identity = result[0].iloc[0]["identity"]
            name = identity.split("\\")[-1].split(".")[0]
        else:
            name = "Unknown"

    except:
        name = "Unknown"

    cv2.putText(frame, name, (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.imshow("Face Recognition Test", frame)

    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
