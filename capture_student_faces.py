
import cv2
import os

student_name = input("Enter student name: ")

save_path = f"students/{student_name}"
os.makedirs(save_path, exist_ok=True)

cap = cv2.VideoCapture(0)

count = 0

print("Press SPACE to capture image")
print("Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    cv2.imshow("Capture Faces", frame)

    key = cv2.waitKey(1)

    if key == 32:  # SPACE key
        img_path = f"{save_path}/{student_name}_{count}.jpg"
        cv2.imwrite(img_path, frame)
        print("Saved:", img_path)
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()