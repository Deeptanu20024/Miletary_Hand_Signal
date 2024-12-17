import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp

# Load the pre-trained model
model_path = 'my_model3.keras'  # Replace with the actual path to your .keras model
model = tf.keras.models.load_model(model_path)


# Initialize MediaPipe Pose and HandLandmarker modelsq
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)





# Function to extract keypoints from frames
def extract_keypoints_from_frame(frame):
    keypoints = []

    pose_results = pose.process(frame)
    if pose_results.pose_landmarks:
        pose_landmarks = []
        for i, lm in enumerate(pose_results.pose_landmarks.landmark):
            if i in [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 15, 17, 19, 21, 23, 24]:  # Only consider specified keypoints
                pose_landmarks.extend([lm.x, lm.y, lm.z])  # Add 'Z' coordinate for pose landmarks
    else:
        pose_landmarks = [0] * 48  # Write zeros if no pose keypoints detected (16 keypoints * 3 coordinates = 48)

    # Get hand landmarks for the left hand only
    hand_results = hands.process(frame)
    if hand_results.multi_hand_landmarks:
        left_hand_landmarks = []
        for i, hand_landmark in enumerate(hand_results.multi_hand_landmarks):
            hand_label = hand_results.multi_handedness[i].classification[0].label
            if hand_label == "Left":  # Check index to determine left hand
                for landmark in hand_landmark.landmark:
                    left_hand_landmarks.extend([landmark.x, landmark.y, landmark.z])  # Add 'Z' coordinate for hand landmarks
    else:
        left_hand_landmarks = [0] * 63  # Write zeros if no left hand landmarks detected (21 keypoints * 3 coordinates = 63)

    # Combine pose and hand keypoints
    if len(pose_landmarks) < 48:
        pose_landmarks.extend([0] * (48 - len(pose_landmarks)))
    # Trim the list to the desired size
    pose_landmarks = pose_landmarks[:48]

    if len(left_hand_landmarks) < 63:
        left_hand_landmarks.extend([0] * (63 - len(left_hand_landmarks)))
    # Trim the list to the desired size
    left_hand_landmarks = left_hand_landmarks[:63]

    keypoints = pose_landmarks + left_hand_landmarks
    return keypoints









# Function to predict the hand sign using the keypoints
def predict_hand_sign(keypoints):
    keypoints= np.array(keypoints)
    keypoints= keypoints.flatten()
    keypoints = np.expand_dims(keypoints, axis=0)  # Expand dimensions to match the model input shape (1, 70, 111)
    prediction = model.predict(keypoints)
    predicted_class = np.argmax(prediction, axis=1)
    return label_map[predicted_class[0]]





prediction_c=[]
sequence=[]
sentence=[]
threshold=0.4
label_map = {
    0: "frez",
    1: "hurry",
    2: "listen",
    3: "stop",
    4: "hostage",
    5: "understand",
    6: "pistol",
    7: "come",
}

print(label_map[0])
# Main logic for continuous capture and processing
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
else:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        frame = cv2.flip(frame, 1)

        
        # Convert frame to RGB for MediaPipe processing
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        

        pre = None 
        keypoints= extract_keypoints_from_frame(frame_rgb)
        sequence.append(keypoints)
        sequence = sequence[-70:]
        # Check if we have collected 70 frames
        if len(sequence) == 70:
            pre=predict_hand_sign(sequence)
        # Define font properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 100)
        fontScale = 3
        color = (255, 0, 0)
        thickness = 5
        # Get the word corresponding to the predicted number
        word = pre
        # Put the word on the frame
        frame = cv2.putText(frame, word, org, font, fontScale, color, thickness, cv2.LINE_AA)
    
        cv2.imshow('frame', frame)

        # Check for 'q' key press to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
