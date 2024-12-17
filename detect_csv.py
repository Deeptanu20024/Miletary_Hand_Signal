import mediapipe as mp
import cv2
import os
import csv

# Function to extract keypoints from a single video file
def extract_keypoints(video_file, output_csv, class_label):
    # Initialize MediaPipe Pose and HandLandmarker models
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Open video file
    cap = cv2.VideoCapture(video_file)

    # Initialize CSV writer
    with open(output_csv, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        
        # Write header for combined landmarks
        header = ['class_label']
        for i in range(33):  # Assuming there are 33 pose keypoints
            header.append(f'pose_keypoint_{i}_x')
            header.append(f'pose_keypoint_{i}_y')
            header.append(f'pose_keypoint_{i}_z')
        for i in range(21):  # Assuming there are 21 landmarks per hand (x, y, z)
            header.append(f'hand_keypoint_{i}_x')
            header.append(f'hand_keypoint_{i}_y')
            header.append(f'hand_keypoint_{i}_z')
        csvwriter.writerow(header)

        # Iterate through frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Get pose landmarks
            pose_results = pose.process(frame_rgb)
            if pose_results.pose_landmarks:
                pose_landmarks = []
                for lm in pose_results.pose_landmarks.landmark:
                    pose_landmarks.extend([lm.x, lm.y, lm.z])  # Add 'Z' coordinate for pose landmarks
            else:
                pose_landmarks = [0] * 99  # Write 'NaN' if no pose keypoints detected (33 keypoints * 3 coordinates = 99)
            
            # Get hand landmarks for the left hand only
            hand_results = hands.process(frame_rgb)     
            if hand_results.multi_hand_landmarks:
                left_hand_landmarks = []
                for i, hand_landmark in enumerate(hand_results.multi_hand_landmarks):
                    hand_label = hand_results.multi_handedness[i].classification[0].label
                    if hand_label == "Left":  # Check index to determine left hand
                        for landmark in hand_landmark.landmark:
                            left_hand_landmarks.extend([landmark.x, landmark.y, landmark.z])  # Add 'Z' coordinate for hand landmarks
            else:
                left_hand_landmarks = [0] * 63  # Write 'NaN' if no left hand landmarks detected (21 keypoints * 3 coordinates = 63)
            
            # Write landmarks to CSV
            csvwriter.writerow([class_label] + pose_landmarks + left_hand_landmarks)
                
        # Release resources
        cap.release()
        cv2.destroyAllWindows()

# Function to process all videos in a directory
def process_videos(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for file in os.listdir(input_dir):
        if file.endswith(".mp4"):
            video_path = os.path.join(input_dir, file)
            class_label = os.path.splitext(file)[0]  # Use video filename as class label
            output_csv = os.path.join(output_dir, file.replace(".mp4", ".csv"))
            extract_keypoints(video_path, output_csv, class_label)
            print(f"Keypoints extracted and saved to {output_csv}")

# Example usage
input_directory = "dataset"
output_directory = "output_keypoints"
process_videos(input_directory, output_directory)
