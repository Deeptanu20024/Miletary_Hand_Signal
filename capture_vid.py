import cv2

def record_video(output_file, video_number, frame_count):
    # Open default camera
    cap = cv2.VideoCapture(0)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Change the codec to 'mp4v' for .mp4 format
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (frame_width, frame_height))

    # Record video for the specified number of frames
    frame_counter = 0
    while frame_counter < frame_count:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Flip the frame horizontally
        flipped_frame = cv2.flip(frame, 1)

        # Write the flipped frame to the output video file
        out.write(flipped_frame)

        # Show the flipped frame on screen with video number
        cv2.putText(flipped_frame, f"Recording {video_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.imshow('Recording', flipped_frame)

        # Increment frame counter
        frame_counter += 1

        # Check for 'q' key press to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Define the number of frames for each video
frame_count_per_video = 70

# Record 40 videos sequentially
for i in range(1, 41):
    output_file = f"{i}.mp4"  # Output file name for each video
    print(f"Start recording {output_file}")
    record_video(output_file, i, frame_count_per_video)
    print(f"Video recording {i} completed. Saved as {output_file}")
    if i < 40:
        print("Waiting for 2 seconds before starting the next recording...")
        cv2.waitKey(2000)  # Wait for 2 seconds between recordings
