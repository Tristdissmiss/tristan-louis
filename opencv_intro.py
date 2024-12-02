import cv2
import numpy as np
import argparse
import pandas as pd

# Load smoothed predictions
predictions_df = pd.read_csv('smoothed_predictions.csv')  # Use smoothed predictions

# Define active frames (frames where the ball is being hit or in motion)
active_frames = predictions_df[predictions_df['value'] == 1]['frame'].tolist()  # Assuming 'value' column indicates activity


def process_frame(frame, frame_number, apply_filters=True, ball_color_transition=False, ball_trail=None):
    """
    Process a single frame of the video with optional filters, tracking, and ball trail visualization.
    """
    # Preserve the original frame size
    output_frame = frame.copy()

    if apply_filters:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(output_frame, cv2.COLOR_BGR2GRAY)
        output_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if ball_color_transition or ball_trail:
        # Add prediction overlay
        prediction = predictions_df[predictions_df['frame'] == frame_number]

        if not prediction.empty:
            # Ensure column names match the DataFrame
            try:
                predicted_x = int(prediction['predicted_x'])
                predicted_y = int(prediction['predicted_y'])
            except KeyError:
                print(f"Error: Missing columns 'predicted_x' or 'predicted_y' in predictions DataFrame.")
                return output_frame
        else:
            print(f"Warning: No prediction for frame {frame_number}. Defaulting to center.")
            predicted_x = output_frame.shape[1] // 2  # Center of the frame horizontally
            predicted_y = output_frame.shape[0] // 2  # Center of the frame vertically

        # Draw the ball at the predicted position
        color = (0, 255, 0)  # Green for predicted position
        cv2.circle(output_frame, (predicted_x, predicted_y), radius=10, color=color, thickness=-1)

        # Update the ball trail
        if ball_trail is not None:
            ball_trail.append((predicted_x, predicted_y))
            for i, point in enumerate(reversed(ball_trail[-30:])):  # Keep the last 30 points for the trail
                fade_color = (255 - i * 8, 255 - i * 8, 255 - i * 8)  # Fading effect
                cv2.circle(output_frame, point, radius=5, color=fade_color, thickness=-1)

    return output_frame



def main():
    parser = argparse.ArgumentParser(description='Process video with effects and tracking')
    parser.add_argument('input_video', type=str, help='Path to the input video file')
    parser.add_argument('--filters', action='store_true', help='Apply black-and-white filter to the video')
    parser.add_argument('--track_ball', action='store_true', help='Track ball color and transition color on movement')
    parser.add_argument('--trail', action='store_true', help='Add a trail to the ball movement')
    args = parser.parse_args()

    # Open the input video
    input_video = cv2.VideoCapture(args.input_video)
    if not input_video.isOpened():
        print(f"Error: Unable to open the video file {args.input_video}.")
        return

    # Get video properties
    original_width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(input_video.get(cv2.CAP_PROP_FPS))

    # Define output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter('output_video_with_active_frames.mp4', fourcc, fps, (original_width, original_height))

    frame_number = 0
    ball_trail = []  # Store ball trail points
    while True:
        ret, frame = input_video.read()
        if not ret:
            break  # Stop if the end of the video is reached

        # Skip frames not in active_frames
        if frame_number not in active_frames:
            frame_number += 1
            continue

        # Process the current frame
        processed_frame = process_frame(
            frame,
            frame_number,
            apply_filters=args.filters,
            ball_color_transition=args.track_ball,
            ball_trail=ball_trail if args.trail else None
        )

        # Write the processed frame to the output video
        output_video.write(processed_frame)

        # Display the processed frame (press 'q' to quit)
        cv2.imshow('Processed Video with Active Frames', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit keyword detected. Stopping processing.")
            break

        frame_number += 1

    # Release resources
    input_video.release()
    output_video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
