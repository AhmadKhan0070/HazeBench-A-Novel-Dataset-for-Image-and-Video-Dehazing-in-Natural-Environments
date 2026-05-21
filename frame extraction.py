import cv2
import os

def extract_frames(input_folder, output_folder, fps=30):
    # Iterate through all subfolders (categories)
    for category in os.listdir(input_folder):
        category_path = os.path.join(input_folder, category)
        
        # Ensure it is a directory
        if os.path.isdir(category_path):
            output_category_path = os.path.join(output_folder, category)  # Maintain folder structure
            os.makedirs(output_category_path, exist_ok=True)

            # Iterate through all videos in the category folder
            for idx, file_name in enumerate(os.listdir(category_path), start=1):
                if file_name.endswith((".mp4", ".avi", ".mov")):
                    video_path = os.path.join(category_path, file_name)
                    
                    # Open the video file
                    video_capture = cv2.VideoCapture(video_path)
                    video_fps = video_capture.get(cv2.CAP_PROP_FPS)  # Original FPS

                    # If video_fps is 0, set a default value
                    if video_fps is None or video_fps <= 0:
                        print(f"Warning: Couldn't read FPS for {file_name}. Using default FPS: {fps}")
                        video_fps = fps  # Set default FPS
                    
                    frame_interval = max(1, int(video_fps / fps))  # Avoid division by zero
                    frame_num = 0
                    saved_frame_num = 0
                    success = True

                    while success:
                        success, frame = video_capture.read()
                        if not success:
                            break
                        
                        # Save frame every 'frame_interval' frames
                        if frame_num % frame_interval == 0:
                            saved_frame_num += 1
                            frame_filename = os.path.join(output_category_path, f"{idx}_{saved_frame_num:04d}.jpg")
                            cv2.imwrite(frame_filename, frame)
                        
                        frame_num += 1

                    # Release resources
                    video_capture.release()

if __name__ == "__main__":
    input_folder = r"orignal videos classified"  # Change to your dataset folder
    output_folder = r"classified images"  # Change to your desired output folder
    fps = 30  # Extract 30 frames per second

    extract_frames(input_folder, output_folder, fps)
