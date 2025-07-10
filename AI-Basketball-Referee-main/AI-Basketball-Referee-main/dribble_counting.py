import cv2
from ultralytics import YOLO
import numpy as np
import csv
import os
from datetime import datetime


class DribbleCounter:
    def __init__(self):
        # Load the YOLO model for ball detection
        self.model = YOLO("basketballModel.pt")

        # Open the webcam or video file
        # Use video file:
        self.cap = cv2.VideoCapture("driblle.mp4")
        # Or use webcam:
        # self.cap = cv2.VideoCapture(0)

        # Initialize variables to store the previous position of the basketball
        self.prev_x_center = None
        self.prev_y_center = None
        self.prev_delta_y = None

        # Initialize the dribble counter
        self.dribble_count = 0

        # Threshold for the y-coordinate change to be considered as a dribble
        self.dribble_threshold = 18

        # Initialize prediction saving
        self.save_predictions = True  # Set to False if you don't want to save
        self.frame_count = 0
        self.predictions_data = []
        
        # Create output directories
        if self.save_predictions:
            os.makedirs("output", exist_ok=True)
            os.makedirs("output/frames", exist_ok=True)
            
            # Setup video writer for saving annotated video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                'output/dribble_detection_output.mp4', 
                fourcc, 
                20.0,  # FPS
                (640, 480)  # Frame size - adjust based on your video
            )

    def run(self):
        # Process frames from the webcam until the user quits
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if success:
                self.frame_count += 1
                results_list = self.model(frame, verbose=False, conf=0.65)

                annotated_frame = frame.copy()
                ball_detected = False
                ball_coordinates = None

                for results in results_list:
                    if results.boxes is not None and len(results.boxes) > 0:
                        for bbox in results.boxes.xyxy:
                            x1, y1, x2, y2 = bbox[:4]

                            x_center = (x1 + x2) / 2
                            y_center = (y1 + y2) / 2
                            ball_coordinates = (float(x_center), float(y_center))

                            print(f"Frame {self.frame_count}: Ball coordinates: (x={x_center:.2f}, y={y_center:.2f})")
                            print(f"Dribble count: {self.dribble_count}")

                            self.update_dribble_count(x_center, y_center)

                            self.prev_x_center = x_center
                            self.prev_y_center = y_center

                            ball_detected = True

                            # Draw bounding box around the ball
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            
                            # Add ball center point
                            cv2.circle(annotated_frame, (int(x_center), int(y_center)), 5, (0, 0, 255), -1)

                        # Use YOLO's built-in plotting for better visualization
                        annotated_frame = results.plot()

                # Save prediction data
                if self.save_predictions:
                    self.save_frame_data(ball_detected, ball_coordinates)

                # Draw the dribble count on the frame
                cv2.putText(
                    annotated_frame,
                    f"Dribble Count: {self.dribble_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Draw frame number
                cv2.putText(
                    annotated_frame,
                    f"Frame: {self.frame_count}",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                # Draw ball detection status
                status_text = "Ball Detected" if ball_detected else "No Ball Detected"
                color = (0, 255, 0) if ball_detected else (0, 0, 255)
                cv2.putText(
                    annotated_frame,
                    status_text,
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                    cv2.LINE_AA,
                )

                # Save annotated frame to video
                if self.save_predictions:
                    # Resize frame to match video writer dimensions
                    resized_frame = cv2.resize(annotated_frame, (640, 480))
                    self.video_writer.write(resized_frame)
                    
                    # Save individual frames every 30 frames (adjust as needed)
                    if self.frame_count % 30 == 0:
                        cv2.imwrite(f'output/frames/frame_{self.frame_count:06d}.jpg', annotated_frame)

                cv2.imshow("Basketball Dribble Counter", annotated_frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                break

        # Release the webcam and destroy the windows
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Save final results
        if self.save_predictions:
            self.save_final_results()

    def save_frame_data(self, ball_detected, ball_coordinates):
        """Save frame data for CSV export"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        data = {
            'timestamp': timestamp,
            'frame_number': self.frame_count,
            'ball_detected': ball_detected,
            'ball_x': ball_coordinates[0] if ball_coordinates else None,
            'ball_y': ball_coordinates[1] if ball_coordinates else None,
            'dribble_count': self.dribble_count,
            'prev_delta_y': self.prev_delta_y if hasattr(self, 'prev_delta_y') else None
        }
        
        self.predictions_data.append(data)

    def save_final_results(self):
        """Save all predictions to files"""
        print(f"\nSaving predictions...")
        
        # Save CSV file with all predictions
        csv_filename = f'output/dribble_predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        with open(csv_filename, 'w', newline='') as csvfile:
            if self.predictions_data:
                fieldnames = self.predictions_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.predictions_data)
        
        # Release video writer
        if hasattr(self, 'video_writer'):
            self.video_writer.release()
        
        # Save summary statistics
        summary_filename = f'output/dribble_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(summary_filename, 'w') as f:
            f.write(f"Dribble Detection Summary\n")
            f.write(f"========================\n")
            f.write(f"Total Frames Processed: {self.frame_count}\n")
            f.write(f"Total Dribbles Detected: {self.dribble_count}\n")
            f.write(f"Frames with Ball Detected: {sum(1 for d in self.predictions_data if d['ball_detected'])}\n")
            f.write(f"Detection Rate: {sum(1 for d in self.predictions_data if d['ball_detected'])/len(self.predictions_data)*100:.1f}%\n")
            f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"✅ Results saved:")
        print(f"   📊 CSV data: {csv_filename}")
        print(f"   📹 Video: output/dribble_detection_output.mp4")
        print(f"   🖼️  Frames: output/frames/")
        print(f"   📋 Summary: {summary_filename}")
        print(f"   📈 Total dribbles detected: {self.dribble_count}")
        print(f"   🎬 Total frames processed: {self.frame_count}")

    def update_dribble_count(self, x_center, y_center):
        if self.prev_y_center is not None:
            delta_y = y_center - self.prev_y_center

            if (
                self.prev_delta_y is not None
                and self.prev_delta_y > self.dribble_threshold
                and delta_y < -self.dribble_threshold
            ):
                self.dribble_count += 1

            self.prev_delta_y = delta_y


if __name__ == "__main__":
    dribble_counter = DribbleCounter()
    dribble_counter.run()
