import cv2
import numpy as np
import time
import os
from ultralytics import YOLO
from collections import deque
from datetime import datetime
import csv

class BasketballRefereeAI:
    def __init__(self, video_source=0, save_output=True):
        # Initialize YOLO models
        self.pose_model = YOLO("yolov8s-pose.pt")
        self.ball_model = YOLO("basketballModel.pt")
        
        # Video source (0 for webcam, or path to video file)
        self.video_source = video_source
        if isinstance(video_source, str) and os.path.exists(video_source):
            self.cap = cv2.VideoCapture(video_source)
        else:
            self.cap = cv2.VideoCapture(int(video_source))
        
        # Frame dimensions
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        
        # Body keypoint indices
        self.body_index = {
            "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
            "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
            "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
            "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
        }
        
        # ===== DOUBLE DRIBBLE DETECTION VARIABLES =====
        self.hold_start_time = None
        self.is_holding = False
        self.was_holding = False
        self.hold_duration = 0.85
        self.hold_threshold = 300
        self.double_dribble_time = None
        
        # ===== BALL TRACKING VARIABLES =====
        self.prev_x_center = None
        self.prev_y_center = None
        self.prev_delta_y = None
        self.dribble_count = 0
        self.total_dribble_count = 0
        self.dribble_threshold = 18
        self.ball_not_detected_frames = 0
        self.max_ball_not_detected_frames = 20
        
        # ===== TRAVEL DETECTION VARIABLES =====
        self.step_count = 0
        self.total_step_count = 0
        self.prev_left_ankle_y = None
        self.prev_right_ankle_y = None
        self.step_threshold = 5
        self.min_wait_frames = 7
        self.wait_frames = 0
        self.travel_detected = False
        self.travel_timestamp = None
        
        # ===== BLOCKING FOUL DETECTION VARIABLES =====
        self.blocking_foul_detected = False
        self.blocking_foul_timestamp = None
        self.player_proximity_threshold = 40
        
        # ===== OUTPUT VARIABLES =====
        self.save_output = save_output
        self.frame_buffer = deque(maxlen=30)
        self.save_frames = 60
        self.frame_save_counter = 0
        self.saving = False
        self.out = None
        self.predictions_data = []
        self.frame_count = 0
        
        # Create output directories
        if self.save_output:
            os.makedirs("output", exist_ok=True)
            os.makedirs("output/frames", exist_ok=True)
            os.makedirs("output/violations", exist_ok=True)
            
            # Setup video writer for saving annotated video
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                'output/referee_output.mp4', 
                fourcc, 
                self.fps,
                (self.frame_width, self.frame_height)
            )
    
    def run(self):
        """Main processing loop"""
        while self.cap.isOpened():
            success, frame = self.cap.read()
            
            if not success:
                break
                
            self.frame_count += 1
            self.frame_buffer.append(frame.copy())
            
            # Process the frame to detect pose and ball
            pose_results, ball_results, annotated_frame = self.process_frame(frame)
            
            # If we have valid detections, check for violations
            if pose_results and ball_results:
                # Detect different violations
                self.detect_double_dribble()
                self.detect_travel()
                self.detect_blocking_foul(pose_results)
                
                # Annotate the frame with violation information
                annotated_frame = self.annotate_violations(annotated_frame)
                
                # Save violation data if needed
                if self.save_output:
                    self.save_frame_data(pose_results, ball_results)
                    self.save_violation_footage()
            
            # Display the frame
            cv2.imshow("Basketball Referee AI", annotated_frame)
            
            # Save to video if enabled
            if self.save_output:
                self.video_writer.write(annotated_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        self.cap.release()
        if self.save_output and hasattr(self, 'video_writer'):
            self.video_writer.release()
        cv2.destroyAllWindows()
        
        # Save final results
        if self.save_output:
            self.save_final_results()
    
    def process_frame(self, frame):
        """Process a frame to detect pose and ball"""
        # Detect human poses
        pose_results = self.pose_model(frame, verbose=False, conf=0.5)
        pose_annotated_frame = pose_results[0].plot()
        
        # Detect basketball
        ball_results_list = self.ball_model(frame, verbose=False, conf=0.65)
        ball_detected = False
        
        # Process ball detection results
        for ball_results in ball_results_list:
            for bbox in ball_results.boxes.xyxy:
                x1, y1, x2, y2 = bbox[:4]
                
                # Calculate ball center
                ball_x_center = (x1 + x2) / 2
                ball_y_center = (y1 + y2) / 2
                
                # Update dribble count and tracking variables
                self.update_dribble_count(ball_x_center, ball_y_center)
                
                self.prev_x_center = ball_x_center
                self.prev_y_center = ball_y_center
                
                ball_detected = True
                self.ball_not_detected_frames = 0
                
                # Check if ball is being held by player
                try:
                    keypoints = pose_results[0].keypoints.data[0].cpu().numpy()
                    left_wrist = keypoints[self.body_index["left_wrist"]][:2]
                    right_wrist = keypoints[self.body_index["right_wrist"]][:2]
                    
                    # Calculate distance from ball to wrists
                    left_distance = np.hypot(ball_x_center - left_wrist[0], ball_y_center - left_wrist[1])
                    right_distance = np.hypot(ball_x_center - right_wrist[0], ball_y_center - right_wrist[1])
                    
                    # Check if player is holding the ball
                    self.check_holding(left_distance, right_distance)
                except Exception as e:
                    print(f"Error processing keypoints: {e}")
                
                # Draw ball bounding box
                cv2.rectangle(
                    pose_annotated_frame, 
                    (int(x1), int(y1)), 
                    (int(x2), int(y2)), 
                    (0, 255, 0), 
                    2
                )
        
        # If no ball was detected
        if not ball_detected:
            self.ball_not_detected_frames += 1
            self.hold_start_time = None
            self.is_holding = False
            
            # Reset step count if ball is not detected for a prolonged period
            if self.ball_not_detected_frames >= self.max_ball_not_detected_frames:
                self.step_count = 0
        
        # Update step count based on pose
        self.update_step_count(pose_results)
        
        return pose_results, ball_results_list, pose_annotated_frame
    
    def check_holding(self, left_distance, right_distance):
        """Check if the player is holding the ball"""
        if min(left_distance, right_distance) < self.hold_threshold:
            if self.hold_start_time is None:
                self.hold_start_time = time.time()
            elif time.time() - self.hold_start_time > self.hold_duration:
                self.is_holding = True
                self.was_holding = True
                self.dribble_count = 0
        else:
            self.hold_start_time = None
            self.is_holding = False
    
    def update_dribble_count(self, x_center, y_center):
        """Update the dribble count based on ball movement"""
        if self.prev_y_center is not None:
            delta_y = y_center - self.prev_y_center
            
            if (self.prev_delta_y is not None and 
                self.prev_delta_y > self.dribble_threshold and 
                delta_y < -self.dribble_threshold):
                self.dribble_count += 1
                self.total_dribble_count += 1
            
            self.prev_delta_y = delta_y
    
    def update_step_count(self, pose_results):
        """Update step count based on ankle movement"""
        try:
            keypoints = pose_results[0].keypoints.numpy()[0]
            left_ankle = keypoints[self.body_index["left_ankle"]]
            right_ankle = keypoints[self.body_index["right_ankle"]]
            left_knee = keypoints[self.body_index["left_knee"]]
            right_knee = keypoints[self.body_index["right_knee"]]
            
            # Check if keypoints are detected with sufficient confidence
            if (left_knee[2] > 0.5 and right_knee[2] > 0.5 and 
                left_ankle[2] > 0.5 and right_ankle[2] > 0.5):
                
                if (self.prev_left_ankle_y is not None and 
                    self.prev_right_ankle_y is not None and 
                    self.wait_frames == 0):
                    
                    left_diff = abs(left_ankle[1] - self.prev_left_ankle_y)
                    right_diff = abs(right_ankle[1] - self.prev_right_ankle_y)
                    
                    if max(left_diff, right_diff) > self.step_threshold:
                        self.step_count += 1
                        self.total_step_count += 1
                        self.wait_frames = self.min_wait_frames
                
                self.prev_left_ankle_y = left_ankle[1]
                self.prev_right_ankle_y = right_ankle[1]
                
                if self.wait_frames > 0:
                    self.wait_frames -= 1
                    
        except Exception as e:
            print(f"Error updating step count: {e}")
    
    def detect_double_dribble(self):
        """Detect double dribble violations"""
        if self.was_holding and self.dribble_count > 0:
            self.double_dribble_time = time.time()
            self.was_holding = False
            self.dribble_count = 0
            print("Double dribble detected!")
    
    def detect_travel(self):
        """Detect travel violations"""
        if self.ball_not_detected_frames < self.max_ball_not_detected_frames and self.step_count >= 3 and self.dribble_count == 0:
            self.travel_detected = True
            self.travel_timestamp = time.time()
            self.step_count = 0
            print("Travel violation detected!")
            
            # Start saving frames when travel is detected
            if self.save_output and not self.saving:
                filename = os.path.join(
                    "output/violations",
                    f"travel_{time.strftime('%Y%m%d-%H%M%S')}.mp4"
                )
                
                # Create a VideoWriter object
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.out = cv2.VideoWriter(
                    filename, 
                    fourcc, 
                    self.fps, 
                    (self.frame_width, self.frame_height)
                )
                
                # Write the buffered frames
                for f in self.frame_buffer:
                    self.out.write(f)
                
                self.saving = True
    
    def detect_blocking_foul(self, pose_results):
        """Detect blocking fouls based on player proximity"""
        try:
            # Need at least two players to detect blocking fouls
            if len(pose_results[0].keypoints.data) >= 2:
                keypoints_data = pose_results[0].keypoints.data.cpu().numpy()
                
                # Get the first two detected people (assuming one is shooter, one is defender)
                shooter = keypoints_data[0]
                defender = keypoints_data[1]
                
                # Check proximity between specific body parts
                def_parts = ["left_wrist", "right_wrist"]
                shoot_parts = ["left_shoulder", "right_shoulder", "left_elbow", "right_elbow", 
                              "left_wrist", "right_wrist"]
                
                for def_part in def_parts:
                    def_idx = self.body_index[def_part]
                    def_keypoint = defender[def_idx][:2]
                    
                    for shoot_part in shoot_parts:
                        shoot_idx = self.body_index[shoot_part]
                        shoot_keypoint = shooter[shoot_idx][:2]
                        
                        distance = np.hypot(def_keypoint[0] - shoot_keypoint[0], 
                                          def_keypoint[1] - shoot_keypoint[1])
                        
                        if distance < self.player_proximity_threshold:
                            self.blocking_foul_detected = True
                            self.blocking_foul_timestamp = time.time()
                            print("Blocking foul detected!")
                            return
        except Exception as e:
            print(f"Error detecting blocking foul: {e}")
    
    def annotate_violations(self, frame):
        """Add visual indicators for detected violations"""
        # Add information text to frame
        cv2.putText(
            frame,
            f"Dribble count: {self.total_dribble_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        
        cv2.putText(
            frame,
            f"Step count: {self.step_count}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        
        cv2.putText(
            frame,
            f"Holding: {'Yes' if self.is_holding else 'No'}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        
        # Add blue tint if player is holding the ball
        if self.is_holding:
            blue_tint = np.full_like(frame, (255, 0, 0), dtype=np.uint8)
            frame = cv2.addWeighted(frame, 0.7, blue_tint, 0.3, 0)
        
        # Add red tint and text for double dribble
        if self.double_dribble_time and time.time() - self.double_dribble_time <= 3:
            red_tint = np.full_like(frame, (0, 0, 255), dtype=np.uint8)
            frame = cv2.addWeighted(frame, 0.7, red_tint, 0.3, 0)
            
            cv2.putText(
                frame,
                "Double Dribble!",
                (self.frame_width - 600, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                4,
                cv2.LINE_AA,
            )
        
        # Add visual indicators for travel violation
        if self.travel_detected and time.time() - self.travel_timestamp <= 3:
            red_tint = np.full_like(frame, (0, 0, 255), dtype=np.uint8)
            frame = cv2.addWeighted(frame, 0.7, red_tint, 0.3, 0)
            
            cv2.putText(
                frame,
                "Travel Violation!",
                (self.frame_width - 600, 220),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                4,
                cv2.LINE_AA,
            )
        
        # Add visual indicators for blocking foul
        if self.blocking_foul_detected and time.time() - self.blocking_foul_timestamp <= 3:
            yellow_tint = np.full_like(frame, (0, 255, 255), dtype=np.uint8)
            frame = cv2.addWeighted(frame, 0.7, yellow_tint, 0.3, 0)
            
            cv2.putText(
                frame,
                "Blocking Foul!",
                (self.frame_width - 600, 290),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 255, 255),
                4,
                cv2.LINE_AA,
            )
        
        return frame
    
    def save_frame_data(self, pose_results, ball_results):
        """Save frame data for analysis"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        
        ball_x, ball_y = None, None
        ball_detected = False
        
        # Extract ball position if detected
        for results in ball_results:
            if len(results.boxes.xyxy) > 0:
                bbox = results.boxes.xyxy[0]
                x1, y1, x2, y2 = bbox[:4]
                ball_x = (x1 + x2) / 2
                ball_y = (y1 + y2) / 2
                ball_detected = True
                break
        
        # Save data
        data = {
            'timestamp': timestamp,
            'frame_number': self.frame_count,
            'ball_detected': ball_detected,
            'ball_x': ball_x,
            'ball_y': ball_y,
            'is_holding': self.is_holding,
            'dribble_count': self.dribble_count,
            'step_count': self.step_count,
            'double_dribble_detected': self.double_dribble_time is not None and time.time() - self.double_dribble_time <= 3,
            'travel_detected': self.travel_detected and time.time() - self.travel_timestamp <= 3,
            'blocking_foul_detected': self.blocking_foul_detected and time.time() - self.blocking_foul_timestamp <= 3
        }
        
        self.predictions_data.append(data)
    
    def save_violation_footage(self):
        """Save footage of detected violations"""
        if self.saving:
            if self.out is not None:
                current_frame = self.frame_buffer[-1].copy()
                self.out.write(current_frame)
                self.frame_save_counter += 1
                
                # Stop saving frames after reaching the limit
                if self.frame_save_counter >= self.save_frames:
                    self.saving = False
                    self.frame_save_counter = 0
                    self.out.release()
                    self.out = None
    
    def save_final_results(self):
        """Save all predictions to files"""
        print(f"\nSaving results and statistics...")
        
        # Save CSV file with all predictions
        csv_filename = f'output/referee_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        with open(csv_filename, 'w', newline='') as csvfile:
            if self.predictions_data:
                fieldnames = self.predictions_data[0].keys()
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.predictions_data)
        
        # Save summary statistics
        summary_filename = f'output/referee_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        with open(summary_filename, 'w') as f:
            f.write(f"Basketball Referee AI Summary\n")
            f.write(f"============================\n")
            f.write(f"Total Frames Processed: {self.frame_count}\n")
            f.write(f"Total Dribbles Detected: {self.total_dribble_count}\n")
            f.write(f"Total Steps Detected: {self.total_step_count}\n")
            
            # Count violations
            double_dribbles = sum(1 for d in self.predictions_data if d['double_dribble_detected'])
            travels = sum(1 for d in self.predictions_data if d['travel_detected'])
            blocking_fouls = sum(1 for d in self.predictions_data if d['blocking_foul_detected'])
            
            f.write(f"Double Dribble Violations: {double_dribbles}\n")
            f.write(f"Travel Violations: {travels}\n")
            f.write(f"Blocking Fouls: {blocking_fouls}\n")
            f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"✅ Results saved:")
        print(f"   📊 CSV data: {csv_filename}")
        print(f"   📹 Video: output/referee_output.mp4")
        print(f"   📋 Summary: {summary_filename}")


# Main execution
if __name__ == "__main__":
    # Use webcam (0) or video file ("path/to/video.mp4")
    # referee = BasketballRefereeAI(video_source=0, save_output=True)
    referee = BasketballRefereeAI(video_source="driblle.mp4",save_output=True)
    referee.run()