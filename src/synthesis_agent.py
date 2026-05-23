import cv2
import numpy as np
from src.utils.logger import logger

class SynthesisAgent:
    """
    Correction Synthesis Agent (Omni Generative Core Simulator)
    Simulates the Omni video-to-video pose guidance diffusion pipeline.
    It reads the original video and outputs a visual comparison video overlaying
    the actual skeletal faults side-by-side with the biomechanically corrected "ideal" pose.
    """
    def __init__(self):
        logger.info("SynthesisAgent initialized.")

    def generate_ideal_video(self, video_path: str, reps_telemetry: list[dict], output_path: str = "output_corrected.mp4") -> str:
        """
        Creates a side-by-side comparison video.
        Left: Original user video with red/orange "faulty" skeleton.
        Right: Original background/user body with a green "Omni-corrected" ideal skeleton.
        """
        logger.info(f"SynthesisAgent: Creating ideal corrected video from {video_path}...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open source video: {video_path}")
            return video_path

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Output video will be side-by-side (2 * width, height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

        # We will parse frame indices of bad reps to know when to apply visual deformation
        faulty_frames_map = {}
        for rep in reps_telemetry:
            # Map every frame inside this rep to its specific faults
            for f_id in range(rep["start_frame"], rep["end_frame"] + 1):
                faulty_frames_map[f_id] = rep

        frame_idx = 0
        logger.info("SynthesisAgent: Commencing frame-by-frame visual pose correction...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Create side-by-side frames
            left_frame = frame.copy()
            right_frame = frame.copy()

            # If the current frame is inside an identified faulty repetition, draw skeletons
            if frame_idx in faulty_frames_map:
                rep = faulty_frames_map[frame_idx]
                faults = rep["faults"]
                
                # Draw "Actual Form" visual guides on the left frame (Red / Amber warning)
                color_actual = (0, 0, 255) # Red in BGR
                cv2.putText(left_frame, f"ACTUAL: Rep {rep['rep_index']}", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_actual, 2, cv2.LINE_AA)
                
                # Draw warnings for detected faults
                y_offset = 90
                for fault_name, triggered in faults.items():
                    if triggered:
                        cv2.putText(left_frame, f"[FAULT] {fault_name.replace('_', ' ').upper()}", 
                                    (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_actual, 2, cv2.LINE_AA)
                        y_offset += 30

                # Draw "Omni Ideal Form" visual guides on the right frame (Green ideal)
                color_ideal = (0, 255, 0) # Green in BGR
                cv2.putText(right_frame, f"OMNI IDEAL: Rep {rep['rep_index']}", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_ideal, 2, cv2.LINE_AA)
                
                cv2.putText(right_frame, "[OMNI] Torso uprighted & depth parallelized", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_ideal, 1, cv2.LINE_AA)

                # Visual effect: draw a clean mock coordinate skeletal overlay
                # Let's draw standard squat landmarks on the left (distorted) and right (corrected)
                self._draw_mock_skeleton(left_frame, has_faults=True, faults=faults, width=width, height=height)
                self._draw_mock_skeleton(right_frame, has_faults=False, faults=faults, width=width, height=height)
            else:
                # Outside repetitions: draw simple green status
                cv2.putText(left_frame, "Status: Set Setup", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(right_frame, "Status: Set Setup", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            # Stitch side-by-side
            combined_frame = np.hstack((left_frame, right_frame))
            out.write(combined_frame)
            frame_idx += 1

        cap.release()
        out.release()
        logger.info(f"SynthesisAgent: Perfect form comparison video compiled and saved to {output_path}")
        return output_path

    def _draw_mock_skeleton(self, img, has_faults: bool, faults: dict, width: int, height: int):
        """
        Draws simulated skeletal stick figures on the frame.
        If has_faults is True, joint positions reflect errors.
        If has_faults is False (Ideal Form), the joint positions are mathematically adjusted to show perfect posture.
        """
        # Base normalized coordinates for a body performing a squat
        # Coordinates: [X, Y] from 0 to 1
        shoulder_l = [0.45, 0.35]
        shoulder_r = [0.55, 0.35]
        hip_l = [0.44, 0.65]
        hip_r = [0.56, 0.65]
        
        knee_l = [0.40, 0.78]
        knee_r = [0.60, 0.78]
        
        ankle_l = [0.42, 0.90]
        ankle_r = [0.58, 0.90]

        # Apply fault deformations on the Left Frame (Actual Form)
        if has_faults:
            if faults.get("shallow_depth"):
                # Shallow: Hips don't sink low enough (y hip is higher/smaller than knee)
                hip_l[1] = 0.58
                hip_r[1] = 0.58
            if faults.get("excessive_forward_lean"):
                # Leaning forward: shoulders move forward/down and hips move back
                shoulder_l[0] -= 0.08
                shoulder_r[0] -= 0.08
                shoulder_l[1] += 0.05
                shoulder_r[1] += 0.05
            if faults.get("knee_valgus"):
                # Knee caving: Knees move closer to each other horizontally
                knee_l[0] = 0.47
                knee_r[0] = 0.53
        else:
            # Ideal form adjustments (Green, perfect posture alignment)
            # Torso upright, hips sunk to parallel, knees tracking correctly
            shoulder_l = [0.45, 0.32] # Upright shoulder
            shoulder_r = [0.55, 0.32]
            hip_l = [0.44, 0.75]      # Deep hip depth (y is larger)
            hip_r = [0.56, 0.75]
            knee_l = [0.38, 0.78]     # Knees pushed out correctly
            knee_r = [0.62, 0.78]

        # Convert normalized coordinates to pixel spaces
        def to_px(pt):
            return int(pt[0] * width), int(pt[1] * height)

        s_l, s_r = to_px(shoulder_l), to_px(shoulder_r)
        h_l, h_r = to_px(hip_l), to_px(hip_r)
        k_l, k_r = to_px(knee_l), to_px(knee_r)
        a_l, a_r = to_px(ankle_l), to_px(ankle_r)

        color = (0, 0, 255) if has_faults else (0, 255, 0)
        thickness = 4

        # Draw Spine/Shoulders
        cv2.line(img, s_l, s_r, color, thickness)
        cv2.line(img, s_l, h_l, color, thickness)
        cv2.line(img, s_r, h_r, color, thickness)
        cv2.line(img, h_l, h_r, color, thickness)

        # Draw Legs
        cv2.line(img, h_l, k_l, color, thickness)
        cv2.line(img, h_r, k_r, color, thickness)
        cv2.line(img, k_l, a_l, color, thickness)
        cv2.line(img, k_r, a_r, color, thickness)

        # Draw Joint nodes
        joints = [s_l, s_r, h_l, h_r, k_l, k_r, a_l, a_r]
        for joint in joints:
            cv2.circle(img, joint, 8, (255, 255, 255), -1)
            cv2.circle(img, joint, 5, color, -1)
