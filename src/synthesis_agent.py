import cv2
import copy
import numpy as np
from src.utils.logger import logger

class SynthesisAgent:
    """
    Correction Synthesis Agent (Omni Generative Core Simulator)
    Simulates the Omni video-to-video pose guidance diffusion pipeline.
    It reads the original video, retrieves the real MediaPipe frame coordinates,
    and overlays the actual skeletal positions side-by-side with the biomechanically corrected "ideal" pose.
    """
    def __init__(self):
        logger.info("SynthesisAgent initialized.")

    def generate_ideal_video(self, video_path: str, raw_frames: list[dict], reps_telemetry: list[dict], output_path: str = "output_corrected.mp4") -> str:
        """
        Creates a side-by-side comparison video.
        Left: Original user video with real red/orange tracked skeleton.
        Right: Original background/user body with a green mathematically corrected skeleton.
        """
        logger.info(f"SynthesisAgent: Creating real skeletal overlay video from {video_path}...")
        
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

        # We map frames inside identified faulty reps
        faulty_frames_map = {}
        for rep in reps_telemetry:
            for f_id in range(rep["start_frame"], rep["end_frame"] + 1):
                faulty_frames_map[f_id] = rep

        # Quick lookup for frame landmarks
        frames_lookup = {f["frame_id"]: f for f in raw_frames}

        frame_idx = 0
        logger.info("SynthesisAgent: Commencing real coordinate skeletal mapping...")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            left_frame = frame.copy()
            right_frame = frame.copy()

            # If we have real landmark coordinates for this frame, draw them!
            if frame_idx in frames_lookup:
                frame_data = frames_lookup[frame_idx]
                lms = frame_data["landmarks"]
                
                # Check if this frame is inside an identified repetition
                if frame_idx in faulty_frames_map:
                    rep = faulty_frames_map[frame_idx]
                    faults = rep["faults"]
                    has_faults = any(faults.values())
                    
                    # Left Frame: Draw actual tracked skeleton (Red if faulty, Orange/Yellow if Warning)
                    color_actual = (0, 0, 255) if has_faults else (0, 165, 255)
                    cv2.putText(left_frame, f"ACTUAL FORM: Rep {rep['rep_index']}", (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_actual, 2, cv2.LINE_AA)
                    
                    y_offset = 90
                    for fault_name, triggered in faults.items():
                        if triggered:
                            cv2.putText(left_frame, f"[ALERT] {fault_name.replace('_', ' ').upper()}", 
                                        (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_actual, 2, cv2.LINE_AA)
                            y_offset += 30
                            
                    # Draw your actual skeleton on the left frame
                    self._draw_real_skeleton(left_frame, lms, color=color_actual, width=width, height=height)
                    
                    # Right Frame: Draw OMNI Corrected skeleton (Green)
                    color_ideal = (0, 255, 0)
                    cv2.putText(right_frame, f"OMNI CORRECTED: Rep {rep['rep_index']}", (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_ideal, 2, cv2.LINE_AA)
                    
                    if has_faults:
                        cv2.putText(right_frame, "[OMNI] Deforming pose coordinates to ideal vertical/depth planes...", (30, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_ideal, 1, cv2.LINE_AA)
                    else:
                        cv2.putText(right_frame, "[OMNI] Perfect textbook alignment maintained", (30, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_ideal, 1, cv2.LINE_AA)
                    
                    # Compute corrected landmarks dynamically based on original landmarks
                    corrected_lms = self._compute_ideal_landmarks(lms, faults)
                    self._draw_real_skeleton(right_frame, corrected_lms, color=color_ideal, width=width, height=height)
                    
                else:
                    # Outside reps / setup: Draw standard cyan/blue tracking overlay
                    color_tracking = (255, 255, 0) # Cyan in BGR
                    cv2.putText(left_frame, "ACTIVE SENSOR SKELETON DETECTED", (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_tracking, 2, cv2.LINE_AA)
                    cv2.putText(right_frame, "ACTIVE SENSOR SKELETON DETECTED", (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color_tracking, 2, cv2.LINE_AA)
                    
                    self._draw_real_skeleton(left_frame, lms, color=color_tracking, width=width, height=height)
                    self._draw_real_skeleton(right_frame, lms, color=color_tracking, width=width, height=height)
            else:
                # No landmarks detected in frame
                cv2.putText(left_frame, "NO TRACKING DETECTED", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(right_frame, "NO TRACKING DETECTED", (30, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

            combined_frame = np.hstack((left_frame, right_frame))
            out.write(combined_frame)
            frame_idx += 1

        cap.release()
        out.release()
        logger.info(f"SynthesisAgent: Perfect form comparison video compiled and saved to {output_path}")
        return output_path

    def _draw_real_skeleton(self, img, landmarks: dict, color: tuple, width: int, height: int):
        """
        Draws the actual joint wireframe skeleton on the frame based on coordinate dicts.
        """
        # Convert normalized coordinates to pixel spaces
        def to_px(pt_name):
            if pt_name not in landmarks:
                return None
            pt = landmarks[pt_name]
            return int(pt["x"] * width), int(pt["y"] * height)

        s_l = to_px("left_shoulder")
        s_r = to_px("right_shoulder")
        h_l = to_px("left_hip")
        h_r = to_px("right_hip")
        k_l = to_px("left_knee")
        k_r = to_px("right_knee")
        a_l = to_px("left_ankle")
        a_r = to_px("right_ankle")

        # Double check none of the core points are missing
        if not all([s_l, s_r, h_l, h_r, k_l, k_r, a_l, a_r]):
            return

        thickness = 4

        # Draw Spine/Shoulders/Hips structure
        cv2.line(img, s_l, s_r, color, thickness)
        cv2.line(img, s_l, h_l, color, thickness)
        cv2.line(img, s_r, h_r, color, thickness)
        cv2.line(img, h_l, h_r, color, thickness)

        # Draw Legs structure
        cv2.line(img, h_l, k_l, color, thickness)
        cv2.line(img, h_r, k_r, color, thickness)
        cv2.line(img, k_l, a_l, color, thickness)
        cv2.line(img, k_r, a_r, color, thickness)

        # Draw Joint nodes
        joints = [s_l, s_r, h_l, h_r, k_l, k_r, a_l, a_r]
        for joint in joints:
            cv2.circle(img, joint, 8, (255, 255, 255), -1)
            cv2.circle(img, joint, 5, color, -1)

    def _compute_ideal_landmarks(self, lms: dict, faults: dict) -> dict:
        """
        Takes the actual landmarks dictionary and returns a mathematically
        adjusted version representing perfect squat postures.
        """
        # Deep copy landmarks to manipulate safely
        corrected = copy.deepcopy(lms)

        # 1. Correct Shallow Depth
        # If hips fail to sink past the knee level (larger y is deeper)
        if faults.get("shallow_depth"):
            # Set hips slightly lower than knee level
            corrected["left_hip"]["y"] = corrected["left_knee"]["y"] + 0.03
            corrected["right_hip"]["y"] = corrected["right_knee"]["y"] + 0.03

        # 2. Correct Excessive Torso Lean
        # In a side profile view, leaning forward means shoulders are horizontally displaced forward.
        # We push shoulders backward horizontally (close to hip vertical line) and raise them up.
        if faults.get("excessive_forward_lean"):
            # Move shoulders vertically above hips
            corrected["left_shoulder"]["x"] = corrected["left_hip"]["x"]
            corrected["right_shoulder"]["x"] = corrected["right_hip"]["x"]
            # Raise shoulders vertically (smaller y)
            corrected["left_shoulder"]["y"] = corrected["left_hip"]["y"] - 0.35
            corrected["right_shoulder"]["y"] = corrected["right_hip"]["y"] - 0.35

        # 3. Correct Knee Valgus (Inward cave-in)
        # We widen knee x-coordinates outwards relative to hips/ankles
        if faults.get("knee_valgus"):
            # Knees should track in line or slightly wider than ankles
            corrected["left_knee"]["x"] = corrected["left_ankle"]["x"] - 0.03
            corrected["right_knee"]["x"] = corrected["right_ankle"]["x"] + 0.03

        return corrected
