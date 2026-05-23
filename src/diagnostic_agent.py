import math
import numpy as np
from src.utils.logger import logger

class DiagnosticAgent:
    """
    Kinematic Diagnostic Agent (Biomechanical Reasoning Expert)
    Translates raw 3D landmark streams into detailed biomechanical metrics,
    segments the workout into individual repetitions, and flags posture faults.
    """
    def __init__(self):
        logger.info("DiagnosticAgent initialized.")

    @staticmethod
    def calculate_angle_3d(p1: dict, p2: dict, p3: dict) -> float:
        """
        Calculates the angle (in degrees) formed at vertex p2 by vectors p1-p2 and p3-p2.
        Input points are dicts with keys 'x', 'y', 'z'.
        """
        v1 = np.array([p1['x'] - p2['x'], p1['y'] - p2['y'], p1['z'] - p2['z']])
        v2 = np.array([p3['x'] - p2['x'], p3['y'] - p2['y'], p3['z'] - p2['z']])
        
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        
        if norm_v1 == 0 or norm_v2 == 0:
            return 180.0
            
        cos_theta = dot_product / (norm_v1 * norm_v2)
        # Clip to prevent numerical overflow outside [-1.0, 1.0]
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        angle_rad = np.arccos(cos_theta)
        return float(np.degrees(angle_rad))

    @staticmethod
    def calculate_torso_angle(shoulder: dict, hip: dict) -> float:
        """
        Calculates the forward lean angle (in degrees) of the torso relative to the vertical vector.
        Vector p2-p1 represents the torso. Vertical vector points straight up (0, -1, 0) in MediaPipe coordinate space.
        """
        # Torso vector points from hip to shoulder
        torso_vector = np.array([shoulder['x'] - hip['x'], shoulder['y'] - hip['y'], shoulder['z'] - hip['z']])
        vertical_vector = np.array([0.0, -1.0, 0.0])  # Negative Y is "up" in MediaPipe
        
        dot_product = np.dot(torso_vector, vertical_vector)
        norm_torso = np.linalg.norm(torso_vector)
        norm_vertical = np.linalg.norm(vertical_vector)
        
        if norm_torso == 0 or norm_vertical == 0:
            return 0.0
            
        cos_theta = dot_product / (norm_torso * norm_vertical)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        angle_rad = np.arccos(cos_theta)
        return float(np.degrees(angle_rad))

    def analyze_kinematics(self, frames_data: list[dict]) -> list[dict]:
        """
        Processes a list of raw frames and appends calculated angles and metrics.
        """
        logger.info("DiagnosticAgent: Starting coordinate kinematic analysis...")
        processed_frames = []
        
        for frame in frames_data:
            # If metrics are already present (e.g. from mock testing generators), preserve them
            if "metrics" in frame:
                processed_frames.append(frame)
                continue
                
            lms = frame["landmarks"]

            
            # Left & Right Knee Angles
            left_knee_angle = self.calculate_angle_3d(lms["left_hip"], lms["left_knee"], lms["left_ankle"])
            right_knee_angle = self.calculate_angle_3d(lms["right_hip"], lms["right_knee"], lms["right_ankle"])
            
            # Torso angle relative to vertical (average of left/right)
            left_torso_angle = self.calculate_torso_angle(lms["left_shoulder"], lms["left_hip"])
            right_torso_angle = self.calculate_torso_angle(lms["right_shoulder"], lms["right_hip"])
            avg_torso_angle = (left_torso_angle + right_torso_angle) / 2.0
            
            # Depth metric: Y position of hip relative to knee (larger Y is lower)
            # depth_ratio > 0 means hips are lower than knees (at or below parallel)
            left_depth_gap = lms["left_hip"]["y"] - lms["left_knee"]["y"]
            right_depth_gap = lms["right_hip"]["y"] - lms["right_knee"]["y"]
            avg_depth_gap = (left_depth_gap + right_depth_gap) / 2.0
            
            # Valgus: lateral caving. If knees are caving in, the width between knees is smaller
            # than the width between hips/ankles. We evaluate knee distance vs ankle distance.
            knee_width = abs(lms["left_knee"]["x"] - lms["right_knee"]["x"])
            ankle_width = abs(lms["left_ankle"]["x"] - lms["right_ankle"]["x"])
            valgus_ratio = knee_width / (ankle_width if ankle_width > 0 else 1.0)
            
            processed_frames.append({
                **frame,
                "metrics": {
                    "left_knee_angle": left_knee_angle,
                    "right_knee_angle": right_knee_angle,
                    "avg_knee_angle": (left_knee_angle + right_knee_angle) / 2.0,
                    "avg_torso_angle": avg_torso_angle,
                    "avg_depth_gap": avg_depth_gap,
                    "valgus_ratio": valgus_ratio
                }
            })
            
        logger.info(f"DiagnosticAgent: Completed processing of {len(processed_frames)} frames.")
        return processed_frames

    def segment_reps(self, frames_data: list[dict]) -> list[dict]:
        """
        Segments continuous squat telemetry into discrete reps using a knee-angle state machine.
        
        States:
        - "STAND" (knee angle > 160)
        - "DESCENT" (knee angle decreasing below 160)
        - "ASCENT" (knee angle rising from bottom flexion)
        """
        logger.info("DiagnosticAgent: Segmenting squats into repetitions...")
        reps = []
        
        # State variables
        current_state = "STAND"
        rep_start_frame = None
        rep_min_knee_angle = 180.0
        rep_bottom_frame = None
        rep_max_torso_angle = 0.0
        rep_min_depth_gap = -1.0 # Tracks deepest hip level (most negative or positive)
        rep_valgus_spikes = 0
        
        # We need processing data with metrics
        analyzed_frames = self.analyze_kinematics(frames_data)
        
        for i, frame in enumerate(analyzed_frames):
            metrics = frame["metrics"]
            angle = metrics["avg_knee_angle"]
            torso = metrics["avg_torso_angle"]
            depth_gap = metrics["avg_depth_gap"]
            valgus = metrics["valgus_ratio"]
            
            if current_state == "STAND":
                # Descent begins if knee flexion goes below 160
                if angle < 160.0:
                    current_state = "DESCENT"
                    rep_start_frame = frame
                    rep_min_knee_angle = angle
                    rep_bottom_frame = frame
                    rep_max_torso_angle = torso
                    rep_min_depth_gap = depth_gap
                    rep_valgus_spikes = 1 if valgus < 0.85 else 0
                    
            elif current_state == "DESCENT":
                # Record metrics at deepest inflection point
                if angle < rep_min_knee_angle:
                    rep_min_knee_angle = angle
                    rep_bottom_frame = frame
                    
                rep_max_torso_angle = max(rep_max_torso_angle, torso)
                rep_min_depth_gap = max(rep_min_depth_gap, depth_gap) # In MediaPipe, larger depth_gap is deeper
                if valgus < 0.85:
                    rep_valgus_spikes += 1
                    
                # Transition to ascent once knee angle starts opening back up significantly
                # E.g., current angle is 5 degrees larger than minimum found so far
                if angle > rep_min_knee_angle + 8.0:
                    current_state = "ASCENT"
                    
            elif current_state == "ASCENT":
                rep_max_torso_angle = max(rep_max_torso_angle, torso)
                if valgus < 0.85:
                    rep_valgus_spikes += 1
                
                # Rep is complete when user returns to a straight standing pose (knee > 165)
                if angle > 165.0:
                    rep_end_frame = frame
                    
                    # Store rep metrics
                    reps.append({
                        "rep_index": len(reps) + 1,
                        "start_time": rep_start_frame["timestamp_sec"],
                        "bottom_time": rep_bottom_frame["timestamp_sec"],
                        "end_time": rep_end_frame["timestamp_sec"],
                        "start_frame": rep_start_frame["frame_id"],
                        "bottom_frame": rep_bottom_frame["frame_id"],
                        "end_frame": rep_end_frame["frame_id"],
                        "metrics": {
                            "min_knee_angle": rep_min_knee_angle,
                            "max_torso_lean_angle": rep_max_torso_angle,
                            "peak_depth_gap": rep_min_depth_gap,
                            "valgus_count": rep_valgus_spikes
                        },
                        "faults": {
                            "shallow_depth": rep_min_depth_gap < -0.05, # hips failed to reach knee plane
                            "excessive_forward_lean": rep_max_torso_angle > 40.0, # torso angled too far forward
                            "knee_valgus": rep_valgus_spikes > 5 # knees caved multiple times during rep
                        }
                    })
                    
                    logger.info(f"DiagnosticAgent: Segmented Rep #{len(reps)} "
                                f"({rep_start_frame['timestamp_sec']:.2f}s -> "
                                f"{rep_bottom_frame['timestamp_sec']:.2f}s -> "
                                f"{rep_end_frame['timestamp_sec']:.2f}s)")
                    
                    # Reset State
                    current_state = "STAND"
                    rep_start_frame = None

        # Post-filter segmented repetitions to eliminate high-frequency tracking noise sways and setup wiggles
        filtered_reps = []
        for rep in reps:
            duration = rep["end_time"] - rep["start_time"]
            min_knee = rep["metrics"]["min_knee_angle"]
            
            # Constraints:
            # 1. Squat duration must be >= 0.5s (anything faster is a MediaPipe coordinate tracking jitter spike).
            # 2. Knee angle must dip below 150 degrees (wiggles where the athlete sways at setup are filtered).
            if duration >= 0.5 and min_knee < 150.0:

                rep["rep_index"] = len(filtered_reps) + 1
                filtered_reps.append(rep)
            else:
                logger.info(f"DiagnosticAgent: Filtering out false positive rep #{rep['rep_index']} "
                            f"(Duration: {duration:.2f}s, Min Knee: {min_knee:.1f}deg) as noise.")
        
        logger.info(f"DiagnosticAgent: Rep segmentation completed. Total reps identified after filtering: {len(filtered_reps)}.")
        return filtered_reps

