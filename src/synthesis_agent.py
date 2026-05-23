import cv2
import copy
import numpy as np
import os
from src.utils.logger import logger

# Import Google GenAI SDK if available
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google-genai SDK not available. SynthesisAgent will run in local skeletal overlay fallback mode.")

class SynthesisAgent:
    """
    Correction Synthesis Agent (Omni Generative U-Net & Veo API Connector)
    Coordinates the video-to-video pose guidance and diffusion pipeline.
    Connects directly to Google GenAI's Gemini Omni and Veo APIs to generate/edit
    perfect-form videos, or falls back to custom static side-by-side skeletal overlays.
    """
    def __init__(self):
        logger.info("SynthesisAgent initialized.")

    def generate_ideal_video(self, video_path: str, raw_frames: list[dict], reps_telemetry: list[dict], coaching_feedback: dict = None, output_path: str = "output_corrected.mp4") -> str:
        """
        Creates a side-by-side perfect-form corrected wireframe comparison video locally.
        Independently, queries Google's premium Veo API to generate a short virtual coach demonstration video
        showing bad form correcting to perfect form based on Gemini's coaching advice.
        """
        logger.info(f"SynthesisAgent: Synthesizing ideal video from {video_path}...")
        
        # ----------------- PART 1: Google Veo Cloud Virtual Coach Video -----------------
        api_key = os.environ.get("GEMINI_API_KEY")
        if GENAI_AVAILABLE and api_key:
            try:
                # Compile specific coaching demonstration prompt based on feedback
                faults_detected = []
                if coaching_feedback and "reps" in coaching_feedback:
                    for rep in coaching_feedback["reps"]:
                        desc = rep.get("posture_evaluation", "").lower() + " " + rep.get("depth_evaluation", "").lower()
                        if "shallow" in desc and "depth" not in faults_detected:
                            faults_detected.append("depth")
                        if "lean" in desc and "lean" not in faults_detected:
                            faults_detected.append("lean")
                        if "valgus" in desc and "valgus" not in faults_detected:
                            faults_detected.append("valgus")
                
                # Fallback to reps_telemetry if coaching_feedback is empty or missing
                if not faults_detected:
                    for rep in reps_telemetry:
                        if rep["faults"].get("shallow_depth") and "depth" not in faults_detected:
                            faults_detected.append("depth")
                        if rep["faults"].get("excessive_forward_lean") and "lean" not in faults_detected:
                            faults_detected.append("lean")
                        if rep["faults"].get("knee_valgus") and "valgus" not in faults_detected:
                            faults_detected.append("valgus")

                veo_prompt = "A high-fidelity video of a professional strength coach in a gym showing how to squat properly. "
                if "lean" in faults_detected:
                    veo_prompt += (
                        "The coach starts at the bottom of the squat with a heavily forward-bent posture (bad form), "
                        "and then smoothly demonstrates the corrective action by straightening their back, raising their chest tall and proud, "
                        "and keeping their torso upright at a safe, natural 20-degree lean relative to vertical. A perfect visual transition showing bad form correcting to textbook form."
                    )
                elif "valgus" in faults_detected:
                    veo_prompt += (
                        "The coach starts the ascent with their knees caving inward (bad knee valgus form), "
                        "and then smoothly demonstrates the corrective action by driving their knees straight out over their toes "
                        "to align perfectly over the ankles. A perfect visual transition showing bad form correcting to textbook form."
                    )
                elif "depth" in faults_detected:
                    veo_prompt += (
                        "The coach starts the squat but stops shallow above parallel (bad form), "
                        "and then smoothly demonstrates the corrective action by sinking their hips back and down to reach full deep parallel depth "
                        "with perfect posture. A perfect visual transition showing bad form correcting to textbook form."
                    )
                else:
                    veo_prompt += (
                        "The coach performs a textbook back squat with flawless form: controlled descent, "
                        "sinking hips deep to parallel, keeping the chest tall, knees tracking perfectly over the toes, and a smooth ascent. "
                        "Extremely educational, demonstrating perfect technique."
                    )

                logger.info(f"SynthesisAgent: Formulated Veo Virtual Coach prompt: {veo_prompt}")
                client = genai.Client(api_key=api_key)
                
                logger.info("SynthesisAgent: Querying Google Veo video generation endpoint ('veo-2.0-generate-001') for virtual coach demo...")
                operation = client.models.generate_videos(
                    model="veo-2.0-generate-001",
                    prompt=veo_prompt,
                    config=types.GenerateVideosConfig(
                        aspect_ratio="16:9",
                        duration_seconds=5,
                    )
                )
                
                # Poll the long-running operation
                import time
                wait_count = 0
                max_wait = 18  # Wait up to 180 seconds (10s intervals)
                
                while not operation.done and wait_count < max_wait:
                    logger.info("SynthesisAgent: Waiting for cloud coach video generation... (polling 10s)")
                    time.sleep(10)
                    operation = client.operations.get(operation)
                    wait_count += 1
                    
                if operation.done:
                    res = operation.result if (hasattr(operation, "result") and operation.result) else getattr(operation, "response", None)
                    if res and res.generated_videos:
                        generated_video = res.generated_videos[0]
                        logger.info("SynthesisAgent: Cloud coach video generation complete. Downloading video file...")
                        video_bytes = client.files.download(file=generated_video.video)
                        
                        veo_output_path = "veo_coaching_demo.mp4"
                        with open(veo_output_path, "wb") as f:
                            f.write(video_bytes)
                        logger.info(f"SynthesisAgent: Cloud coach video generated and saved successfully to {veo_output_path}")
                    else:
                        logger.warning("SynthesisAgent: No generated videos found in operation response.")
                else:
                    logger.warning("SynthesisAgent: Cloud coach video generation timed out.")
            except Exception as cloud_err:
                logger.error(f"SynthesisAgent: Google cloud video API call failed: {cloud_err}.")

        # ----------------- PART 2: Local Skeletal Overlay Side-by-Side Path -----------------
        logger.info("SynthesisAgent: Commencing local side-by-side skeletal overlay rendering...")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open source video: {video_path}")
            return video_path

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Output video will be side-by-side (2 * width, height)
        # We write to a temporary file first, then transcode to H.264
        self.temp_path = "temp_synthesis_output.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(self.temp_path, fourcc, fps, (width * 2, height))

        # Map frame indices inside identified repetitions
        faulty_frames_map = {}
        for rep in reps_telemetry:
            for f_id in range(rep["start_frame"], rep["end_frame"] + 1):
                faulty_frames_map[f_id] = rep

        # Quick lookup for frame landmarks
        frames_lookup = {f["frame_id"]: f for f in raw_frames}
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            left_frame = frame.copy()
            right_frame = frame.copy()

            if frame_idx in frames_lookup:
                frame_data = frames_lookup[frame_idx]
                lms = frame_data["landmarks"]
                
                if frame_idx in faulty_frames_map:
                    rep = faulty_frames_map[frame_idx]
                    faults = rep["faults"]
                    has_faults = any(faults.values())
                    
                    # Draw only the simplified Rep count on both frames
                    color_actual = (0, 0, 255) if has_faults else (0, 165, 255)
                    cv2.putText(left_frame, f"Rep {rep['rep_index']}", (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_actual, 2, cv2.LINE_AA)
                    
                    # Draw actual skeleton on the left frame
                    self._draw_real_skeleton(left_frame, lms, color=color_actual, width=width, height=height)
                    
                    # Draw only the simplified Rep count on corrected frame
                    color_ideal = (0, 255, 0)
                    cv2.putText(right_frame, f"Rep {rep['rep_index']}", (30, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color_ideal, 2, cv2.LINE_AA)
                    
                    # Compute corrected landmarks dynamically based on original landmarks
                    corrected_lms = self._compute_ideal_landmarks(lms, faults)
                    
                    # Draw OMNI Corrected skeleton overlay directly on top of the original body (NO warping!)
                    self._draw_real_skeleton(right_frame, corrected_lms, color=color_ideal, width=width, height=height)
                    
                else:
                    # Outside reps / setup: Draw standard cyan tracking skeleton
                    color_tracking = (255, 255, 0)
                    self._draw_real_skeleton(left_frame, lms, color=color_tracking, width=width, height=height)
                    self._draw_real_skeleton(right_frame, lms, color=color_tracking, width=width, height=height)
            else:
                pass

            combined_frame = np.hstack((left_frame, right_frame))
            out.write(combined_frame)
            frame_idx += 1

        cap.release()
        out.release()

        # Transcode video to browser-playable H.264
        import subprocess
        try:
            logger.info("SynthesisAgent: Transcoding compiled video to browser-playable H.264 codec...")
            cmd = [
                "ffmpeg", "-y", "-i", self.temp_path,
                "-vcodec", "libx264", "-pix_fmt", "yuv420p",
                output_path
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            if os.path.exists(self.temp_path):
                os.remove(self.temp_path)
            logger.info(f"SynthesisAgent: Browser-friendly H.264 video transcoded successfully to {output_path}")
        except Exception as err:
            logger.warning(f"SynthesisAgent: FFMpeg transcode failed: {err}. Falling back directly to MPEG-4.")
            if os.path.exists(self.temp_path):
                if os.path.exists(output_path):
                    os.remove(output_path)
                os.rename(self.temp_path, output_path)
                
        return output_path

    def _draw_badge(self, img, text: str, coord: tuple, color: tuple):
        """
        Draws a premium, semi-transparent text badge for biomechanical annotations.
        """
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        
        # Get size of text
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        x, y = coord
        x_pad, y_pad = 6, 4
        
        # Box corners
        x1 = x - x_pad
        y1 = y - text_height - y_pad
        x2 = x + text_width + x_pad
        y2 = y + y_pad + 2
        
        # Create transparent overlay box for maximum premium readability
        overlay = img.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (15, 15, 20), -1)
        cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        
        # Draw badge border
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
        
        # Draw white text
        cv2.putText(img, text, (x, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    def _draw_real_skeleton(self, img, landmarks: dict, color: tuple, width: int, height: int):
        """
        Draws the actual joint wireframe skeleton on the frame based on coordinate dicts
        alongside premium annotations of joint names and degrees.
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

        # Calculate exact 3D angles to display next to the skeleton overlay
        from src.diagnostic_agent import DiagnosticAgent
        
        left_knee_angle = DiagnosticAgent.calculate_angle_3d(landmarks["left_hip"], landmarks["left_knee"], landmarks["left_ankle"])
        right_knee_angle = DiagnosticAgent.calculate_angle_3d(landmarks["right_hip"], landmarks["right_knee"], landmarks["right_ankle"])
        
        left_torso_angle = DiagnosticAgent.calculate_torso_angle(landmarks["left_shoulder"], landmarks["left_hip"])
        right_torso_angle = DiagnosticAgent.calculate_torso_angle(landmarks["right_shoulder"], landmarks["right_hip"])
        
        # Display joint labeling and angle degree badges next to each joint (rounded to integers, standard ASCII "deg" units)
        # Left-side labels are offset leftwards, right-side labels offset rightwards to avoid overlap
        self._draw_badge(img, "L Shoulder", (s_l[0] - 80, s_l[1] - 8), color)
        self._draw_badge(img, "R Shoulder", (s_r[0] + 12, s_r[1] - 8), color)
        
        self._draw_badge(img, f"L Torso: {int(round(left_torso_angle))} deg", (h_l[0] - 110, h_l[1] - 8), color)
        self._draw_badge(img, f"R Torso: {int(round(right_torso_angle))} deg", (h_r[0] + 12, h_r[1] - 8), color)
        
        self._draw_badge(img, f"L Knee: {int(round(left_knee_angle))} deg", (k_l[0] - 110, k_l[1] + 12), color)
        self._draw_badge(img, f"R Knee: {int(round(right_knee_angle))} deg", (k_r[0] + 12, k_r[1] + 12), color)
        
        self._draw_badge(img, "L Ankle", (a_l[0] - 65, a_l[1] + 12), color)
        self._draw_badge(img, "R Ankle", (a_r[0] + 12, a_r[1] + 12), color)

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
        # In a back squat, aiming for a completely vertical spine (0 degrees) is biomechanically unrealistic
        # and physically impossible due to the need to balance the center of gravity over the midfoot.
        # We correct the excessive lean to a safe, natural, and realistic 20-degree forward lean.
        # We preserve the actual torso length and direction of the user's squat lean (facing left or right).
        if faults.get("excessive_forward_lean"):
            target_lean_rad = np.radians(20.0)  # Realistic 20-degree lean target
            for side in ["left", "right"]:
                sh_name = f"{side}_shoulder"
                hip_name = f"{side}_hip"
                
                dx = lms[sh_name]["x"] - lms[hip_name]["x"]
                dy = lms[sh_name]["y"] - lms[hip_name]["y"]
                
                # Conserve the user's physical torso length (L)
                torso_len = np.sqrt(dx**2 + dy**2)
                
                if torso_len > 0:
                    # Dynamically preserve the forward lean direction (left vs right profile)
                    sign_x = np.sign(dx) if dx != 0 else 1.0
                    
                    corrected[sh_name]["x"] = lms[hip_name]["x"] + sign_x * torso_len * np.sin(target_lean_rad)
                    corrected[sh_name]["y"] = lms[hip_name]["y"] - torso_len * np.cos(target_lean_rad)

        # 3. Correct Knee Valgus (Inward cave-in)
        # We widen knee x-coordinates outwards relative to hips/ankles
        if faults.get("knee_valgus"):
            # Knees should track in line or slightly wider than ankles
            corrected["left_knee"]["x"] = corrected["left_ankle"]["x"] - 0.03
            corrected["right_knee"]["x"] = corrected["right_ankle"]["x"] + 0.03

        return corrected
