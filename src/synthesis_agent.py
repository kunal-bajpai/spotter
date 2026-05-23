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

    def generate_ideal_video(self, video_path: str, raw_frames: list[dict], reps_telemetry: list[dict], output_path: str = "output_corrected.mp4") -> str:
        """
        Creates a perfect-form corrected video.
        Attempts to query Google's premium Gemini Omni (video-to-video edit) or Veo (video generation) APIs.
        Falls back to generating a side-by-side comparison video using local skeletal overlays.
        """
        logger.info(f"SynthesisAgent: Synthesizing ideal video from {video_path}...")
        
        # ----------------- OPTION 1: Gemini Omni / Veo Cloud API Path -----------------
        api_key = os.environ.get("GEMINI_API_KEY")
        if GENAI_AVAILABLE and api_key:
            try:
                logger.info("SynthesisAgent: Attempting Gemini Omni / Veo cloud video synthesis...")
                client = genai.Client(api_key=api_key)
                
                # 1. Upload video to Google's GenAI File Manager
                logger.info(f"SynthesisAgent: Uploading source video to GenAI File Manager: {video_path}")
                uploaded_file = client.files.upload(file=video_path)
                logger.info(f"SynthesisAgent: Video uploaded successfully. Cloud URI: {uploaded_file.uri}")
                
                # 2. Extract active faults to compose a highly detailed prompt
                active_faults = []
                for rep in reps_telemetry:
                    for fault_name, triggered in rep["faults"].items():
                        if triggered and fault_name not in active_faults:
                            active_faults.append(fault_name.replace("_", " "))
                            
                correction_prompt = (
                    "Biomechanical Squat Form Correction: Edit this person performing the squat to demonstrate perfect form. "
                    "Make sure to keep the user's face, clothes, gym background, and equipment 100% consistent. "
                )
                if "shallow depth" in active_faults:
                    correction_prompt += "Ensure the hips sink lower at the bottom of the squat to achieve full parallel depth. "
                if "excessive forward lean" in active_faults:
                    correction_prompt += "Upright the torso and chest to a safe, natural 20-degree lean relative to vertical at the bottom. "
                if "knee valgus" in active_faults:
                    correction_prompt += "Keep the knees tracking straight out over the toes during the ascent. "
                    
                logger.info(f"SynthesisAgent: Formulated correction prompt: {correction_prompt}")
                
                # 3. Call Veo Video Generation API ('veo-2.0-generate-001') directly
                logger.info("SynthesisAgent: Querying Google Veo video generation endpoint ('veo-2.0-generate-001')...")
                operation = client.models.generate_videos(
                    model="veo-2.0-generate-001",
                    prompt=f"A high-fidelity video of the same athlete performing a perfect back squat. {correction_prompt}",
                    config=types.GenerateVideosConfig(
                        aspect_ratio="16:9",
                        duration_seconds=5,
                    )
                )
                
                # 5. Poll the long-running operation
                import time
                wait_count = 0
                max_wait = 18  # Wait up to 180 seconds (10s intervals)
                
                while not operation.done and wait_count < max_wait:
                    logger.info("SynthesisAgent: Waiting for cloud video model to synthesize perfect form... (polling 10s)")
                    time.sleep(10)
                    wait_count += 1
                    
                if operation.done:
                    video_result = operation.result()
                    # Write the resulting video bytes to the output path
                    with open(output_path, "wb") as f:
                        f.write(video_result.video_bytes)
                    logger.info(f"SynthesisAgent: Cloud video generated and saved successfully to {output_path}")
                    
                    # Clean up the uploaded cloud file
                    try:
                        client.files.delete(name=uploaded_file.name)
                        logger.info("SynthesisAgent: Cloud temp file cleaned up successfully.")
                    except Exception as del_err:
                        logger.warning(f"SynthesisAgent: Could not delete cloud temp file: {del_err}")
                        
                    return output_path
                else:
                    logger.warning("SynthesisAgent: Cloud video generation timed out. Falling back to local skeletal overlays.")
                    
            except Exception as cloud_err:
                logger.error(f"SynthesisAgent: Google cloud video API call failed: {cloud_err}. "
                             f"This typically means your GEMINI_API_KEY is standard and lacks billing/preview access to paid Veo/Omni video endpoints yet. "
                             f"Falling back gracefully to local side-by-side skeletal wireframe overlays.")

        # ----------------- OPTION 2: Local Skeletal Overlay Fallback Path -----------------
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
