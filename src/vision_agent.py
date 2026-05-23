import cv2
import os
import urllib.request
import numpy as np
from src.utils.logger import logger

# Import modern MediaPipe Tasks modules
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, PoseLandmark, RunningMode
import mediapipe as mp

class VisionAgent:
    """
    Vision Sensor Agent (MediaPipe Tasks Core)
    Responsible for extracting frame-by-frame 3D skeletal landmarks from a video file.
    Applies exponential smoothing to joint tracking to eliminate sensor jitter.
    Loads and configures the modern MediaPipe PoseLandmarker model lazily.
    """
    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.landmarker = None
        self.model_path = "pose_landmarker_full.task"
        logger.info("VisionAgent: Initialized with lazy loading configurations.")

    def _ensure_model_exists(self):
        """Downloads the standard pose landmark model if not present in the workspace."""
        if not os.path.exists(self.model_path):
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
            logger.info(f"VisionAgent: Downloading PoseLandmarker model weights from Google CDN...")
            try:
                urllib.request.urlretrieve(url, self.model_path)
                logger.info(f"VisionAgent: Successfully downloaded model weights to {self.model_path}")
            except Exception as e:
                logger.error(f"VisionAgent: Failed to download model weights: {e}")
                raise RuntimeError(f"Could not retrieve PoseLandmarker weights file: {e}")

    def _initialize_landmarker(self):
        """Instantiates the PoseLandmarker model dynamically."""
        if self.landmarker is not None:
            return

        self._ensure_model_exists()
        
        # Configure modern MediaPipe options
        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.model_path),
            running_mode=RunningMode.VIDEO,
            min_pose_detection_confidence=self.min_detection_confidence,
            min_pose_presence_confidence=self.min_tracking_confidence,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)
        logger.info("VisionAgent: Lazily initialized PoseLandmarker model.")

    def process_video(self, video_path: str, smooth_factor: float = 0.6) -> list[dict]:
        """
        Processes a raw video, extracts joint coordinate frames, and returns smoothed skeletal telemetry.
        """
        # Lazily initialize model only when processing starts
        self._initialize_landmarker()

        logger.info(f"VisionAgent: Starting analysis on: {video_path}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            raise ValueError(f"Could not open video file: {video_path}")

        # Extract video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0  # Fallback
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video parsed: {total_frames} frames at {fps} FPS (~{total_frames / fps:.2f} seconds)")

        frames_data = []
        frame_idx = 0

        # Memory buffer for exponential smoothing
        smoothed_landmarks = {}

        # Joint list we care about for Squat Kinematics
        tracked_indices = {
            "left_shoulder": PoseLandmark.LEFT_SHOULDER.value,
            "right_shoulder": PoseLandmark.RIGHT_SHOULDER.value,
            "left_hip": PoseLandmark.LEFT_HIP.value,
            "right_hip": PoseLandmark.RIGHT_HIP.value,
            "left_knee": PoseLandmark.LEFT_KNEE.value,
            "right_knee": PoseLandmark.RIGHT_KNEE.value,
            "left_ankle": PoseLandmark.LEFT_ANKLE.value,
            "right_ankle": PoseLandmark.RIGHT_ANKLE.value,
        }

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR (OpenCV) to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Wrap as MediaPipe Image
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            
            timestamp_sec = frame_idx / fps
            
            # Perform detection (frame timestamp must be in milliseconds)
            result = self.landmarker.detect_for_video(mp_image, int(timestamp_sec * 1000))

            frame_landmarks = {}

            # If a person is detected
            if result.pose_landmarks and len(result.pose_landmarks) > 0:
                landmarks = result.pose_landmarks[0]
                
                for name, idx in tracked_indices.items():
                    lm = landmarks[idx]
                    raw_coords = np.array([lm.x, lm.y, lm.z, lm.visibility])

                    # Apply exponential smoothing filter: S_t = alpha * S_{t-1} + (1 - alpha) * X_t
                    if name not in smoothed_landmarks:
                        smoothed_landmarks[name] = raw_coords
                    else:
                        smoothed_landmarks[name] = (smooth_factor * smoothed_landmarks[name]) + \
                                                   ((1.0 - smooth_factor) * raw_coords)

                    # Store smoothed coords
                    frame_landmarks[name] = {
                        "x": float(smoothed_landmarks[name][0]),
                        "y": float(smoothed_landmarks[name][1]),
                        "z": float(smoothed_landmarks[name][2]),
                        "visibility": float(smoothed_landmarks[name][3])
                    }

                frames_data.append({
                    "frame_id": frame_idx,
                    "timestamp_sec": timestamp_sec,
                    "landmarks": frame_landmarks
                })
            else:
                logger.warning(f"No landmarks detected at frame {frame_idx} (t={timestamp_sec:.2f}s)")
                if frames_data:
                    # Append previous frame's landmarks with updated timestamp to keep continuity
                    frames_data.append({
                        "frame_id": frame_idx,
                        "timestamp_sec": timestamp_sec,
                        "landmarks": frames_data[-1]["landmarks"]
                    })

            frame_idx += 1
            if frame_idx % 100 == 0:
                logger.info(f"Processed {frame_idx}/{total_frames} frames...")

        cap.release()
        logger.info(f"VisionAgent successfully finished parsing. Extracted {len(frames_data)} landmark frames.")
        return frames_data
