import argparse
import json
import os
import sys
from src.utils.logger import logger, setup_logger
from src.vision_agent import VisionAgent
from src.diagnostic_agent import DiagnosticAgent
from src.coach_agent import CoachAgent
from src.synthesis_agent import SynthesisAgent

class MasterOrchestrator:
    """
    Master Orchestrator Agent
    Controls the entire AI Squat Coach pipeline: Coordinates data flow between 
    Vision, Diagnostic, Coach, and Synthesis agents.
    """
    def __init__(self, gemini_model: str = "gemini-2.5-flash", api_key: str = None):

        logger.info("MasterOrchestrator: Initializing orchestrator pipeline...")
        self.vision_agent = VisionAgent()
        self.diagnostic_agent = DiagnosticAgent()
        self.coach_agent = CoachAgent(model_name=gemini_model, api_key=api_key)
        self.synthesis_agent = SynthesisAgent(api_key=api_key)
        logger.info("MasterOrchestrator: All agent components initialized successfully.")

    def run_coaching_flow(self, video_path: str, output_video_path: str = "outputs/output_corrected.mp4") -> dict:
        """
        Executes the full pipeline:
        Video -> MediaPipe Extraction -> Kinematic Rules -> Gemini Feedback -> Omni Visual Synthesis.
        """
        logger.info(f"MasterOrchestrator: Ingesting video: {video_path}")
        
        # Remove any stale veo coaching files from previous runs
        for stale_file in ["outputs/veo_coaching_demo.mp4"]:
            if os.path.exists(stale_file):
                try:
                    os.remove(stale_file)
                    logger.info(f"MasterOrchestrator: Stale {stale_file} deleted.")
                except Exception as e:
                    logger.warning(f"MasterOrchestrator: Could not delete stale file {stale_file}: {e}")
        
        if not os.path.exists(video_path):
            logger.error(f"MasterOrchestrator: Video file does not exist: {video_path}")
            raise FileNotFoundError(f"Source video file not found: {video_path}")
 
        # 1. Vision Sensor Agent: extract 3D landmarks
        logger.info("-------------------- STEP 1: Vision Extraction --------------------")
        raw_frames = self.vision_agent.process_video(video_path)
        
        if not raw_frames:
            logger.error("MasterOrchestrator: Vision Agent failed to extract any landmarks.")
            raise ValueError("No landmarks detected in the input video. Cannot proceed.")
 
        # 2. Kinematic Diagnostic Agent: segment reps & assess physical geometry
        logger.info("-------------------- STEP 2: Kinematic Diagnostics --------------------")
        reps_telemetry = self.diagnostic_agent.segment_reps(raw_frames)
        
        if not reps_telemetry:
            logger.warning("MasterOrchestrator: No squat repetitions detected by the diagnostic state machine.")
            # If no reps are segmented, we provide an empty set payload
            return {
                "success": False,
                "message": "No squat repetitions were detected. Please ensure your full body is visible in the frame.",
                "analysis": {
                    "workout_summary": "No reps detected. Check camera framing.",
                    "perfect_reps_count": 0,
                    "reps": []
                },
                "corrected_video_path": None
            }
 
        # 3. Cognitive Coach Agent: invoke Gemini for rep critiques with multimodal video understanding
        logger.info("-------------------- STEP 3: Cognitive Coaching --------------------")
        coaching_feedback = self.coach_agent.generate_feedback(reps_telemetry, video_path=video_path)

        # 4. Correction Synthesis Agent: render the side-by-side corrected form video and Veo demo video
        logger.info("-------------------- STEP 4: Visual Perfect-Form Synthesis --------------------")
        final_video_path = self.synthesis_agent.generate_ideal_video(
            video_path=video_path,
            raw_frames=raw_frames,
            reps_telemetry=reps_telemetry,
            coaching_feedback=coaching_feedback,
            output_path=output_video_path
        )

        logger.info("MasterOrchestrator: Coaching pipeline completed successfully.")
        
        return {
            "success": True,
            "message": "Coaching feedback and corrected visual synthesis generated.",
            "analysis": coaching_feedback,
            "corrected_video_path": final_video_path
        }

if __name__ == "__main__":
    # Configure argument parsing for CLI usage
    parser = argparse.ArgumentParser(description="AI Squat Coach - Multi-Agent Command Line Interface")
    parser.add_argument("--video", type=str, required=True, help="Path to input squat video")
    parser.add_argument("--output", type=str, default="outputs/output_corrected.mp4", help="Path for corrected output video")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash", help="Gemini model name")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging")

    args = parser.parse_args()

    # Configure logs level based on debug flag
    main_logger = setup_logger()
    if args.debug:
        main_logger.setLevel(10) # DEBUG

    try:
        orchestrator = MasterOrchestrator(gemini_model=args.model, api_key=os.environ.get("GEMINI_API_KEY"))
        result = orchestrator.run_coaching_flow(args.video, args.output)
        
        print("\n================== SQUAT COACH FEEDBACK ==================")
        print(f"Summary: {result['analysis']['workout_summary']}")
        print(f"Perfect Reps: {result['analysis']['perfect_reps_count']}\n")
        
        for rep in result['analysis']['reps']:
            print(f"Rep #{rep['rep_index']}:")
            print(f"  - Depth: {rep['depth_evaluation']}")
            print(f"  - Posture: {rep['posture_evaluation']}")
            print(f"  - Coach Cue: {rep['coaching_cue']}")
            print(f"  - Safety: {rep['safety_rating']}\n")
            
        print(f"Corrected side-by-side video saved to: {result['corrected_video_path']}")
        print("==========================================================\n")
        
    except Exception as ex:
        main_logger.exception(f"Orchestrator run failed with error: {ex}")
        sys.exit(1)
