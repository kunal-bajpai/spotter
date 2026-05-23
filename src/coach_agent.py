import os
from pydantic import BaseModel, Field
from src.utils.logger import logger

# Import Google GenAI SDK if available
try:
    from google import genai
    from google.genai import types
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False
    logger.warning("google-genai SDK not available. Using local rule-based feedback engine fallback.")

# ----------------- Structured Output Schemas -----------------

class RepFeedbackSchema(BaseModel):
    rep_index: int = Field(description="The index of the repetition (1-based).")
    depth_evaluation: str = Field(description="Critique on parallel squat depth (adequate vs. shallow).")
    posture_evaluation: str = Field(description="Critique on torso tilt, spine alignment, and knee alignment.")
    coaching_cue: str = Field(description="A specific, actionable physiological coaching cue to correct the form, e.g. 'screw your feet into the floor'.")
    safety_rating: str = Field(description="One of: 'SAFE', 'WARNING', 'DANGEROUS'.")

class WorkoutFeedbackSchema(BaseModel):
    workout_summary: str = Field(description="Brief high-level summary of the overall set performance, tempo, and main focus area.")
    perfect_reps_count: int = Field(description="Count of perfectly executed reps with good depth and stability.")
    reps: list[RepFeedbackSchema] = Field(description="List of rep-by-rep analysis feedback details.")

# -------------------------------------------------------------

class CoachAgent:
    """
    Cognitive Coach Agent (Gemini Cognitive Core)
    Translates quantitative kinematic telemetry into professional, physiological,
    and encouraging coaching feedback using modern Gemini multimodal models.
    """
    def __init__(self, model_name: str = "gemini-3.5-flash"):

        self.model_name = model_name
        self.api_key = os.environ.get("GEMINI_API_KEY")
        self.client = None

        if GENAI_AVAILABLE and self.api_key:
            try:
                # Initialize modern Google GenAI Client
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"CoachAgent: Gemini Client successfully initialized with model {self.model_name}.")
            except Exception as e:
                logger.error(f"CoachAgent: Failed to initialize Gemini client: {e}")
        else:
            if not self.api_key:
                logger.warning("CoachAgent: GEMINI_API_KEY env variable is missing. "
                               "Coach will operate in local biomechanical fallback mode.")

    def generate_feedback(self, reps_telemetry: list[dict]) -> dict:
        """
        Generates expert-level feedback. Connects to Gemini if API key is present;
        otherwise, runs a deterministic heuristic feedback generator.
        """
        if self.client:
            return self._generate_gemini_feedback(reps_telemetry)
        else:
            return self._generate_fallback_feedback(reps_telemetry)

    def _generate_gemini_feedback(self, reps_telemetry: list[dict]) -> dict:
        """
        Calls Gemini using the google-genai SDK, passing telemetry and requesting structured output.
        """
        logger.info("CoachAgent: Querying Gemini API for structured coaching feedback...")
        
        prompt = f"""
        You are a World-Class Strength & Conditioning Coach. Analyze the following physical telemetry logs 
        extracted from a user's squat repetition video.
        
        SQUAT TELEMETRY DATA:
        {reps_telemetry}
        
        BIOMECHANICAL REFERENCE STANDARDS:
        - Torso Lean: In a standard back squat, some forward torso lean is physically necessary to maintain balance. A lean angle of up to 35 degrees is completely normal, safe, and expected. Do NOT flag torso lean as a fault or tell the lifter they leaned too much unless the max torso lean angle exceeds 40 degrees.
        - Squat Depth: Adequate depth is achieved when the hip joint sinks to the level of the knee joint or lower (peak_depth_gap >= -0.05). If the peak_depth_gap is less than -0.05, it is shallow.
        - Knee Valgus: A valgus count of 5 or fewer is safe. Only flag knee valgus (knee caving) if the valgus count exceeds 5.
        
        Your objective:
        1. Evaluate each repetition thoroughly.
        2. Analyze the physical joint telemetry trends organically using the Biomechanical Reference Standards to diagnose real movement faults.
        3. Provide exactly one high-impact, physiological coaching cue for each repetition (e.g. "push the ground away", "keep your eyes on the horizon", "imagine sitting back into a chair").
        4. Rate safety: DANGEROUS if extreme forward lean (> 40 degrees) or knee cave occurred, WARNING if shallow, SAFE if clean and within standard ranges.
        
        CRITICAL TONE INSTRUCTION:
        Adopt an exceptionally motivational, positive, and encouraging coaching persona. Write all evaluations and cues in a supportive, empowering tone that builds the athlete's confidence, inspires them to improve, and drives them to execute better on their next set, while keeping their safety as the absolute highest priority.
        
        Provide your critique in the exact structured JSON schema requested.
        """


        try:
            # Call modern Gemini SDK with structured Pydantic schema validation
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=WorkoutFeedbackSchema,
                    temperature=0.2
                )
            )
            # Parse the JSON response
            import json
            feedback_data = json.loads(response.text)
            logger.info("CoachAgent: Received validated structured response from Gemini.")
            return feedback_data
        except Exception as e:
            logger.error(f"CoachAgent: Gemini API call failed: {e}. Falling back to rule-based engine.")
            return self._generate_fallback_feedback(reps_telemetry)

    def _generate_fallback_feedback(self, reps_telemetry: list[dict]) -> dict:
        """
        Fallback biomechanical linguistic cue engine.
        Generates clean, professional coaching outputs purely from heuristic telemetry.
        """
        logger.info("CoachAgent: Generating heuristic feedback...")
        reps_feedback = []
        perfect_count = 0

        for rep in reps_telemetry:
            idx = rep["rep_index"]
            metrics = rep["metrics"]
            faults = rep["faults"]

            # Depth Evaluation
            if faults["shallow_depth"]:
                depth_eval = "Your squat was shallow. Hips did not reach the parallel line formed with your knees."
                cue_depth = "Imagine sinking your hips back and down until your thighs are parallel to the floor."
                safety = "WARNING"
            else:
                depth_eval = f"Excellent depth! Sunk past parallel (peak depth gap: {metrics['peak_depth_gap']:.2f})."
                cue_depth = ""
                safety = "SAFE"

            # Posture & Valgus Evaluation
            posture_issues = []
            cue_posture = ""
            if faults["excessive_forward_lean"]:
                posture_issues.append("You displayed excessive forward torso lean, shifting load onto your lower back.")
                cue_posture = "Keep your chest tall and proud, and pull your shoulders down into your back pockets."
                safety = "WARNING" if safety == "SAFE" else "DANGEROUS"
            
            if faults["knee_valgus"]:
                posture_issues.append("Significant knee valgus (cave-in) was detected as you ascended.")
                cue_posture = "Screw your feet into the floor and drive your knees outward over your pinky toes."
                safety = "DANGEROUS"

            if not posture_issues:
                posture_eval = "Torso remained stable and knees tracked perfectly aligned over your ankles."
                cue_posture = "Maintain this great vertical alignment and foot pressure."
            else:
                posture_eval = " ".join(posture_issues)

            # Synthesize coaching cue
            chosen_cue = cue_posture if cue_posture else (cue_depth if cue_depth else "Press the floor away evenly through your whole foot.")
            
            if safety == "SAFE":
                perfect_count += 1

            reps_feedback.append(RepFeedbackSchema(
                rep_index=idx,
                depth_evaluation=depth_eval,
                posture_evaluation=posture_eval,
                coaching_cue=chosen_cue,
                safety_rating=safety
            ).model_dump())

        # Overall summary
        if perfect_count == len(reps_telemetry) and len(reps_telemetry) > 0:
            summary = "Incredible work! Every single rep was performed with textbook kinematics. Perfect depth and rock-solid core stability."
        elif perfect_count > 0:
            summary = f"Good set! You completed {perfect_count} clean reps. However, watch out for fatigue on the other repetitions leading to knee caving or torso lean."
        else:
            summary = "We detected multiple biomechanical deviations. Prioritize keeping your torso upright and pushing your knees out to lift safely."

        result = WorkoutFeedbackSchema(
            workout_summary=summary,
            perfect_reps_count=perfect_count,
            reps=reps_feedback
        ).model_dump()

        logger.info("CoachAgent: Heuristic feedback generation complete.")
        return result
