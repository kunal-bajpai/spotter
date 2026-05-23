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
    person_description: str = Field(description="An extremely detailed physical description of the person performing the squat in the original video. Visually inspect the video and describe the person's gender, approximate age, hair style and color, physical build/physique, and the precise details of their attire (e.g., style, colors, and features of their shirt, shorts or pants, shoes). Do NOT include any racial or ethnic details in this description. If no video file is attached or the video upload fails, default to a standard description of a professional athletic trainee.")
    veo_coaching_prompt: str = Field(description="A highly detailed and descriptive video generation prompt (like for Google Veo) to generate a short, 5-second video. It must describe a professional athletic coach in a well-lit gym performing a squat. The coach must exactly match the physical features, hair style, build, and clothing described in the `person_description` field (excluding any racial or ethnic details). The coach must NOT talk, open their mouth, or move their lips; they must remain completely silent and perform the squat with closed lips and a focused expression. It must describe the coach starting by demonstrating the primary biomechanical fault observed in the telemetry (e.g. leaning forward, caving knees, or a shallow depth), and then smoothly demonstrating the corrective action to achieve textbook perfect form. Make sure the prompt is extremely detailed, describing visual style, smooth transition, clothing, gym environment, lighting, and explicit biomechanics, avoiding any dynamic text references like 'reps' or 'telemetry'.")

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

    def generate_feedback(self, reps_telemetry: list[dict], video_path: str = None) -> dict:
        """
        Generates expert-level feedback. Connects to Gemini if API key is present;
        otherwise, runs a deterministic heuristic feedback generator.
        """
        if self.client:
            return self._generate_gemini_feedback(reps_telemetry, video_path=video_path)
        else:
            return self._generate_fallback_feedback(reps_telemetry)

    def _generate_gemini_feedback(self, reps_telemetry: list[dict], video_path: str = None) -> dict:
        """
        Calls Gemini using the google-genai SDK, passing telemetry and the reference video
        for multimodal actor identification and visual squat analysis.
        """
        logger.info("CoachAgent: Querying Gemini API for structured coaching feedback...")
        
        prompt = f"""
        You are a World-Class Strength & Conditioning Coach. Analyze the following physical telemetry logs 
        extracted from a user's squat repetition video.
        
        SQUAT TELEMETRY DATA:
        {reps_telemetry}
        
        BIOMECHANICAL REFERENCE STANDARDS:
        - Torso Lean: Torso lean is the forward tilt of the spine relative to vertical (0 degrees = perfectly vertical spine).
          * 0° to 15°: Extremely upright. Typical in front squats, goblet squats, or high-bar squats by lifters with short femurs.
          * 15° to 35°: Standard, safe forward torso lean for high-bar back squats.
          * 35° to 50°: Completely safe, expected, and physically necessary for low-bar back squats, or lifters with long femurs and short torsos. Do NOT flag this as a fault or say they leaned forward too much. This is proper, stable powerlifting form to keep the bar centered over the midfoot.
          * Above 50°: Heavy torso lean. It indicates a "Good Morning squat" trend where hips rise faster than shoulders. Do NOT label this as "DANGEROUS" simply because of the angle. A lean of up to 60 degrees is stable and standard in powerlifting if the spine remains flat and neutral. Only label as WARNING. Reserve "DANGEROUS" for extreme collapse (> 60 degrees) or if combined with severe knee valgus.
        - Squat Depth: Adequate depth is achieved when the hip joint sinks to the level of the knee joint or lower (peak_depth_gap >= -0.05). If the peak_depth_gap is less than -0.05, it is shallow.
        - Knee Valgus: A valgus count of 5 or fewer is safe. Only flag knee valgus (knee caving) if the valgus count exceeds 5.
        
        Your objective:
        1. Evaluate each repetition thoroughly.
        2. Analyze the physical joint telemetry trends organically using the Biomechanical Reference Standards to diagnose real movement faults.
        3. Provide exactly one high-impact, physiological coaching cue for each repetition (e.g. "push the ground away", "keep your eyes on the horizon", "imagine sitting back into a chair").
        4. Rate safety: DANGEROUS only if extreme spinal collapse (> 60 degrees) or severe knee caving occurred; WARNING if shallow or moderate/heavy deviations are present; SAFE if clean and within standard anatomical ranges.
        5. Generate a highly detailed and precise `person_description` by visually inspecting the attached video file to identify the user's gender, approximate age, clothing (e.g., color and style of t-shirt, shorts/pants), hair color/style, and physical features. Do NOT include any racial or ethnic details in this description.
        6. Generate a highly detailed and descriptive `veo_coaching_prompt` to guide a video generation model (like Google Veo) to create a 5-second video.
           - Incorporate the visual details from the `person_description` field organically into the prompt's description of the coach (e.g., 'A professional athletic male coach in his [age range] with [hair style], wearing a [description of clothes]...'), ensuring that the coach generated by Veo matches the user's physical appearance and attire in the original video for high-fidelity identity consistency (excluding any racial or ethnic details).
           - Describe the coach starting by demonstrating the primary biomechanical fault observed in the set (e.g., caving knees or excessive forward lean), and then showing a smooth transition as they correct their posture (e.g., driving their knees out or straightening their spine) to achieve perfect textbook squat form. The coach must NOT talk, open their mouth, or move their lips; they must perform the entire squat silently with closed lips and a focused athletic expression. Do not mention reps or JSON properties in the prompt; make it a pure visual scene description for a video model.
        
        CRITICAL TONE INSTRUCTION:
        Adopt an exceptionally motivational, positive, and encouraging coaching persona. Write all evaluations and cues in a supportive, empowering tone that builds the athlete's confidence, inspires them to improve, and drives them to execute better on their next set, while keeping their safety as the absolute highest priority.
        
        Provide your critique in the exact structured JSON schema requested.
        """

        uploaded_file = None
        contents = [prompt]
        
        try:
            if video_path and os.path.exists(video_path):
                try:
                    logger.info(f"CoachAgent: Uploading video to GenAI File Manager for multimodal analysis: {video_path}")
                    uploaded_file = self.client.files.upload(file=video_path)
                    logger.info(f"CoachAgent: Video uploaded successfully. Initial state: {uploaded_file.state}")
                    
                    # Wait for video file to transition to ACTIVE state
                    import time
                    wait_count = 0
                    while uploaded_file.state == "PROCESSING" and wait_count < 10:
                        logger.info("CoachAgent: Waiting for video processing... (polling 2s)")
                        time.sleep(2)
                        uploaded_file = self.client.files.get(name=uploaded_file.name)
                        wait_count += 1
                        
                    if uploaded_file.state == "ACTIVE":
                        contents.insert(0, uploaded_file)
                        logger.info("CoachAgent: Video is ACTIVE and attached to Gemini prompt.")
                    else:
                        logger.warning(f"CoachAgent: Video file is in state {uploaded_file.state}, proceeding with text-only prompt.")
                except Exception as upload_err:
                    logger.warning(f"CoachAgent: Could not upload video for visual description: {upload_err}")

            # Call modern Gemini SDK with structured Pydantic schema validation
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
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
            
        finally:
            if uploaded_file:
                try:
                    self.client.files.delete(name=uploaded_file.name)
                    logger.info("CoachAgent: Cloud temp file cleaned up successfully from File Manager.")
                except Exception as del_err:
                    logger.warning(f"CoachAgent: Could not delete cloud temp file: {del_err}")

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

        # Compile a fallback veo coaching prompt based on detected faults
        faults_detected = []
        for rep in reps_telemetry:
            if rep["faults"].get("shallow_depth") and "depth" not in faults_detected:
                faults_detected.append("depth")
            if rep["faults"].get("excessive_forward_lean") and "lean" not in faults_detected:
                faults_detected.append("lean")
            if rep["faults"].get("knee_valgus") and "valgus" not in faults_detected:
                faults_detected.append("valgus")

        fallback_person_desc = "A professional athletic trainee with a strong build, dark hair, wearing a sleek black performance shirt and athletic grey training shorts."

        fallback_veo_prompt = f"A high-fidelity silent video of a professional strength coach in a gym showing how to squat properly, performing the exercise silently with closed lips and a focused face without opening their mouth or talking. The coach matches the following description: {fallback_person_desc}. "
        if "lean" in faults_detected:
            fallback_veo_prompt += (
                "The coach starts at the bottom of the squat with a heavily forward-bent posture (bad form), "
                "and then smoothly demonstrates the corrective action by straightening their back, raising their chest tall and proud, "
                "and keeping their torso upright at a safe, natural 20-degree lean relative to vertical. A perfect visual transition showing bad form correcting to textbook form."
            )
        elif "valgus" in faults_detected:
            fallback_veo_prompt += (
                "The coach starts the ascent with their knees caving inward (bad knee valgus form), "
                "and then smoothly demonstrates the corrective action by driving their knees straight out over their toes "
                "to align perfectly over the ankles. A perfect visual transition showing bad form correcting to textbook form."
            )
        elif "depth" in faults_detected:
            fallback_veo_prompt += (
                "The coach starts the squat but stops shallow above parallel (bad form), "
                "and then smoothly demonstrates the corrective action by sinking their hips back and down to reach full deep parallel depth "
                "with perfect posture. A perfect visual transition showing bad form correcting to textbook form."
            )
        else:
            fallback_veo_prompt += (
                "The coach performs a textbook back squat with flawless form: controlled descent, "
                "sinking hips deep to parallel, keeping the chest tall, knees tracking perfectly over the toes, and a smooth ascent. "
                "Extremely educational, demonstrating perfect technique."
            )

        result = WorkoutFeedbackSchema(
            workout_summary=summary,
            perfect_reps_count=perfect_count,
            reps=reps_feedback,
            person_description=fallback_person_desc,
            veo_coaching_prompt=fallback_veo_prompt
        ).model_dump()

        logger.info("CoachAgent: Heuristic feedback generation complete.")
        return result

    def generate_audio_commentary(self, text: str, output_path: str = "veo_coaching_audio.wav") -> str:
        """
        Queries Gemini to perform text-to-speech conversion of the coaching summary
        and saves the audio bytes wrapped in a standard WAV header.
        """
        logger.info(f"CoachAgent: Generating native audio commentary for text: '{text}'")
        if not self.client:
            logger.warning("CoachAgent: No Gemini client available. Skipping audio commentary generation.")
            return None

        try:
            # We use gemini-2.0-flash as it supports high-fidelity audio generation natively
            response = self.client.models.generate_content(
                model="gemini-2.0-flash",
                contents=f"Please read the following athletic squat coaching feedback in an encouraging, professional, and clear coaching voice without saying anything else: {text}",
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Kore", # High-fidelity athletic tone
                            )
                        )
                    )
                )
            )

            # Extract the raw PCM bytes from response parts
            pcm_bytes = None
            for part in response.candidates[0].content.parts:
                if part.inline_data and part.inline_data.data:
                    pcm_bytes = part.inline_data.data
                    break

            if pcm_bytes:
                import wave
                # Gemini outputs 24kHz, 16-bit, mono little-endian PCM
                sample_rate = 24000
                sample_width = 2  # 16-bit = 2 bytes
                channels = 1  # mono

                with wave.open(output_path, 'wb') as wav_file:
                    wav_file.setnchannels(channels)
                    wav_file.setsampwidth(sample_width)
                    wav_file.setframerate(sample_rate)
                    wav_file.writeframes(pcm_bytes)

                logger.info(f"CoachAgent: Native coaching PCM audio successfully wrapped and saved as WAV to {output_path}")
                return output_path

            logger.warning("CoachAgent: No inline audio data found in response parts.")
            return None
        except Exception as e:
            logger.error(f"CoachAgent: Failed to generate audio commentary using Gemini: {e}")
            return None

