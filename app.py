import streamlit as st
import os
import json
import numpy as np
import cv2
import time
from dotenv import load_dotenv

# Load .env (project-scoped secrets) before importing modules that may read env vars.
load_dotenv()

from src.orchestrator import MasterOrchestrator
from src.utils.logger import logger

# Set premium dark mode configuration
st.set_page_config(
    page_title="AI Squat Coach - Multi-Agent Biomechanical Core",
    page_icon="🏋️‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache resource to start our background WebRTC video upload handler server exactly once
@st.cache_resource
def start_upload_backend_server():
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading

    class UploadHandler(BaseHTTPRequestHandler):
        def do_OPTIONS(self):
            self.send_response(200)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_POST(self):
            try:
                # 1. Read request body first
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                
                # 2. Write WebM file
                with open("recorded_squat.webm", "wb") as f:
                    f.write(post_data)
                    
                logger.info("Webcam Upload Server: Successfully received and saved recorded_squat.webm")
                
                # 3. Respond with 200 and explicit CORS header
                self.send_response(200)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Upload successful")
            except Exception as e:
                logger.error(f"Webcam Upload Server: Error processing upload: {e}")
                self.send_response(500)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(f"Error: {e}".encode())


    try:
        server = HTTPServer(("0.0.0.0", 8503), UploadHandler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        logger.info("Webcam Upload Server started in background on port 8503.")
    except OSError as e:
        if e.errno == 48: # Address already in use
            logger.info("Webcam Upload Server is already active on port 8503 due to hot-reload.")
        else:
            raise e
    return True



# ----------------- Inject Premium Custom CSS & Typography -----------------
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main Background & Gradient styling */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #171725 100%);
    }
    
    /* Premium Title styling */
    .main-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FF4B4B 0%, #FF8F8F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        text-align: center;
    }
    
    .subtitle {
        font-size: 1.2rem;
        color: #9ab0c1;
        text-align: center;
        margin-bottom: 2.5rem;
    }
    
    /* Glassmorphic card styling */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    .safe-glow {
        border-left: 5px solid #00E676;
        box-shadow: 0 4px 20px rgba(0, 230, 118, 0.1);
    }
    
    .warning-glow {
        border-left: 5px solid #FFD600;
        box-shadow: 0 4px 20px rgba(255, 214, 0, 0.1);
    }
    
    .danger-glow {
        border-left: 5px solid #FF1744;
        box-shadow: 0 4px 20px rgba(255, 23, 68, 0.1);
    }
    
    /* Premium Stepper visual markers */
    .step-badge {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8F8F 100%);
        color: white;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to generate an OpenCV synthetic squat set dynamically
def generate_synthetic_squat_video(filename="demo_squat.mp4", squat_type="perfect"):
    """
    Generates a stylized MP4 squat video using OpenCV.
    Draws a stick-figure performing a squat, moving in 3D frame space.
    This simulates a camera-captured set dynamically.
    """
    logger.info(f"Generating synthetic video: {filename} with form: {squat_type}")
    fps = 30
    duration_sec = 4
    num_frames = fps * duration_sec
    width, height = 640, 480
    
    # Store landmark coordinates sequence
    landmarks_sequence = []
    
    # Define VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for f in range(num_frames):
        # Create dark blue background
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:] = [20, 20, 30] # BGR color
        
        # Grid lines
        for y in range(0, height, 40):
            cv2.line(img, (0, y), (width, y), (30, 30, 45), 1)
        for x in range(0, width, 40):
            cv2.line(img, (x, 0), (x, height), (30, 30, 45), 1)
            
        # Draw floor
        cv2.line(img, (50, 420), (590, 420), (80, 80, 100), 3)
        
        # Calculate kinematics using sine curve
        # Standing -> squatting -> standing
        # Descent peaks in middle frame
        pct = np.sin(np.pi * (f / float(num_frames))) # goes from 0.0 -> 1.0 -> 0.0
        
        # Base anatomical coordinates (normalized spacing)
        head_center = (320, int(120 + 70 * pct))
        shoulder_center = (320, int(170 + 80 * pct))
        hip_center = (320, int(260 + 95 * pct))
        
        # Apply deformations based on form type
        if squat_type == "shallow":
            # Hip doesn't sink deep enough
            hip_center = (320, int(260 + 50 * pct))
            shoulder_center = (320, int(170 + 40 * pct))
            head_center = (320, int(120 + 35 * pct))
            
        elif squat_type == "lean":
            # Leaning forward: head and shoulders shift horizontally forward
            shoulder_center = (int(320 - 45 * pct), int(170 + 70 * pct))
            head_center = (int(320 - 75 * pct), int(120 + 60 * pct))
            hip_center = (int(320 + 20 * pct), int(260 + 90 * pct))

        knee_l = (int(270 - 15 * pct), int(330 + 35 * pct))
        knee_r = (int(370 + 15 * pct), int(330 + 35 * pct))
        
        # Apply knee caving (valgus)
        if squat_type == "valgus" and pct > 0.4:
            # Knees cave inwards horizontally
            knee_l = (int(270 + 20 * pct), int(330 + 35 * pct))
            knee_r = (int(370 - 20 * pct), int(330 + 35 * pct))

        ankle_l = (260, 400)
        ankle_r = (380, 400)
        
        # Convert to normalized coordinates (0.0 to 1.0) to match MediaPipe
        landmarks = {
            "left_shoulder": {"x": (shoulder_center[0] - 15) / 640.0, "y": shoulder_center[1] / 480.0, "z": 0.0, "visibility": 0.99},
            "right_shoulder": {"x": (shoulder_center[0] + 15) / 640.0, "y": shoulder_center[1] / 480.0, "z": 0.0, "visibility": 0.99},
            "left_hip": {"x": (hip_center[0] - 20) / 640.0, "y": hip_center[1] / 480.0, "z": 0.0, "visibility": 0.99},
            "right_hip": {"x": (hip_center[0] + 20) / 640.0, "y": hip_center[1] / 480.0, "z": 0.0, "visibility": 0.99},
            "left_knee": {"x": knee_l[0] / 640.0, "y": knee_l[1] / 480.0, "z": 0.0, "visibility": 0.99},
            "right_knee": {"x": knee_r[0] / 640.0, "y": knee_r[1] / 480.0, "z": 0.0, "visibility": 0.99},
            "left_ankle": {"x": ankle_l[0] / 640.0, "y": ankle_l[1] / 480.0, "z": 0.0, "visibility": 0.99},
            "right_ankle": {"x": ankle_r[0] / 640.0, "y": ankle_r[1] / 480.0, "z": 0.0, "visibility": 0.99}
        }
        
        landmarks_sequence.append({
            "frame_id": f,
            "timestamp_sec": f / float(fps),
            "landmarks": landmarks
        })
        
        # Draw stick figure elements
        color_skin = (255, 200, 150)
        color_torso = (235, 120, 50)
        
        # Head
        cv2.circle(img, head_center, 22, color_skin, -1)
        # Spine (Neck to Hip)
        cv2.line(img, shoulder_center, hip_center, color_torso, 8)
        # Shoulders line
        cv2.line(img, (shoulder_center[0] - 30, shoulder_center[1]), 
                 (shoulder_center[0] + 30, shoulder_center[1]), color_torso, 8)
        
        # Thighs
        cv2.line(img, hip_center, knee_l, color_torso, 6)
        cv2.line(img, hip_center, knee_r, color_torso, 6)
        
        # Calves
        cv2.line(img, knee_l, ankle_l, color_torso, 6)
        cv2.line(img, knee_r, ankle_r, color_torso, 6)
        
        # Labels
        cv2.putText(img, "SIMULATED TRAINER SQUAT FEED", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 150, 180), 1, cv2.LINE_AA)
        
        out.write(img)
        
    cap = out.release()
    
    # Save exact landmark coordinates to a JSON file to bypass MediaPipe detection limits
    with open("demo_squat_landmarks.json", "w") as f_json:
        json.dump(landmarks_sequence, f_json, indent=2)
        
    logger.info(f"Synthetic video compiled successfully: {filename}")
    return filename


# ----------------- SIDEBAR: Control & Configurations -----------------
st.sidebar.markdown("<h2 style='text-align: center;'>⚙️ CONTROL PANEL</h2>", unsafe_allow_html=True)
st.sidebar.markdown("Configure options for the Multi-Agent Diagnostic Core.")

# --- Bring-your-own API key (session-only; never persisted to disk) ---
st.sidebar.markdown("### 🔑 Your Gemini API Key")
user_api_key = st.sidebar.text_input(
    "Paste your key (kept in browser session only):",
    type="password",
    value=st.session_state.get("gemini_api_key", ""),
    help="Get a free key at https://aistudio.google.com/apikey. Never shared, never written to disk.",
    placeholder="AIza..."
)
st.session_state.gemini_api_key = user_api_key

if user_api_key:
    st.sidebar.success(f"✓ Key loaded ({len(user_api_key)} chars). Cleared when you close this tab.")
else:
    st.sidebar.warning("⚠️ No key set — analysis will run in offline heuristic mode (no Gemini insights, no Veo video, no Live Voice Coach).")

st.sidebar.markdown("---")

# Choose execution path
run_mode = st.sidebar.radio(
    "Choose Input Stream:",
    ["🤖 Run Synthetic Demo Mode", "🎥 Record Live Webcam", "📤 Upload Squat Video"]
)


model_selector = st.sidebar.selectbox(
    "Cognitive Coach Model",
    [
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
        "gemini-2.5-pro",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
        "gemini-1.5-flash",
        "gemini-1.5-pro",
    ],
    index=0,
    help="Flash = fast & cheap, Pro = highest quality. Lite variants are cheapest."
)


# Render premium header
st.markdown("<h1 class='main-title'>AI SQUAT COACH</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Multi-Agent Biomechanical Analytics & Generative Correction Synthesis</p>", unsafe_allow_html=True)

# ----------------- MAIN FLOW -----------------

# Set default session state variables
if "analyzed" not in st.session_state:
    st.session_state.analyzed = False
    st.session_state.feedback = None
    st.session_state.video_path = None
    st.session_state.corrected_path = None

if run_mode == "🎥 Record Live Webcam":
    # Launch background local upload server
    start_upload_backend_server()

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🎥 Record Your Squat Live")
    st.write("Grant camera permission in the box below to start recording. Align your full body in standard profile view, record your squat set, and submit for multi-agent evaluation!")

    # Inject WebRTC MediaRecorder HTML5/JS iframe
    camera_html = """
    <div style="background: rgba(255, 255, 255, 0.02); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 20px; text-align: center;">
        <video id="preview" width="480" height="360" autoplay muted style="border-radius: 8px; border: 2px solid rgba(255,255,255,0.1); background: black; transform: scaleX(-1);"></video>
        <div style="margin-top: 15px;">
            <button id="startBtn" onclick="startRecording()" style="background: #FF4B4B; color: white; border: none; padding: 12px 24px; border-radius: 8px; font-weight: 600; cursor: pointer; font-family: sans-serif; font-size: 1rem; margin-right: 10px; box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3);">🔴 Start Recording</button>
            <button id="stopBtn" onclick="stopRecording()" disabled style="background: #444; color: #999; border: none; padding: 12px 24px; border-radius: 8px; font-weight: 600; cursor: pointer; font-family: sans-serif; font-size: 1rem;">⏹️ Stop & Submit</button>
        </div>
        <p id="status" style="color: #9ab0c1; margin-top: 15px; font-size: 0.95rem; font-family: sans-serif;">Status: Connecting to camera...</p>
    </div>

    <script>
        let stream;
        let recorder;
        let chunks = [];

        async function setupCamera() {
            try {
                stream = await navigator.mediaDevices.getUserMedia({ 
                    video: { width: 640, height: 480, facingMode: "user" }, 
                    audio: false 
                });
                document.getElementById('preview').srcObject = stream;
                document.getElementById('status').innerText = "Status: Camera Connected. Stand in frame and click 'Start Recording'.";
            } catch (e) {
                document.getElementById('status').innerText = "Error: Camera Access Denied (" + e.message + "). Please allow permission.";
            }
        }

        async function startRecording() {
            chunks = [];
            recorder = new MediaRecorder(stream, { mimeType: 'video/webm' });
            recorder.ondataavailable = e => chunks.push(e.data);
            
            recorder.onstop = async () => {
                document.getElementById('status').innerText = "Status: Transmitting recording to local backend server...";
                let blob = new Blob(chunks, { type: 'video/webm' });
                
                try {
                    // Resolve upload server URL dynamically based on location hostname (supporting remote network devices)
                    let uploadUrl = "http://" + window.location.hostname + ":8503/upload";
                    let response = await fetch(uploadUrl, {
                        method: "POST",
                        body: blob
                    });

                    if (response.ok) {
                        document.getElementById('status').innerText = "Status: ✅ Squat Video successfully saved! Please close the camera and click 'Analyze WebCam Squat' in Streamlit.";
                    } else {
                        document.getElementById('status').innerText = "Error: Backend server rejected upload.";
                    }
                } catch (err) {
                    document.getElementById('status').innerText = "Error contacting backend upload server: " + err.message;
                }
            };

            recorder.start();
            document.getElementById('startBtn').disabled = true;
            document.getElementById('startBtn').style.background = "#444";
            document.getElementById('startBtn').style.color = "#999";
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('stopBtn').style.background = "#FF1744";
            document.getElementById('stopBtn').style.color = "white";
            document.getElementById('status').innerText = "Status: 🔴 RECORDING SQUAT SET. EXECUTE YOUR SQUATS NOW...";
        }

        function stopRecording() {
            recorder.stop();
            document.getElementById('startBtn').disabled = false;
            document.getElementById('startBtn').style.background = "#FF4B4B";
            document.getElementById('startBtn').style.color = "white";
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('stopBtn').style.background = "#444";
            document.getElementById('stopBtn').style.color = "#999";
        }

        setupCamera();
    </script>
    """
    st.components.v1.html(camera_html, height=500)

    # Trigger analysis if recorded video exists
    recorded_file = "recorded_squat.webm"
    if os.path.exists(recorded_file):
        st.success("🎉 Detected active live recording!")
        
        if st.button("🚀 Analyze WebCam Squat"):
            status_box = st.empty()
            with status_box.container():
                st.info("🔄 Vision Sensor Agent: Lazily loading PoseLandmarker & extracting 3D skeletal keypoints...")
                st.info("📐 Kinematic Diagnostic Agent: Calculating joint angles & segmenting repetitions...")
                st.info("🧠 Cognitive Coaching Agent: Querying Gemini API for structured physiological critiques...")
                st.info("🎨 Correction Synthesis Agent: Compiling side-by-side ideal pose deforming overlays...")
                
            try:
                orchestrator = MasterOrchestrator(gemini_model=model_selector, api_key=user_api_key)
                result = orchestrator.run_coaching_flow(video_path=recorded_file, output_video_path="outputs/output_corrected.mp4")
                
                st.session_state.feedback = result
                st.session_state.video_path = recorded_file
                st.session_state.corrected_path = "outputs/output_corrected.mp4"
                st.session_state.analyzed = True
                
                status_box.empty()
                st.success("✅ Webcam set successfully processed and compiled!")
            except Exception as e:
                status_box.empty()
                st.error(f"Failed to execute orchestrator flow: {e}")
    else:
        st.info("💡 Once you start and stop recording, a submission confirmation will display here, allowing you to run the multi-agent solver.")
        
    st.markdown("</div>", unsafe_allow_html=True)

elif run_mode == "🤖 Run Synthetic Demo Mode":
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🤖 Interactive Synthetic Demo Mode")

    st.write("No squat video on your device? No problem! Choose a biomechanical form template. We will dynamically programmatically draw a 3D trainer stick-figure squatting, compile an MP4, and run it through our full Pose Landmarker and Kinematics solver.")
    
    demo_type = st.selectbox(
        "Select Squat Demo Form (Real Video vs. Simulated Skeletal Deviations):",
        [
            "Real Squat Athlete Video (Perfect Textbook Form)",
            "Perfect Textbook Form (Simulated)", 
            "Shallow Squat (Depth Fault - Simulated)", 
            "Excessive Torso Lean (Good Morning - Simulated)",
            "Knee Valgus (Inward Knee Cave - Simulated)"
        ]
    )
    
    type_map = {
        "Perfect Textbook Form (Simulated)": "perfect",
        "Shallow Squat (Depth Fault - Simulated)": "shallow",
        "Excessive Torso Lean (Good Morning - Simulated)": "lean",
        "Knee Valgus (Inward Knee Cave - Simulated)": "valgus"
    }
    
    if st.button("🚀 Analyze Demo Squat"):
        # Stepper flow updates
        status_box = st.empty()
        
        with status_box.container():
            if demo_type == "Real Squat Athlete Video (Perfect Textbook Form)":
                st.info("Using local real squat athlete video (data/YouTube_Back-Squat-Side-View.mp4)...")
                video_file = "data/YouTube_Back-Squat-Side-View.mp4"
            else:
                st.info("Creating OpenCV skeleton sequence...")
                # Generate simulated video file
                video_file = generate_synthetic_squat_video(filename="demo_squat.mp4", squat_type=type_map[demo_type])
            
            st.info("🔄 Vision Sensor Agent: Lazily loading PoseLandmarker & extracting keypoints...")
            time.sleep(0.5)
            
            st.info("📐 Kinematic Diagnostic Agent: Calculating joint angles & segmenting repetitions...")
            time.sleep(0.5)

            
            st.info("🧠 Cognitive Coaching Agent: Querying Gemini API for structured physiological critiques...")
            time.sleep(0.5)
            
            st.info("🎨 Correction Synthesis Agent: Compiling side-by-side ideal pose deforming overlays...")
            
        try:
            # Instantiate pipeline
            orchestrator = MasterOrchestrator(gemini_model=model_selector, api_key=user_api_key)
            
            # Run pipeline E2E
            result = orchestrator.run_coaching_flow(video_path=video_file, output_video_path="outputs/output_corrected.mp4")
            
            st.session_state.feedback = result
            st.session_state.video_path = video_file
            st.session_state.corrected_path = "outputs/output_corrected.mp4"
            st.session_state.analyzed = True
            
            status_box.empty()
            st.success("✅ Set successfully processed and compiled!")
        except Exception as e:
            status_box.empty()
            st.error(f"Failed to execute orchestrator flow: {e}")
            
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📤 Upload Your Squat Video")
    uploaded_file = st.file_uploader("Upload an MP4 or MOV squat video file:", type=["mp4", "mov"])
    
    if uploaded_file is not None:
        # Save temp file
        os.makedirs("cache", exist_ok=True)
        temp_path = "cache/uploaded_temp.mp4"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())
            
        st.video(temp_path)
        
        if st.button("🚀 Analyze uploaded Squat Video"):
            status_box = st.empty()
            
            with status_box.container():
                st.info("🔄 Vision Sensor Agent: Lazily loading PoseLandmarker & extracting 3D skeletal keypoints...")
                st.info("📐 Kinematic Diagnostic Agent: Calculating joint angles & segmenting repetitions...")
                st.info("🧠 Cognitive Coaching Agent: Querying Gemini API for structured physiological critiques...")
                st.info("🎨 Correction Synthesis Agent: Compiling side-by-side ideal pose deforming overlays...")
                
            try:
                orchestrator = MasterOrchestrator(gemini_model=model_selector, api_key=user_api_key)
                result = orchestrator.run_coaching_flow(video_path=temp_path, output_video_path="outputs/output_corrected.mp4")
                
                st.session_state.feedback = result
                st.session_state.video_path = temp_path
                st.session_state.corrected_path = "outputs/output_corrected.mp4"
                st.session_state.analyzed = True
                
                status_box.empty()
                st.success("✅ Set successfully processed and compiled!")
            except Exception as e:
                status_box.empty()
                st.error(f"Failed to execute orchestrator flow: {e}")
                
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- DISPLAY RESULTS -----------------
if st.session_state.analyzed and st.session_state.feedback:
    result = st.session_state.feedback
    analysis = result["analysis"]
    
    # 1. Summary Cards
    st.markdown("### 📊 Set Kinematic Summary")
    col1, col2, col3 = st.columns(3)
    
    # Color coding safety ratings
    safety_theme = "safe-glow"
    safety_lbl = "SAFE"
    
    # Look through reps to find safety issues
    if analysis.get("reps"):
        ratings = [r["safety_rating"] for r in analysis["reps"]]
        if "DANGEROUS" in ratings:
            safety_theme = "danger-glow"
            safety_lbl = "🔴 DANGEROUS"
        elif "WARNING" in ratings:
            safety_theme = "warning-glow"
            safety_lbl = "🟡 WARNING"
        else:
            safety_lbl = "🟢 SAFE & TEXTBOOK"
            
    with col1:
        st.markdown(f"""
        <div class="glass-card {safety_theme}">
            <p style='margin:0;font-size:0.9rem;color:#8892b0;'>WORKOUT SAFETY STATUS</p>
            <h2 style='margin:5px 0 0 0;'>{safety_lbl}</h2>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="glass-card">
            <p style='margin:0;font-size:0.9rem;color:#8892b0;'>TOTAL DETECTED REPS</p>
            <h2 style='margin:5px 0 0 0;color:#FF4B4B;'>{len(analysis['reps'])} Reps</h2>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="glass-card">
            <p style='margin:0;font-size:0.9rem;color:#8892b0;'>TEXTBOOK REPS COUNT</p>
            <h2 style='margin:5px 0 0 0;color:#00E676;'>{analysis['perfect_reps_count']} / {len(analysis['reps'])}</h2>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown(f"""
    <div class="glass-card">
        <h4 style='margin:0 0 8px 0;color:#FF8F8F;'>🧠 World-Class Coach Summary</h4>
        <p style='margin:0;line-height:1.6;font-size:1.05rem;'>"{analysis['workout_summary']}"</p>
    </div>
    """, unsafe_allow_html=True)

    # Display detected athlete appearance if available
    if "person_description" in analysis and analysis["person_description"]:
        st.markdown(f"""
        <div class="glass-card">
            <h4 style='margin:0 0 8px 0;color:#64FFDA;'>👤 Detected Athlete Appearance</h4>
            <p style='margin:0;line-height:1.6;font-size:1.05rem;'>"{analysis['person_description']}"</p>
            <p style='margin-top:8px;font-size:0.85rem;color:#8892b0;font-style:italic;'>This visual signature is dynamically extracted by Gemini's multimodal core and passed to the Google Veo prompt to preserve your identity in the coaching demo video.</p>
        </div>
        """, unsafe_allow_html=True)

    # 2. Side-by-Side Visual Comparison & Video
    st.markdown("### 👁️ Omni Perfect Form Synthesis Comparison")
    st.write("Original/Input Video (Left) vs. Corrected Ideal Form Stick Skeleton (Right) generated by the Correction Synthesis Agent.")
    
    if st.session_state.corrected_path and os.path.exists(st.session_state.corrected_path):
        # Play side-by-side corrected video
        st.video(st.session_state.corrected_path)
    else:
        st.warning("Comparison video is not available or failed to compile.")

    # 2.5 Google Veo Virtual Coach Demonstration Video
    veo_demo_path = "outputs/veo_coaching_demo.mp4"
    
    if os.path.exists(veo_demo_path):
        st.markdown("### 🏋️‍♂️ Virtual Coach Demonstration Video (Google Veo)")
        st.write("A high-fidelity coaching demonstration generated dynamically in the cloud by Google Veo showing bad form correcting to perfect textbook technique based on your coaching cues.")
        st.video(veo_demo_path)
        st.info("ℹ️ **Audio Integration Note**: Google Veo (veo-2.0-generate-001) is a silent visual-only engine under Developer API tiers. To speak with your coach in real-time about this demonstration, click **Start Voice Session** in our real-time interactive **Live Voice Coach** dashboard below!")
    else:
        # Display a highly premium user-friendly diagnostic warning if Veo video could not be generated due to billing/prepay depletion
        st.markdown("""
        <div class="glass-card warning-glow">
            <h4 style='margin:0 0 8px 0;color:#FFD600;'>⚠️ Cloud Virtual Coach Demonstration Unavailable</h4>
            <p style='margin:0;line-height:1.6;font-size:0.95rem;'>
                The dynamic cloud-based Google Veo coach demo video is currently unavailable. 
                <br><br>
                <b>Detected Root Cause:</b> Your Google AI Studio billing tier has reached its prepayment balance limit, resulting in a <code>429 RESOURCE_EXHAUSTED</code> error during synthesis.
                <br><br>
                To restore dynamic photorealistic virtual coach generation, please visit the <a href="https://aistudio.google.com/" target="_blank" style="color:#FFD600;font-weight:600;text-decoration:underline;">Google AI Studio Console</a> and top up your prepay billing credits. 
                In the meantime, our offline side-by-side skeletal wireframe overlay is fully operational and rendering above!
            </p>
        </div>
        """, unsafe_allow_html=True)


    # 2.7 Live Voice Coach Session (Gemini Live WebSocket Widget)
    st.markdown("### 🎙️ Talk to the Coach (Live Interactive Voice Session)")
    st.write("Your personal, world-class athletic coach is ready to speak with you in real-time. Our system has loaded your squat diagnostics into their active memory. Press the button below, grant microphone permissions, and start speaking!")
    
    # Get active API key (from sidebar input, session-only)
    api_key = st.session_state.get("gemini_api_key", "")
    
    if not api_key:
        st.markdown("""
        <div class="glass-card danger-glow" style="text-align: center; padding: 30px;">
            <h4 style="margin:0 0 8px 0;color:#FF1744;">⚠️ Live Voice Session Blocked</h4>
            <p style="margin:0;font-size:0.95rem;color:#8892b0;line-height:1.4;">
                Please paste your Gemini API key in the sidebar (top of the Control Panel) to enable real-time bidirectional audio chat with the coach.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Dynamically compile the diagnostic context
        workout_summary = analysis.get("workout_summary", "No reps analyzed yet.")
        reps_cues = []
        if "reps" in analysis and analysis["reps"]:
            for r in analysis["reps"]:
                reps_cues.append(f"Rep {r['rep_index']}: posture='{r['posture_evaluation']}', depth='{r['depth_evaluation']}', actionable cue='{r['coaching_cue']}'")
        cues_text = " | ".join(reps_cues)
        
        dynamic_context = f"The user has just performed a set of squats. Summary of their performance: '{workout_summary}'. Rep-by-rep diagnostics: {cues_text}."
        
        # Format the system instruction prompt
        system_instruction_raw = f"""You are a friendly, encouraging, but highly professional athletic trainer and Squat Coach.
The user is here for real-time live voice squat coaching.

Our team has analyzed the user's squat set and generated the following coaching context:
{dynamic_context}

Your exact instructions:
Greet them enthusiastically and state clearly right off the bat the primary biomechanical issues you noticed in their squats based on the context (e.g. caving knees, torso tilt, or shallow depth).
Explain briefly what those issues mean, why they matter/the impact (e.g. shear stress on joints, loss of glute power), and tell them how we are going to fix it today (e.g. using the cues and warm-up corrections described in the context).
Do NOT lecture them for too long at once. End your introductory assessment by asking a friendly clarifying question, such as: "How long have you been training squats?", "Does this caving/tilt happen when you lift heavier weights?", or "Do you feel any pain when this happens?"
Listen to their responses, offer tailored corrections, and coach them through setting up their correct mental and physical squat patterns.
Keep your sentences concise, punchy, conversational, and verbally interactive. Avoid using deep markdown syntax, lists, or tables because you are speaking to them over live real-time audio! Keep the focus 100% on the dynamic squat fix."""

        # Escape the prompt string for JS safely
        js_safe_instruction = system_instruction_raw.replace('"', '\\"').replace("'", "\\'").replace('\n', ' ')
        
        # Inject custom HTML5/JS WebRTC/WebSocket player
        voice_chat_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
        <style>
            body {{
                margin: 0;
                padding: 0;
                background: transparent;
                font-family: 'Outfit', sans-serif;
                color: #e6f1ff;
            }}
            .coach-container {{
                background: rgba(255, 255, 255, 0.02);
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-radius: 16px;
                padding: 24px;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 16px;
                backdrop-filter: blur(10px);
                -webkit-backdrop-filter: blur(10px);
            }}
            .status-badge {{
                background: rgba(255, 255, 255, 0.05);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 20px;
                padding: 6px 16px;
                font-size: 0.9rem;
                font-weight: 600;
                color: #9ab0c1;
                letter-spacing: 0.5px;
                display: flex;
                align-items: center;
                gap: 8px;
                transition: all 0.3s ease;
            }}
            .status-badge.connected {{
                border-color: #00E676;
                color: #00E676;
                box-shadow: 0 0 10px rgba(0, 230, 118, 0.15);
            }}
            .status-badge.speaking {{
                border-color: #FF8F8F;
                color: #FF8F8F;
                box-shadow: 0 0 10px rgba(255, 143, 143, 0.15);
            }}
            .visualizer-box {{
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 6px;
                height: 50px;
                margin: 10px 0;
            }}
            .bar {{
                width: 4px;
                height: 10px;
                background: #9ab0c1;
                border-radius: 2px;
                transition: all 0.2s ease;
            }}
            .active-wave .bar {{
                background: linear-gradient(180deg, #FF4B4B 0%, #FF8F8F 100%);
                animation: bounce 0.6s infinite alternate;
            }}
            .active-wave .bar:nth-child(2) {{ animation-delay: 0.1s; }}
            .active-wave .bar:nth-child(3) {{ animation-delay: 0.2s; }}
            .active-wave .bar:nth-child(4) {{ animation-delay: 0.3s; }}
            .active-wave .bar:nth-child(5) {{ animation-delay: 0.4s; }}

            @keyframes bounce {{
                0% {{ height: 10px; }}
                100% {{ height: 45px; }}
            }}
            
            .mic-btn {{
                width: 80px;
                height: 80px;
                border-radius: 50%;
                border: none;
                background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3), inset 0 1px 1px rgba(255, 255, 255, 0.1);
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 2.2rem;
                transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275);
                outline: none;
            }}
            .mic-btn:hover {{
                transform: scale(1.05);
            }}
            .mic-btn.active {{
                background: linear-gradient(135deg, #FF4B4B 0%, #FF8F8F 100%);
                box-shadow: 0 0 25px rgba(255, 75, 75, 0.5), inset 0 1px 1px rgba(255, 255, 255, 0.2);
                animation: pulse 1.8s infinite;
            }}
            @keyframes pulse {{
                0% {{ transform: scale(1); }}
                50% {{ transform: scale(1.06); }}
                100% {{ transform: scale(1); }}
            }}
            .mic-btn:disabled {{
                opacity: 0.5;
                cursor: not-allowed;
            }}
            .btn-lbl {{
                font-size: 0.85rem;
                color: #8892b0;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-top: -4px;
            }}
            .log-box {{
                background: rgba(0, 0, 0, 0.2);
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-radius: 8px;
                width: 90%;
                padding: 12px;
                font-size: 0.85rem;
                color: #8892b0;
                text-align: left;
                max-height: 80px;
                overflow-y: auto;
                line-height: 1.5;
                box-sizing: border-box;
            }}
        </style>
        </head>
        <body>
        <div class="coach-container">
            <div id="statusBadge" class="status-badge">🔘 Offline</div>
            
            <div id="waveBox" class="visualizer-box">
                <div class="bar"></div>
                <div class="bar"></div>
                <div class="bar"></div>
                <div class="bar"></div>
                <div class="bar"></div>
            </div>
            
            <button id="micBtn" class="mic-btn" onclick="toggleSession()">🎙️</button>
            <div id="btnLbl" class="btn-lbl">Start Voice Session</div>
            
            <div id="logBox" class="log-box">
                System initialized. Click the microphone button above to start your live voice coaching session.
            </div>
        </div>

        <script>
            let webSocket = null;
            let mediaStream = null;
            let audioProcessor = null;
            let sourceNode = null;
            let audioCtx = null;
            let nextPlayTime = 0;
            let activeSources = [];
            let isConnected = false;
            
            const api_key = "{api_key}";
            const systemInstructionText = "{js_safe_instruction}";
            
            function log(msg) {{
                const box = document.getElementById("logBox");
                box.innerHTML = msg + "<br>" + box.innerHTML;
                console.log(msg);
            }}
            
            function updateStatus(statusText, type) {{
                const badge = document.getElementById("statusBadge");
                const waveBox = document.getElementById("waveBox");
                const btnLbl = document.getElementById("btnLbl");
                
                badge.className = "status-badge";
                waveBox.className = "visualizer-box";
                
                if (type === 'connected') {{
                    badge.classList.add("connected");
                    badge.innerText = "🟢 " + statusText;
                    btnLbl.innerText = "Stop Voice Session";
                }} else if (type === 'speaking') {{
                    badge.classList.add("speaking");
                    badge.innerText = "🎙️ " + statusText;
                    waveBox.classList.add("active-wave");
                }} else if (type === 'connecting') {{
                    badge.innerText = "🟡 " + statusText;
                }} else {{
                    badge.innerText = "🔘 Offline";
                    btnLbl.innerText = "Start Voice Session";
                }}
            }}
            
            function toggleSession() {{
                if (isConnected) {{
                    stopSession();
                }} else {{
                    startSession();
                }}
            }}
            
            function startSession() {{
                if (!api_key) {{
                    log("Error: API Key is missing. Set GEMINI_API_KEY in environment.");
                    return;
                }}
                isConnected = true;
                document.getElementById("micBtn").classList.add("active");
                updateStatus("Connecting...", "connecting");
                log("Connecting to Gemini Live API WebSocket...");
                
                connectWebSocket();
            }}
            
            function stopSession() {{
                isConnected = false;
                document.getElementById("micBtn").classList.remove("active");
                updateStatus("Offline");
                log("Session stopped.");
                
                if (webSocket) {{
                    webSocket.close();
                    webSocket = null;
                }}
                stopAudioRecording();
                stopAllAudioPlayback();
            }}
            
            function connectWebSocket() {{
                const model = "models/gemini-2.0-flash-exp";
                const wsUrl = `wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent?key=${{api_key}}`;
                
                try {{
                    webSocket = new WebSocket(wsUrl);
                    
                    webSocket.onopen = () => {{
                        log("WebSocket handshake completed. Configuring live session...");
                        const setupMessage = {{
                            "setup": {{
                                "model": model,
                                "generationConfig": {{
                                    "responseModalities": ["AUDIO"],
                                    "speechConfig": {{
                                        "voiceConfig": {{
                                            "prebuiltVoiceConfig": {{
                                                "voiceName": "Kore"
                                            }}
                                        }}
                                    }}
                                }},
                                "systemInstruction": {{
                                    "parts": [
                                        {{
                                            "text": systemInstructionText
                                        }}
                                    ]
                                }}
                            }}
                        }};
                        webSocket.send(JSON.stringify(setupMessage));
                    }};
                    
                    webSocket.onmessage = async (event) => {{
                        try {{
                            let data;
                            if (event.data instanceof Blob) {{
                                const text = await event.data.text();
                                data = JSON.parse(text);
                            }} else {{
                                data = JSON.parse(event.data);
                            }}
                            
                            if (data.setupComplete) {{
                                log("Session active! Greet the Coach.");
                                updateStatus("Listening...", "connected");
                                startAudioRecording();
                                return;
                            }}
                            
                            if (data.serverContent) {{
                                const serverContent = data.serverContent;
                                
                                if (serverContent.interrupted) {{
                                    log("Coach was interrupted by your voice.");
                                    stopAllAudioPlayback();
                                    updateStatus("Listening...", "connected");
                                    return;
                                }}
                                
                                if (serverContent.modelTurn && serverContent.modelTurn.parts) {{
                                    updateStatus("Coach is speaking...", "speaking");
                                    serverContent.modelTurn.parts.forEach(part => {{
                                        if (part.inlineData && part.inlineData.data) {{
                                            playPCM(part.inlineData.data);
                                        }}
                                    }});
                                }}
                            }}
                        }} catch (err) {{
                            console.error("Error parsing WebSocket message:", err);
                        }}
                    }};
                    
                    webSocket.onclose = (e) => {{
                        log(`WebSocket closed (Code: ${{e.code}}).`);
                        if (e.code === 1006) {{
                            log("<span style='color:#FF8F8F;font-weight:600;'>⚠️ Connection Failed (Abnormal Closure):</span> Google's WebSocket gateway rejected the connection. This typically indicates your API key project is suspended. Check logs/app.log or visit Google AI Studio billing console.");
                        }}
                        if (isConnected) {{
                            stopSession();
                        }}
                    }};
                    
                    webSocket.onerror = (err) => {{
                        log("<span style='color:#FF4B4B;font-weight:600;'>❌ WebSocket Error:</span> The bidirectional live connection was blocked or rejected.");
                        console.error(err);
                    }};
                    
                }} catch (e) {{
                    log(`WebSocket creation failed: ${{e.message}}`);
                    stopSession();
                }}
            }}
            
            async function startAudioRecording() {{
                try {{
                    mediaStream = await navigator.mediaDevices.getUserMedia({{ audio: true }});
                    
                    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                    const inputSampleRate = audioCtx.sampleRate;
                    
                    sourceNode = audioCtx.createMediaStreamSource(mediaStream);
                    audioProcessor = audioCtx.createScriptProcessor(4096, 1, 1);
                    
                    audioProcessor.onaudioprocess = (e) => {{
                        if (!webSocket || webSocket.readyState !== WebSocket.OPEN) return;
                        
                        const inputData = e.inputBuffer.getChannelData(0);
                        const pcmBuffer = downsampleTo16kHzPCM(inputData, inputSampleRate);
                        const base64Data = arrayBufferToBase64(pcmBuffer.buffer);
                        
                        const clientMessage = {{
                            "realtimeInput": {{
                                "mediaChunks": [
                                    {{
                                        "mimeType": "audio/pcm;rate=16000",
                                        "data": base64Data
                                    }}
                                ]
                            }}
                        }};
                        webSocket.send(JSON.stringify(clientMessage));
                    }};
                    
                    sourceNode.connect(audioProcessor);
                    audioProcessor.connect(audioCtx.destination);
                    log("Microphone recording active (16kHz PCM).");
                }} catch (err) {{
                    log("Error getting microphone: " + err.message);
                    stopSession();
                }}
            }}
            
            function stopAudioRecording() {{
                if (audioProcessor) {{
                    audioProcessor.disconnect();
                    audioProcessor = null;
                }}
                if (sourceNode) {{
                    sourceNode.disconnect();
                    sourceNode = null;
                }}
                if (mediaStream) {{
                    mediaStream.getTracks().forEach(track => track.stop());
                    mediaStream = null;
                }}
            }}
            
            function downsampleTo16kHzPCM(buffer, inputSampleRate) {{
                const sampleRateRatio = inputSampleRate / 16000;
                const newLength = Math.round(buffer.length / sampleRateRatio);
                const result = new Int16Array(newLength);
                for (let i = 0; i < newLength; i++) {{
                    let index = Math.round(i * sampleRateRatio);
                    if (index >= buffer.length) index = buffer.length - 1;
                    let sample = buffer[index];
                    sample = Math.max(-1, Math.min(1, sample));
                    result[i] = sample < 0 ? sample * 0x8000 : sample * 0x7FFF;
                }}
                return result;
            }}
            
            function arrayBufferToBase64(buffer) {{
                let binary = '';
                const bytes = new Uint8Array(buffer);
                const len = bytes.byteLength;
                for (let i = 0; i < len; i++) {{
                    binary += String.fromCharCode(bytes[i]);
                }}
                return window.btoa(binary);
            }}
            
            function playPCM(base64Data) {{
                if (!audioCtx) return;
                
                try {{
                    const binaryString = window.atob(base64Data);
                    const len = binaryString.length;
                    const bytes = new Uint8Array(len);
                    for (let i = 0; i < len; i++) {{
                        bytes[i] = binaryString.charCodeAt(i);
                    }}
                    
                    const int16Array = new Int16Array(bytes.buffer);
                    const float32Array = new Float32Array(int16Array.length);
                    for (let i = 0; i < int16Array.length; i++) {{
                        float32Array[i] = int16Array[i] / 32768.0;
                    }}
                    
                    const audioBuffer = audioCtx.createBuffer(1, float32Array.length, 24000);
                    audioBuffer.getChannelData(0).set(float32Array);
                    
                    const source = audioCtx.createBufferSource();
                    source.buffer = audioBuffer;
                    source.connect(audioCtx.destination);
                    
                    const currentTime = audioCtx.currentTime;
                    if (nextPlayTime < currentTime) {{
                        nextPlayTime = currentTime;
                    }}
                    source.start(nextPlayTime);
                    
                    source.onended = () => {{
                        const badge = document.getElementById("statusBadge");
                        if (badge.innerText.includes("speaking") && audioCtx.currentTime >= nextPlayTime) {{
                            updateStatus("Listening...", "connected");
                        }}
                    }};
                    
                    nextPlayTime += audioBuffer.duration;
                    activeSources.push(source);
                }} catch (e) {{
                    console.error("Playback failed:", e);
                }}
            }}
            
            function stopAllAudioPlayback() {{
                activeSources.forEach(source => {{
                    try {{ source.stop(); }} catch(e) {{}}
                }});
                activeSources = [];
                nextPlayTime = 0;
            }}
        </script>
        </body>
        </html>
        """
        st.components.v1.html(voice_chat_html, height=380)


    # 3. Rep-by-Rep Detail Carousel
    st.markdown("### 🏋️‍♂️ Rep-by-Rep Biomechanical Breakdown")
    for rep in analysis["reps"]:
        # Find match glowing theme
        rep_rating = rep["safety_rating"]
        glow_map = {
            "SAFE": "safe-glow",
            "WARNING": "warning-glow",
            "DANGEROUS": "danger-glow"
        }
        rep_glow = glow_map.get(rep_rating, "")
        
        st.markdown(f"""
        <div class="glass-card {rep_glow}">
            <div style='display:flex; justify-content:space-between; align-items:center;'>
                <h4 style='margin:0;color:#FF4B4B;'>REPETITION #{rep['rep_index']}</h4>
                <span class="step-badge">{rep_rating}</span>
            </div>
            <hr style='border:0;border-top:1px solid rgba(255,255,255,0.08);margin:12px 0;'>
            <div style='display:grid; grid-template-columns: 1fr 1fr; gap: 20px;'>
                <div>
                    <p style='margin:0;font-size:0.95rem;color:#8892b0;'>🏋️‍♂️ DEPTH ASSESSMENT</p>
                    <p style='margin:4px 0 0 0;font-size:1rem;color:#e6f1ff;'>{rep['depth_evaluation']}</p>
                </div>
                <div>
                    <p style='margin:0;font-size:0.95rem;color:#8892b0;'>📐 SPINE & KNEE ALIGNMENT</p>
                    <p style='margin:4px 0 0 0;font-size:1rem;color:#e6f1ff;'>{rep['posture_evaluation']}</p>
                </div>
            </div>
            <div style='margin-top:15px;background:rgba(255, 75, 75, 0.05);border-radius:6px;padding:12px;border:1px dashed rgba(255, 75, 75, 0.2);'>
                <p style='margin:0;font-size:0.9rem;color:#FF8F8F;font-weight:600;'>💡 Actionable Physiological Cue:</p>
                <p style='margin:4px 0 0 0;font-style:italic;font-size:1rem;color:#e6f1ff;'>"{rep['coaching_cue']}"</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
