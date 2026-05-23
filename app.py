import streamlit as st
import os
import json
import numpy as np
import cv2
import time
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

# Choose execution path
run_mode = st.sidebar.radio(
    "Choose Input Stream:",
    ["🤖 Run Synthetic Demo Mode", "🎥 Record Live Webcam", "📤 Upload Squat Video"]
)


model_selector = st.sidebar.selectbox(
    "Cognitive Coach Model",
    ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-1.5-flash"],
    index=0
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
                orchestrator = MasterOrchestrator(gemini_model=model_selector)
                result = orchestrator.run_coaching_flow(video_path=recorded_file, output_video_path="output_corrected.mp4")
                
                st.session_state.feedback = result
                st.session_state.video_path = recorded_file
                st.session_state.corrected_path = "output_corrected.mp4"
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
                st.info("Using downloaded real squat athlete video (YouTube: txnwoJz-Rno)...")
                video_file = "real_demo_squat.mp4"
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
            orchestrator = MasterOrchestrator(gemini_model=model_selector)
            
            # Run pipeline E2E
            result = orchestrator.run_coaching_flow(video_path=video_file, output_video_path="output_corrected.mp4")
            
            st.session_state.feedback = result
            st.session_state.video_path = video_file
            st.session_state.corrected_path = "output_corrected.mp4"
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
        temp_path = "uploaded_temp.mp4"
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
                orchestrator = MasterOrchestrator(gemini_model=model_selector)
                result = orchestrator.run_coaching_flow(video_path=temp_path, output_video_path="output_corrected.mp4")
                
                st.session_state.feedback = result
                st.session_state.video_path = temp_path
                st.session_state.corrected_path = "output_corrected.mp4"
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
        
    # High level coaching summary
    st.markdown(f"""
    <div class="glass-card">
        <h4 style='margin:0 0 8px 0;color:#FF8F8F;'>🧠 World-Class Coach Summary</h4>
        <p style='margin:0;line-height:1.6;font-size:1.05rem;'>"{analysis['workout_summary']}"</p>
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
    veo_demo_path = "veo_coaching_demo.mp4"
    if os.path.exists(veo_demo_path):
        st.markdown("### 🏋️‍♂️ Virtual Coach Demonstration Video (Google Veo)")
        st.write("A high-fidelity coaching demonstration generated dynamically in the cloud by Google Veo showing bad form correcting to perfect textbook technique based on your coaching cues.")
        st.video(veo_demo_path)

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
