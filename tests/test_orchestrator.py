import os
import pytest
from unittest.mock import MagicMock, patch
from src.orchestrator import MasterOrchestrator

@pytest.fixture
def mock_landmarks():
    """Generates standard standing landmarks for mock testing."""
    return {
        "left_shoulder": {"x": 0.45, "y": 0.35, "z": 0.0, "visibility": 0.99},
        "right_shoulder": {"x": 0.55, "y": 0.35, "z": 0.0, "visibility": 0.99},
        "left_hip": {"x": 0.44, "y": 0.65, "z": 0.0, "visibility": 0.99},
        "right_hip": {"x": 0.56, "y": 0.65, "z": 0.0, "visibility": 0.99},
        "left_knee": {"x": 0.40, "y": 0.78, "z": 0.0, "visibility": 0.99},
        "right_knee": {"x": 0.60, "y": 0.78, "z": 0.0, "visibility": 0.99},
        "left_ankle": {"x": 0.42, "y": 0.90, "z": 0.0, "visibility": 0.99},
        "right_ankle": {"x": 0.58, "y": 0.90, "z": 0.0, "visibility": 0.99}
    }

@patch('src.vision_agent.VisionAgent.process_video')
@patch('src.synthesis_agent.SynthesisAgent.generate_ideal_video')
@patch('os.path.exists')
def test_orchestrator_successful_flow(mock_exists, mock_generate_ideal_video, mock_process_video, mock_landmarks):
    # Setup Mocks
    mock_exists.return_value = True
    mock_generate_ideal_video.return_value = "output_corrected.mp4"
    
    # Simulate a single full squat rep (60 frames)
    frames = []
    for f in range(60):
        # We start at standing (175 deg knee angle)
        # Drop down to deep parallel (90 deg knee angle) around frame 30
        # Return to standing at frame 60
        if f < 15:
            k_angle = 175.0
            depth = -0.10
        elif f <= 30:
            pct = (f - 15) / 15.0
            k_angle = 175.0 - (85.0 * pct)
            depth = -0.10 + (0.13 * pct)
        elif f <= 45:
            pct = (f - 30) / 15.0
            k_angle = 90.0 + (85.0 * pct)
            depth = 0.03 - (0.13 * pct)
        else:
            k_angle = 175.0
            depth = -0.10
            
        # Recreate a simple landmark layout to trick calculate_angle formulas
        # Hip at y=0.65 -> y=0.78 -> y=0.65
        # Knee at y=0.75, Ankle at y=0.90
        hip_y = 0.75 - depth
        
        # When hip_y = 0.78 (deep), depth gap is +0.03
        f_landmarks = {
            "left_shoulder": {"x": 0.45, "y": 0.35, "z": 0.0, "visibility": 0.99},
            "right_shoulder": {"x": 0.55, "y": 0.35, "z": 0.0, "visibility": 0.99},
            "left_hip": {"x": 0.44, "y": hip_y, "z": 0.0, "visibility": 0.99},
            "right_hip": {"x": 0.56, "y": hip_y, "z": 0.0, "visibility": 0.99},
            "left_knee": {"x": 0.40, "y": 0.75, "z": 0.0, "visibility": 0.99},
            "right_knee": {"x": 0.60, "y": 0.75, "z": 0.0, "visibility": 0.99},
            "left_ankle": {"x": 0.42, "y": 0.90, "z": 0.0, "visibility": 0.99},
            "right_ankle": {"x": 0.58, "y": 0.90, "z": 0.0, "visibility": 0.99}
        }
        
        frames.append({
            "frame_id": f,
            "timestamp_sec": f * 0.033,
            "landmarks": f_landmarks,
            "metrics": {
                "avg_knee_angle": k_angle,
                "avg_torso_angle": 10.0,
                "avg_depth_gap": depth,
                "valgus_ratio": 1.0
            }
        })

        
    mock_process_video.return_value = frames

    # Run Orchestrator (without GEMINI_API_KEY, invoking fallback rule-engine)
    with patch.dict(os.environ, {}, clear=True):
        orchestrator = MasterOrchestrator()
        result = orchestrator.run_coaching_flow("sample.mp4")

    # Assertions
    assert result["success"] is True
    assert result["corrected_video_path"] == "output_corrected.mp4"
    
    analysis = result["analysis"]
    assert "workout_summary" in analysis
    assert analysis["perfect_reps_count"] == 1
    assert len(analysis["reps"]) == 1
    
    rep = analysis["reps"][0]
    assert rep["rep_index"] == 1
    assert rep["safety_rating"] == "SAFE"
    assert "depth" in rep["depth_evaluation"].lower()
