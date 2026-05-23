import pytest
import math
from src.diagnostic_agent import DiagnosticAgent

def generate_frame(frame_id: int, knee_angle: float, torso_angle_deg: float, depth_gap: float, knee_caving: bool = False) -> dict:
    """
    Generates a synthetic skeleton frame with landmarks mathematically positioned
    to yield the targeted knee angle, torso angle, and depth gap.
    """
    # Normalized coordinates
    # We will adjust Y of hip to control depth gap
    # Y increases downwards. So ankle is at bottom, knee is in middle, hip is at top (if standing)
    ankle_y = 0.90
    knee_y = 0.75
    hip_y = knee_y - depth_gap  # standing = 0.65 (gap is -0.10), parallel = 0.75 (gap is 0.0), deep = 0.78 (gap is +0.03)

    # Torso tilt: shoulder offset
    # Vertical torso: shoulder directly above hip
    # If shoulder tilts forward, X moves left (for left-to-right alignment)
    # We can calculate shoulder coordinate based on torso_angle_deg
    torso_length = 0.35
    lean_rad = math.radians(torso_angle_deg)
    
    # We construct left side
    shoulder_x = 0.45 - torso_length * math.sin(lean_rad)
    shoulder_y = hip_y - torso_length * math.cos(lean_rad)

    # Knee width: normal is 0.20 separation. Caving (valgus) is 0.06 separation.
    knee_dist = 0.08 if knee_caving else 0.22

    # We build standard coordinates
    landmarks = {
        "left_shoulder": {"x": shoulder_x, "y": shoulder_y, "z": 0.0, "visibility": 0.99},
        "right_shoulder": {"x": shoulder_x + 0.1, "y": shoulder_y, "z": 0.0, "visibility": 0.99},
        
        "left_hip": {"x": 0.44, "y": hip_y, "z": 0.0, "visibility": 0.99},
        "right_hip": {"x": 0.56, "y": hip_y, "z": 0.0, "visibility": 0.99},
        
        # Knee caving affects x distance
        "left_knee": {"x": 0.5 - (knee_dist / 2), "y": knee_y, "z": 0.0, "visibility": 0.99},
        "right_knee": {"x": 0.5 + (knee_dist / 2), "y": knee_y, "z": 0.0, "visibility": 0.99},
        
        "left_ankle": {"x": 0.40, "y": ankle_y, "z": 0.0, "visibility": 0.99},
        "right_ankle": {"x": 0.60, "y": ankle_y, "z": 0.0, "visibility": 0.99}
    }

    # In our calculation:
    # For testing, we mock the joint calculation results inside the test by feeding frame_id, timestamp, and metrics.
    return {
        "frame_id": frame_id,
        "timestamp_sec": frame_id * 0.033,  # 30 FPS
        "landmarks": landmarks,
        "metrics": {
            "avg_knee_angle": knee_angle,
            "avg_torso_angle": torso_angle_deg,
            "avg_depth_gap": depth_gap,
            "valgus_ratio": 0.70 if knee_caving else 1.0
        }
    }


def test_segment_single_perfect_squat():
    # Construct a synthetic squat curve: standing -> sinking to parallel -> standing
    frames = []
    
    # 60 frames = 2 seconds of video
    # Frame 0 to 15: Standing (Knee angle 175)
    # Frame 16 to 30: Descent (Knee angle 175 -> 95)
    # Frame 30: Bottom (Knee angle 90)
    # Frame 31 to 45: Ascent (Knee angle 90 -> 175)
    # Frame 46 to 60: Standing (Knee angle 175)
    
    for f in range(60):
        if f < 15:
            # Stand
            k_angle = 175.0
            t_angle = 10.0
            depth = -0.10
        elif f <= 30:
            # Sinking
            pct = (f - 15) / 15.0
            k_angle = 175.0 - (85.0 * pct)  # 175 down to 90
            t_angle = 10.0 + (15.0 * pct)   # Torso leans slightly to 25 deg
            depth = -0.10 + (0.13 * pct)    # Hips sink past parallel: -0.10 to +0.03
        elif f <= 45:
            # Rising
            pct = (f - 30) / 15.0
            k_angle = 90.0 + (85.0 * pct)   # 90 up to 175
            t_angle = 25.0 - (15.0 * pct)
            depth = 0.03 - (0.13 * pct)
        else:
            # Stand
            k_angle = 175.0
            t_angle = 10.0
            depth = -0.10
            
        frames.append(generate_frame(f, k_angle, t_angle, depth, knee_caving=False))

    agent = DiagnosticAgent()
    reps = agent.segment_reps(frames)
    
    # Assertions
    assert len(reps) == 1, "Should detect exactly one squat repetition."
    rep = reps[0]
    
    assert rep["rep_index"] == 1
    assert rep["start_frame"] > 10 and rep["start_frame"] < 20
    assert rep["bottom_frame"] == 30
    assert rep["end_frame"] > 40 and rep["end_frame"] < 50
    
    # Check that no faults are flagged
    assert rep["faults"]["shallow_depth"] is False
    assert rep["faults"]["excessive_forward_lean"] is False
    assert rep["faults"]["knee_valgus"] is False

def test_segment_shallow_depth_squat():
    frames = []
    
    for f in range(60):
        if f < 15:
            k_angle = 175.0
            t_angle = 10.0
            depth = -0.10
        elif f <= 30:
            # Shallow sinking
            pct = (f - 15) / 15.0
            k_angle = 175.0 - (45.0 * pct)  # Only sinks to 130 degrees
            t_angle = 10.0 + (10.0 * pct)
            depth = -0.10 + (0.02 * pct)    # Hips stop above knee plane (-0.08 depth gap)
        elif f <= 45:
            # Rising
            pct = (f - 30) / 15.0
            k_angle = 130.0 + (45.0 * pct)
            t_angle = 20.0 - (10.0 * pct)
            depth = -0.08 - (0.02 * pct)
        else:
            k_angle = 175.0
            t_angle = 10.0
            depth = -0.10
        frames.append(generate_frame(f, k_angle, t_angle, depth, knee_caving=False))

    agent = DiagnosticAgent()
    reps = agent.segment_reps(frames)
    
    assert len(reps) == 1, "Should segment the shallow squat."
    rep = reps[0]
    assert rep["faults"]["shallow_depth"] is True, "Shallow depth must be flagged."


def test_segment_unrack_crouched_start_ignored():
    """
    Ensures that starting crouched (unracking the bar) then standing up
    does not register as a rep, but a subsequent real squat does.
    """
    frames = []
    
    # 90 frames = 3 seconds of video
    # Frame 0 to 20: Crouched setup/unrack (Knee angle starts at 120 and rises to 175)
    # Frame 21 to 40: Standing straight (Knee angle 175)
    # Frame 41 to 55: Descent (Knee angle 175 -> 90)
    # Frame 55: Bottom of real squat (Knee angle 90)
    # Frame 56 to 70: Ascent (Knee angle 90 -> 175)
    # Frame 71 to 90: Standing straight (Knee angle 175)
    
    for f in range(90):
        if f <= 20:
            # Crouched unrack setup rising up
            pct = f / 20.0
            k_angle = 120.0 + (55.0 * pct)  # 120 to 175
            t_angle = 20.0 - (10.0 * pct)
            depth = -0.05 - (0.05 * pct)
        elif f < 41:
            # Stand
            k_angle = 175.0
            t_angle = 10.0
            depth = -0.10
        elif f <= 55:
            # Descent
            pct = (f - 40) / 14.0
            k_angle = 175.0 - (85.0 * pct)
            t_angle = 10.0 + (15.0 * pct)
            depth = -0.10 + (0.13 * pct)
        elif f <= 70:
            # Ascent
            pct = (f - 55) / 15.0
            k_angle = 90.0 + (85.0 * pct)
            t_angle = 25.0 - (15.0 * pct)
            depth = 0.03 - (0.13 * pct)
        else:
            # Stand
            k_angle = 175.0
            t_angle = 10.0
            depth = -0.10
            
        frames.append(generate_frame(f, k_angle, t_angle, depth, knee_caving=False))

    agent = DiagnosticAgent()
    reps = agent.segment_reps(frames)
    
    # We expect exactly 1 rep (the one starting at frame 41).
    # The initial unrack rising (0-20) must be ignored completely.
    assert len(reps) == 1, "Should filter out initial unrack crouch and only detect the real subsequent squat."
    # Allowing a small frame window (41-45) where the state machine detects descent crossing below 160°
    assert reps[0]["start_frame"] >= 41 and reps[0]["start_frame"] <= 45, f"Should start between 41 and 45, but got {reps[0]['start_frame']}"


def test_segment_first_two_seconds_ignored_in_long_video():
    """
    Ensures that any rep starting in the first 2 seconds of a video longer than 4 seconds
    is filtered out as part of the initial walkout/setup phase.
    """
    frames = []
    
    # 180 frames = 6 seconds of video
    # False rep: starts at frame 10 (0.33s), bottom at frame 25 (0.83s), ends at frame 40 (1.32s)
    # Real rep: starts at frame 90 (2.97s), bottom at frame 110 (3.63s), ends at frame 130 (4.29s)
    for f in range(180):
        # Default standing straight
        k_angle = 175.0
        t_angle = 10.0
        depth = -0.10
        
        # False rep in setup window (starts < 2s)
        if 10 <= f <= 40:
            if f <= 25:
                pct = (f - 10) / 15.0
                k_angle = 175.0 - (85.0 * pct)
                t_angle = 10.0 + (15.0 * pct)
                depth = -0.10 + (0.13 * pct)
            else:
                pct = (f - 25) / 15.0
                k_angle = 90.0 + (85.0 * pct)
                t_angle = 25.0 - (15.0 * pct)
                depth = 0.03 - (0.13 * pct)
                
        # Real rep after setup window (starts > 2s)
        elif 90 <= f <= 130:
            if f <= 110:
                pct = (f - 90) / 20.0
                k_angle = 175.0 - (85.0 * pct)
                t_angle = 10.0 + (15.0 * pct)
                depth = -0.10 + (0.13 * pct)
            else:
                pct = (f - 110) / 20.0
                k_angle = 90.0 + (85.0 * pct)
                t_angle = 25.0 - (15.0 * pct)
                depth = 0.03 - (0.13 * pct)
                
        frames.append(generate_frame(f, k_angle, t_angle, depth, knee_caving=False))
        
    agent = DiagnosticAgent()
    reps = agent.segment_reps(frames)
    
    # We expect exactly 1 rep (the second one). The first one was filtered out because start_time < 2.0s in a 6s video.
    assert len(reps) == 1, "Should filter out the first rep as setup noise and only return the second rep."
    assert reps[0]["start_frame"] >= 90 and reps[0]["start_frame"] <= 95, f"Expected rep to start between 90 and 95, got {reps[0]['start_frame']}"


