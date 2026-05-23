import pytest
import math
from src.diagnostic_agent import DiagnosticAgent

def test_calculate_angle_3d_90_degrees():
    # p2 is the vertex (e.g. knee) at origin
    p2 = {"x": 0.0, "y": 0.0, "z": 0.0}
    # p1 is hip, extending straight up along positive Y
    p1 = {"x": 0.0, "y": 1.0, "z": 0.0}
    # p3 is ankle, extending straight right along positive X
    p3 = {"x": 1.0, "y": 0.0, "z": 0.0}
    
    angle = DiagnosticAgent.calculate_angle_3d(p1, p2, p3)
    assert math.isclose(angle, 90.0, abs_tol=1e-5)

def test_calculate_angle_3d_180_degrees():
    # Straight line along Y-axis
    p2 = {"x": 0.0, "y": 0.0, "z": 0.0}
    p1 = {"x": 0.0, "y": 1.0, "z": 0.0}
    p3 = {"x": 0.0, "y": -1.0, "z": 0.0}
    
    angle = DiagnosticAgent.calculate_angle_3d(p1, p2, p3)
    assert math.isclose(angle, 180.0, abs_tol=1e-5)

def test_calculate_torso_angle_straight_up():
    # Shoulder is directly above hip in MediaPipe coordinates (remember, Y is downward)
    # Hip at Y=1.0, Shoulder at Y=0.0
    hip = {"x": 0.5, "y": 1.0, "z": 0.0}
    shoulder = {"x": 0.5, "y": 0.0, "z": 0.0}
    
    # Vertically aligned = 0 degrees forward lean
    angle = DiagnosticAgent.calculate_torso_angle(shoulder, hip)
    assert math.isclose(angle, 0.0, abs_tol=1e-5)

def test_calculate_torso_angle_45_degrees():
    # Torso vector points from hip (0.0, 1.0) to shoulder (-1.0, 0.0)
    # The vertical vector is (0, -1, 0)
    hip = {"x": 0.0, "y": 1.0, "z": 0.0}
    shoulder = {"x": -1.0, "y": 0.0, "z": 0.0}
    
    angle = DiagnosticAgent.calculate_torso_angle(shoulder, hip)
    # Torso vector is (-1.0, -1.0, 0.0) which is 45 degrees relative to (0.0, -1.0, 0.0)
    assert math.isclose(angle, 45.0, abs_tol=1e-5)
