"""Numerical integration (RK4) and quaternion helpers. Not physics, just maths."""
import numpy as np


def rk4_step(f, t, s, dt):
    """One 4th-order Runge-Kutta step. f(t, s) -> (ds/dt, info). Returns (new state, info at start of step)."""
    k1, info = f(t, s)
    k2, _ = f(t + dt / 2, s + dt / 2 * k1)
    k3, _ = f(t + dt / 2, s + dt / 2 * k2)
    k4, _ = f(t + dt, s + dt * k3)
    s_new = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    s_new[6:10] /= np.linalg.norm(s_new[6:10])      # keep the quaternion a unit quaternion
    return s_new, info


def quat_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw*bw - ax*bx - ay*by - az*bz,
                     aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx,
                     aw*bz + ax*by - ay*bx + az*bw])


def quat_to_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
                     [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                     [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]])


def matrix_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        q = [S / 4, (R[2, 1] - R[1, 2]) / S, (R[0, 2] - R[2, 0]) / S, (R[1, 0] - R[0, 1]) / S]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / S, S / 4, (R[0, 1] + R[1, 0]) / S, (R[0, 2] + R[2, 0]) / S]
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / S, (R[0, 1] + R[1, 0]) / S, S / 4, (R[1, 2] + R[2, 1]) / S]
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / S, (R[0, 2] + R[2, 0]) / S, (R[1, 2] + R[2, 1]) / S, S / 4]
    q = np.array(q)
    return q / np.linalg.norm(q)


def launch_attitude(tilt_deg: float, heading_deg: float):
    """Quaternion for a rocket pointing `tilt_deg` off vertical toward compass `heading_deg`."""
    tilt, heading = np.radians(tilt_deg), np.radians(heading_deg)
    x_b = np.array([np.sin(tilt) * np.sin(heading), np.sin(tilt) * np.cos(heading), np.cos(tilt)])
    y_b = np.cross([0.0, 0.0, 1.0], x_b)
    y_b = np.array([0.0, 1.0, 0.0]) if np.linalg.norm(y_b) < 1e-6 else y_b / np.linalg.norm(y_b)
    z_b = np.cross(x_b, y_b)
    return matrix_to_quat(np.column_stack([x_b, y_b, z_b]))
