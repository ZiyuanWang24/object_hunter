from __future__ import annotations

import numpy as np


class ConstantVelocityKalman:
    """2-D constant-velocity filter with continuous white-acceleration noise."""

    def __init__(self, position, covariance=None, process_noise=1.0, measurement_noise=0.02):
        p = np.asarray(position, dtype=float)
        self.x = np.array([p[0], p[1], 0.0, 0.0], dtype=float)
        self.P = np.asarray(covariance, dtype=float) if covariance is not None else np.diag([0.05, 0.05, 1.0, 1.0])
        self.q = float(process_noise)
        self.r = float(measurement_noise)
        self.last_timestamp: float | None = None

    def predict(self, dt: float):
        dt = max(0.0, float(dt))
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        q = self.q
        Q = q * np.array([[dt**4/4, 0, dt**3/2, 0], [0, dt**4/4, 0, dt**3/2], [dt**3/2, 0, dt**2, 0], [0, dt**3/2, 0, dt**2]], float)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        self.P = (self.P + self.P.T) / 2
        return self.x.copy(), self.P.copy()

    def update(self, measurement, measurement_noise=None):
        z = np.asarray(measurement, dtype=float).reshape(2)
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        R = np.eye(2) * (self.r if measurement_noise is None else measurement_noise)
        innovation = z - H @ self.x
        S = H @ self.P @ H.T + R
        # Solve rather than explicitly invert S.
        K = np.linalg.solve(S, (self.P @ H.T).T).T
        self.x = self.x + K @ innovation
        I = np.eye(4)
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
        self.P = (self.P + self.P.T) / 2
        return innovation, S, K

    def mahalanobis_squared(self, measurement):
        z = np.asarray(measurement, dtype=float).reshape(2)
        H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        d = z - H @ self.x
        S = H @ self.P @ H.T + np.eye(2) * self.r
        return float(d.T @ np.linalg.solve(S, d))

    def covariance_ellipse(self, confidence=0.95, samples=48):
        chi2 = 5.991464547 if confidence == 0.95 else 1.0
        vals, vecs = np.linalg.eigh(self.P[:2, :2])
        order = np.argsort(vals)[::-1]
        vals, vecs = vals[order], vecs[:, order]
        theta = np.linspace(0, 2*np.pi, samples)
        circle = np.vstack([np.cos(theta), np.sin(theta)])
        return (self.x[:2, None] + vecs @ (np.sqrt(np.maximum(vals, 0) * chi2)[:, None] * circle)).T
