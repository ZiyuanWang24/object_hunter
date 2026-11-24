import numpy as np
from objecthunter.kalman import ConstantVelocityKalman

def test_prediction_and_correction():
    k=ConstantVelocityKalman([0,0]); p0=k.P.copy(); k.predict(1); assert np.trace(k.P)>np.trace(p0); k.update([1,0]); assert np.linalg.norm(k.x[:2]-[1,0])<.5

def test_covariance_grows_on_missed_observations():
    k=ConstantVelocityKalman([0,0]); a=k.P.copy(); k.predict(1); b=k.P.copy(); k.predict(1); assert np.trace(k.P)>np.trace(b)>np.trace(a)

def test_mahalanobis():
    k=ConstantVelocityKalman([0,0]); assert k.mahalanobis_squared([0,0]) < k.mahalanobis_squared([10,10])
