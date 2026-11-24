import numpy as np
from objecthunter.geometry import HomographyCalibrator

def test_homography_roundtrip():
    image=np.array([[100,100],[500,100],[500,400],[100,400]],float); world=np.array([[0,0],[2,0],[2,1],[0,1]],float)
    h=HomographyCalibrator.from_points(image,world,(640,480)); q=np.array([300,250]); assert np.allclose(h.world_to_pixel(h.pixel_to_world(q)),q,atol=1e-3)
