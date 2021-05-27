import numpy as np

def solid_cuboid(w,d,h,m):
    ixx = (1.0/12.0)*m*(h**2  + d**2)
    iyy = (1.0/12.0)*m*(w**2  + d**2)
    izz = (1.0/12.0)*m*(w**2  + h**2)
    print(f"ixx: {ixx}\niyy: {iyy}\nizz: {izz}")

solid_cuboid(1, 1, 0.05, 1)

def camera_angle(b, h):
    return 180.0 - np.degrees(np.arctan(b/h))


print(camera_angle(0.5, 0.5))
print(camera_angle(0.5, 0.75))
print(camera_angle(np.sqrt(0.5**2 + 0.5**2), 0.5))
