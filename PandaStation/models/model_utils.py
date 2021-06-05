import numpy as np

def solid_cuboid(w,d,h,m):
    ixx = (1.0/12.0)*m*(h**2  + d**2)
    iyy = (1.0/12.0)*m*(w**2  + d**2)
    izz = (1.0/12.0)*m*(w**2  + h**2)
    print(f"ixx: {ixx}\niyy: {iyy}\nizz: {izz}")


def camera_angle(b, h):
    return 180.0 - np.degrees(np.arctan(b/h))

def solid_cylinder(r, l, m):
    ixx = (1.0/12.0)*m*(3*r**2 + l**2)  
    iyy = (1.0/12.0)*m*(3*r**2 + l**2)  
    izz = (1.0/2.0)*m*r**2
    print(f"ixx: {ixx}\niyy: {iyy}\nizz: {izz}")

def solid_sphere(r, m):
    i = (2.0/5.0)*m*r**2
    print(f"ixx: {i}\niyy: {i}\nizz: {i}")

solid_cylinder(0.03, 0.03, 0.1)

