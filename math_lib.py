"""
This is a part of the autonomous driving car project.
This simulates how to apply Reinforcement Learning in dynamic obstacles avoidance for a self-driving car.
author: Binh Tran Thanh / email:thanhbinh@hcmut.edu.vn or thanhbinh.hcmut@gmail.com
"""

import numpy as np
import math 

def matrix_rotation(matrix, yaw):
    rotate_matrix = np.array([[math.cos(yaw), math.sin(yaw)],
                            [-math.sin(yaw), math.cos(yaw)]])
    return (matrix.T.dot(rotate_matrix)).T

def matrix_translation(matrix, vector):
    return matrix + vector.T

def vector_translation(vector, translate_vec):
    return vector + translate_vec

def point_dist(p, q):
    '''
    calculate distance of 2 point
    '''
    return math.hypot(q[0] - p[0], q[1] - p[1])

def intersection(x, y, radius, line_segment):
    '''
    find two different points where a line intersects with a circle
    '''

    is_pt = None
    p1x, p1y = line_segment[0]
    p2x, p2y = line_segment[1]
    dx, dy = np.subtract(p2x, p1x), np.subtract(p2y, p1y)
    a = dx ** 2 + dy ** 2
    b = 2 * (dx * (p1x - x) + dy * (p1y - y))
    c = (p1x - x) ** 2 + (p1y - y) ** 2 - radius ** 2
    discriminant = b ** 2 - 4 * a * c
    if discriminant > 0:
        t1 = (-b + discriminant ** 0.5) / (2 * a)
        t2 = (-b - discriminant ** 0.5) / (2 * a)
        pt1 = (dx * t1 + p1x, dy * t1 + p1y)
        pt2 = (dx * t2 + p1x, dy * t2 + p1y)
        is_pt = pt1, pt2

    return is_pt

def inside_line_segment(point, line_segment):
    '''
    check if a point is whether inside given line segment 
    return True if inside, otherwise return False
    using total distance to check, 
    '''
    dp1 = point_dist(point, line_segment[0])
    dp2 = point_dist(point, line_segment[1])
    dls = point_dist(line_segment[1], line_segment[0])
    dp = dp1 + dp2
    return math.isclose(dp, dls)

def unsigned_angle(v1, v2):
    '''
    Find unsigned angle between two vectors
    '''
    usa = signed_angle(v1, v2)
    if usa < 0:
        usa = 2 * math.pi + usa
    return usa

def unsigned_angle(center, ptA, ptB):
    '''
    Find unsigned angle among 3 points (ptA- center- ptB)
    '''
    vector_a = np.subtract(ptA, center)
    vector_b = np.subtract(ptB, center)
    usa = signed_angle(vector_a, vector_b)
    if usa < 0:
        usa = 2 * math.pi + usa
    return usa
    
def signed_angle_xAxis(point):
    '''
    Finds signed angle between point and x axis
    return angle (+) for anti clockwise, and (-) for clockwise
    '''
    angle = math.atan2(point[1], point[0])
    # print (math.degrees(angle))
    return angle


def signed_angle(v1, v2):
    '''
    Finds angle between two vectors
    return angle (+) for anti clockwise, and (-) for clockwise
    '''
    angle = signed_angle_xAxis(v1)  # cal angle between vector (A) and ox axis
    v_b = rotate_vector(v2, angle)  # rotate vector B according to rotation_radians 
    return signed_angle_xAxis(v_b)

def rotate_vector(v, radian):
    '''
    rotate vector with a angle of radians around (0,0)
    '''
    x, y = v
    rx = x * math.cos(radian) + y * math.sin(radian)
    ry = -x * math.sin(radian) + y * math.cos(radian)
    return rx, ry


def line_across(line1, line2):
    '''
    check whether 2 lines are cross each others or not
    return cross point if they are
    return None if not
    '''
    is_pt = line_intersection(line1, line2)
    ret_result = None
    if is_pt is not None and inside_line_segment(is_pt, line1) and inside_line_segment(is_pt, line2):
        ret_result = is_pt
    return ret_result


def line_intersection(line1, line2):
    '''
    return intersection point of 2 lines
    if that point does not exist, return none
    '''
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None
    # raise Exception('lines do not intersect')
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y