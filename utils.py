# utils.py
import numpy as np

from config import *
ECC2 = 6.69437999014e-3  # 第一偏心率平方


def wrap_angle(angle):
    """将角度归一化到 [-π, π]"""
    return (angle + np.pi) % (2 * np.pi) - np.pi

def great_circle_distance(lat1, lon1, lat2, lon2, R=R_EARTH):
    """计算两点间的大圆距离（单位：米）"""
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def geodetic_to_ecef(lat, lon, alt):
    """
    将经纬度高度（弧度，米）转为 ECEF 坐标（米）
    """
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    N = R_EARTH / np.sqrt(1 - ECC2 * sin_lat ** 2)

    x = (N + alt) * cos_lat * cos_lon
    y = (N + alt) * cos_lat * sin_lon
    z = (N * (1 - ECC2) + alt) * sin_lat

    return np.array([x, y, z])