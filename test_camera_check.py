"""카메라 없이 find_objects 확인: 기울어진 테이블 위에 상자 2개를 합성한다."""
import numpy as np

from camera_check import find_objects


def synthetic_cloud():
    h, w = 200, 300
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    x, y = (u - w / 2) * 2.0, (v - h / 2) * 2.0   # 2mm/px
    z = 800.0 + 0.1 * x                           # 약간 기울어진 테이블, 카메라에서 ~800mm
    z[50:90, 40:100] -= 30.0                      # 높이 30mm 상자
    z[120:170, 180:260] -= 60.0                   # 높이 60mm 상자
    cloud = np.dstack([x, y, z])
    cloud[0:5, 0:5] = np.nan                      # 무효 점
    return cloud


def test_finds_two_boxes():
    objects, _ = find_objects(synthetic_cloud())
    assert len(objects) == 2, objects
    heights = sorted(o["height_mm"] for o in objects)
    assert abs(heights[0] - 30) < 3 and abs(heights[1] - 60) < 6, heights


if __name__ == "__main__":
    test_finds_two_boxes()
    print("ok")
