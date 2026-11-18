"""Mech-Eye 카메라로 한 장 촬영해서 바닥(테이블) 위 물체를 찾는지 확인한다.

실행: python camera_check.py [카메라 IP]
결과: captures/ 에 color.png, depth.png, detect.png, cloud.ply 저장 + 물체 목록 출력
"""
import sys
from pathlib import Path

import cv2
import numpy as np

MIN_HEIGHT_MM = 5.0   # 테이블 면에서 이만큼 솟아 있어야 물체로 본다
MIN_AREA_PX = 300     # 이보다 작은 덩어리는 노이즈로 버린다


def fit_plane(pts, iters=200, tol=3.0, seed=0):
    """RANSAC으로 가장 큰 평면(테이블)을 찾는다. 반환: (법선 n, d), n·p + d = 0."""
    rng = np.random.default_rng(seed)
    sample = pts[rng.choice(len(pts), min(len(pts), 20000), replace=False)]
    best, best_count = None, -1
    for _ in range(iters):
        a, b, c = sample[rng.choice(len(sample), 3, replace=False)]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-6:
            continue
        n /= norm
        d = -n @ a
        count = np.count_nonzero(np.abs(sample @ n + d) < tol)
        if count > best_count:
            best, best_count = (n, d), count
    n, d = best
    if d < 0:  # 카메라(원점) 쪽이 +가 되도록: 물체 높이가 양수로 나온다
        n, d = -n, -d
    return n, d


def find_objects(cloud):
    """정렬된 포인트클라우드 (H, W, 3) mm → 물체 목록과 라벨 이미지."""
    valid = np.isfinite(cloud).all(axis=2) & (cloud[..., 2] > 0)
    if np.count_nonzero(valid) < 100:
        raise RuntimeError("유효한 3D 점이 거의 없습니다. 노출/거리/조명을 확인하세요.")
    n, d = fit_plane(cloud[valid])

    height = np.full(valid.shape, np.nan, np.float32)
    height[valid] = cloud[valid] @ n + d
    mask = (np.nan_to_num(height) > MIN_HEIGHT_MM).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    objects = []
    for i in range(1, count):
        x, y, w, h, area = stats[i]
        if area < MIN_AREA_PX:
            continue
        sel = labels == i
        objects.append({
            "center_mm": cloud[sel].mean(axis=0),        # 카메라 좌표계 기준 중심
            "height_mm": float(np.nanmax(height[sel])),  # 테이블에서 가장 높은 점
            "bbox_px": (int(x), int(y), int(w), int(h)),
            "area_px": int(area),
        })
    return objects, height


def draw(height, objects):
    vis = cv2.applyColorMap(
        cv2.normalize(np.nan_to_num(height), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_JET)
    for k, o in enumerate(objects):
        x, y, w, h = o["bbox_px"]
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 255, 255), 2)
        cv2.putText(vis, f"#{k} h={o['height_mm']:.0f}mm", (x, max(y - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


def connect(ip=None):
    from mecheye.area_scan_3d_camera import Camera

    camera = Camera()
    if ip:
        status = camera.connect(ip)
    else:
        infos = Camera.discover_cameras()
        if not infos:
            raise RuntimeError("카메라를 찾지 못했습니다. 전원, 랜선, PC IP 대역을 확인하세요.")
        print(f"연결: {infos[0].model} ({infos[0].serial_number}) @ {infos[0].ip_address}")
        status = camera.connect(infos[0])
    if not status.is_ok():
        raise RuntimeError(f"연결 실패: {status.error_description}")
    return camera


def capture(ip=None):
    from mecheye.area_scan_3d_camera import Frame2DAnd3D, ColorTypeOf2DCamera_Monochrome
    from mecheye.shared import FileFormat_PLY

    camera = connect(ip)
    try:
        frame = Frame2DAnd3D()
        status = camera.capture_2d_and_3d(frame)
        if not status.is_ok():
            raise RuntimeError(f"촬영 실패: {status.error_description}")
        f2d, f3d = frame.frame_2d(), frame.frame_3d()
        mono = f2d.color_type() == ColorTypeOf2DCamera_Monochrome
        image = (f2d.get_gray_scale_image() if mono else f2d.get_color_image()).data().copy()
        depth = f3d.get_depth_map().data().copy()
        cloud = f3d.get_untextured_point_cloud().data().copy()  # data()는 프레임 메모리를 가리켜서 복사
        out = Path("captures")
        out.mkdir(exist_ok=True)
        f3d.save_untextured_point_cloud(FileFormat_PLY, str(out / "cloud.ply"))
        return image, depth, cloud
    finally:
        camera.disconnect()


def main():
    image, depth, cloud = capture(sys.argv[1] if len(sys.argv) > 1 else None)
    out = Path("captures")
    cv2.imwrite(str(out / "color.png"), image)
    cv2.imwrite(str(out / "depth.png"), cv2.applyColorMap(
        cv2.normalize(np.nan_to_num(depth), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8),
        cv2.COLORMAP_JET))

    objects, height = find_objects(cloud)
    cv2.imwrite(str(out / "detect.png"), draw(height, objects))

    print(f"물체 {len(objects)}개 발견")
    for k, o in enumerate(objects):
        x, y, z = o["center_mm"]
        print(f"  #{k}: 중심 ({x:.1f}, {y:.1f}, {z:.1f}) mm, 높이 {o['height_mm']:.1f} mm, "
              f"bbox {o['bbox_px']}")
    print(f"결과 이미지: {out.resolve()}")


if __name__ == "__main__":
    main()
