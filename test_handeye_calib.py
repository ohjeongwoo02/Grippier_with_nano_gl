"""카메라/로봇 없이 핸드-아이 계산 확인."""
import cv2
import numpy as np

from handeye_calib import (BOARD, SQUARE_MM, _axis, board_pose, calib_poses, cam_to_base, euler_zyx, far_enough,
                           lookat_poses, move_cmd, pose_to_T, solve)


def test_euler_zyx_roundtrip():
    for pose in ([0, 0, 0, -171.6, 3.4, -110.5], [0, 0, 0, 20, -35, 150], [0, 0, 0, 179, 10, -179]):
        back = euler_zyx(pose_to_T(pose, "Rz*Ry*Rx (deg)")[:3, :3])
        assert np.allclose(pose_to_T([0, 0, 0, *back], "Rz*Ry*Rx (deg)"), pose_to_T(pose, "Rz*Ry*Rx (deg)"), atol=1e-9)


def test_lookat_keeps_board_center_in_view():
    conv = "Rz*Ry*Rx (deg)"
    X = pose_to_T([30, -10, 80, 5, -3, 90], conv)
    board_in_base = pose_to_T([380, -140, 0, 0, 0, 20], conv)
    pose0 = [373, -138, 330, -171.6, 3.4, -110.5]
    Tc0 = np.linalg.inv(X) @ np.linalg.inv(pose_to_T(pose0, conv)) @ board_in_base
    calib = {"convention": conv, "T_flange_cam": X.tolist()}
    center = board_in_base @ np.array([7 * SQUARE_MM / 2, 5 * SQUARE_MM / 2, 0, 1])
    c0 = np.linalg.inv(pose_to_T(pose0, conv) @ X) @ center
    poses = lookat_poses([{"pose": pose0, "T_cam_board": Tc0.tolist()}], calib, lift=0)
    for p in poses:
        c = np.linalg.inv(pose_to_T(p, conv) @ X) @ center           # 보드 중심을 카메라 좌표로
        assert np.allclose(c, c0, atol=1e-6), (c, c0)                # 처음과 같은 화면 위치에 계속 보임
        assert all(abs(a - b) <= 180 for a, b in zip(p[3:], pose0[3:]))  # 각도가 기준에 연속
    assert max(abs((a - b + 180) % 360 - 180) for p in poses for a, b in zip(p[3:], pose0[3:])) > 15  # 실제로 크게 기울임


def test_calib_poses_stay_safe():
    base = [373.2, -138.5, 265.7, -171.6, 3.4, 178.0]
    for p in calib_poses(base):
        assert p[2] >= base[2]                                   # 아래로 내려가지 않음
        assert max(abs(p[0] - base[0]), abs(p[1] - base[1])) <= 40
        assert all(-180 <= a < 180 for a in p[3:])               # 각도 범위 정리됨 (178+15 → -167)
        assert not far_enough(p, base, mm=80, deg=16)             # 기준에서 크게 벗어나지 않음
    assert move_cmd(base).endswith(", 20, 80)")                  # 느린 속도로 명령


def test_far_enough():
    base = [0, 0, 0, 179, 0, 0]
    assert not far_enough(base, [5, 0, 0, -179, 0, 0])   # 179°와 -179°는 2° 차이
    assert far_enough(base, [40, 0, 0, 179, 0, 0])       # 40 mm 이동
    assert far_enough(base, [0, 0, 0, 179, 15, 0])       # 15° 회전


def test_solve_recovers_flange_to_camera():
    rng = np.random.default_rng(1)
    conv = "Rz*Ry*Rx (deg)"
    X = pose_to_T([30, -10, 80, 5, -3, 90], conv)          # 정답: 플랜지 ← 카메라
    board_in_base = pose_to_T([500, 0, 0, 0, 0, 20], conv)
    samples = []
    for _ in range(12):
        pose = [500 + rng.uniform(-80, 80), rng.uniform(-80, 80), rng.uniform(350, 450),
                180 + rng.uniform(-25, 25), rng.uniform(-25, 25), rng.uniform(-60, 60)]
        Tc = np.linalg.inv(X) @ np.linalg.inv(pose_to_T(pose, conv)) @ board_in_base
        samples.append({"pose": pose, "T_cam_board": Tc.tolist()})

    best_conv, X_est, err, sign = solve(samples)
    assert best_conv == conv and sign == 0, (best_conv, sign)
    assert err < 0.1, err
    assert np.allclose(X_est[:3, :3], X[:3, :3], atol=1e-3) and np.allclose(X_est[:3, 3], X[:3, 3], atol=0.1)

    # 카메라가 본 보드 원점 → 베이스 좌표가 실제 보드 위치와 같아야 함
    calib = {"convention": conv, "T_flange_cam": X_est.tolist()}
    p_base = cam_to_base(np.array(samples[0]["T_cam_board"])[:3, 3], samples[0]["pose"], calib)
    assert np.allclose(p_base, board_in_base[:3, 3], atol=0.5), p_base


def test_solve_camera_mounted_before_j6():
    """카메라가 J6 앞쪽에 고정: 플랜지가 J6로 돌아도 카메라는 안 돈다. J6를 빼는 방식이 정답을 찾아야 함."""
    rng = np.random.default_rng(2)
    conv = "Rz*Ry*Rx (deg)"
    X = pose_to_T([-8, -66, -150, 2, -5, 25], conv)          # 정답: 마운트 ← 카메라
    board_in_base = pose_to_T([380, -140, 0, 0, 0, 20], conv)
    samples = []
    for _ in range(14):
        pose = [380 + rng.uniform(-60, 60), -140 + rng.uniform(-60, 60), rng.uniform(260, 320),
                180 + rng.uniform(-20, 20), rng.uniform(-20, 20), rng.uniform(-150, -70)]
        q6 = rng.uniform(-40, 80)
        s = {"pose": pose, "joints": [0, 0, 0, 0, 0, q6]}
        mount = pose_to_T(pose, conv) @ np.r_[np.c_[_axis(2, -q6), [0, 0, 0]], [[0, 0, 0, 1]]]
        s["T_cam_board"] = (np.linalg.inv(X) @ np.linalg.inv(mount) @ board_in_base).tolist()
        samples.append(s)

    best_conv, X_est, err, sign = solve(samples)
    assert best_conv == conv and sign == 1, (best_conv, sign)
    assert err < 0.1, err
    calib = {"convention": conv, "q6_sign": sign, "T_flange_cam": X_est.tolist()}
    p_base = cam_to_base(np.array(samples[3]["T_cam_board"])[:3, 3], samples[3]["pose"], calib, samples[3]["joints"])
    assert np.allclose(p_base, board_in_base[:3, 3], atol=0.5), p_base
    shifted = cam_to_base(np.array(samples[3]["T_cam_board"])[:3, 3], samples[3]["pose"],
                          {**calib, "base_offset_mm": [4, -0.5, 0]}, samples[3]["joints"])
    assert np.allclose(shifted - p_base, [4, -0.5, 0])  # 보정값이 그대로 더해짐
    try:
        cam_to_base([0, 0, 0], samples[0]["pose"], calib)
    except ValueError:
        pass
    else:
        raise AssertionError("J6 각도 없이 호출하면 거부해야 함")


def test_board_pose_on_flat_cloud():
    px, margin = 4, 40  # 이미지 4 px/mm, 평면 z=500mm에 보드를 그대로 깔았다고 가정
    gray = cv2.copyMakeBorder(BOARD.generateImage((int(7 * SQUARE_MM * px), int(5 * SQUARE_MM * px))),
                              *[margin] * 4, cv2.BORDER_CONSTANT, value=255)
    h, w = gray.shape
    v, u = np.mgrid[0:h, 0:w].astype(np.float32)
    cloud = np.dstack([u / px, v / px, np.full_like(u, 500.0)])
    T, rms = board_pose(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), cloud)
    assert rms < 0.5, rms
    assert abs(T[2, 2]) > 0.99 and abs(T[2, 3] - 500) < 1, T


if __name__ == "__main__":
    test_far_enough()
    test_calib_poses_stay_safe()
    test_euler_zyx_roundtrip()
    test_lookat_keeps_board_center_in_view()
    test_solve_recovers_flange_to_camera()
    test_solve_camera_mounted_before_j6()
    test_board_pose_on_flat_cloud()
    print("ok")
