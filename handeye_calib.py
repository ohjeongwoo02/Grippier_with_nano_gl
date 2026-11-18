"""손목 카메라(eye-in-hand) 핸드-아이 캘리브레이션: 플랜지 → 카메라 변환을 구한다.

1) python handeye_calib.py board          board.png 생성 → 100% 크기로 인쇄, 평평한 판에 붙여 작업대에 고정
                                          인쇄 후 칸 크기를 자로 재서 SQUARE_MM, MARKER_MM 수정
2) python handeye_calib.py robot          로봇을 자동으로 15개 자세로 천천히 옮기며 촬영 + 실시간 창 (ESC = 즉시 정지)
   python handeye_calib.py auto [개수]     로봇을 펜던트로 움직여 멈출 때마다 자동 촬영 (플랜지 좌표는 로봇에서 자동으로 읽음)
   python handeye_calib.py capture [IP]   (수동) 자세마다 펜던트의 플랜지 좌표를 직접 입력 → 촬영
3) python handeye_calib.py solve          handeye.json 저장 + 오차(mm) 출력

이후 cam_to_base()로 camera_check.py의 물체 좌표를 로봇 베이스 좌표로 바꾼다.
"""
import json
import socket
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

SQUARES_X, SQUARES_Y = 7, 5
SQUARE_MM, MARKER_MM = 24.0, 17.6  # 인쇄된 실제 크기 (80% 인쇄: 30x0.8, 22x0.8)
BOARD = cv2.aruco.CharucoBoard((SQUARES_X, SQUARES_Y), SQUARE_MM, MARKER_MM,
                               cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))
SAMPLES, RESULT = Path("calib_samples.json"), Path("handeye.json")
# 로봇 내부 상태 스트림 (공식 문서 없음: 접속하면 JSON이 \x0c로 구분돼 흘러나오는 걸 읽기만 확인함)
ROBOT_HOST, ROBOT_STATE_PORT = "192.168.10.100", 8001
# 게이트웨이 스크립트 명령 포트 (Rodi RemoteCommand 플러그인이 쓰는 창구: 스크립트 한 줄 + \n → JSON ACK)
ROBOT_CMD_PORT = 20001
SLOW_V, SLOW_A = 20, 80  # moveLinear 최대 속도 mm/s, 가속도 mm/s^2 (기본값 100, 1000)


def _axis(i, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    j, k = [(1, 2), (2, 0), (0, 1)][i]
    R = np.eye(3)
    R[j, j], R[j, k], R[k, j], R[k, k] = c, -s, s, c
    return R


# HCR 펜던트의 rx ry rz 표기법을 확인하지 못해서, solve가 후보를 모두 시험하고 오차가 가장 작은 걸 고른다
CONVENTIONS = {
    "Rz*Ry*Rx (deg)": lambda r: _axis(2, r[2]) @ _axis(1, r[1]) @ _axis(0, r[0]),
    "Rx*Ry*Rz (deg)": lambda r: _axis(0, r[0]) @ _axis(1, r[1]) @ _axis(2, r[2]),
    "ZYZ (deg)": lambda r: _axis(2, r[0]) @ _axis(1, r[1]) @ _axis(2, r[2]),
    "rotvec (rad)": lambda r: cv2.Rodrigues(np.asarray(r, float))[0],
}


def pose_to_T(pose, convention):
    """펜던트 좌표 [x, y, z(mm), rx, ry, rz] → 4x4 (베이스 ← 플랜지)."""
    T = np.eye(4)
    T[:3, :3] = CONVENTIONS[convention](pose[3:])
    T[:3, 3] = pose[:3]
    return T


def kabsch(a, b):
    """b ≈ R a + t 인 R, t."""
    ca, cb = a.mean(0), b.mean(0)
    U, _, Vt = np.linalg.svd((a - ca).T @ (b - cb))
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    return R, cb - R @ ca


def board_pose(colors, cloud):
    """정렬된 컬러 (H,W,3) + 포인트클라우드 (H,W,3) → 4x4 (카메라 ← 보드), 맞춤 오차 mm.

    2D로 코너를 찾고 같은 픽셀의 3D 점을 바로 써서 내부 파라미터가 필요 없다."""
    corners, ids, _, _ = cv2.aruco.CharucoDetector(BOARD).detectBoard(
        cv2.cvtColor(colors, cv2.COLOR_BGR2GRAY))
    if ids is None or len(ids) < 6:
        raise ValueError("보드 코너를 6개 이상 찾지 못했습니다. 보드가 화면에 다 들어오는지 확인하세요.")
    model = np.asarray(BOARD.getChessboardCorners(), float)[ids.ravel()]
    h, w = cloud.shape[:2]
    meas = []
    for u, v in corners.reshape(-1, 2):
        c, r = int(round(u)), int(round(v))
        patch = cloud[max(r - 1, 0):min(r + 2, h), max(c - 1, 0):min(c + 2, w)].reshape(-1, 3)
        patch = patch[np.isfinite(patch).all(1)]  # 3x3 평균으로 노이즈 완화, 무효 점 제외
        meas.append(patch.mean(0) if len(patch) else [np.nan] * 3)
    meas = np.array(meas)
    ok = np.isfinite(meas).all(1)
    if ok.sum() < 6:
        raise ValueError("코너 위치의 3D 점이 비어 있습니다. 노출이나 거리를 확인하세요.")
    R, t = kabsch(model[ok], meas[ok])
    rms = float(np.sqrt(np.mean(np.sum((model[ok] @ R.T + t - meas[ok]) ** 2, axis=1))))
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T, rms


def mount_T(sample, convention, q6_sign):
    """카메라가 붙은 부분의 자세 (베이스 ← 마운트).
    카메라가 J6 앞쪽(J6가 돌아도 안 도는 곳)에 달린 경우: 플랜지 자세에서 J6 회전을 뺀다 (J6 축 = 플랜지 z축 가정).
    샘플에 관절각이 없으면(q6_sign=0) 플랜지 자체."""
    T = pose_to_T(sample["pose"], convention)
    if q6_sign:
        U = np.eye(4)
        U[:3, :3] = _axis(2, -q6_sign * sample["joints"][5])
        T = T @ U
    return T


def solve(samples):
    """→ (표기법, 4x4 마운트 ← 카메라, 오차 mm, J6 부호). 오차 = 각 촬영에서 계산한 보드 코너 위치(베이스 기준)가 흩어진 정도.
    샘플에 관절각이 있으면 J6 회전을 뺀 경우(부호 +/-)와 플랜지 그대로인 경우를 모두 시험한다."""
    if len(samples) < 3:
        raise ValueError("샘플이 3개 이상 필요합니다 (10개 이상 권장).")
    corners = np.c_[np.asarray(BOARD.getChessboardCorners(), float), np.ones(len(BOARD.getChessboardCorners()))]
    Tc = [np.array(s["T_cam_board"]) for s in samples]
    methods = {"TSAI": cv2.CALIB_HAND_EYE_TSAI, "PARK": cv2.CALIB_HAND_EYE_PARK, "HORAUD": cv2.CALIB_HAND_EYE_HORAUD,
               "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF, "DANIILIDIS": cv2.CALIB_HAND_EYE_DANIILIDIS}
    best = None
    signs = (0, 1, -1) if all("joints" in s for s in samples) else (0,)
    for conv, sign in [(c, g) for c in CONVENTIONS for g in signs]:
        Tb = [mount_T(s, conv, sign) for s in samples]
        errs = {}
        for name, m in methods.items():  # 데이터에 따라 잘 맞는 방법이 달라서 모두 시험
            try:
                R, t = cv2.calibrateHandEye([T[:3, :3] for T in Tb], [T[:3, 3] for T in Tb],
                                            [T[:3, :3] for T in Tc], [T[:3, 3] for T in Tc], method=m)
            except cv2.error:
                continue
            X = np.eye(4)
            X[:3, :3], X[:3, 3] = R, t.ravel()
            pts = np.array([(B @ X @ C @ corners.T).T[:, :3] for B, C in zip(Tb, Tc)])
            err = float(np.linalg.norm(pts - pts.mean(0), axis=2).mean())
            if not np.isfinite(err):
                continue
            errs[name] = err
            if best is None or err < best[2]:
                best = (conv, X, err, sign)
        label = conv + ("" if not sign else f", J6 {'+' if sign > 0 else '-'}")
        print(f"  {label:22s} " + "  ".join(f"{k} {v:.2f}" for k, v in errs.items()) + "  (mm)")
    return best


def euler_zyx(R):
    """R = Rz(rz)·Ry(ry)·Rx(rx) 의 (rx, ry, rz) [도]. solve가 고른 HCR 표기법('Rz*Ry*Rx (deg)')의 역변환."""
    ry = np.degrees(np.arcsin(-np.clip(R[2, 0], -1, 1)))
    rx = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
    rz = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    return [rx, ry, rz]


def lookat_poses(samples, calib, tilt=20.0, spin=35.0, lift=30.0):
    """대략 구한 카메라 위치로, 보드 중심을 계속 바라보며 크게 기울인 플랜지 자세들을 만든다.
    카메라를 보드 중심을 축으로 tilt°만큼 앞뒤좌우로 돌리고, 광축 둘레로 ±spin° 돌린다. lift mm만큼 위로 띄워 여유를 둔다."""
    assert calib["convention"] == "Rz*Ry*Rx (deg)", "euler_zyx는 이 표기법 전용"
    X = np.array(calib["T_flange_cam"])
    Tb0 = pose_to_T(samples[0]["pose"], calib["convention"])
    C0 = Tb0 @ X                                                   # 첫 샘플의 카메라 자세 (베이스 기준)
    board = Tb0 @ X @ np.array(samples[0]["T_cam_board"])
    center = (board @ np.array([SQUARES_X * SQUARE_MM / 2, SQUARES_Y * SQUARE_MM / 2, 0, 1]))[:3]

    def rot(axis, deg):
        return cv2.Rodrigues(np.asarray(axis, float) / np.linalg.norm(axis) * np.radians(deg))[0]

    cam_z = C0[:3, 2]                                               # 카메라 광축
    side = np.cross(cam_z, [0, 0, 1]); side /= np.linalg.norm(side)
    up = np.cross(cam_z, side)
    moves = [(side, tilt, 0), (side, -tilt, 0), (up, tilt, 0), (up, -tilt, 0),
             (side, tilt, spin), (side, -tilt, -spin), (up, tilt, -spin), (up, -tilt, spin),
             (side, 0, spin), (side, 0, -spin)]
    poses = []
    for axis, t_deg, s_deg in moves:
        R = rot(axis, t_deg) @ rot(cam_z, s_deg)
        C = np.eye(4)
        C[:3, :3] = R @ C0[:3, :3]
        C[:3, 3] = center + R @ (C0[:3, 3] - center)               # 보드 중심을 축으로 회전
        F = C @ np.linalg.inv(X)
        # 각도를 기준 자세에 연속되게 (177.7 대신 -182.3): ±180 경계에서 손목이 반대로 크게 도는 걸 막는다
        ang = [b + (a - b + 180) % 360 - 180 for a, b in zip(euler_zyx(F[:3, :3]), samples[0]["pose"][3:])]
        poses.append([*(F[:3, 3] + [0, 0, lift]), *ang])
    return poses


def cam_to_base(p_cam, pose, calib, joints=None):
    """카메라 좌표(mm) → 로봇 베이스 좌표(mm). pose는 촬영한 순간의 플랜지 좌표, joints는 그 순간의 관절각
    (캘리브레이션이 J6를 뺀 방식(q6_sign≠0)이면 필요)."""
    sign = calib.get("q6_sign", 0)
    if sign and joints is None:
        raise ValueError("이 캘리브레이션은 J6 각도가 필요합니다 (카메라가 J6 앞쪽에 장착됨).")
    T = mount_T({"pose": pose, "joints": joints}, calib["convention"], sign) @ np.array(calib["T_flange_cam"])
    # base_offset_mm: 실물 검증에서 눈으로 맞춘 차이 (한 점 기준 상수 보정)
    return (T @ np.append(p_cam, 1.0))[:3] + np.asarray(calib.get("base_offset_mm", [0, 0, 0]), float)


def far_enough(a, b, mm=30.0, deg=10.0):
    """두 플랜지 자세가 새 샘플로 쓸 만큼 다른가 (각도는 ±180 경계를 넘어도 올바르게 비교)."""
    dpos = np.linalg.norm(np.subtract(a[:3], b[:3]))
    drot = np.abs((np.subtract(a[3:], b[3:]) + 180) % 360 - 180).max()
    return dpos > mm or drot > deg


class RobotPose:
    """로봇이 내보내는 motion/flange/position 메시지를 계속 읽어 최신 플랜지 좌표를 들고 있는다 (읽기 전용)."""

    def __init__(self, host=ROBOT_HOST, port=ROBOT_STATE_PORT):
        self.sock = socket.create_connection((host, port), timeout=3)
        self.history = []  # (시각, [x y z rx ry rz])
        self.joints = None  # 최신 관절각 [base shoulder elbow wrist1 wrist2 wrist3] (도)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        buf = b""
        while True:
            try:
                d = self.sock.recv(65536)
            except socket.timeout:  # 잠깐 조용한 것뿐: 끊긴 게 아니다 (이걸 끊김으로 처리하면 좌표가 멈춘다)
                continue
            except OSError:
                return
            if not d:
                return
            *msgs, buf = (buf + d).split(b"\x0c")
            for m in msgs:
                if b"motion/flange/position" not in m and b"motion/joint/position" not in m:
                    continue
                try:
                    j = json.loads(m)
                    p = j["data"]["data"]["data"]
                    if j.get("type") == "motion/joint/position":
                        self.joints = [float(p[k]) for k in ("base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3")]
                        continue
                    if j.get("type") != "motion/flange/position":
                        continue
                    pose = [float(p[k]) for k in ("x", "y", "z", "rx", "ry", "rz")]
                except (ValueError, KeyError, TypeError):  # 모양이 다른 메시지 하나 때문에 읽기가 죽지 않게
                    continue
                self.history = self.history[-50:] + [(time.time(), pose)]

    def still_pose(self, window=1.0):
        """최근 window초 동안 거의 안 움직였으면 그 자세, 아니면 None."""
        now = time.time()
        recent = [p for t, p in self.history if now - t <= window]
        if len(recent) < 3 or now - self.history[-1][0] > 0.5:
            return None
        return recent[-1] if not far_enough(recent[0], recent[-1], mm=0.5, deg=0.2) else None


def shoot(camera, on_image=None):
    """한 장 찍어서 (카메라 ← 보드) 4x4와 맞춤 오차를 돌려준다. 실패하면 ValueError."""
    from mecheye.area_scan_3d_camera import Frame2DAnd3D

    frame = Frame2DAnd3D()
    status = camera.capture_2d_and_3d(frame)
    if not status.is_ok():
        raise ValueError(f"촬영 실패: {status.error_description}")
    tpc = frame.get_textured_point_cloud()
    colors = tpc.colors()
    if on_image:
        on_image(colors)
    return board_pose(colors, tpc.vertices())


def grab_2d(camera):
    """미리보기용 2D 한 장 (3D보다 훨씬 빠름)."""
    from mecheye.area_scan_3d_camera import Frame2D, ColorTypeOf2DCamera_Monochrome

    f = Frame2D()
    if not camera.capture_2d(f).is_ok():
        return None
    if f.color_type() == ColorTypeOf2DCamera_Monochrome:
        return cv2.cvtColor(f.get_gray_scale_image().data().copy(), cv2.COLOR_GRAY2BGR)
    return f.get_color_image().data().copy()


def send_script(line, timeout=10):
    """로봇 게이트웨이에 Rodi 스크립트 한 줄을 보내고 ACK를 확인한다. 응답이 늦으면 socket.timeout."""
    with socket.create_connection((ROBOT_HOST, ROBOT_CMD_PORT), timeout=timeout) as s:
        s.sendall((line + "\n").encode())
        data, reply = b"", None
        while reply is None:  # 응답이 여러 조각으로 와도 JSON이 완성될 때까지 받는다
            chunk = s.recv(4096)
            if not chunk:
                raise RuntimeError(f"로봇이 응답 없이 연결을 닫았습니다: {data!r}")
            data += chunk
            try:
                reply = json.loads(data.decode("utf-8", "replace"))
            except ValueError:
                continue
    if not reply.get("isSuccess"):
        raise RuntimeError(f"로봇이 명령을 거부했습니다: {reply}")
    return reply


def move_cmd(pose, v=SLOW_V, a=SLOW_A):
    x, y, z, rx, ry, rz = pose
    return f"moveLinear('flange', {{x:{x:.2f}, y:{y:.2f}, z:{z:.2f}, rx:{rx:.2f}, ry:{ry:.2f}, rz:{rz:.2f}}}, {v}, {a})"


def calib_poses(base):
    """base(현재 자세) 주변 15개 자세. 안전: 아래로는 내려가지 않고(dz>=0), 이동 ±40 mm, 기울기 ±8°, 손목 ±15° 이내."""
    deltas = [  # dx dy dz drx dry drz
        (0, 0, 0, 0, 0, 0), (40, 0, 0, 0, 0, 0), (-40, 0, 0, 0, 0, 0), (0, 40, 0, 0, 0, 0), (0, -40, 0, 0, 0, 0),
        (0, 0, 40, 0, 0, 0), (0, 0, 0, 8, 0, 0), (0, 0, 0, -8, 0, 0), (0, 0, 0, 0, 8, 0), (0, 0, 0, 0, -8, 0),
        (0, 0, 0, 0, 0, 15), (0, 0, 0, 0, 0, -15), (25, 25, 20, 6, -6, 10), (-25, -25, 20, -6, 6, -10),
        (25, -25, 30, -6, -6, 12),
    ]
    wrap = lambda a: (a + 180) % 360 - 180
    return [[base[0] + d[0], base[1] + d[1], base[2] + d[2],
             wrap(base[3] + d[3]), wrap(base[4] + d[4]), wrap(base[5] + d[5])] for d in deltas]


def save_sample(samples, pose, T, rms, joints=None):
    samples.append({"pose": pose, "T_cam_board": T.tolist(), **({"joints": joints} if joints else {})})
    SAMPLES.write_text(json.dumps(samples, indent=1))  # 매번 저장해서 중간에 끊겨도 안 날아감
    print(f"#{len(samples)} 저장  자세 {np.round(pose, 1).tolist()}  (보드 맞춤 오차 {rms:.2f} mm)", flush=True)


def capture_samples(ip=None):
    from camera_check import connect

    samples = json.loads(SAMPLES.read_text()) if SAMPLES.exists() else []
    print(f"기존 샘플 {len(samples)}개. 매번 보드가 보이게 위치와 각도를 크게 바꿔 주세요.")
    camera = connect(ip)
    try:
        while True:
            line = input("로봇을 멈춘 뒤 플랜지 좌표 'x y z rx ry rz' 입력 (엔터만 치면 종료): ").split()
            if not line:
                break
            try:
                pose = [float(v) for v in line]
                assert len(pose) == 6
            except (ValueError, AssertionError):
                print("숫자 6개를 공백으로 띄워 입력하세요.")
                continue
            try:
                save_sample(samples, pose, *shoot(camera))
            except ValueError as e:
                print(e)
    finally:
        camera.disconnect()


def auto_capture(target=15, timeout_s=1200):
    """로봇이 새 자세에서 1초 이상 멈추면 자동으로 촬영. 로봇은 사람이 펜던트로 움직인다."""
    from camera_check import connect

    samples = json.loads(SAMPLES.read_text()) if SAMPLES.exists() else []
    robot = RobotPose()
    camera = connect()
    last = samples[-1]["pose"] if samples else None
    print(f"기존 샘플 {len(samples)}개, 목표 {target}개. 로봇을 새 자세로 옮기고 1~2초 멈춰 주세요.", flush=True)
    end = time.time() + timeout_s
    try:
        while len(samples) < target and time.time() < end:
            time.sleep(0.2)
            pose = robot.still_pose()
            if pose is None or (last is not None and not far_enough(pose, last)):
                continue
            try:
                save_sample(samples, pose, *shoot(camera), joints=robot.joints)
            except ValueError as e:
                print(f"건너뜀: {e}", flush=True)
            last = pose  # 실패해도 같은 자세에서 반복 촬영하지 않게
    finally:
        camera.disconnect()
    print(f"끝: 샘플 {len(samples)}개", flush=True)


def robot_calib(timeout_per_move=60, poses=None):
    """로봇을 calib_poses(또는 주어진 poses)로 천천히 옮기며 촬영. 메인 스레드는 실시간 창(좌표 + 카메라), ESC = 정지."""
    from camera_check import connect

    samples = json.loads(SAMPLES.read_text()) if SAMPLES.exists() else []
    robot = RobotPose()
    time.sleep(1.5)
    # 다시 실행할 때도 같은 자세 세트를 쓰도록, 기존 샘플이 있으면 첫 샘플 자세를 기준으로 한다
    base = samples[0]["pose"] if samples else robot.still_pose()
    if base is None:
        raise RuntimeError("로봇이 멈춰 있지 않거나 좌표를 못 읽었습니다.")
    poses = poses or calib_poses(base)
    cam = {"c": connect()}  # 끊기면 다시 연결해서 바꿔 끼운다
    st = {"frame": None, "status": "start", "abort": False, "done": False}

    def reconnect():
        try:
            cam["c"].disconnect()
        except Exception:
            pass
        cam["c"] = connect()

    def shoot_retry():
        try:
            return shoot(cam["c"], lambda im: st.update(frame=im))
        except ValueError as e:
            if not str(e).startswith("촬영 실패"):
                raise
            print(f"카메라 오류, 다시 연결합니다: {e}", flush=True)
            reconnect()
            return shoot(cam["c"], lambda im: st.update(frame=im))

    def go(target):
        """이동 명령 + 도착 확인. ACK가 늦어도 실제 도착으로 판단하고, 안 오면 같은 절대 목표를 한 번 더 보낸다."""
        for attempt in (1, 2):
            try:
                send_script(move_cmd(target))
            except socket.timeout:
                print(f"로봇 응답 지연 (시도 {attempt}), 실제 이동 여부로 확인합니다", flush=True)
            if wait_arrive(target):
                return True
            if st["abort"]:
                return False
        return False

    def wait_arrive(target):
        end = time.time() + timeout_per_move
        while time.time() < end and not st["abort"]:
            img = grab_2d(cam["c"])
            if img is not None:
                st["frame"] = img
            cur = robot.history[-1][1]
            if not far_enough(cur, target, mm=1.0, deg=0.5) and robot.still_pose() is not None:
                return True
            time.sleep(0.1)
        return False

    def work():
        try:
            for i, pose in enumerate(poses):
                if st["abort"]:
                    break
                if any(not far_enough(s["pose"], pose, mm=1.0, deg=0.5) for s in samples):
                    continue  # 이미 찍은 자세 (다시 실행할 때 중복 방지)
                st["status"] = f"moving {i + 1}/{len(poses)}"
                print(st["status"], flush=True)
                if not go(pose):
                    if not st["abort"]:
                        st["status"] = f"timeout at pose {i + 1} -> stopped"
                        print(st["status"], flush=True)
                        send_script("moveStop()")
                        st["abort"] = True
                    break
                time.sleep(0.5)
                st["status"] = f"capturing {i + 1}/{len(poses)}"
                try:
                    save_sample(samples, robot.history[-1][1], *shoot_retry(), joints=robot.joints)
                except ValueError as e:
                    print(f"pose {i + 1} 건너뜀: {e}", flush=True)
            if not st["abort"]:
                st["status"] = "returning to start"
                go(base)
                st["status"] = f"done: {len(samples)} samples (press any key)"
        except Exception as e:  # 통신 오류 등: 로봇을 세우고 화면에 표시
            st["status"] = f"error: {e}"
            print(st["status"], flush=True)
            try:
                send_script("moveStop()")
            except Exception:
                pass
        finally:
            st["done"] = True

    threading.Thread(target=work, daemon=True).start()
    detector = cv2.aruco.CharucoDetector(BOARD)
    try:
        while True:
            img = st["frame"].copy() if st["frame"] is not None else np.zeros((1024, 1280, 3), np.uint8)
            corners, ids, _, _ = detector.detectBoard(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            if ids is not None:
                cv2.aruco.drawDetectedCornersCharuco(img, corners, ids, (0, 255, 0))
            p = robot.history[-1][1] if robot.history else [0] * 6
            lines = [f"flange  x {p[0]:8.1f}  y {p[1]:8.1f}  z {p[2]:8.1f}",
                     f"        rx {p[3]:7.1f}  ry {p[4]:7.1f}  rz {p[5]:7.1f}",
                     f"board corners: {0 if ids is None else len(ids)}   samples: {len(samples)}",
                     st["status"], "ESC = STOP ROBOT"]
            for k, t in enumerate(lines):
                y = 40 + 38 * k
                cv2.putText(img, t, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
                cv2.putText(img, t, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255) if k < 3 else (0, 128, 255), 2)
            cv2.imshow("Hand-eye calibration", cv2.resize(img, None, fx=0.6, fy=0.6))
            key = cv2.waitKey(100)
            if key == 27 and not st["done"]:
                st["abort"] = True
                send_script("moveStop()")
                st["status"] = "STOPPED by ESC"
            elif key != -1 and st["done"]:
                break
            if cv2.getWindowProperty("Hand-eye calibration", cv2.WND_PROP_VISIBLE) < 1:  # 창 X 버튼
                if not st["done"]:
                    st["abort"] = True
                    send_script("moveStop()")
                break
    finally:
        cv2.destroyAllWindows()
        while not st["done"]:
            time.sleep(0.1)
        cam["c"].disconnect()
    print(f"끝: {st['status']}, 샘플 {len(samples)}개", flush=True)


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "board":
        px = 10  # 10 px/mm ≈ 254 dpi
        img = BOARD.generateImage((int(SQUARES_X * SQUARE_MM * px), int(SQUARES_Y * SQUARE_MM * px)))
        # marginSize를 쓰면 보드가 줄어들어 칸 크기가 틀어지므로 흰 여백은 바깥에 따로 붙인다
        img = cv2.copyMakeBorder(img, *[10 * px] * 4, cv2.BORDER_CONSTANT, value=255)
        cv2.imwrite("board.png", img)
        print(f"board.png 저장. 한 칸 {SQUARE_MM} mm가 되도록 인쇄 배율 100%로 출력하세요.")
    elif cmd == "robot":
        robot_calib()
    elif cmd == "robot2":  # 1차 결과(handeye.json)로 보드를 바라보며 크게 기울인 자세를 추가 촬영
        robot_calib(poses=lookat_poses(json.loads(SAMPLES.read_text()), json.loads(RESULT.read_text())))
    elif cmd == "rerun":  # 이미 보드가 잘 보였던 25개 자세(1차 15 + 2차 10)를 관절각을 기록하며 다시 촬영
        r1 = json.loads(Path("calib_samples_round1.json").read_text())
        c1 = json.loads(Path("handeye_round1.json").read_text())
        robot_calib(poses=calib_poses(r1[0]["pose"]) + lookat_poses(r1, c1))
    elif cmd == "auto":
        auto_capture(int(sys.argv[2]) if len(sys.argv) > 2 else 15)
    elif cmd == "capture":
        capture_samples(sys.argv[2] if len(sys.argv) > 2 else None)
    elif cmd == "solve":
        conv, X, err, sign = solve(json.loads(SAMPLES.read_text()))
        # T_flange_cam: q6_sign이 0이 아니면 '플랜지에서 J6 회전을 뺀 마운트' 기준 (cam_to_base가 처리)
        RESULT.write_text(json.dumps({"convention": conv, "q6_sign": sign, "T_flange_cam": X.tolist(), "error_mm": err}, indent=1))
        print(f"선택: {conv}, J6 처리 {sign:+d}, 오차 {err:.2f} mm → {RESULT}")
        print("np.round(T_flange_cam, 2) =\n", np.round(X, 2))
        if err > 5:
            print("오차가 큽니다. 칸 크기 값, 펜던트 좌표가 플랜지 기준인지(툴 오프셋 0), 자세 다양성을 확인하세요.")
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
