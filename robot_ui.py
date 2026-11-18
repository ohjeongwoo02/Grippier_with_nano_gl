"""HCR 제어 화면. 이 PC가 Modbus TCP 서버가 되고, 로봇이 이 PC를 "장비"로 등록해서 레지스터를 읽고 쓴다.
(HCR 매뉴얼 12.1: 로봇은 Modbus 마스터라 PC에서 로봇으로 직접 명령을 보낼 수 없다.)

레지스터 (우리가 정한 주소표, 로봇 프로그램이 같은 규칙으로 읽어야 한다. 값은 모두 0~65535):
    0~5  x y z rx ry rz   목표 자세. 레지스터 = round(값 x 10) + 32768   (mm 또는 도, 0.1 단위, ±3276.7 범위)
    6    SEQ              이동 명령마다 1씩 증가. 로봇은 값이 바뀌면 새 목표로 이동
    7    STOP             1이면 정지 상태. 로봇은 이 값이 1이면 이동하지 않아야 한다
    8    ACK              로봇이 이동을 끝내면 처리한 SEQ 값을 여기에 쓴다 (로봇이 쓰는 칸)
실행: python robot_ui.py   (로봇의 장비 등록 IP = 이 PC의 이더넷 IP, 포트 502)
"""
import asyncio
import threading
import time
import tkinter as tk

from pymodbus.datastore import ModbusSequentialDataBlock, ModbusServerContext, ModbusSlaveContext
from pymodbus.server import StartAsyncTcpServer

OFFSET, SCALE = 32768, 10
SEQ, STOP, ACK = 6, 7, 8


class WatchedBlock(ModbusSequentialDataBlock):
    """로봇이 읽은 시각을 기록해서 '로봇이 연결 중인지' 알 수 있게 한다."""
    last_read = 0.0

    def getValues(self, address, count=1):
        self.last_read = time.time()
        return super().getValues(address, count)

    def peek(self, address, count=1):  # 화면용: 읽기 시각을 갱신하지 않는다
        return super().getValues(address, count)


def encode(v):
    r = round(v * SCALE) + OFFSET
    if not 0 <= r <= 65535:
        raise ValueError(f"값 {v}이(가) 범위(±3276.7)를 벗어났습니다.")
    return r


def decode(r):
    return (r - OFFSET) / SCALE


class Bridge:
    def __init__(self):
        self.block = WatchedBlock(0, [0] * 16)
        self.block.setValues(0, [OFFSET] * 6)  # 초기 자세 = 0, 0, 0, 0, 0, 0

    def start(self, port, on_error):
        # 같은 블록을 holding/input 양쪽에 연결: 로봇이 어느 타입으로 읽어도 같은 값
        ctx = ModbusServerContext(ModbusSlaveContext(hr=self.block, ir=self.block, zero_mode=True), single=True)

        def run():
            try:
                asyncio.run(StartAsyncTcpServer(context=ctx, address=("0.0.0.0", port)))
            except OSError as e:
                on_error(f"서버 시작 실패: {e}")

        threading.Thread(target=run, daemon=True).start()

    def move(self, pose):
        if len(pose) != 6:
            raise ValueError("값 6개(x y z rx ry rz)가 필요합니다.")
        if self.block.peek(STOP)[0]:
            raise RuntimeError("정지 상태입니다. '정지 해제' 후 이동하세요.")
        seq = (self.block.peek(SEQ)[0] + 1) % 65536
        self.block.setValues(0, [encode(v) for v in pose] + [seq])  # 자세와 SEQ를 한 번에 써서 로봇이 반쪽 값을 읽지 않게 함
        return seq

    def set_stop(self, on):
        self.block.setValues(STOP, [1 if on else 0])

    def status(self):
        v = self.block.peek(0, 9)
        return {"pose": [decode(r) for r in v[:6]], "seq": v[SEQ], "stop": v[STOP], "ack": v[ACK],
                "robot_age": time.time() - self.block.last_read if self.block.last_read else None}


def main():
    root = tk.Tk()
    root.title("HCR 제어 (Modbus)")
    bridge = Bridge()
    msg = tk.StringVar(value="서버가 꺼져 있습니다.")
    state = tk.StringVar()

    top = tk.Frame(root)
    top.pack(pady=6)
    tk.Label(top, text="포트").pack(side="left")
    port = tk.Entry(top, width=6)
    port.insert(0, "502")
    port.pack(side="left", padx=4)

    def listen():
        try:
            bridge.start(int(port.get()), lambda m: msg.set(m))
            msg.set(f"대기 중: 포트 {port.get()}. 로봇에서 이 PC를 장비로 등록하세요.")
            listen_btn.config(state="disabled")
        except ValueError:
            msg.set("포트는 숫자여야 합니다.")

    listen_btn = tk.Button(top, text="서버 시작", command=listen)
    listen_btn.pack(side="left")

    pose = tk.Frame(root)
    pose.pack(pady=6, padx=8)
    fields = []
    for name in ("x", "y", "z", "rx", "ry", "rz"):
        tk.Label(pose, text=name).pack(side="left")
        e = tk.Entry(pose, width=8)
        e.insert(0, "0")
        e.pack(side="left", padx=2)
        fields.append(e)

    def act(fn):
        try:
            msg.set(fn())
        except (ValueError, RuntimeError) as e:
            msg.set(str(e))

    btns = tk.Frame(root)
    btns.pack(pady=4)
    tk.Button(btns, text="이동", command=lambda: act(
        lambda: f"이동 명령 SEQ={bridge.move([float(e.get()) for e in fields])} 기록. 로봇이 읽어가야 움직입니다.")).pack(side="left", padx=6)
    tk.Button(btns, text="정지", bg="#c33", fg="white", command=lambda: act(
        lambda: bridge.set_stop(True) or "정지 신호 ON")).pack(side="left", padx=6)
    tk.Button(btns, text="정지 해제", command=lambda: act(
        lambda: bridge.set_stop(False) or "정지 신호 OFF")).pack(side="left", padx=6)

    tk.Label(root, textvariable=msg, wraplength=520).pack(pady=4)
    tk.Label(root, textvariable=state, font=("Consolas", 10), justify="left").pack(pady=4, padx=8)
    tk.Label(root, text="비상정지는 항상 실물 버튼을 쓰세요. 정지 신호는 로봇 프로그램이 읽어야 효과가 있습니다.").pack(pady=(0, 8))

    def refresh():
        s = bridge.status()
        link = "연결 안 됨" if s["robot_age"] is None or s["robot_age"] > 3 else f"로봇이 읽는 중 ({s['robot_age']:.1f}s 전)"
        state.set(f"로봇: {link}\nSEQ={s['seq']}  ACK(로봇 완료)={s['ack']}  STOP={s['stop']}\n"
                  "자세: " + "  ".join(f"{v:g}" for v in s["pose"]))
        root.after(300, refresh)

    refresh()
    root.mainloop()


if __name__ == "__main__":
    main()
