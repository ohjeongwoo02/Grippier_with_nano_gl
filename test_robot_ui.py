"""화면 없이 Bridge를 가짜 로봇(Modbus 클라이언트)으로 확인."""
import socket
import time

from pymodbus.client import ModbusTcpClient

from robot_ui import ACK, OFFSET, Bridge


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_roundtrip():
    bridge, errors, port = Bridge(), [], free_port()
    bridge.start(port, errors.append)
    robot = ModbusTcpClient("127.0.0.1", port=port)
    for _ in range(50):
        if robot.connect():
            break
        time.sleep(0.1)
    assert robot.connected, errors

    assert bridge.status()["robot_age"] is None  # 아직 로봇이 읽지 않음
    seq = bridge.move([100.5, -12.5, 300, 180, 0, -90])
    for fc in (robot.read_holding_registers, robot.read_input_registers):  # 어느 타입으로 읽어도 같은 값
        regs = fc(0, 9, slave=1).registers
        assert regs[:7] == [OFFSET + 1005, OFFSET - 125, OFFSET + 3000, OFFSET + 1800, OFFSET, OFFSET - 900, seq], regs
    assert bridge.status()["robot_age"] < 2  # 로봇이 읽는 중으로 표시됨

    robot.write_register(ACK, seq, slave=1)  # 로봇이 완료를 알림
    assert bridge.status()["ack"] == seq

    bridge.set_stop(True)
    assert robot.read_holding_registers(7, 1, slave=1).registers == [1]
    try:
        bridge.move([0] * 6)
    except RuntimeError:
        pass
    else:
        raise AssertionError("정지 중에는 이동이 거부돼야 함")
    bridge.set_stop(False)
    assert bridge.move([0] * 6) == seq + 1

    try:
        bridge.move([4000, 0, 0, 0, 0, 0])
    except ValueError:
        pass
    else:
        raise AssertionError("범위를 넘는 값은 거부돼야 함")
    robot.close()


if __name__ == "__main__":
    test_roundtrip()
    print("ok")
