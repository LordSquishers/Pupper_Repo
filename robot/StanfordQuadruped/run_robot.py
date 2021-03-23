import selectors
import socket
import types

import numpy as np
import time

from src.Utilities import deadband, clipped_first_order_filter
from src.IMU import IMU
from src.Controller import Controller
from src.JoystickInterface import JoystickInterface
from src.State import State
from pupper.HardwareInterface import HardwareInterface
from pupper.Config import Configuration
from pupper.Kinematics import four_legs_inverse_kinematics

TCP_IP = '192.168.1.240'
TCP_PORT = 5005
BUFFER_SIZE = 32

sel = selectors.DefaultSelector()
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen()
print('Listening on', (TCP_IP, TCP_PORT))
s.setblocking(False)
sel.register(s, selectors.EVENT_READ, data=None)

latest_data = b''
remote_ctl_flag = False
previous_gait_toggle = 0

def accept_wrapper(sock):
    conn, addr = sock.accept()
    print('Accepted connection from', addr)
    conn.setblocking(False)
    data = types.SimpleNamespace(addr=addr, inb=b'', outb=b'')
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    sel.register(conn, events, data=data)


def service_connection(key, mask):
    global is_connected, conn_time, latest_data
    sock = key.fileobj
    data = key.data
    if mask & selectors.EVENT_READ:
        try:
            recv_data = sock.recv(128)
        except ConnectionResetError:
            print('Lost connection to', data.addr, " Exiting...")
            sel.unregister(sock)
            sock.close()

        if recv_data is not None and recv_data:  # we have a connection!
            data.outb += recv_data
            latest_data = recv_data
            is_connected = True
            # print(latest_data)
            conn_time = time.time()
        else:  # connection closed.
            print('Closing connection to', data.addr)
            sel.unregister(sock)
            sock.close()
            is_connected = False
    if mask & selectors.EVENT_WRITE:
        if data.outb:
            # print('Echoing', repr(data.outb), 'to', data.addr)
            sent = sock.send(data.outb)
            data.outb = data.outb[sent:]


def parse_command(old_com, state, latest_data, config):
    global remote_ctl_flag, previous_gait_toggle
    command = old_com

    lx = 0
    ly = 0
    rx = 0
    ry = 0
    dpadx = 0
    dpady = 0
    trot = 0

    for entry in latest_data.split(b'@'):
        raw_pair = entry.split(b'#')
        if len(raw_pair) < 2: continue
        key = raw_pair[0]
        try:
            value = float(raw_pair[1])
        except:
            print('Invalid data received. Skipping...')
            print(entry)
            continue

        if value != 0:
            remote_ctl_flag = True

        if key == b'rx':
            rx = value
        elif key == b'ry':
            ry = value
        elif key == b'lx':
            lx = value
        elif key == b'ly':
            ly = value
        elif key == b'dpadx':
            dpadx = value
        elif key == b'dpady':
            dpady = value
        elif key == b'trot':
            trot = value

    ####### Handle continuous commands ########
    x_vel = ly * config.max_x_velocity  # for back
    # print(x_vel)
    y_vel = lx * -config.max_y_velocity  # left right
    command.horizontal_velocity = np.array([x_vel, y_vel])
    command.yaw_rate = rx * -config.max_yaw_rate  # twist and shout

    message_rate = 50
    message_dt = 1.0 / message_rate

    command.trot_event = (trot == 1 and previous_gait_toggle == 0)

    previous_gait_toggle = trot

    pitch = ry * config.max_pitch

    deadbanded_pitch = deadband(
        pitch, config.pitch_deadband
    )

    pitch_rate = clipped_first_order_filter(
        state.pitch,
        deadbanded_pitch,
        config.max_pitch_rate,
        config.pitch_time_constant,
    )
    command.pitch = state.pitch + message_dt * pitch_rate

    height_movement = dpady
    command.height = state.height - message_dt * config.z_speed * height_movement

    roll_movement = - dpadx
    command.roll = state.roll + message_dt * config.roll_speed * roll_movement

    return command


def main(use_imu=False):
    global remote_ctl_flag
    """Main program
    """
    last_poll = time.time()
    poll_cooldown = 0.05
    # Create config
    config = Configuration()
    hardware_interface = HardwareInterface()

    # Create imu handle
    if use_imu:
        imu = IMU(port="/dev/ttyACM0")
        imu.flush_buffer()

    # Create controller and user input handles
    controller = Controller(
        config,
        four_legs_inverse_kinematics,
    )
    state = State()
    print("Creating joystick listener...")
    joystick_interface = JoystickInterface(config)
    print("Done.")

    last_loop = time.time()

    print("Summary of gait parameters:")
    print("overlap time: ", config.overlap_time)
    print("swing time: ", config.swing_time)
    print("z clearance: ", config.z_clearance)
    print("x shift: ", config.x_shift)

    # Wait until the activate button has been pressed
    while True:
        print("Waiting for L1 to activate robot.")
        while True:
            command = joystick_interface.get_command(state)
            joystick_interface.set_color(config.ps4_deactivated_color)
            if command.activate_event == 1:
                break
            time.sleep(0.1)
        print("Robot activated.")
        joystick_interface.set_color(config.ps4_color)

        while True:
            now = time.time()
            if now - last_loop < config.dt:
                continue
            last_loop = time.time()

            if time.time() - last_poll > poll_cooldown:
                last_poll = time.time()
                # print(latest_data)
                events = sel.select(timeout=0.5)
                for key, mask in events:
                    if key.data is None:
                        accept_wrapper(key.fileobj)
                    else:
                        service_connection(key, mask)

            command = parse_command(command, state, latest_data, config)

            # Parse the udp joystick commands and then update the robot controller's parameters
            if not remote_ctl_flag:
                command = joystick_interface.get_command(state)

            remote_ctl_flag = False

            if command.activate_event == 1:
                print("Deactivating Robot")
                break

            # Read imu data. Orientation will be None if no data was available
            quat_orientation = (
                imu.read_orientation() if use_imu else np.array([1, 0, 0, 0])
            )
            state.quat_orientation = quat_orientation

            # Step the controller forward by dt
            controller.run(state, command)

            # Update the pwm widths going to the servos
            hardware_interface.set_actuator_postions(state.joint_angles)


main()
