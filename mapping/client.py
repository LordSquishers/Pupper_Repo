import math
import socket
import selectors
import time
import types
import pykalman as KalmanFilter
import numpy as np
import struct
import FaBo9Axis_MPU9250

TCP_IP = '192.168.1.193'
TCP_PORT = 5005
BUFFER_SIZE = 128

sel = selectors.DefaultSelector()
ds = None

imu = FaBo9Axis_MPU9250.MPU9250()


def start_connection():
    global ds
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setblocking(False)
    s.connect_ex((TCP_IP, TCP_PORT))
    events = selectors.EVENT_READ | selectors.EVENT_WRITE
    ds = types.SimpleNamespace(connid=1, msg_total=0, recv_total=0, messages=list(), outb=b'')
    sel.register(s, events, data=ds)


def service_connection(key, mask):
    sock = key.fileobj
    data = key.data
    if mask & selectors.EVENT_READ:
        recv_data = sock.recv(128)  # Should be ready to read
        if recv_data:
            # print('received', repr(recv_data), 'from connection', data.connid)
            data.recv_total += len(recv_data)
        if not recv_data or data.recv_total == data.msg_total:
            print('closing connection', data.connid)
            sel.unregister(sock)
            sock.close()
    if mask & selectors.EVENT_WRITE:
        if not data.outb and data.messages:
            # data.outb = b''
            data.outb = data.messages.pop(0)
        if data.outb:
            # print('sending', repr(data.outb), 'to connection', data.connid)
            sent = sock.send(data.outb)  # Should be ready to write
            data.outb = data.outb[sent:]


start_connection()
last_time = time.time()
cooldown = 0.05
calibration = True

AX_OFFSET = 0
AY_OFFSET = 0
AZ_OFFSET = 0

ROLL_OFFSET = 0
PITCH_OFFSET = 0
YAW_OFFSET = 0

vx = 0
vy = 0
vz = 0

x = 0
y = 0
z = 0

ox = 0
oy = 0
oz = 0

hdg = 0
pitch = 0
roll = 0

KEY = b'#!$(#)'
SEP = b'!@#(#)'

class KalmanFilter(object):
    def __init__(self, dt, u, std_acc, std_meas):
        self.dt = dt
        self.u = u
        self.std_acc = std_acc
        self.std_meas = std_meas

        self.A = np.matrix([[1, self.dt],
                            [0, 1]])
        self.B = np.matrix([[(self.dt**2)/2], [self.dt]])
        self.H = np.matrix([[1, 0]])
        self.Q = np.matrix([[(self.dt**4)/4, (self.dt**3)/2],
                            [(self.dt**3)/2, self.dt**2]]) * self.std_acc**2
        self.R = std_meas**2
        self.P = np.eye(self.A.shape[1])
        self.x = np.matrix([[0], [0]])

    def predict(self):
        # update time + state
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(sel.P, self.H.T), np.linalg.inv(S))

        self.x = np.round(self.x + np.dot(K, (z - np.dot(self.H, self.x))))

        I = np.eye(self.H.shape[1])
        self.P = (I - (K * self.H)) * self.P


def calculate_tel():
    global ds, vx, vy, vz, x, y, z, hdg, pitch, roll, ox, oy, oz
    acc_data = imu.readAccel()
    rot_data = imu.readGyro()
    # x: forward back     y: left right     z: up down

    hdg -= (rot_data['z'] + YAW_OFFSET) * (cooldown)
    roll += (rot_data['x'] + ROLL_OFFSET) * (cooldown)
    pitch -= (rot_data['y'] + PITCH_OFFSET) * (cooldown)

    x_off = math.sin(math.radians(pitch)) * -1
    y_off = math.sin(math.radians(roll)) * -1

    # print('x_off, y_off, spitch, sroll', x_off, y_off, math.sin(math.radians(pitch)), math.sin(math.radians(roll)))

    ax = (acc_data['x']) + x_off + AX_OFFSET - ox
    ay = (acc_data['y']) + y_off + AY_OFFSET - oy
    az = (acc_data['z']) + (AZ_OFFSET - (x_off + y_off)) - oz

    vx += (ax) * cooldown
    vy += (ay) * cooldown
    vz += (az) * cooldown

    x += (vx) * (cooldown)
    y += (vy) * (cooldown)
    z += (vz) * (cooldown)

    ox = ax
    oy = ay
    oz = oz

    return (ax, ay, az)


def send_telemetry(vals):
    ax, ay, az = vals
    ds.messages.append(b'x' + KEY + bytearray(struct.pack('f', ax))
                       + SEP + b'y' + KEY + bytearray(struct.pack('f', ay))
                       + SEP + b'z' + KEY + bytearray(struct.pack('f', az))
                       + SEP + b'hdg' + KEY + bytearray(struct.pack('f', hdg))
                       + SEP + b'pit' + KEY + bytearray(struct.pack('f', pitch))
                       + SEP + b'rol' + KEY + bytearray(struct.pack('f', roll)))

tot_ax = 0
tot_ay = 0
tot_az = 0

tot_rz = 0
tot_ry = 0
tot_rx = 0

calib_runs = 0
TOTAL_CALIBRATION_RUNS = 1000
notified_calibration = False

kf = KalmanFilter(cooldown, 2, 0.01, 0.5)
predictions = []
measurements = []

try:
    while 1:
        if calibration:  # runs for n loops
            if not notified_calibration:  # this statement runs once.
                print('Calibrating...')
                notified_calibration = True

            # calibrate acceleration
            acc = imu.readAccel()
            tot_ax += acc['x']
            tot_ay += acc['y']
            tot_az += acc['z'] - 1
            # g is the +1 value.

            # cal. rotation
            rot = imu.readGyro()
            tot_rx += rot['x']
            tot_ry += rot['y']
            tot_rz += rot['z']
            calib_runs += 1

            if calib_runs > TOTAL_CALIBRATION_RUNS:  # this statement runs once.
                calibration = False
                # set vars
                AX_OFFSET = -tot_ax / calib_runs
                AY_OFFSET = -tot_ay / calib_runs
                AZ_OFFSET = -tot_az / calib_runs

                YAW_OFFSET = -tot_rz / calib_runs
                ROLL_OFFSET = -tot_rx / calib_runs
                PITCH_OFFSET = -tot_ry / calib_runs

                # print(AX_OFFSET, AY_OFFSET, AZ_OFFSET)
                # print(PITCH_OFFSET, YAW_OFFSET, ROLL_OFFSET)



                print('Sending telemetry...')

            continue

        vals = calculate_tel()
        if time.time() - last_time > cooldown:  # so we dont spam
            send_telemetry(vals)
            last_time = time.time()
        events = sel.select(timeout=1)
        if events:
            for key, mask in events:
                service_connection(key, mask)
        if not sel.get_map():
            break
except KeyboardInterrupt:
    print('Exiting')
finally:
    sel.close()
