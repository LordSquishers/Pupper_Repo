import math
import socket
import selectors
import time
import types
# noinspection PyUnresolvedReferences
from Kalman import KalmanAngle
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

start_connection()
last_time = time.time()
cooldown = 0.05
calibration = True

KEY = b'#!$(#)'
SEP = b'!@#(#)'

radToDeg = 180 / math.pi

accel_0 = imu.readAccel()
accX = accel_0['x']
accY = accel_0['y']
accZ = accel_0['z']

kalAngleX = 0
kalAngleY = 0
kalAngleZ = 0
kalmanX = KalmanAngle()
kalmanY = KalmanAngle()
kalmanZ = KalmanAngle()

roll = math.atan(accY / math.sqrt((accX ** 2) + (accZ ** 2))) * radToDeg
pitch = math.atan2(-accX, accZ) * radToDeg
hdg = 0
# print(roll)
kalmanX.setAngle(roll)
kalmanY.setAngle(pitch)
kalmanZ.setAngle(hdg)
gyroXAngle = roll
gyroYAngle = pitch
gyroZAngle = hdg
compAngleX = roll
compAngleY = pitch
compAngleZ = hdg

timer = time.time()
flag = 0


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
            if not notified_calibration:
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

            if calib_runs > TOTAL_CALIBRATION_RUNS:
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

        if (flag > 100):  # Problem with the connection
            print("There is a problem with the connection")
            flag = 0
            continue
        try:
            acl = imu.readAccel()
            gyro = imu.readGyro()
            # Read Accelerometer raw value
            accX = acl['x']
            accY = acl['y']
            accZ = acl['z']

            # Read Gyroscope raw value
            gyroX = gyro['x']
            gyroY = gyro['y']
            gyroZ = gyro['z']

            dt = time.time() - timer
            timer = time.time()

            roll = math.atan(accY / math.sqrt((accX ** 2) + (accZ ** 2))) * radToDeg
            pitch = math.atan2(-accX, accZ) * radToDeg
            hdg -= (gyroZ + YAW_OFFSET) * dt

            gyroXRate = gyroX / 131
            gyroYRate = gyroY / 131
            gyroZRate = gyroZ / 131
            if ((pitch < -90 and kalAngleY > 90) or (pitch > 90 and kalAngleY < -90)):
                kalmanY.setAngle(pitch)
                complAngleY = pitch
                kalAngleY = pitch
                gyroYAngle = pitch
            else:
                kalAngleY = kalmanY.getAngle(pitch, gyroYRate, dt)

            if (abs(kalAngleY) > 90):
                gyroXRate = -gyroXRate
                kalAngleX = kalmanX.getAngle(roll, gyroXRate, dt)

            kalAngleZ = kalmanZ.getAngle(hdg, gyroZRate, dt)

            # angle = (rate of change of angle) * change in time
            gyroXAngle = gyroXRate * dt
            gyroYAngle = gyroYAngle * dt
            gyroZAngle = gyroZRate * dt

            # compAngle = constant * (old_compAngle + angle_obtained_from_gyro) + constant * angle_obtained from accelerometer
            compAngleX = 0.93 * (compAngleX + gyroXRate * dt) + 0.07 * roll
            compAngleY = 0.93 * (compAngleY + gyroYRate * dt) + 0.07 * pitch
            compAngleZ = 0.93 * (compAngleZ + gyroZRate * dt) + 0.07 * hdg

            if ((gyroXAngle < -180) or (gyroXAngle > 180)):
                gyroXAngle = kalAngleX
            if ((gyroYAngle < -180) or (gyroYAngle > 180)):
                gyroYAngle = kalAngleY
            if ((gyroZAngle < -180) or (gyroZAngle > 180)):
                gyroZAngle = kalAngleZ

            # print("Angle X: " + str(kalAngleX) + "   " + "Angle Y: " + str(kalAngleY))
            # print(str(roll)+"  "+str(gyroXAngle)+"  "+str(compAngleX)+"  "+str(kalAngleX)+"  "+str(pitch)+"  "+str(gyroYAngle)+"  "+str(compAngleY)+"  "+str(kalAngleY))
            # print("Pitch:", pitch, "Roll:", roll, "Hdg:", hdg)



            time.sleep(0.005)

        except Exception as exc:
            flag += 1

        if time.time() - last_time > cooldown:  # so we dont spam
            send_telemetry((0, 0, 0))
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
