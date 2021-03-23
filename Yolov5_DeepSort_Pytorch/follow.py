import math
import socket
import selectors
import sys
import time
import types

DEBUG_MODE = False

WALK_SENS = 0.85
ROT_SENS = 0.75
HEIGHT_SENS = 0.9
ROLL_SENS = 0.6

TCP_IP = '192.168.1.240'
TCP_PORT = 5005
BUFFER_SIZE = 128

KEY = b'#'
SEP = b'@'

vals = None
sel = None
ds = None
last_time = time.time()

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
            print('sending', repr(data.outb), 'to connection', data.connid)
            sent = sock.send(data.outb)  # Should be ready to write
            data.outb = data.outb[sent:]


def send_commands(inputs):
    ds.messages.append(b'rx' + KEY + bytes(str(inputs['rx']), 'utf-8')
                       + SEP + b'ry' + KEY + bytes(str(inputs['ry']), 'utf-8')
                       + SEP + b'lx' + KEY + bytes(str(inputs['lx']), 'utf-8')
                       + SEP + b'ly' + KEY + bytes(str(inputs['ly']), 'utf-8')
                       + SEP + b'dpadx' + KEY + bytes(str(inputs['dpadx']), 'utf-8')
                       + SEP + b'dpady' + KEY + bytes(str(inputs['dpady']), 'utf-8')
                       + SEP + b'trot' + KEY + bytes(str(inputs['trot']), 'utf-8') + SEP)

def update(cooldown):  # called each frame
    global last_time
    # print('updated!')
    if not DEBUG_MODE:
        if time.time() - last_time > cooldown:  # so we dont spam
            send_commands(vals)
            last_time = time.time()
        events = sel.select(timeout=1)
        if events:
            for key, mask in events:
                service_connection(key, mask)
        if not sel.get_map():
            return

def initialize():  # initialization
    global vals, sel, ds
    vals = {
        'rx': 0,
        'ry': 0,
        'lx': 0,
        'ly': 0,
        'dpadx': 0,
        'dpady': 0,
        'trot': 0
    }

    # inits
    sel = selectors.DefaultSelector()
    ds = None

    start_connection()