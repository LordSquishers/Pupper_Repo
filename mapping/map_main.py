import sys
import math
import socket
import time
import selectors
import types
import struct

import pygame
from pygame.locals import *

# init
pygame.init()
DISPLAY_SURF = pygame.display.set_mode((800, 600))
DISPLAY_SURF.fill(Color(255, 255, 255))
MAP_BOUNDS = (600, 600)
pygame.display.set_caption("Pupper Map")

FPS = pygame.time.Clock()


class Text():
    def __init__(self, text, color, fontSize):
        font = pygame.font.SysFont("Verdana", fontSize)
        self.text = font.render(text, True, color)


class Pupper(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load("pupper_icon.png")
        self.surf = pygame.Surface((100, 100))
        self.rect = self.surf.get_rect(center=(MAP_BOUNDS[0] / 2, MAP_BOUNDS[1] / 2))

    def draw(self, surface):
        surface.blit(self.image, self.rect)


# real variables
pupper = Pupper()
scale = 1  # m/s to map space
panel_x = 0
panel_y = 0
pupper_hdg = 0
pupper_pit = 0
pupper_roll = 0
is_connected = False

# server deets
TCP_IP = '192.168.1.193'
TCP_PORT = 5005
BUFFER_SIZE = 32

sel = selectors.DefaultSelector()
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((TCP_IP, TCP_PORT))
s.listen()
print('Listening on', (TCP_IP, TCP_PORT))
s.setblocking(False)
sel.register(s, selectors.EVENT_READ, data=None)

conn_max_ping = 5
conn_time = time.time()

latest_data = b''


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
            pygame.quit()
            sys.exit()

        if recv_data:  # we have a connection!
            data.outb += recv_data
            latest_data = recv_data
            is_connected = True
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


def get_pupper_telemetry():
    global panel_x, panel_y, pupper_hdg, pupper_pit, pupper_roll

    events = sel.select(timeout=None)
    for key, mask in events:
        if key.data is None:
            accept_wrapper(key.fileobj)
        else:
            service_connection(key, mask)

    x = 0
    y = 0
    z = 0
    hdg = 0
    pit = 0
    rol = 0

    # decode latest data
    for entry in latest_data.split(b'!@#(#)'):
        rawPair = entry.split(b'#!$(#)')
        if len(rawPair) < 2: continue
        key = rawPair[0]
        try:
            value = struct.unpack('f', rawPair[1])[0]
        except struct.error:
            print('Invalid data received. Skipping...')
            print(entry)
            continue

        # print(key, value)

        if key == b'x':
            x = value
        elif key == b'y':
            y = value
        elif key == b'z':
            z = value
        elif key == b'hdg':
            hdg = value
        elif key == b'pit':
            pit = value
        elif key == b'rol':
            rol = value

    # do some processing / integrals to turn accel. into vel.

    # load velocity and update pupper pos
    # z velocity shouldn't matter.
    # pupper.rect.move_ip(x * scale, y * scale)
    # update panel data
    panel_x = round(x, 1)
    panel_y = round(y, 1)
    # print('\n')

    pupper_hdg = hdg
    pupper_hdg = round(pupper_hdg % 360)

    pupper_pit = pit
    pupper_pit = round(pupper_pit % 360)

    pupper_roll = rol
    pupper_roll = round(pupper_roll % 360)


# Game Loop
while True:
    is_connected = time.time() - conn_time < conn_max_ping
    get_pupper_telemetry()

    # events
    for event in pygame.event.get():
        if event.type == QUIT:
            sel.close()
            pygame.quit()
            sys.exit()

    # redraw
    DISPLAY_SURF.fill(Color(255, 255, 255))

    # panel update
    pygame.draw.rect(DISPLAY_SURF, Color(65, 65, 65), (MAP_BOUNDS[0], 0, 200, 600))
    DISPLAY_SURF.blit(Text("Pupper", Color(245, 245, 245), 50).text, (612, 15))
    DISPLAY_SURF.blit(Text("X : " + str(panel_x) + " m/s", Color(245, 245, 245), 20).text, (615, 100))
    DISPLAY_SURF.blit(Text("Y : " + str(panel_y) + " m/s", Color(245, 245, 245), 20).text, (615, 125))
    DISPLAY_SURF.blit(Text("Heading : " + str(pupper_hdg) + " deg", Color(245, 245, 245), 20).text, (615, 150))
    DISPLAY_SURF.blit(Text("Pitch : " + str(pupper_pit) + " deg", Color(245, 245, 245), 20).text, (615, 175))
    DISPLAY_SURF.blit(Text("Roll : " + str(pupper_roll) + " deg", Color(245, 245, 245), 20).text, (615, 200))
    pygame.draw.line(DISPLAY_SURF, Color(255, 0, 0), pupper.rect.center,
                     (pupper.rect.centerx + math.sin(math.radians(pupper_hdg)) * 100,
                      pupper.rect.centery - math.cos(math.radians(pupper_hdg)) * 100),
                     width=2)

    DISPLAY_SURF.blit(
        Text("Connected" if is_connected else "Not Found", Color(0, 255, 0) if is_connected else Color(255, 0, 0),
             25).text, (635, 550))

    # pupper update
    pupper.draw(DISPLAY_SURF)

    # render
    pygame.display.update()
    FPS.tick(60)
