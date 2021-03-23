import math
import socket
import selectors
import sys
import time
import types
import pygame
from pygame.locals import *

DEBUG_MODE = False

# real variables
scale = 1  # m/s to map space
panel_x = 0
panel_y = 0

cooldown = 0.05
WALK_SENS = 0.85
ROT_SENS = 0.75
HEIGHT_SENS = 0.9
ROLL_SENS = 0.6

TCP_IP = '192.168.1.240'
TCP_PORT = 5005
BUFFER_SIZE = 128

KEY = b'#'
SEP = b'@'

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


class Button():
    def __init__(self, x, y, width, height, active_color, select_color, color):
        # Rendering
        self.rect = Rect(x, y, width, height)
        self.text = Text('', [0, 0, 0], 32)

        # Logic
        self.selected = False
        self.active = False
        self.flag = 0
        self.flag2 = 0

        # Color
        self.color = color
        self.norm_color = color
        self.active_color = active_color
        self.select_color = select_color

    def update(self):
        if self.active:
            self.color = self.active_color
        elif self.selected:
            self.color = self.select_color
        else:
            self.color = self.norm_color


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

def key_input(vals):
    keys = pygame.key.get_pressed()

    if keys[K_w]:
        vals['ly'] = WALK_SENS
    elif keys[K_s]:
        vals['ly'] = -WALK_SENS
    else:
        vals['ly'] = 0

    if keys[K_a]:
        vals['lx'] = -WALK_SENS
    elif keys[K_d]:
        vals['lx'] = WALK_SENS
    else:
        vals['lx'] = 0

    if keys[K_UP]:
        vals['ry'] = -ROT_SENS
    elif keys[K_DOWN]:
        vals['ry'] = ROT_SENS
    else:
        vals['ry'] = 0

    if keys[K_LEFT]:
        vals['rx'] = -ROT_SENS
    elif keys[K_RIGHT]:
        vals['rx'] = ROT_SENS
    else:
        vals['rx'] = 0

    if keys[K_LSHIFT]:
        vals['dpady'] = HEIGHT_SENS
    elif keys[K_LCTRL]:
        vals['dpady'] = -HEIGHT_SENS
    else:
        vals['dpady'] = 0

    if keys[K_q]:
        vals['dpadx'] = -ROLL_SENS
    elif keys[K_e]:
        vals['dpadx'] = ROLL_SENS
    else:
        vals['dpadx'] = 0

    if keys[K_SPACE]:
        vals['trot'] = 1
    else:
        vals['trot'] = 0

    return vals


try:
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

    pygame.init()
    DISPLAY_SURF = pygame.display.set_mode((800, 600))
    DISPLAY_SURF.fill(Color(255, 255, 255))
    MAP_BOUNDS = (600, 600)
    pygame.display.set_caption("Pupper")

    FPS = pygame.time.Clock()

    start_connection()
    last_time = time.time()
    pupper = Pupper()

    move_forward_btn = Button(620, 100, 165, 60, [25, 255, 25], [200, 200, 200], [147, 147, 147])
    move_forward_btn.text = Text('Forward (2s)', [255, 255, 255], 22)

    # main loop
    while 1:
        # events
        for event in pygame.event.get():
            vals = key_input(vals)
            if event.type == QUIT:
                sel.close()
                pygame.quit()
                sys.exit()

            if event.type == MOUSEMOTION:
                mouse_pos = event.pos

                # button logic
                move_forward_btn.selected = move_forward_btn.rect.collidepoint(mouse_pos)

            if event.type == MOUSEBUTTONDOWN:
                mouse_pos = event.pos

                # button logic
                if move_forward_btn.rect.collidepoint(mouse_pos):  # on-only activation
                    move_forward_btn.active = True

        # redraw
        DISPLAY_SURF.fill(Color(255, 255, 255))
        move_forward_btn.update()  # button render update

        # custom button logic
        if move_forward_btn.active and not move_forward_btn.flag:
            move_forward_btn.duration = 2
            move_forward_btn.start_time = time.time()
            move_forward_btn.flag = 1
            move_forward_btn.flag2 = 1

            vals['trot'] = 1
        elif move_forward_btn.active:
            # command logic
            vals['ly'] = WALK_SENS

            # timer logic
            if time.time() - move_forward_btn.start_time > move_forward_btn.duration:
                move_forward_btn.flag = 0
                move_forward_btn.active = 0
                vals['ly'] = 0
                vals['trot'] = 1
            elif time.time() - move_forward_btn.start_time > 0.5:
                vals['trot'] = 0

        if move_forward_btn.flag2 and time.time() - move_forward_btn.start_time > move_forward_btn.duration + 0.5:
            vals['trot'] = 0
            flag2 = 0

        # panel update
        pygame.draw.rect(DISPLAY_SURF, Color(65, 65, 65), (MAP_BOUNDS[0], 0, 200, 600))
        pygame.draw.rect(DISPLAY_SURF, move_forward_btn.color, move_forward_btn.rect)
        DISPLAY_SURF.blit(Text("Pupper", Color(245, 245, 245), 50).text, (612, 15))
        DISPLAY_SURF.blit(move_forward_btn.text.text, (632, 115))

        # update
        pupper.draw(DISPLAY_SURF)

        # render
        pygame.display.update()
        FPS.tick(60)
        if not DEBUG_MODE:
            if time.time() - last_time > cooldown:  # so we dont spam
                send_commands(vals)
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
