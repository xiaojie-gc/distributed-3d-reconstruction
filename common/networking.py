import socket
import struct
import threading
import time
from common.logger import logger


def recv_msg(c):
    # Read message length and unpack it into an integer
    raw_msglen = recvall(4, c)
    if not raw_msglen:
        return None
    msglen = struct.unpack('>I', raw_msglen)[0]
        # Read the message data
        # print("mes len", ms
    # glen)
    #print("length=", msglen)
    return recvall(msglen, c)


def recvall(n, c):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = c.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    #print(len(data))
    return data


def send_msg(c, msg):
    # Prefix each message with a 4-byte length (network byte order)
    msg = struct.pack('>I', len(msg)) + msg
    c.sendall(msg)


class setInterval :
    def __init__(self,interval,action) :
        self.interval=interval
        self.action=action
        self.stopEvent=threading.Event()
        thread=threading.Thread(target=self.__setInterval)
        thread.start()

    def __setInterval(self) :
        nextTime=time.time()+self.interval
        while not self.stopEvent.wait(nextTime-time.time()) :
            nextTime+=self.interval
            self.action()

    def cancel(self) :
        self.stopEvent.set()


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


def server_discovery(port):
    server_channel = []
    while True:
        try:
            dispatcher_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            dispatcher_socket.bind(("", port))
            break
        except:
            port += 1
            print(port)
    dispatcher_socket.listen(1)
    print(logger() + "start to listening:{} -> server......".format(port))

    while True:
        try:
            channel, server_addr = dispatcher_socket.accept()
            server_channel.append(channel)
            print(logger() + "server connected, start to communicate with server [{}]".format(server_addr))
            break
        except:
            break
    return server_channel


def user_discovery(port=8003):
    user_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    user_socket.bind(("", port))
    user_socket.listen(1)
    print(logger() + "start to listening:{} -> user......".format(port))
    user_channel = None
    while True:
        try:
            user_channel, user_addr = user_socket.accept()
            print(logger() + "user connected, start to communicate with user [{}]".format(user_addr))
            break
        except:
            break
    return user_channel