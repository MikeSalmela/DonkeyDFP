import socket
import numpy as np
import pickle
from io import StringIO


HEADERSIZE=16

class sender():
    def __init__(self, address, port):
        self.address_ = address
        self.port_ = port
        self.socket_ = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clientsocket_ = 0
        self.socket_.bind((self.address_, self.port_)) 
        self.socket_.listen(10)

    def __del__(self):
        self.socket_.shutdown(1)
        self.socket_.close()
        print("Connection closed")

    def connect(self):
        print("Waiting for connection")
        self.clientsocket_, address = self.socket_.accept()
        print("Connection from" , address)

    def send(self, img):
        msg = pickle.dumps(img)
        msg = bytes(f'{len(msg):<{HEADERSIZE}}', "utf-8") + msg 

        self.clientsocket_.send(msg)

class reciever():
    def __init__(self, address, port):
        self.address_ = address
        self.port_ = port
        self.socket_ = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    def __del__(self):
        self.socket_.shutdown(1)
        self.socket_.close()

    def connect(self):
        self.socket_.connect((self.address_, self.port_))
        print("Connected")

    def recieve(self):
        full_msg = b''
        new = True
        while True:
            msg = self.socket_.recv(2048)
            if new:
                msg_len = int(msg[:HEADERSIZE])
                new = False
            full_msg += msg

            if len(full_msg)-HEADERSIZE == msg_len:
                return pickle.loads(full_msg[HEADERSIZE:])
                

