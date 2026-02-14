import socket
import time

HOST = "127.0.0.1"
PORT = 14580

CALLSIGN = "DSCBRIDGE"
PASSCODE = "-1"

sock = None
last_connect = 0


def connect():
    global sock, last_connect

    now = time.time()

    if sock:
        return

    if now - last_connect < 5:
        return

    last_connect = now

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((HOST, PORT))

        login = f"user {CALLSIGN} pass {PASSCODE} vers DSC-Bridge 1.0\r\n"
        s.sendall(login.encode())

        sock = s
        print("[APRS-LOCAL] Connected")

    except Exception as e:
        print("[APRS-LOCAL] connect failed:", e)
        sock = None


def send_frame(frame: str):
    global sock

    if not sock:
        connect()

    if not sock:
        return

    try:
        sock.sendall((frame + "\r\n").encode())
    except Exception:
        try:
            sock.close()
        except:
            pass
        sock = None
