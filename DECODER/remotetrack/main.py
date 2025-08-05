#!/usr/bin/python3
# (c) Bluemark Innovations BV
# MIT license

import time
from pymavlink import mavutil
from bitstruct import *
import threading
import zmq
import json

import config # config.py with settings

from modules import odid # opendroneID functions
from modules import adsb # ADSB vehicle functions
from modules import log_file # log opendroneID data to a CSV file
from modules import sbs # for SBS export

# Imposta ZMQ PUSH per invio al bridge_uploader
context = zmq.Context()
push_socket = context.socket(zmq.PUSH)
push_socket.connect("tcp://127.0.0.1:5556")

if hasattr(config, 'sbs_server_ip_address'): # only enable SBS export if relevant vars have been defined
	if hasattr(config, 'sbs_server_port'):
		print("SBS export thread started")
		sbs_thread = threading.Thread(target=sbs.connect, args=(1,))
		sbs_thread.daemon = True
		sbs_thread.start()

#setup MAVLink serial link and wait until a heartbeat is received
master = mavutil.mavlink_connection(config.interface, config.baudrate)
master.wait_heartbeat()

# if this var is not defined, make it true
if not hasattr(config, 'print_messages'):
    config.print_messages = True
        
while True:
    try:
        msg = master.recv_match()
    except:
        pass
    if not msg:
        time.sleep(0.1)
        continue
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)

    if msg.get_type() == 'HEARTBEAT':
        if config.print_messages == True:
            print("%s MAVLink heartbeat received" % current_time)

    if msg.get_type() == 'ADSB_VEHICLE':
        if config.print_messages == True:
            print("\n%s MAVLink ADS-B vehicle message received" % current_time)
        adsb.print_payload(msg)

        # Invia al bridge tramite ZMQ
        try:
            data = {
                "icao": msg.ICAO_address,
                "lat": msg.lat / 1e7,
                "lon": msg.lon / 1e7,
                "alt": msg.altitude / 1000.0,
                "heading": msg.heading / 100.0,
                "hor_velocity": msg.hor_velocity / 100.0
            }
            push_socket.send_json(data)
        except Exception as e:
            print(f"[ZMQ PUSH] Errore invio: {e}")

    if msg.get_type() == 'OPEN_DRONE_ID_MESSAGE_PACK':
        if config.print_messages == True:
            print("\n%s MAVLink OpenDroneID Message Pack message received" % current_time)
            odid.print_message_pack(msg.messages, msg.msg_pack_size)

        if hasattr(config, 'log_path'):
            if 'filename' not in globals():
                global filename
                filename = log_file.open_csv(config.log_path)
            log_file.write_csv(msg.messages, msg.msg_pack_size, filename)

        if hasattr(config, 'sbs_server_ip_address'):
            try:
                sbs.export(msg.messages, msg.msg_pack_size)
            except:
                pass

