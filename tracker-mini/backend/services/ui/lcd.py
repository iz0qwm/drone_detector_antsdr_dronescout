from threading import Thread
from time import sleep
from services.logger import log
from config import SETTINGS
from services.readsb import is_receiving
from services.ds110 import is_alive as ds110_is_alive
from services.meshtastic_service import is_alive as meshtastic_is_alive
from services.network import (
    get_admin_lan_status,
    has_internet
)
import psutil
import json
from pathlib import Path

from services.ds110 import get_aircraft as get_remoteid_aircraft
from services.meshtastic_service import get_nodes as get_meshtastic_nodes
from services.gps import get_gps_status

class LCDService:

    def __init__(self):

        self.page = "boot"
        self.lcd = None
        self.available = False
        self.running = False
        self.thread = None
        self.encoder = None
        self.button = None

        self.pages = [
            "status",
            "traffic",
            "network",
            "gps",
            "system"
        ]

        self.page_index = 0


    def start(self):

        try:
            from RPLCD.i2c import CharLCD

            log(
                "LCD",
                "Initializing display..."
            )
            
            self.lcd = CharLCD(
                i2c_expander="PCF8574",
                address=0x27,
                port=1,
                cols=20,
                rows=4,
                charmap="A02",
            )

            self.available = True
            self.running = True
            
            self.thread = Thread(
                target=self._worker,
                daemon=True
            )

            self.thread.start()

            try:
                from gpiozero import RotaryEncoder, Button

                self.encoder = RotaryEncoder(
                    a=5,
                    b=6,
                    max_steps=0
                )

                self.button = Button(
                    16,
                    pull_up=True,
                    bounce_time=0.05
                )

                self.encoder.when_rotated = (
                    self._encoder_rotated
                )

                self.button.when_pressed = (
                    self._button_pressed
                )

            except Exception as e:
                self.encoder = None
                self.button = None

                log(
                    "LCD",
                    "Rotary encoder not available:",
                    str(e),
                    level="WARNING"
                )

            log(
                "LCD",
                "Service started"
            )
                    

        except Exception as e:
            log(
                "LCD",
                "Initialization failed:",
                str(e),
                level="ERROR"
            )
            return

    def _encoder_rotated(self):

        if self.page == "boot":
            return

        if self.encoder is None:
            return

        direction = (
            1
            if self.encoder.steps > 0
            else -1
        )

        self.encoder.steps = 0

        self.page_index = (
            self.page_index + direction
        ) % len(self.pages)

        self.page = self.pages[
            self.page_index
        ]

        log(
            "LCD",
            f"Page changed to {self.page}"
        )


    def _button_pressed(self):

        log(
            "LCD",
            f"Button pressed on page {self.page}"
        )

        
    def _worker(self):

        self.show_boot()

        sleep(3)

        self.page_index = 0
        self.page = self.pages[self.page_index]

        while self.running:

            try:

                if self.page == "status":

                    self.refresh_status()
                    self.show_status()

                elif self.page == "traffic":

                    self.refresh_traffic()
                    self.show_traffic()

                elif self.page == "network":

                    self.refresh_network()
                    self.show_network()

                elif self.page == "gps":

                    self.refresh_gps()
                    self.show_gps()

                elif self.page == "system":

                    self.refresh_system()
                    self.show_system()

                else:

                    self.show_page_placeholder()

            except Exception as e:

                log(
                    "LCD",
                    f"Page {self.page} update failed: {e}",
                    level="ERROR"
                )

                try:
                    self.lcd.clear()
                    self.lcd.write_string(
                        "LCD PAGE ERROR"
                    )

                    self.lcd.cursor_pos = (1, 0)
                    self.lcd.write_string(
                        self.page.upper()[:20]
                    )

                except Exception:
                    pass

            sleep(1)

    def show_boot(self):

        if not self.available:
            return
        
        self.lcd.clear()

        self.lcd.write_string("Drone Sky Check")

        self.lcd.cursor_pos = (1, 0)
        self.lcd.write_string("Mini Tracker")

        self.lcd.cursor_pos = (3, 0)
        self.lcd.write_string("Starting...")



    def show_page_placeholder(self):

        if not self.available:
            return

        self.lcd.clear()

        self.lcd.write_string(
            self.page.upper()[:20]
        )

        self.lcd.cursor_pos = (2, 0)

        self.lcd.write_string(
            "Page not available"
        )

        
    def refresh_status(self):

        self.adsb = is_receiving()

        self.rid = ds110_is_alive()

        self.mesh = meshtastic_is_alive()

        admin_lan = get_admin_lan_status()

        self.ip = (
            admin_lan.get("ip")
            or "---"
        )

        self.ap = SETTINGS.get(
            "ap_ssid",
            ""
        )

    def show_status(self):

        if not self.available:
            return

        self.lcd.clear()

        adsb = "OK" if self.adsb else "--"
        rid = "OK" if self.rid else "--"

        mesh = "OK" if self.mesh else "--"

        self.lcd.write_string(
            f"ADSB:{adsb} RID:{rid}"
        )

        self.lcd.cursor_pos = (1, 0)

        self.lcd.write_string(
            f"MESH:{mesh}"
        )

        self.lcd.cursor_pos = (2, 0)

        self.lcd.write_string(
            f"IP:{self.ip}"
        )

        self.lcd.cursor_pos = (3, 0)

        self.lcd.write_string(
            f"AP:{self.ap[:17]}"
        )

    def refresh_network(self):

        admin_lan = get_admin_lan_status()

        self.admin_ip = (
            admin_lan.get("ip")
            or "---"
        )

        self.admin_connected = bool(
            admin_lan.get("connected")
        )

        self.internet = has_internet()

        self.ap_ssid = SETTINGS.get(
            "ap_ssid",
            "---"
        )


    def show_network(self):

        if not self.available:
            return

        self.lcd.clear()

        lan_status = (
            "UP"
            if self.admin_connected
            else "--"
        )

        internet = (
            "OK"
            if self.internet
            else "--"
        )

        self.lcd.write_string(
            "NETWORK"
        )

        self.lcd.cursor_pos = (1, 0)
        self.lcd.write_string(
            f"LAN:{lan_status} WWW:{internet}"
        )

        self.lcd.cursor_pos = (2, 0)
        self.lcd.write_string(
            f"IP:{self.admin_ip}"[:20]
        )

        self.lcd.cursor_pos = (3, 0)
        self.lcd.write_string(
            f"AP:{self.ap_ssid}"[:20]
        )
    
    def refresh_system(self):

        self.cpu_percent = psutil.cpu_percent(
            interval=None
        )

        self.memory_percent = psutil.virtual_memory().percent

        self.disk_percent = psutil.disk_usage(
            "/"
        ).percent

        try:
            temperature = psutil.sensors_temperatures()

            cpu_temperature = temperature.get(
                "cpu_thermal",
                []
            )

            if cpu_temperature:
                self.cpu_temperature = (
                    cpu_temperature[0].current
                )
            else:
                self.cpu_temperature = None

        except Exception:
            self.cpu_temperature = None


    def show_system(self):

        if not self.available:
            return

        self.lcd.clear()

        self.lcd.write_string(
            "SYSTEM"
        )

        self.lcd.cursor_pos = (1, 0)
        self.lcd.write_string(
            f"CPU:{self.cpu_percent:3.0f}%"
        )

        self.lcd.cursor_pos = (2, 0)
        self.lcd.write_string(
            f"RAM:{self.memory_percent:3.0f}% "
            f"DSK:{self.disk_percent:3.0f}%"
        )

        self.lcd.cursor_pos = (3, 0)

        if self.cpu_temperature is not None:
            self.lcd.write_string(
                f"TEMP:{self.cpu_temperature:.1f} C"
            )
        else:
            self.lcd.write_string(
                "TEMP:---"
            )

    def refresh_traffic(self):

        self.adsb_active = is_receiving()
        self.rid_active = ds110_is_alive()
        self.mesh_active = meshtastic_is_alive()

        self.adsb_count = 0

        try:

            aircraft_file = Path(
                "/run/readsb/aircraft.json"
            )

            if aircraft_file.exists():

                with open(
                    aircraft_file,
                    "r",
                    encoding="utf-8"
                ) as file:

                    data = json.load(file)

                self.adsb_count = len(
                    data.get("aircraft", [])
                )

        except Exception as e:

            log(
                "LCD",
                f"ADS-B count failed: {e}",
                level="WARNING"
            )

            self.adsb_count = 0

        try:

            self.rid_count = len(
                get_remoteid_aircraft()
            )

        except Exception as e:

            log(
                "LCD",
                f"Remote ID count failed: {e}",
                level="WARNING"
            )

            self.rid_count = 0

        try:

            self.mesh_count = len(
                get_meshtastic_nodes()
            )

        except Exception as e:

            log(
                "LCD",
                f"Meshtastic count failed: {e}",
                level="WARNING"
            )

            self.mesh_count = 0

            
    def show_traffic(self):

        if not self.available:
            return

        self.lcd.clear()

        adsb_status = (
            "OK"
            if self.adsb_active
            else "--"
        )

        rid_status = (
            "OK"
            if self.rid_active
            else "--"
        )

        mesh_status = (
            "OK"
            if self.mesh_active
            else "--"
        )

        self.lcd.write_string(
            "TRAFFIC"
        )

        self.lcd.cursor_pos = (1, 0)
        self.lcd.write_string(
            f"ADS-B {adsb_status}  {self.adsb_count}"
            [:20]
        )

        self.lcd.cursor_pos = (2, 0)
        self.lcd.write_string(
            f"RID   {rid_status}  {self.rid_count}"
            [:20]
        )

        self.lcd.cursor_pos = (3, 0)
        self.lcd.write_string(
            f"MESH  {mesh_status}  {self.mesh_count}"
            [:20]
        )


    def refresh_gps(self):

        try:

            gps = get_gps_status()

        except Exception as e:

            log(
                "LCD",
                f"GPS refresh failed: {e}",
                level="WARNING"
            )

            gps = {
                "available": False,
                "fix": False
            }

        self.gps_available = bool(
            gps.get("available")
        )

        self.gps_fix = bool(
            gps.get("fix")
        )

        self.gps_mode = gps.get(
            "mode"
        )

        self.gps_satellites = gps.get(
            "satellites"
        )

        self.gps_hdop = gps.get(
            "hdop"
        )

        
    def show_gps(self):

        if not self.available:
            return

        self.lcd.clear()

        self.lcd.write_string(
            "GPS"
        )

        if not self.gps_available:

            fix_text = "GPSD OFF"

        elif not self.gps_fix:

            fix_text = "NO FIX"

        elif self.gps_mode == 3:

            fix_text = "FIX:3D"

        elif self.gps_mode == 2:

            fix_text = "FIX:2D"

        else:

            fix_text = "FIX:OK"

        satellites = (
            str(self.gps_satellites)
            if self.gps_satellites is not None
            else "--"
        )

        hdop = (
            f"{self.gps_hdop:.1f}"
            if isinstance(
                self.gps_hdop,
                (int, float)
            )
            else "--"
        )

        self.lcd.cursor_pos = (1, 0)
        self.lcd.write_string(
            fix_text[:20]
        )

        self.lcd.cursor_pos = (2, 0)
        self.lcd.write_string(
            f"SAT:{satellites}"[:20]
        )

        self.lcd.cursor_pos = (3, 0)
        self.lcd.write_string(
            f"HDOP:{hdop}"[:20]
        )

    def show_shutdown(self):

        if not self.available:
            return

        self.lcd.clear()

        self.lcd.write_string(
            "Drone Sky Check"
        )

        self.lcd.cursor_pos = (1, 0)
        self.lcd.write_string(
            "Mini Tracker"
        )

        self.lcd.cursor_pos = (3, 0)
        self.lcd.write_string(
            "Safe shutdown..."
        )
        
lcd = LCDService()
