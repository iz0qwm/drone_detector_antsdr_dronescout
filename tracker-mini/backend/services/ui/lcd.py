from threading import Thread
from time import sleep
from services.logger import log
from config import SETTINGS
from services.readsb import is_receiving


class LCDService:

    def __init__(self):

        self.page = "boot"
        self.lcd = None
        self.available = False
        self.running = False
        self.thread = None

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

    def _worker(self):

        self.show_boot()

        sleep(3)

        self.page = "status"

        while self.running:

            if self.page == "status":

                self.refresh_status()
                self.show_status()

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


    def refresh_status(self):

        self.adsb = is_receiving()

        #
        # Per ora li lasciamo fissi.
        # Li sistemiamo uno alla volta.
        #

        self.rid = True

        self.mesh = True

        self.ip = "192.168.1.115"

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

lcd = LCDService()