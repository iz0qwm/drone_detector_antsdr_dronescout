# Mini Tracker — Hardware Integration

## Platform

- **SBC**: Raspberry Pi (model to be confirmed on physical device)
- **OS**: Debian-based Raspberry Pi OS
- **Hostname**: `dsc-node02`
- **Admin LAN IP**: `192.168.1.115` (fixed, on `eth0`)

## Network Interfaces

| Interface | Role | Notes |
|-----------|------|-------|
| `eth0` | Admin LAN (fixed) + optional User LAN | `192.168.1.115/24` always present |
| `wlan0` | Wi-Fi Access Point | SSID `Portable-Air-Node`, managed by `nmcli hotspot` |
| `wlan1` | Wi-Fi Client (optional) | External USB adapter, used for Internet |

## Serial Devices

| Device | Purpose | Baud | Config Key |
|--------|---------|------|------------|
| `/dev/serial0` | DS110 Remote ID receiver (UART) | 115200 | `settings.ds110.device` |
| `/dev/serial/by-id/usb-Espressif_Systems_LilyGo_TBeam-S3-Core_48CA435BFC80-if00` | Meshtastic gateway (USB) | — | `settings.meshtastic.device` |

## I2C Devices

| Address | Device | Library |
|---------|--------|---------|
| `0x27` | PCF8574 LCD expander (20x4 display) | RPLCD, port 1 |

## GPIO Assignments

| GPIO | Function | Library |
|------|----------|---------|
| 5 | Rotary encoder pin A | gpiozero `RotaryEncoder` |
| 6 | Rotary encoder pin B | gpiozero `RotaryEncoder` |
| 16 | Rotary encoder button | gpiozero `Button` (pull_up, bounce 50ms) |

## System Services

| Service | Purpose |
|---------|---------|
| `tracker-mini.service` | Main Mini Tracker Flask application |
| `readsb-local.service` | ADS-B decoder (controlled by app via systemctl) |
| `gpsd` | GPS daemon on 127.0.0.1:2947 |
| NetworkManager | Manages eth0, wlan0, wlan1 via nmcli |

## Sudoers Requirements

The `pi` user must be able to run without password:
- `/usr/bin/nmcli` (network management)
- `/usr/bin/systemctl start/stop readsb-local.service`
- `/usr/bin/systemctl restart tracker-mini.service`
- `/usr/sbin/reboot`
- `/usr/sbin/shutdown -h now`

Sudoers file: `/etc/sudoers.d/tracker-mini`

## Hardware Safety Rules

Never change without inspecting current usage and documenting the reason:
- GPIO assignments (5, 6, 16)
- Serial ports and baud rates
- I2C addresses
- Network interface roles
- System services
- Startup/shutdown procedures
- Hardware initialization order

## Graceful Degradation

All hardware access must handle unavailability:
- LCD: `self.available = False` if init fails
- Rotary encoder: `self.encoder = None` if GPIO fails
- DS110: reconnect loop with 5s backoff
- Meshtastic: reconnect loop with 5s backoff
- GPS: `_connected = False`, retry on next call
- Wi-Fi Client: reported as missing if `/sys/class/net/wlan1` absent

## Power Considerations

- Input: 12VDC external regulated supply
- Internal conversion to Pi supply levels
- Unexpected power loss can corrupt filesystem writes
- Dashboard provides shutdown command for safe power-off
- LCD shows "Safe shutdown..." message before halt

## References

- `frontend/help/docs/hardware/overview.md`
- `frontend/help/docs/hardware/raspberry-pi.md`
- `frontend/help/docs/hardware/networking.md`
- `backend/services/ui/lcd.py`
- `backend/services/ds110.py`
- `backend/services/meshtastic_service.py`
