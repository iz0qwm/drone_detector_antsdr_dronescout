import subprocess
import re
import ipaddress

from config import SETTINGS

AP_SSID = SETTINGS["ap_ssid"]

def list_connections():
    try:
        result = subprocess.check_output(
            ["/usr/bin/nmcli", "-t", "-f", "NAME,TYPE,DEVICE", "connection", "show"]
        ).decode().splitlines()

        result = clean_nmcli_output(result)
        connections = []

        for line in result:
            parts = line.rsplit(":", 2)

            if len(parts) >= 3:
                if parts[1] == "loopback":
                    continue

                connections.append({
                    "name": parts[0],
                    "type": parts[1],
                    "device": parts[2]
                })

        return connections

    except Exception as e:
        return {"error": str(e)}


def scan_wifi():

    saved_connections = set()
    try:
        saved = subprocess.check_output(
            [
                "/usr/bin/nmcli",
                "-t",
                "-f",
                "NAME,TYPE",
                "connection",
                "show"
            ]
        ).decode().splitlines()

        for line in saved:
            parts = line.rsplit(":", 1)

            if len(parts) == 2:
                name, conn_type = parts

                if conn_type == "802-11-wireless":
                    saved_connections.add(name)

    except:
        pass

    try:
        result = subprocess.check_output(
            ["/usr/bin/nmcli", "-t", "-f", "SSID,SIGNAL,SECURITY", "device", "wifi", "list"]
        ).decode().splitlines()

        networks = []
        seen = set()

        for line in result:
            parts = line.rsplit(":", 2)

            if len(parts) >= 3:
                ssid = parts[0].strip()

                if not ssid:
                    continue

                if ssid in seen:
                    continue

                seen.add(ssid)

                networks.append({
                    "ssid": ssid,
                    "signal": int(parts[1]) if parts[1].isdigit() else 0,
                    "security": parts[2],
                    "saved": ssid in saved_connections
                })

        return networks

    except Exception as e:
        return {"error": str(e)}


def connect_wifi(ssid, password):
    try:
        cmd = [
            "/usr/bin/sudo",
            "/usr/bin/nmcli",
            "device",
            "wifi",
            "connect",
            ssid
        ]

        if password:
            cmd.extend(["password", password])

        result = subprocess.check_output(
            cmd,
            stderr=subprocess.STDOUT
        ).decode()

        return {
            "success": True,
            "message": result.strip()
        }

    except subprocess.CalledProcessError as e:
        error_msg = e.output.decode().strip()

        try:
            connections = subprocess.check_output(
                [
                    "/usr/bin/nmcli",
                    "-t",
                    "-f",
                    "NAME,TYPE",
                    "connection",
                    "show"
                ]
            ).decode().splitlines()

            for line in connections:
                parts = line.rsplit(":", 1)

                if len(parts) == 2:
                    name, conn_type = parts

                    if conn_type == "802-11-wireless" and ssid in name:
                        subprocess.run(
                            [
                                "/usr/bin/sudo",
                                "/usr/bin/nmcli",
                                "connection",
                                "delete",
                                name
                            ],
                            stderr=subprocess.DEVNULL
                        )

            retry_cmd = [
                "/usr/bin/sudo",
                "/usr/bin/nmcli",
                "device",
                "wifi",
                "connect",
                ssid
            ]

            if password:
                retry_cmd.extend(["password", password])

            retry_result = subprocess.check_output(
                retry_cmd,
                stderr=subprocess.STDOUT
            ).decode()

            return {
                "success": True,
                "message": retry_result.strip()
            }

        except Exception:
            return {
                "success": False,
                "message": error_msg
            }


def disconnect_wifi():
    try:
        result = subprocess.check_output(
            [  
                "/usr/bin/sudo",
                "/usr/bin/nmcli",
                "device",
                "disconnect",
                "wlan0"
            ],
            stderr=subprocess.STDOUT
        ).decode()

        return {
            "success": True,
            "message": result.strip()
        }

    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "message": e.output.decode().strip()
        }

def clean_nmcli_output(lines):
    ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

    cleaned = []
    for line in lines:
        line = ansi_escape.sub('', line).replace('\r', '').strip()
        if line:
            cleaned.append(line)

    return cleaned


def start_hotspot():
    status = hotspot_status()

    if status["active"]:
        return {
            "success": True,
            "message": "Hotspot already active"
        }

    try:
        result = subprocess.check_output(
            [
                "/usr/bin/sudo",
                "/usr/bin/nmcli",
                "device",
                "wifi",
                "hotspot",
                "ifname",
                "wlan0",
                "ssid",
                AP_SSID,
                "password",
                "tracker123"
            ],
            stderr=subprocess.STDOUT
        ).decode()

        return {
            "success": True,
            "message": result.strip()
        }

    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "message": e.output.decode().strip()
        }


def stop_hotspot():
    try:
        connections = subprocess.check_output(
            [
                "/usr/bin/nmcli",
                "-t",
                "-f",
                "NAME,TYPE",
                "connection",
                "show",
                "--active"
            ]
        ).decode().splitlines()

        for line in connections:
            parts = line.rsplit(":", 1)

            if len(parts) == 2:
                name, conn_type = parts

                if conn_type == "802-11-wireless" and (
                    "Hotspot" in name or name == AP_SSID
                ):
                    subprocess.check_output(
                        [
                            "/usr/bin/sudo",
                            "/usr/bin/nmcli",
                            "connection",
                            "down",
                            name
                        ],
                        stderr=subprocess.STDOUT
                    )

                    return {
                        "success": True,
                        "message": f"Hotspot {name} stopped"
                    }

        return {
            "success": False,
            "message": "No hotspot active"
        }

    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "message": e.output.decode().strip()
        }


def hotspot_status():
    try:
        connections = subprocess.check_output(
            [
                "/usr/bin/nmcli",
                "-t",
                "-f",
                "NAME,TYPE,DEVICE",
                "connection",
                "show",
                "--active"
            ]
        ).decode().splitlines()

        for line in connections:
            parts = line.rsplit(":", 2)

            if len(parts) == 3:
                name, conn_type, device = parts

                if conn_type == "802-11-wireless" and device == "wlan0":
                    if name == "Portable-Air-Node" or "Hotspot" in name:
                        return {
                            "active": True,
                            "ssid": name
                        }

        return {
            "active": False,
            "ssid": None
        }

    except Exception:
        return {
            "active": False,
            "ssid": None
        }


def mask_to_prefix(mask):
    try:
        return ipaddress.IPv4Network(f"0.0.0.0/{mask}").prefixlen
    except Exception:
        return None


def prefix_to_mask(prefix):
    try:
        return str(ipaddress.IPv4Network(f"0.0.0.0/{prefix}").netmask)
    except Exception:
        return ""


def get_lan_config():
    try:
        result = subprocess.check_output(
            [
                "/usr/bin/nmcli",
                "-g",
                "ipv4.addresses,ipv4.gateway",
                "connection",
                "show",
                "netplan-eth0"
            ]
        ).decode().splitlines()

        addresses = result[0].split(",") if result else []

        secondary_ip = ""
        secondary_mask = "255.255.255.0"

        for addr in addresses:
            addr = addr.strip()

            if not addr:
                continue

            if addr.startswith("192.168.1.115"):
                continue

            if "/" in addr:
                ip, prefix = addr.split("/", 1)
                secondary_ip = ip
                secondary_mask = prefix_to_mask(prefix)
            else:
                secondary_ip = addr

        gateway = result[1] if len(result) > 1 else ""

        return {
            "success": True,
            "ip": secondary_ip,
            "mask": secondary_mask,
            "gateway": gateway
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }


def set_secondary_lan(ip, mask, gateway=None):
    try:
        if not ip:
            return {
                "success": False,
                "message": "User LAN IP missing"
            }

        prefix = mask_to_prefix(mask)

        if prefix is None:
            return {
                "success": False,
                "message": "Invalid subnet mask"
            }

        ipaddress.IPv4Address(ip)

        if gateway:
            ipaddress.IPv4Address(gateway)

        management_ip = "192.168.1.115/24"
        secondary_ip = f"{ip}/{prefix}"

        addresses = f"{management_ip},{secondary_ip}"

        subprocess.check_output(
            [
                "/usr/bin/sudo",
                "/usr/bin/nmcli",
                "connection",
                "modify",
                "netplan-eth0",
                "ipv4.method",
                "manual",
                "ipv4.addresses",
                addresses
            ],
            stderr=subprocess.STDOUT
        )

        if gateway:

            subprocess.check_output(
                [
                    "/usr/bin/sudo",
                    "/usr/bin/nmcli",
                    "connection",
                    "modify",
                    "netplan-eth0",
                    "ipv4.gateway",
                    gateway
                ],
                stderr=subprocess.STDOUT
            )

        else:

            subprocess.check_output(
                [
                    "/usr/bin/sudo",
                    "/usr/bin/nmcli",
                    "connection",
                    "modify",
                    "netplan-eth0",
                    "ipv4.gateway",
                    ""
                ],
                stderr=subprocess.STDOUT
            )

        subprocess.check_output(
            [
                "/usr/bin/sudo",
                "/usr/bin/nmcli",
                "connection",
                "up",
                "netplan-eth0"
            ],
            stderr=subprocess.STDOUT
        )

        return {
            "success": True,
            "message": "User LAN configuration updated"
        }

    except subprocess.CalledProcessError as e:
        return {
            "success": False,
            "message": e.output.decode().strip()
        }

    except Exception as e:
        return {
            "success": False,
            "message": str(e)
        }