import psutil
import socket
import subprocess

from config import SETTINGS

ADMIN_LAN_IP = SETTINGS["admin_lan_ip"]

AP_INTERFACE = "wlan0"
CLIENT_INTERFACE = "wlan1"
AP_SSID = SETTINGS["ap_ssid"]


def get_interface_ipv4_addresses(interface):
    addrs = psutil.net_if_addrs()

    if interface not in addrs:
        return []

    ips = []

    for addr in addrs[interface]:
        if addr.family == socket.AF_INET:
            ips.append(addr.address)

    return ips


def get_wifi_ssid(interface):
    try:
        result = subprocess.check_output(
            [
                "/usr/bin/nmcli",
                "-t",
                "-f",
                "DEVICE,ACTIVE,SSID",
                "device",
                "wifi"
            ],
            stderr=subprocess.DEVNULL
        ).decode().splitlines()

        for line in result:
            parts = line.split(":", 2)

            if len(parts) == 3:
                device, active, ssid = parts

                if device == interface and active == "yes":
                    return ssid

        return None

    except Exception:
        return None


def has_internet():
    try:
        socket.create_connection(
            ("8.8.8.8", 53),
            timeout=2
        )
        return True

    except Exception:
        return False


def get_admin_lan_status():
    stats = psutil.net_if_stats()

    eth_up = (
        stats.get("eth0").isup
        if "eth0" in stats
        else False
    )

    eth_ips = get_interface_ipv4_addresses(
        "eth0"
    )

    return {
        "connected": (
            eth_up and
            ADMIN_LAN_IP in eth_ips
        ),
        "ip": (
            ADMIN_LAN_IP
            if ADMIN_LAN_IP in eth_ips
            else None
        )
    }


def get_network_status():
    admin_lan = get_admin_lan_status()

    stats = psutil.net_if_stats()

    eth_up = (
        stats.get("eth0").isup
        if "eth0" in stats
        else False
    )

    ap_up = (
        stats.get(AP_INTERFACE).isup
        if AP_INTERFACE in stats
        else False
    )

    client_up = (
        stats.get(CLIENT_INTERFACE).isup
        if CLIENT_INTERFACE in stats
        else False
    )

    eth_ips = get_interface_ipv4_addresses("eth0")
    ap_ips = get_interface_ipv4_addresses(AP_INTERFACE)
    client_ips = get_interface_ipv4_addresses(CLIENT_INTERFACE)

    user_lan_ip = None

    for ip in eth_ips:
        if ip != ADMIN_LAN_IP:
            user_lan_ip = ip
            break

    ap_ssid = get_wifi_ssid(AP_INTERFACE)
    client_ssid = get_wifi_ssid(CLIENT_INTERFACE)

    return {
        "admin_lan": admin_lan,

        "user_lan": {
            "connected": eth_up and user_lan_ip is not None,
            "ip": user_lan_ip
        },

        "wifi_ap": {
            "connected": ap_ssid is not None,
            "ip": ap_ips[0] if ap_ips else None,
            "ssid": ap_ssid or AP_SSID,
            "ap_mode": ap_ssid == AP_SSID
        },

        "wifi_client": {
            "connected": client_up and client_ssid is not None,
            "ip": client_ips[0] if client_ips else None,
            "ssid": client_ssid
        },

        # Compatibilità temporanea con il frontend attuale
        "wifi": {
            "connected": client_up and client_ssid is not None,
            "ip": client_ips[0] if client_ips else None,
            "ssid": client_ssid,
            "ap_mode": False
        },

        "internet": has_internet()
    }