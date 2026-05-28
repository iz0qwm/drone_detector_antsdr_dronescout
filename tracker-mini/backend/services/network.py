import psutil
import socket
import subprocess

from config import SETTINGS

ADMIN_LAN_IP = SETTINGS["admin_lan_ip"]


def get_interface_ipv4_addresses(interface):
    addrs = psutil.net_if_addrs()

    if interface not in addrs:
        return []

    ips = []

    for addr in addrs[interface]:
        if addr.family == socket.AF_INET:
            ips.append(addr.address)

    return ips


def get_wifi_ssid():
    try:
        result = subprocess.check_output(
            ["/usr/bin/nmcli", "-t", "-f", "ACTIVE,SSID", "device", "wifi"],
            stderr=subprocess.DEVNULL
        ).decode().splitlines()

        for line in result:
            parts = line.split(":", 1)
            if len(parts) == 2:
                active, ssid = parts
                if active == "yes":
                    return ssid

        return None

    except:
        return None


def has_internet():
    try:
        socket.create_connection(("8.8.8.8", 53), timeout=2)
        return True
    except:
        return False


def get_network_status():
    stats = psutil.net_if_stats()

    eth_up = stats.get("eth0").isup if "eth0" in stats else False
    wlan_up = stats.get("wlan0").isup if "wlan0" in stats else False

    eth_ips = get_interface_ipv4_addresses("eth0")
    wlan_ips = get_interface_ipv4_addresses("wlan0")

    user_lan_ip = None

    for ip in eth_ips:
        if ip != ADMIN_LAN_IP:
            user_lan_ip = ip
            break

    wifi_ssid = get_wifi_ssid()

    return {
        "admin_lan": {
            "connected": eth_up,
            "ip": ADMIN_LAN_IP
        },
        "user_lan": {
            "connected": eth_up and user_lan_ip is not None,
            "ip": user_lan_ip
        },
        "wifi": {
            "connected": wlan_up,
            "ip": wlan_ips[0] if wlan_ips else None,
            "ssid": wifi_ssid,
            "ap_mode": wifi_ssid == "Portable-Air-Node"
        },
        "internet": has_internet()
    }