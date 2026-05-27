import psutil
import socket
import subprocess


def get_interface_ip(interface):
    addrs = psutil.net_if_addrs()
    if interface not in addrs:
        return None

    for addr in addrs[interface]:
        if addr.family == socket.AF_INET:
            return addr.address

    return None


def get_wifi_ssid():
    try:
        result = subprocess.check_output(
            [
                "/usr/bin/nmcli",
                "-t",
                "-f",
                "ACTIVE,SSID",
                "device",
                "wifi"
            ],
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

    return {
        "ethernet": {
            "connected": eth_up,
            "ip": get_interface_ip("eth0")
        },
        "wifi": {
            "connected": wlan_up,
            "ip": get_interface_ip("wlan0"),
            "ssid": get_wifi_ssid()
        },
        "internet": has_internet(),
        "ap_mode": False
    }