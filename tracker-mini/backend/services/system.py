import psutil
import socket
import subprocess

def get_system_status():
    return {
        "hostname": socket.gethostname(),
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage("/").percent,
        "uptime": psutil.boot_time()
    }


def restart_tracker():

    subprocess.run(
        [
            "/usr/bin/sudo",
            "/usr/bin/systemctl",
            "restart",
            "tracker-mini.service"
        ],
        check=True
    )


def reboot_system():

    subprocess.run(
        [
            "/usr/bin/sudo",
            "/usr/sbin/reboot"
        ],
        check=True
    )


def shutdown_system():

    subprocess.run(
        [
            "/usr/bin/sudo",
            "/usr/sbin/shutdown",
            "-h",
            "now"
        ],
        check=True
    )