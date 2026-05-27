import psutil
import socket

def get_system_status():
    return {
        "hostname": socket.gethostname(),
        "cpu": psutil.cpu_percent(interval=0.2),
        "ram": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage("/").percent,
        "uptime": psutil.boot_time()
    }