import psutil
import socket

def get_system_status():
    return {
        "hostname": socket.gethostname(),
        "cpu": psutil.cpu_percent(),
        "ram": psutil.virtual_memory().percent,
        "disk": psutil.disk_usage("/").percent,
        "uptime": psutil.boot_time()
    }