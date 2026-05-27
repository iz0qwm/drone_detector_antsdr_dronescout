import subprocess
import re

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
                "Portable-Air-Node",
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
                    "Hotspot" in name or name == "Portable-Air-Node"
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


