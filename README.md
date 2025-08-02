# 📡 Drone Detector con ANTSDR E200 e Raspberry Pi

Questo progetto permette di utilizzare l'SDR **ANTSDR E200** insieme a una **Raspberry Pi** per rilevare droni DJI con DroneID e droni con Remote ID, visualizzandoli su una mappa web e inviandoli a una rete Firebase (es. Drone Pilot App).

---

## ✅ Caratteristiche principali

* **Rilevamento DJI DroneID** tramite ANTSDR E200
* **Rilevamento Remote ID** tramite DroneScout Bridge
* **Server Web con mappa** integrato su Raspberry Pi
* **Invio automatico dei dati** alla rete locale o a Firebase

---

## 📦 Struttura del progetto su Raspberry Pi

```
/home/pi/
├── bridge_uploader.py          # Server web + ricezione dati da DJI e Remote ID
├── start_all.sh                # Script di avvio completo
├── stop_all.sh                 # Script di arresto completo
├── static/                     # Cartella con la mappa e frontend web
│
├── trackers/                  
│   ├── dji_receiver.py         # Riceve dati UDP da ANTSDR e li invia a bridge
│   └── service_controller.sh   # Avvio/stop servizi tracker
│
└── remotetrack/
    └── ... (dronescout mod)    # Codice modificato per invio dati Remote ID a bridge
```

---

## 🛠️ Installazione firmware ANTSDR DJI

1. Imposta la porta LAN del Raspberry Pi a IP **192.168.1.9**
2. Inserisci la SD card con i file del firmware DJI forniti dal produttore
3. L'ANTSDR sarà disponibile su IP **192.168.1.10** e inizierà a trasmettere i pacchetti DJI DroneID sulla porta UDP 41030

---

## ▶️ Avvio del sistema

Per avviare tutto il sistema:

```bash
cd /home/pi
./start_all.sh
```

Per arrestarlo:

```bash
./stop_all.sh
```

Questi script gestiscono sia il `dji_receiver.py` che i servizi del Remote ID.

---

## 🔁 DJI DroneID Tracker: `dji_receiver.py`

Si trova in:

```
/home/pi/trackers/dji_receiver.py
```

Riceve pacchetti UDP da ANTSDR e li invia via HTTP al server locale (`bridge_uploader.py`).

---

## 🌐 Web Server e Bridge: `bridge_uploader.py`

File principale:

```
/home/pi/bridge_uploader.py
```

Funzionalità:

* Riceve dati via HTTP da `dji_receiver.py` e `remotetrack`
* Visualizza in tempo reale i droni rilevati su una **mappa web** accessibile da browser
* Espone la mappa su `http://<IP_RPI>:8080`

La parte frontend è contenuta nella cartella:

```
/home/pi/static/
```

---

## 📡 Remote ID Tracker: `remotetrack/`

Questa cartella contiene la versione modificata del software DroneScout Bridge, in grado di:

* Ricevere pacchetti Remote ID via USB
* Inviarli via HTTP a `bridge_uploader.py`

---

## 📌 Note

* L'indirizzo IP **fisso** dell'ANTSDR E200 deve essere `192.168.1.10`
* Il Raspberry Pi deve avere IP `192.168.1.9`
* Non è necessario modificare l'indirizzo IP del firmware

---

## ⚠️ Legale

L'utilizzo di questo sistema può essere soggetto a regolamenti locali in materia di radiofrequenze e privacy. Verificare sempre la conformità normativa prima dell'utilizzo in campo.
