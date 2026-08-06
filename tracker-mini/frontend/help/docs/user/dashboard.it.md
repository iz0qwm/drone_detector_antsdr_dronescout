# Dashboard

### Parte della Guida Utente Mini Tracker

---

## Scopo

Questo documento descrive la Dashboard, l'interfaccia operativa principale di Mini Tracker.

La Dashboard è una pagina web accessibile dal browser dell'operatore quando è connesso al Mini Tracker tramite Ethernet o Access Point WiFi.

---

## Panoramica

La Dashboard presenta il quadro operativo unificato: mappa, traffico aereo, droni, team e stato del sistema in un'unica vista.

Elementi principali:

- **Barra superiore**: indicatori LED dello stato dei servizi (NET, ADSB Rx, ADSB Net, RID, OGN, MESH, DSC)
- **Mappa**: vista Leaflet a schermo intero con mappe offline o online
- **Drawer laterale**: pannelli per Network, Maps, System, DSC e Missions
- **Pannello Traffico Vicino**: informazioni di prossimità drone-aeromobile (quando attivo)
- **Barra missione**: strumenti di disegno per la pianificazione missione

---

## Accesso

La Dashboard è raggiungibile a:

```
http://192.168.1.115:5000
```

oppure tramite l'Access Point WiFi `Portable-Air-Node`.

---

## Indicatori di stato

La barra superiore mostra LED colorati per ogni servizio:

| LED | Servizio | Verde | Rosso |
|-----|----------|-------|-------|
| NET | Internet | Connesso | Non disponibile |
| ADSB Rx | Ricevitore locale | Attivo e riceve | Non attivo |
| ADSB Net | Traffico da rete | Dati disponibili | Non disponibile |
| RID | Remote ID | Ricevitore attivo | Non attivo |
| OGN | OGN/FLARM | Dati disponibili | Non disponibile |
| MESH | Meshtastic | Gateway connesso | Non attivo |
| DSC | Drone Sky Check | Sincronizzato | Non connesso |

---

## Mappa

La mappa supporta:

- **Modalità automatica**: usa mappe online quando Internet è disponibile, offline altrimenti
- **Mappe offline**: file MBTiles precaricati per operare senza Internet
- **Mappe online**: OpenTopoMap quando la connettività lo permette
- **Modalità scura**: overlay per ridurre la luminosità

---

## Drawer laterale

Aprire con il pulsante ☰ in alto a destra. Contiene:

- **Network**: stato rete, configurazione WiFi, LAN utente
- **Maps**: gestione mappe, download, sorgente mappa
- **System**: stato CPU/RAM/disco, sorgenti traffico, impostazioni hardware, aggiornamenti, power
- **DSC**: configurazione nodo Drone Sky Check
- **Missions**: creazione missione, pianificazione, team

---

## Sorgenti traffico

Dal pannello System è possibile abilitare/disabilitare:

- ADS-B Network
- OGN Network
- Remote ID (DS110)
- Meshtastic
- ADS-B Local (RTL-SDR)
- Mostra aeromobili sopra 1000m

---

## Pulsanti rapidi

In fondo al drawer:

- **📜 Logs**: visualizzatore log dell'applicazione
- **📖 Help**: apre questo manuale in una nuova scheda

---

## Consapevolezza di Prossimità

Quando droni e aeromobili sono presenti, la Dashboard mostra automaticamente:

- Linea di distanza verso l'aeromobile più critico
- Anelli colorati sugli aeromobili in prossimità
- Pannello "Nearby Traffic" in basso a destra

Per dettagli completi vedere la sezione **Monitoraggio Traffico**.
