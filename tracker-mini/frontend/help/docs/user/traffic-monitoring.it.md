# Monitoraggio Traffico

### Parte della Guida Utente Mini Tracker

---

## Scopo

Questo documento descrive come Mini Tracker supporta la consapevolezza situazionale dell'operatore combinando sorgenti indipendenti di traffico e posizione in un unico quadro operativo.

Il monitoraggio traffico aiuta l'operatore a comprendere la posizione di aeromobili, droni, alianti e operatori del team rispetto all'area operativa.

L'obiettivo non è sostituire sistemi avionici specializzati, ma presentare informazioni rilevanti sul campo in un'unica vista operativa pratica.

---

## Panoramica

Mini Tracker presenta informazioni sul traffico e sul team sulla mappa operativa come parte di un unico quadro situazionale.

La Dashboard può visualizzare:

- Aeromobili ADS-B
- Droni Remote ID
- Traffico OGN / FLARM
- Posizioni operatori Meshtastic

Ogni sorgente rimane indipendente, ma l'operatore le visualizza insieme in un'unica immagine operativa.

---

## Sorgenti di traffico

| Sorgente | Scopo operativo |
|----------|-----------------|
| **ADS-B** | Consapevolezza degli aeromobili cooperativi dotati di trasmettitore ADS-B |
| **Remote ID** | Consapevolezza dei droni che trasmettono dati di identificazione supportati |
| **OGN / FLARM** | Consapevolezza di alianti e aviazione leggera da sorgenti di rete |
| **Operatori Meshtastic** | Consapevolezza delle posizioni degli operatori di missione |

---

### ADS-B

ADS-B fornisce consapevolezza degli aeromobili cooperativi.

Mini Tracker supporta sia informazioni ADS-B locali che di rete:

- **ADS-B locale (ADSBRx)**: dati dal ricevitore e decodificatore locali — funziona offline
- **ADS-B di rete (ADSBNet)**: dati da provider Internet — opzionale, richiede connettività

Per impostazione predefinita, il traffico è filtrato per focalizzarsi sulle quote più basse. Gli elicotteri sono sempre visualizzati.

### Remote ID

Remote ID fornisce consapevolezza dei droni rilevati dal ricevitore supportato (DS110).

I marker dei droni possono includere modello, produttore, numero di serie e sorgente.

### OGN / FLARM

Traffico OGN/FLARM da sorgenti di rete supportate (FLARM, SafeSky, FreeFlight, FANET). Richiede connettività Internet.

### Operatori Meshtastic

Quando Meshtastic è abilitato, Mini Tracker visualizza le posizioni degli operatori del team sulla mappa.

---

## Consapevolezza di Prossimità del Traffico

Mini Tracker fornisce la **Consapevolezza di Prossimità del Traffico** per aiutare l'operatore a capire quanto sono vicini i droni rilevati agli aeromobili nelle vicinanze.

> **La Consapevolezza di Prossimità del Traffico è solo informativa. Non è un sistema certificato di prevenzione delle collisioni e non deve essere utilizzata come unica base per decisioni di separazione.**

### Cosa mostra

Quando uno o più droni sono rilevati e ci sono aeromobili nel raggio di valutazione configurato, Mini Tracker calcola e visualizza automaticamente informazioni di prossimità orizzontale:

- Una **linea di distanza** dalla coppia drone-aeromobile più critica
- **Anelli di prossimità** attorno a fino a cinque aeromobili in stati di prossimità
- Un **pannello Traffico Vicino** che elenca le coppie a maggiore severità

### Stati di prossimità

| Stato | Significato | Indicazione visiva |
|-------|-------------|-------------------|
| MONITOR (MON) | Aeromobile entro soglia esterna | Linea tratteggiata blu |
| CAUTION (CTN) | Aeromobile entro soglia intermedia | Linea tratteggiata arancione |
| WARNING (WRN) | Aeromobile entro soglia interna | Linea continua rossa (con pulsazione) |
| STALE (STL) | Dati del tracciato obsoleti | Linea puntinata grigia |

Ogni stato è distinguibile per colore, tipo di linea E etichetta testuale (non solo colore).

### Pannello Traffico Vicino

Un pannello compatto appare in basso a destra nella mappa quando esistono coppie in prossimità. Mostra:

- Identificativo drone → Callsign aeromobile
- Distanza orizzontale (metri o chilometri)
- Badge stato (MON, CTN, WRN, STL)
- Tendenza movimento: **APR** (in avvicinamento), **DIV** (in allontanamento), **STB** (stabile), **—** (dati insufficienti)

Cliccando una voce del pannello la mappa si centra sull'aeromobile corrispondente.

### Tendenza di movimento

Quando è disponibile una cronologia di posizione sufficiente (almeno 10 secondi di osservazioni), il sistema determina se un aeromobile si sta avvicinando, allontanando o mantiene una distanza stabile da un drone.

### Funzionamento offline

La Consapevolezza di Prossimità funziona completamente offline usando il ricevitore ADS-B locale (ADSBRx) e Remote ID. La connettività Internet non è necessaria.

Quando Internet è disponibile e ADSBNet è abilitato, i dati di rete arricchiscono il quadro locale.

### Configurazione

Soglie predefinite di prossimità:

| Stato | Distanza di ingresso | Distanza di uscita |
|-------|---------------------|-------------------|
| MONITOR | 3000 m | 3300 m |
| CAUTION | 1500 m | 1800 m |
| WARNING | 500 m | 700 m |

Le soglie sono configurabili. Sono parametri operativi, non limiti regolamentari.

### Limitazioni

- Solo distanza orizzontale — nessuna valutazione di separazione verticale
- I riferimenti altimetrici tra ADS-B e Remote ID sono incompatibili
- La tendenza di movimento richiede almeno 10 secondi di cronologia posizione
- Dati mancanti o obsoleti sono indicati anziché nascosti

---

## Flusso operativo tipico

1. Aprire la Dashboard
2. Verificare la copertura della mappa sull'area operativa
3. Controllare gli indicatori delle sorgenti traffico
4. Abilitare le sorgenti necessarie per l'operazione
5. Monitorare aeromobili, droni, alianti e posizioni team sulla mappa
6. Selezionare i marker per ulteriori dettagli
7. Regolare la visibilità quando la mappa diventa troppo affollata

---

## Note operative

- ADS-B locale richiede il ricevitore e decodificatore attivi
- ADS-B di rete e OGN / FLARM richiedono connettività Internet
- Remote ID dipende dallo stato del ricevitore e dalla portata radio
- Gli oggetti traffico possono scomparire quando i dati diventano obsoleti
- La Dashboard aggiorna le informazioni sul traffico periodicamente
- Trattare l'assenza di traffico visibile come una limitazione operativa, non come conferma che l'area sia sgombra
