#!/bin/bash
# Ferma il servizio RF Explorer dual scan
set -e
systemctl --user stop rfe-dual-scan.service
echo "RF Explorer dual scan fermato"
