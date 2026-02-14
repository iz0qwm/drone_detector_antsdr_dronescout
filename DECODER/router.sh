#!/bin/bash
echo -
echo - Controlliamo l interfaccia
echo -
ip -4 addr show
echo -
echo - puliamo le regole
echo -
sudo iptables -F
sudo iptables -t nat -F
echo -
echo - Attiviamo il forward
echo -
sudo sysctl -w net.ipv4.ip_forward=1
echo -
echo - Attiviamo iptables
echo -
sudo iptables -t nat -A POSTROUTING -o usb0 -j MASQUERADE
sudo iptables -A FORWARD -i usb0 -o eth0 -m state --state RELATED,ESTABLISHED -j ACCEPT
sudo iptables -A FORWARD -i eth0 -o usb0 -j ACCEPT
#sudo apt install iptables-persistent
echo -
echo - Controlliamo
echo -
sudo iptables -L -nv
