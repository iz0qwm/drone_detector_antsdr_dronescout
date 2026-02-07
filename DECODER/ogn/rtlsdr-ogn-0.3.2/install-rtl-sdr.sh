cd
sudo apt-get -y install libusb-1.0-0-dev cmake make
git clone https://github.com/rtlsdrblog/rtl-sdr
cd rtl-sdr
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/usr -DINSTALL_UDEV_RULES=ON
make
sudo make install
cd

sudo cat >> /etc/modprobe.d/rtl-glidernet-blacklist.conf <<EOF
blacklist rtl2832
blacklist r820t
blacklist rtl2830
blacklist dvb_usb_rtl28xxu
EOF


