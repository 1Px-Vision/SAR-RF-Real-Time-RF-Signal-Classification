#192.168.137.10/24
#Ip static

# Detect the full path to `ip` binary
IP_BIN=$(command -v ip)

# Create the systemd unit
sudo tee /etc/systemd/system/set-static-eth0.service > /dev/null <<EOF
[Unit]
Description=Set static IP on KR260 eth0
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c '${IP_BIN} addr flush dev eth0; ${IP_BIN} addr add 192.168.137.10/24 dev eth0; ${IP_BIN} link set eth0 up; ${IP_BIN} route add default via 192.168.137.1'
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF


#Enable and start the service

sudo systemctl daemon-reload
sudo systemctl enable set-static-eth0.service
sudo systemctl start set-static-eth0.service


#Verify it worked
ip addr show dev eth0
ip route show

#DHCP won’t overwrite the IP
ps aux | grep -E "udhcpc|dhclient|dhcpcd"

sudo pkill udhcpc
sudo pkill dhclient

