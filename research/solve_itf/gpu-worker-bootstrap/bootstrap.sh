#!/bin/bash
set -euo pipefail
trap 'python3 /root/itf-stage.py error "line=$LINENO,status=$?"' ERR
python3 /root/itf-stage.py begin
if command -v ufw >/dev/null; then ufw allow 80/tcp; ufw allow 443/tcp; fi
systemctl daemon-reload
systemctl enable --now itf-bootstrap-probe
systemctl enable --now itf-worker
python3 /root/itf-stage.py download-caddy
curl --connect-timeout 15 --max-time 120 --fail --location --silent --show-error https://github.com/caddyserver/caddy/releases/download/v2.10.2/caddy_2.10.2_linux_amd64.tar.gz -o /root/caddy.tar.gz
printf '%s  %s\n' '5c218bc34c9197369263da7e9317a83acdbd80ef45d94dca5eff76e727c67cdd' /root/caddy.tar.gz | sha256sum -c -
tar -xzf /root/caddy.tar.gz -C /usr/local/bin caddy
mkdir -p /etc/caddy
cat > /etc/systemd/system/caddy.service <<'UNIT'
[Unit]
After=network-online.target
Wants=network-online.target
[Service]
ExecStart=/usr/local/bin/caddy run --config /etc/caddy/Caddyfile --adapter caddyfile
Restart=on-failure
Environment=HOME=/root
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
python3 /root/itf-stage.py configure-tls
ip=$(curl --max-time 10 -fsS http://169.254.169.254/metadata/v1/interfaces/public/0/ipv4/address)
host="${ip//./-}.sslip.io"
cat > /etc/caddy/Caddyfile <<EOF
http://$host {
    respond /bootstrap-status "caddy-configured" 200
}
https://$host {
    reverse_proxy 127.0.0.1:8765
}
EOF
caddy validate --config /etc/caddy/Caddyfile
systemctl stop itf-bootstrap-probe
systemctl enable --now caddy
systemctl restart caddy
python3 /root/itf-stage.py ready
