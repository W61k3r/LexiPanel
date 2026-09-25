#!/bin/bash
# LAN-only exposure. Right now llama-server (8081) and the telemetry server
# (8082) listen on 0.0.0.0 with NO authentication - anything that can reach
# this box can use the model. This restricts the raw ports to the local subnet
# and leaves 443 (the authenticated panel) as the general entry point.
set -e
sudo ufw allow from 192.0.2.0/24 to any port 22   proto tcp comment 'ssh lan'
sudo ufw allow from 192.0.2.0/24 to any port 443  proto tcp comment 'panel lan'
sudo ufw allow from 192.0.2.0/24 to any port 8081 proto tcp comment 'llama api lan'
sudo ufw deny  8081
sudo ufw deny  8082
sudo ufw --force enable
sudo ufw status verbose
