#!/usr/bin/env bash
# One-time setup on a team member's own Mac or Linux laptop so `ssh hpc` reaches the CoE HPC (coe-hpc2).
#   bash scripts/hpc_ssh_setup.sh <your SJSU ID, e.g. 012345678>
# Needs the SJSU VPN (Cisco Secure Client, vpn.sjsu.edu, group Student-SSO) or the campus network.
# Safe to rerun: it never overwrites an existing key or config entry. You type your SJSU password once,
# when ssh-copy-id installs the key; after that the key is used and no password is asked.
set -euo pipefail
ID=${1:?usage: bash scripts/hpc_ssh_setup.sh <your SJSU ID>}
KEY="$HOME/.ssh/hpc_ed25519"
CONF_DIR="$HOME/.ssh/config.d"
CONF="$CONF_DIR/hpc.conf"
mkdir -p "$HOME/.ssh" "$CONF_DIR" "$HOME/.ssh/cm"
chmod 700 "$HOME/.ssh" "$CONF_DIR" "$HOME/.ssh/cm"

[ -f "$KEY" ] || ssh-keygen -t ed25519 -f "$KEY" -C "$ID-hpc-$(date +%Y%m%d)"
if [ "$(uname)" = Darwin ]; then ssh-add --apple-use-keychain "$KEY" 2>/dev/null || true; fi

if [ ! -f "$CONF" ]; then
  cat > "$CONF" <<CFG
# SJSU CoE HPC (CMPE 295A). Needs the SJSU VPN (or campus network).
Host hpc coe-hpc2
  HostName coe-hpc2.sjsu.edu
  User $ID
  IdentityFile ~/.ssh/hpc_ed25519
  IdentitiesOnly yes
  AddKeysToAgent yes
  IgnoreUnknown UseKeychain
  UseKeychain yes
  StrictHostKeyChecking accept-new
  ServerAliveInterval 60
  ConnectTimeout 15
  ControlMaster auto
  ControlPath ~/.ssh/cm/%C
  ControlPersist 30m
CFG
  chmod 600 "$CONF"
fi
touch "$HOME/.ssh/config"; chmod 600 "$HOME/.ssh/config"
if ! grep -q '^Include config.d/\*' "$HOME/.ssh/config"; then
  cp "$HOME/.ssh/config" "$HOME/.ssh/config.bak.$(date +%Y%m%d%H%M%S)"
  { echo 'Include config.d/*'; echo; cat "$HOME/.ssh/config"; } > "$HOME/.ssh/config.new"
  mv "$HOME/.ssh/config.new" "$HOME/.ssh/config"; chmod 600 "$HOME/.ssh/config"
fi

if ssh -o BatchMode=yes hpc true 2>/dev/null; then
  echo "key already installed"
else
  echo "Installing the key on the cluster: type your SJSU password when asked (only this once)."
  ssh-copy-id -i "$KEY.pub" hpc
fi
ssh hpc 'echo "OK: $(hostname) as $(whoami); groups: $(id -nG)"; sacctmgr -n show assoc where user=$USER format=account -P | head -1 | sed "s/^/slurm account: /"'
