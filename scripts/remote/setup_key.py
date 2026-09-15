# -*- coding: utf-8 -*-
"""scripts/remote/setup_key.py — 用密码登录一次，把本机公钥写入 authorized_keys，实现免密。"""
import os, sys
import paramiko

HOST = os.environ.get("TERED_SSH_HOST", "your-server-host")
PORT = int(os.environ.get("TERED_SSH_PORT", "22"))
USER = os.environ.get("TERED_SSH_USER", "user")
PW = os.environ["SRV_PW"]
PUB = open(os.path.expanduser("~/.ssh/id_ed25519.pub")).read().strip()

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(HOST, port=PORT, username=USER, password=PW, timeout=15)
cmd = (
    'mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && '
    'chmod 600 ~/.ssh/authorized_keys && '
    f'grep -qF "{PUB}" ~/.ssh/authorized_keys || echo "{PUB}" >> ~/.ssh/authorized_keys; '
    'echo KEY_OK'
)
_, out, err = c.exec_command(cmd)
print(out.read().decode(), err.read().decode())
c.close()
