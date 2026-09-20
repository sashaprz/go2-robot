"""Local certificates for the phone page (Windows side). iOS only lets a web page use the microphone and motion sensors over HTTPS,
and it only trusts a certificate you have told it to trust. So:

  * a small private ROOT certificate ("go2-robot local CA") is made once and lives in .phone_tls/. YOU install it on the iPhone once
    (the phone page's setup screen walks through it). It can only vouch for what this program signs with its key, which never leaves
    this folder (git-ignored).
  * a SERVER certificate signed by it, listing this PC's host name and every IP address it has right now. It is remade automatically
    when those change (a different Wi-Fi), and the phone doesn't have to be touched again because it trusts the root.

Uses the openssl that comes with Git for Windows (or any openssl on PATH).
"""
from __future__ import annotations

import os
import re
import shutil
import socket
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
TLS_DIR = os.path.join(HERE, ".phone_tls")
OPENSSL_CANDIDATES = (r"C:\Program Files\Git\usr\bin\openssl.exe", r"C:\Program Files (x86)\Git\usr\bin\openssl.exe")


def find_openssl() -> str:
    for c in (shutil.which("openssl"),) + OPENSSL_CANDIDATES:
        if c and os.path.exists(c):
            return c
    raise RuntimeError("openssl not found (it ships with Git for Windows: C:\\Program Files\\Git\\usr\\bin\\openssl.exe)")


def local_ipv4s() -> list[str]:
    """Every IPv4 address this PC has (ipconfig on Windows; falls back to the host-name lookup)."""
    ips: list[str] = []
    try:
        out = subprocess.run(["ipconfig"], capture_output=True, text=True, timeout=10).stdout
        ips = re.findall(r"IPv4[^:\r\n]*:\s*(\d+\.\d+\.\d+\.\d+)", out)
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        ips += [a[4][0] for a in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)]
    except OSError:
        pass
    return sorted({ip for ip in ips if not ip.startswith("169.254.")})


def _run(args: list[str], cwd: str) -> None:
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(os.path.basename(a) for a in args[:3])} failed: {r.stderr.strip()[:300]}")


def ensure_tls(tls_dir: str = TLS_DIR, ips: list[str] | None = None, hostname: str | None = None) -> dict:
    """Make (or reuse) the root and the server certificate. Returns the paths and the names/addresses the certificate covers."""
    os.makedirs(tls_dir, exist_ok=True)
    ssl_bin = find_openssl()
    host = (hostname or socket.gethostname()).lower()
    ips = sorted(set(ips if ips is not None else local_ipv4s()) | {"127.0.0.1"})
    names = [f"{host}.local", host, "localhost"]
    p = {k: os.path.join(tls_dir, v) for k, v in {"ca_key": "ca.key", "ca_pem": "ca.pem", "ca_der": "ca.cer", "key": "server.key",
                                                    "cert": "server.pem", "san": "server.san", "csr": "server.csr", "ext": "server.ext"}.items()}
    if not (os.path.exists(p["ca_key"]) and os.path.exists(p["ca_pem"])):
        _run([ssl_bin, "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", p["ca_key"], "-out", p["ca_pem"], "-days", "3650",
              "-subj", "/CN=go2-robot local CA", "-addext", "basicConstraints=critical,CA:TRUE",
              "-addext", "keyUsage=critical,keyCertSign,cRLSign", "-addext", "subjectKeyIdentifier=hash",
              # NAME CONSTRAINTS: this root can only vouch for local names and private addresses, never for a real website
              "-addext", "nameConstraints=critical,permitted;DNS:.local,permitted;DNS:localhost,permitted;DNS:" + host
              + ",permitted;IP:10.0.0.0/255.0.0.0,permitted;IP:172.16.0.0/255.240.0.0,permitted;IP:192.168.0.0/255.255.0.0,permitted;IP:127.0.0.0/255.0.0.0"], tls_dir)
        _run([ssl_bin, "x509", "-in", p["ca_pem"], "-outform", "DER", "-out", p["ca_der"]], tls_dir)
    want = "\n".join([f"DNS:{n}" for n in names] + [f"IP:{i}" for i in ips])
    have = open(p["san"], encoding="ascii").read() if os.path.exists(p["san"]) else ""
    if want != have or not (os.path.exists(p["cert"]) and os.path.exists(p["key"])):
        with open(p["ext"], "w", encoding="ascii") as f:
            f.write("basicConstraints=CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\n"
                    "subjectAltName=" + ",".join(want.split("\n")) + "\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid\n")
        _run([ssl_bin, "req", "-newkey", "rsa:2048", "-nodes", "-keyout", p["key"], "-out", p["csr"], "-subj", f"/CN={host}.local"], tls_dir)
        _run([ssl_bin, "x509", "-req", "-in", p["csr"], "-CA", p["ca_pem"], "-CAkey", p["ca_key"], "-CAcreateserial", "-out", p["cert"],
              "-days", "800", "-sha256", "-extfile", p["ext"]], tls_dir)
        with open(p["san"], "w", encoding="ascii") as f:
            f.write(want)
    return {**p, "names": names, "ips": ips, "host": host}


if __name__ == "__main__":
    info = ensure_tls()
    print("root certificate to install on the iPhone:", info["ca_der"])
    print("server certificate covers:", ", ".join(info["names"] + info["ips"]))
