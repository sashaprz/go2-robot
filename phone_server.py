"""Phone link server (Windows side). The iPhone opens a secure web page from this PC and streams its microphone and motion sensors;
this hands them to the dog app (in WSL) over the same kind of protected local socket winmic.py uses.

  phone -- HTTPS page + WSS (port 8443, needs the secret in the link) --> this server --+--> audio  (16 kHz mono PCM16) on the WSL address :48125
                                                                                        +--> motion (one JSON line per update)                          :48126
  phone -- plain HTTP (port 8080): only the setup page and the public root certificate (nothing secret)

Run by go2.bat. First time on the phone: open http://<this PC's address>:8080 , install and trust the root certificate, then open the
link (or scan the QR code) in phone_qr.html / phone_url.txt.

Safety: what the phone sends can move the dog (a spoken "ernest, sit down" is a command), so (1) the page and the WebSocket both need a
random secret that is in the link only; (2) the WSL-facing ports listen only on the WSL virtual address and need the .phone_wsl_token
secret; (3) it exits by itself when the app has gone.
"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import http
import json
import os
import secrets
import socket
import ssl
import sys
import time
from urllib.parse import parse_qs, urlsplit

from websockets.asyncio.server import serve
from websockets.datastructures import Headers
from websockets.http11 import Response

import phone_tls
import winmic

HERE = os.path.dirname(os.path.abspath(__file__))
PAGE = os.path.join(HERE, "phone.html")
PHONE_TOKEN_PATH = os.path.join(HERE, ".phone_token")
LOG_PATH = os.path.join(HERE, "phone_server.log")
URL_PATH = os.path.join(HERE, "phone_url.txt")
QR_PATH = os.path.join(HERE, "phone_qr.html")
HTTPS_PORT, HTTP_PORT, AUDIO_PORT, MOTION_PORT = 8443, 8080, 48125, 48126      # (48123 / .winmic_token belong to winmic.py, the AirPods helper)
WSL_TOKEN_PATH = os.path.join(HERE, ".phone_wsl_token")
IDLE_NEVER_CONNECTED = 900        # s: exit if the app never connects
IDLE_AFTER_CLIENT = 60            # s: exit this long after the app has gone


def log(msg: str) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def phone_token(path: str = PHONE_TOKEN_PATH) -> str:
    """The secret in the phone's link. Kept between runs so the phone's bookmark stays valid; delete the file to change it."""
    try:
        with open(path, encoding="ascii") as f:
            t = f.read().strip()
        if len(t) >= 12:
            return t
    except OSError:
        pass
    t = secrets.token_urlsafe(9)
    with open(path, "w", encoding="ascii") as f:
        f.write(t)
    return t


class Hub:
    """Who is connected, and the two streams from the phone to the app."""

    def __init__(self):
        self.audio_clients: set = set()
        self.motion_clients: set = set()
        self.phone = None                    # the current phone connection
        self.phone_ua = ""
        self.audio_chunks = 0
        self.last_diag = ""
        self.motion_msgs = 0
        self.last_client_at: float | None = None
        self.started = time.time()

    def _push(self, clients: set, data: bytes) -> None:
        for w in list(clients):
            try:
                if w.transport.get_write_buffer_size() > 1_000_000:          # the app isn't keeping up: drop, never grow without limit
                    continue
                w.write(data)
            except Exception:  # noqa: BLE001
                clients.discard(w)

    def push_audio(self, data: bytes) -> None:
        self.audio_chunks += 1
        self._push(self.audio_clients, data)

    def push_motion(self, msg: dict) -> None:
        self.motion_msgs += 1
        self._push(self.motion_clients, (json.dumps(msg, separators=(",", ":")) + "\n").encode())

    @property
    def app_listening(self) -> bool:
        return bool(self.audio_clients or self.motion_clients)


def make_page_response(status: int, body: bytes, ctype: str, extra: dict | None = None) -> Response:
    h = Headers({"Content-Type": ctype, "Content-Length": str(len(body)), "Cache-Control": "no-store", "Connection": "close"})
    for k, v in (extra or {}).items():
        h[k] = v
    return Response(status, http.HTTPStatus(status).phrase, h, body)


def token_ok(query: str, token: str) -> bool:
    got = (parse_qs(query).get("t") or [""])[0]
    return hmac.compare_digest(got, token)


async def run(https_port: int, http_port: int, audio_port: int, motion_port: int, bind_wsl: str | None, tls_dir: str, token_file: str,
              phone_tok_path: str) -> int:
    tls = phone_tls.ensure_tls(tls_dir)
    token = phone_token(phone_tok_path)
    hub = Hub()
    wsl_token = secrets.token_hex(16)
    host = bind_wsl or winmic.wsl_adapter_ip() or "127.0.0.1"

    # ---- the phone-facing side ---------------------------------------------------------------------------
    async def process_https(connection, request):
        u = urlsplit(request.path)
        if u.path == "/ws":
            if not token_ok(u.query, token):
                log(f"refused a WebSocket with a wrong token from {connection.remote_address[0]}")
                return make_page_response(403, b"forbidden\n", "text/plain")
            return None                                                        # go on with the WebSocket handshake
        if u.path in ("/phone", "/phone/"):
            if not token_ok(u.query, token):
                return make_page_response(403, b"This link needs its secret: use the link or QR code the computer shows.\n", "text/plain")
            with open(PAGE, "rb") as f:
                return make_page_response(200, f.read(), "text/html; charset=utf-8")
        if u.path == "/ping":
            return make_page_response(200, b"ok\n", "text/plain")
        return make_page_response(404, b"not found\n", "text/plain")

    async def phone_ws(ws):
        peer = ws.remote_address[0]
        if hub.phone is not None:
            try:
                await hub.phone.close(1000, "another phone connected")
            except Exception:  # noqa: BLE001
                pass
        hub.phone = ws
        log(f"phone connected from {peer}")

        async def status():
            while True:
                try:
                    await ws.send(json.dumps({"type": "status", "app": hub.app_listening}))
                except Exception:  # noqa: BLE001
                    return
                await asyncio.sleep(1.0)

        st = asyncio.create_task(status())
        try:
            async for msg in ws:
                if isinstance(msg, (bytes, bytearray)):
                    hub.push_audio(bytes(msg[: len(msg) - len(msg) % 2]))
                else:
                    try:
                        m = json.loads(msg)
                    except ValueError:
                        continue
                    if m.get("type") == "hello":
                        hub.phone_ua = str(m.get("ua", ""))[:120]
                        log(f"phone says hello: {hub.phone_ua} sign={m.get('sign')}")
                    if m.get("type") == "diag":                                # the page's own view of its microphone: for finding out why no sound arrives
                        line = (f"phone mic: engine {m.get('ctx')}, {m.get('mode')}, track {m.get('track')}, {m.get('samples')} samples captured, page sent {m.get('sentAudio')} "
                                f"chunks, level {m.get('level')}, this server received {hub.audio_chunks}")
                        if line != hub.last_diag:
                            hub.last_diag = line
                            log(line)
                        continue
                    m["st"] = time.time()
                    hub.push_motion(m)
        finally:
            st.cancel()
            if hub.phone is ws:
                hub.phone = None
            log(f"phone {peer} disconnected")

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(tls["cert"], tls["key"])
    https = await serve(phone_ws, "0.0.0.0", https_port, ssl=ctx, process_request=process_https, max_size=2 ** 20, ping_interval=20)

    # ---- plain-HTTP setup page + the PUBLIC root certificate (nothing secret here) --------------------------
    async def process_http(connection, request):
        u = urlsplit(request.path)
        if u.path == "/ca.cer":
            with open(tls["ca_der"], "rb") as f:
                return make_page_response(200, f.read(), "application/x-x509-ca-cert", {"Content-Disposition": 'attachment; filename="go2-robot-ca.cer"'})
        if u.path in ("/", "/setup"):
            return make_page_response(200, SETUP_HTML.replace("{HOST}", tls["host"]).replace("{HTTPS}", str(https_port)).encode(), "text/html; charset=utf-8")
        return make_page_response(404, b"not found\n", "text/plain")

    async def never(ws):
        await ws.close()

    http_srv = await serve(never, "0.0.0.0", http_port, process_request=process_http)

    # ---- the WSL-facing side ------------------------------------------------------------------------------
    def wsl_handler(clients: set, name: str):
        async def handle(reader, writer):
            try:
                got = (await asyncio.wait_for(reader.read(64), 5)).decode("ascii", "ignore").strip()
            except Exception:  # noqa: BLE001
                got = ""
            if not hmac.compare_digest(got, wsl_token):
                log(f"refused a {name} client: wrong token")
                writer.close()
                return
            clients.add(writer)
            hub.last_client_at = time.time()
            log(f"the app connected for {name}")
            try:
                while await reader.read(1):
                    pass
            except Exception:  # noqa: BLE001
                pass
            finally:
                clients.discard(writer)
                hub.last_client_at = time.time()
                log(f"the app disconnected ({name})")
                writer.close()
        return handle

    a_srv = await asyncio.start_server(wsl_handler(hub.audio_clients, "audio"), host, audio_port)
    m_srv = await asyncio.start_server(wsl_handler(hub.motion_clients, "motion"), host, motion_port)
    with open(token_file, "w", encoding="ascii") as f:
        f.write(wsl_token)                                                     # only written once we own the ports

    urls = phone_urls(tls, https_port, token)
    with open(URL_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(urls) + "\n")
    write_qr(urls[0], tls, http_port)
    log(f"phone link ready. WSL side on {host}:{audio_port} (audio) and :{motion_port} (motion)")
    log("open on the phone: " + urls[0])

    while True:                                                                # exit when the app has gone
        await asyncio.sleep(1.0)
        idle = time.time() - (hub.last_client_at if hub.last_client_at else hub.started)
        if not hub.app_listening and idle > (IDLE_AFTER_CLIENT if hub.last_client_at else IDLE_NEVER_CONNECTED):
            log("no app any more: exiting")
            for s in (https, http_srv, a_srv, m_srv):
                s.close()
            return 0


def phone_urls(tls: dict, port: int, token: str) -> list[str]:
    """Links for the phone, best first: the LAN address on the dog's Wi-Fi (192.168.12.x) if there is one, then the .local name."""
    ips = [i for i in tls["ips"] if i != "127.0.0.1"]
    ips.sort(key=lambda i: (not i.startswith("192.168.12."), i.startswith(("192.168.160.", "172.22.", "10.5.")), i))
    return [f"https://{i}:{port}/phone?t={token}" for i in ips[:2]] + [f"https://{tls['host']}.local:{port}/phone?t={token}"]


def write_qr(url: str, tls: dict, http_port: int) -> None:
    try:
        import segno
        svg = segno.make(url, error="m").svg_inline(scale=8, border=2)
    except Exception as e:  # noqa: BLE001
        svg = f"<p>(couldn't draw a QR code: {e})</p>"
    ips = [i for i in tls["ips"] if i != "127.0.0.1"]
    with open(QR_PATH, "w", encoding="utf-8") as f:
        f.write(f"<!doctype html><meta charset=utf-8><title>Go2 phone link</title><body style='font:16px sans-serif;max-width:640px;margin:24px auto'>"
                f"<h2>Go2 phone link</h2><p><b>1. Once only:</b> on the phone (on the same Wi-Fi as this PC) open "
                + " or ".join(f"<code>http://{i}:{http_port}</code>" for i in ips[:3]) + f" and follow the steps to trust the certificate.</p>"
                f"<p><b>2. Every time:</b> scan this with the iPhone camera, or type the link:</p><div>{svg}</div>"
                f"<p style='word-break:break-all'><code>{url}</code></p><p style='color:#a00'>The link contains a secret: don't share it.</p>")


SETUP_HTML = """<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Go2 phone setup</title><style>body{font:17px/1.45 -apple-system,system-ui,sans-serif;margin:18px;max-width:640px}
a.b{display:block;background:#3b82f6;color:#fff;padding:14px;border-radius:10px;text-align:center;text-decoration:none;font-weight:600;margin:12px 0}
li{margin:8px 0}code{background:#eee;padding:1px 5px;border-radius:4px}</style></head><body>
<h2>Go2 phone setup (once)</h2>
<p>iPhones only allow the microphone and motion sensors on secure pages, and only trust certificates you install. This installs a small
private certificate that can vouch <b>only</b> for local names and addresses on this network, never for a real website.</p>
<a class=b href="/ca.cer">1. Download the certificate</a>
<ol>
<li>Tap <b>Allow</b> when Safari asks to download a configuration profile.</li>
<li>Open <b>Settings</b>: tap <b>Profile Downloaded</b> near the top (or General &gt; VPN &amp; Device Management), then <b>Install</b> (twice), enter your passcode.</li>
<li>Then <b>Settings &gt; General &gt; About &gt; Certificate Trust Settings</b> and switch <b>go2-robot local CA</b> ON.</li>
<li>Now open the secure link the computer shows (<code>phone_qr.html</code> on the PC, or the line in the go2.bat window). Scan its QR code with the camera.</li>
<li>On the page tap <b>Start</b>, allow the microphone and Motion &amp; Orientation, then do the one-time <b>turning set-up</b>.</li>
</ol>
<p>Computer name: <code>{HOST}.local</code>, secure port <code>{HTTPS}</code>.</p></body></html>"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--https-port", type=int, default=HTTPS_PORT)
    ap.add_argument("--http-port", type=int, default=HTTP_PORT)
    ap.add_argument("--audio-port", type=int, default=AUDIO_PORT)
    ap.add_argument("--motion-port", type=int, default=MOTION_PORT)
    ap.add_argument("--bind-wsl", default=None, help="address for the WSL-facing ports (default: this PC's WSL virtual address)")
    ap.add_argument("--tls-dir", default=phone_tls.TLS_DIR)
    ap.add_argument("--token-file", default=WSL_TOKEN_PATH, help="where the WSL-side secret is written")
    ap.add_argument("--phone-token-file", default=PHONE_TOKEN_PATH)
    a = ap.parse_args()
    try:
        return asyncio.run(run(a.https_port, a.http_port, a.audio_port, a.motion_port, a.bind_wsl, a.tls_dir, a.token_file, a.phone_token_file))
    except OSError as e:
        log(f"could not start ({e}): is another phone_server.py already running?")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
