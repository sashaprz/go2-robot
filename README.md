# go2-robot

Launchers for controlling a **Unitree Go2 Air** using [DimOS](https://github.com/dimensionalOS/dimos).

**Platform support:**
- **macOS**: Native support - run scripts directly
- **Windows**: Requires WSL2 (Ubuntu 24.04) - DimOS runs inside WSL, launched via `.bat` files
- **Linux**: Native support (Ubuntu 24.04+)

## One-time setup (per machine)

### macOS Setup

1. **Install DimOS directly** (Python 3.12 required):
   ```bash
   curl -fsSL https://raw.githubusercontent.com/dimensionalOS/dimos/main/scripts/install.sh | bash
   ```
   This creates `~/dimensional-applications` with a Python venv.

2. **Get the dog's AES key.** Go2 firmware >= 1.1.15 needs a per-device key for LAN handshake.
   Run `./fetch-aes-key` **on a network with internet** (it prompts for Unitree account email/password):
   ```bash
   ./fetch-aes-key
   ```
   Or manually add the key to `~/.dimos.env`:
   ```
   UNITREE_AES_128_KEY=<32 hex characters>
   ```
   **Never commit the key.** `.gitignore` excludes `*.env`.

3. Clone this repo (if you haven't already).

### Windows Setup

1. **WSL2 + Ubuntu 24.04**, then install DimOS inside it with the official installer:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/dimensionalOS/dimos/main/scripts/install.sh | bash
   ```
   This creates `~/dimensional-applications` with a Python venv.
   (DimOS needs Python 3.12 and supports Ubuntu / macOS only, hence WSL on Windows)

2. **Get the dog's AES key.** Go2 firmware >= 1.1.15 needs a per-device key for LAN handshake.
   Run `fetch-aes-key.bat` **on a network with internet** (prompts for Unitree account email/password), or
   manually add the key to `~/.dimos.env` inside WSL:
   ```
   UNITREE_AES_128_KEY=<32 hex characters>
   ```
   **Never commit the key.** `.gitignore` excludes `*.env`.

3. Clone this repo. The `.bat` files call scripts by an absolute path
   (`/mnt/c/Users/Sasha/go2-robot/...`); edit the path in `dimos-go2.bat` and `fetch-aes-key.bat`
   if your clone lives elsewhere or your Windows username differs.

## All-in-one: camera + driving + tricks

One window with the dog's live camera (FPV), keyboard driving, and poses/tricks. Needs no internet; join the dog's
Wi-Fi (see below), close the phone app, then:
- **macOS/Linux**: `./go2` (or `./go2 --demo` for preview without dog)
- **Windows**: double-click `go2.bat` (or `go2.bat --demo`)

**Click the window** for keyboard focus.

| Keys | Action |
|---|---|
| W / S | forward / back |
| Q / E | strafe left / right |
| A / D | turn left / right |
| Shift / Ctrl (held) | faster x1.5 / slower x0.5 (base 0.4 m/s, 0.8 rad/s; `--linear` / `--angular` change it) |
| Space | emergency stop (also aborts a routine) |
| 1-6 | stand up, balance, lie down, recovery stand, sit, rise from sit |
| 7 8 9 0 F | hello, stretch, content, wiggle hips, finger heart |
| N / M, R, then **Y** | dance 1 / dance 2 / greeting routine, each needs a confirming Y within 4 s |
| Esc | quit |

Driving sends BalanceStand first automatically, and is locked out while a trick is running (Space clears it).
Losing window focus stops the dog. Flips/handstand are deliberately not included. Which tricks an Air accepts depends
on model and firmware. `--motion-mode mcf` (DimOS notes that mode is the one that traverses stairs) is an untested opt-in.

## Driving the dog with DimOS blueprints

The full DimOS stack: keyboard teleop plus optional mapping/navigation and agent blueprints. It has no camera in the
keyboard blueprint. `go2` and `dimos-go2` can't run at the same time: the dog accepts one controller.

1. Power on the dog. Close the Unitree Go phone app (only one controller can connect at a time).
2. Join the dog's Wi-Fi: **Go2_61331_29d4be72**, password **88888888** (Unitree's default). No internet while connected; that's expected.
3. Run the launcher:
   - **macOS/Linux**: `./dimos-go2` (pings dog at `192.168.12.1`, starts `unitree-go2-webrtc-keyboard-teleop`)
   - **Windows**: double-click `dimos-go2.bat`
4. **Click the small pygame window that opens** so it has keyboard focus. Keys are read from that window, not the terminal.

| Key | Action |
|---|---|
| W / S | forward / back |
| Q / E | strafe left / right |
| A / D | turn left / right |
| Space | emergency stop (zero velocity) |
| Shift / Ctrl (held) | boost x2 / slow x0.5 |
| Esc | quit |

Default speed is 0.5 m/s forward and 0.8 rad/s turning (Shift doubles it), so keep the area clear.
The window shows the live twist being sent, which helps tell "keys not registering" from "dog not responding".

Other blueprints:
- **macOS/Linux**: `./dimos-go2 <blueprint> [robot-ip]`
- **Windows**: `dimos-go2.bat <blueprint> [robot-ip]`

| Blueprint | Notes |
|---|---|
| `unitree-go2-webrtc-keyboard-teleop` | default; keyboard driving |
| `unitree-go2-basic` | visualization only, **no control** |
| `unitree-go2-agentic` | natural-language control; needs an LLM key (`OPENAI_API_KEY` in `~/.dimos.env`) and internet |
| `unitree-go2-agentic-ollama` | natural-language control with a local LLM (Ollama not installed yet) |

List them all with `dimos list` inside WSL. If the dog is on another network (STA mode), pass its IP as the second argument.

## Troubleshooting

- **`AesKeyRequiredError`**: the AES key is missing or not being loaded. See setup step 2.
- **`Robot at 192.168.12.1 is not exposing a signaling port`**: the PC isn't on the dog's Wi-Fi, or the dog is off. Windows may
  hop back to a saved network with internet; re-select the dog's Wi-Fi.
- **Nothing moves / keys ignored**: you're on `unitree-go2-basic` (view-only, no control), or the pygame window doesn't have focus.
  Also check no other DimOS run is still holding the dog's connection (`dimos status` / `dimos stop` in WSL); only one client can connect.
- **Ping works but WebRTC still fails**: WSL2 NAT can break WebRTC. Try mirrored networking
  (`networkingMode=mirrored` under `[wsl2]` in `%UserProfile%\.wslconfig`, then `wsl --shutdown`).

## Files

**macOS/Linux launchers:**
- `go2` / `run_go2.sh` / `go2.py`: all-in-one FPV + driving + tricks window (recommended)
- `dimos-go2` / `run_dimos.sh`: launcher for DimOS blueprints
- `fetch-aes-key` / `fetch_aes_key.sh`: fetch and save the AES key

**Windows launchers:**
- `go2.bat`: launches `run_go2.sh` inside WSL
- `dimos-go2.bat`: launches `run_dimos.sh` inside WSL
- `fetch-aes-key.bat`: launches `fetch_aes_key.sh` inside WSL

**Legacy files:**
- `dog.py`, `dog.bat`, `connect_test.py`: earlier standalone controller. **Obsolete** (no AES-key support). Kept for reference.
