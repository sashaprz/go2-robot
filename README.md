# go2-robot

Launchers for controlling a **Unitree Go2 Air** from a Windows PC using [DimOS](https://github.com/dimensionalOS/dimos)
running inside WSL2 (Ubuntu 24.04).

## One-time setup (per machine)

1. **WSL2 + Ubuntu 24.04**, then install DimOS inside it with the official installer
   (it creates `~/dimensional-applications` with a venv):
   ```bash
   curl -fsSL https://raw.githubusercontent.com/dimensionalOS/dimos/main/scripts/install.sh | bash
   ```
   DimOS needs Python 3.12 and supports Ubuntu / macOS only, hence WSL on Windows.
2. **Get the dog's AES key.** Go2 firmware >= 1.1.15 needs a per-device key to do the LAN handshake.
   It belongs to the Unitree Go app account the dog is registered to. Either:
   - run `fetch-aes-key.bat` **on a network with internet** (it prompts for the Unitree account email/password), or
   - ask whoever owns the dog for the key and put this line in `~/.dimos.env` inside WSL:
     ```
     UNITREE_AES_128_KEY=<32 hex characters>
     ```
   **Never commit the key.** `.gitignore` excludes `*.env`; keep it that way.
3. Clone this repo. The `.bat` files call `run_dimos.sh` by an absolute path
   (`/mnt/c/Users/Sasha/go2-robot/...`); edit the path in `dimos-go2.bat` and `fetch-aes-key.bat`
   if your clone lives elsewhere or your Windows username differs.

## Driving the dog

1. Power on the dog. Close the Unitree Go phone app (only one controller can connect at a time).
2. Join the dog's Wi-Fi: **Go2_61331_29d4be72**, password **88888888** (Unitree's default). The PC will have
   no internet while connected; that's expected.
3. Double-click **`dimos-go2.bat`**. It pings the dog at `192.168.12.1`, then starts the
   `unitree-go2-webrtc-keyboard-teleop` blueprint.
4. **Click the small pygame window that opens** so it has keyboard focus. Keys are read from that window, not the terminal.

| Key | Action |
|---|---|
| W / S | forward / back |
| A / D | strafe left / right |
| Y / H | turn (yaw) |
| Esc | quit |

(The keyboard module is shared with arm teleop, so Q/E/R/F/T/G do nothing useful on the dog.)

Other blueprints: `dimos-go2.bat <blueprint> [robot-ip]`

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
- **Nothing moves / WASD ignored**: you're on `unitree-go2-basic` (no control), or the pygame window doesn't have focus.
  The default speed is very low (about 5 cm/s), so motion can be subtle.
- **Ping works but WebRTC still fails**: WSL2 NAT can break WebRTC. Try mirrored networking
  (`networkingMode=mirrored` under `[wsl2]` in `%UserProfile%\.wslconfig`, then `wsl --shutdown`).

## Files

- `dimos-go2.bat` / `run_dimos.sh`: launcher (Windows entry point / script that runs inside WSL)
- `fetch-aes-key.bat` / `fetch_aes_key.sh`: fetch and save the AES key
- `dog.py`, `dog.bat`, `connect_test.py`: an earlier standalone controller. **Obsolete**: it uses a library without
  AES-key support, so it can't connect to current firmware. Kept for reference.
