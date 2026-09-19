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

## All-in-one: camera + driving + tricks (`go2.bat`)

One window with the dog's live camera (FPV), keyboard driving, and poses/tricks. Needs no internet; join the dog's
Wi-Fi (see below), close the phone app, then double-click **`go2.bat`** and **click the window** for keyboard focus.
`go2.bat --demo` previews the window with a fake robot and synthetic video (no dog needed).

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
| T, then **Y** | follow the nearest person (see below); T again / Space / any drive or trick key stops it |
| V (hold) | push-to-talk voice commands (see below) |
| O | toggle object-detection overlay (see below) |
| Esc | quit |

### Voice commands (hold `V`)

Hold **V**, speak, release. Speech is transcribed **offline** by a local Whisper model (faster-whisper `base.en`, CPU), so
it works on the dog's own Wi-Fi and your audio never leaves the PC. The transcript is matched to actions, and the window
shows what it heard and what it matched.

- **Tricks and poses:** "say hello" / "wave", "dance" / "dance two", "sit", "lie down", "stand up", "stretch",
  "wiggle your hips", "make a heart", "good boy", "recover", "greeting".
- **Walking and turning (timed moves):** "walk forward", "go back", "turn left", "turn right", "turn around",
  "step left" / "step right" (sideways). Add an amount if you like: "walk forward two seconds", "go forward one meter",
  "turn right 45 degrees", "go back a little" (half), "walk forward a lot" (double). Defaults: 1.5 s forward/back
  (~0.6 m at 0.4 m/s), 1 s sideways, 90 degrees for a turn, 180 for "turn around". A spoken move is **capped at 5 s
  (~2 m)** walking or 8 s turning. A new move replaces the current one.
- **Follow:** "follow me" starts follow mode straight away (the keyboard `T` still asks for `Y`); "stop following" ends it.
- **"stop"** (or halt / freeze) is an emergency stop and always wins. Space, or pressing any drive key, also cancels a
  spoken move.

- **One-time download (needs internet):** `go2.bat --fetch-model` fetches the object detector and the Whisper model
  (~150 MB, cached in `~/.cache/`). After that no internet is needed.
- **Measured on the dev PC (12-core ARM, CPU only):** `base.en` understood 32/32 spoken test commands (two synthetic
  voices) in ~0.4 s each; silence and noise produced no commands. Not yet measured with real people's voices or a noisy
  room. `small.en` (`--whisper-model small.en`) is slower (~1.5 s) with no gain on those clips.
- **Optional cloud backend:** `go2.bat --stt elevenlabs` uses ElevenLabs speech-to-text instead (needs internet and an
  API key: run `set-elevenlabs-key.bat` once; saved to `~/.dimos.env` in WSL, never in the repo). Untested against the
  real API.
- **Microphone:** captured through WSLg's PulseAudio (`libpulse-simple`); Windows must allow microphone access.
  A transcription that takes over 12 s is abandoned with a message.

### Object detection (`O`)

Toggles labelled boxes and a "seeing: ..." summary over the live video for 80 everyday object types (person, chair, cup,
sports ball, tv, laptop, bottle, dog, ...), using the same YOLOX-tiny model as follow mode (one shared inference, ~25 ms
per frame on CPU). Detection only: it has no depth or distance, and it doesn't drive anything. It's a small model:
expect missed small/far objects (a person a few metres away can be missed) and the odd false positive. Needs the model
from `go2.bat --fetch-model` (see follow mode below).

### Follow mode (`T`)

The dog turns to keep the nearest person centred and walks toward them slowly (max 0.35 m/s, `--follow-speed`),
stopping when they fill ~60% of the frame height. It gives up after ~1.2 s without seeing them, and Space, `T` again,
any drive/trick key, or losing window focus cancel it. **It has no obstacle avoidance**: keep the path clear, and stay
in front of it. The detector is YOLOX-tiny on CPU via onnxruntime (`follow.py`), so it works on the dog's hotspot with
no cloud or API keys. Fetch the model once while you have internet: `go2.bat --fetch-model` (stored in
`~/.cache/go2/`, not in the repo). Tests in this repo only used still photos: it has not been run on a real dog.

Driving sends BalanceStand first automatically, and is locked out while a trick is running (Space clears it).
Losing window focus stops the dog. Flips/handstand are deliberately not included. Which tricks an Air accepts depends
on model and firmware. `--motion-mode mcf` (DimOS notes that mode is the one that traverses stairs) is an untested opt-in.

## Driving the dog with DimOS instead (`dimos-go2.bat`)

The full DimOS stack: keyboard teleop plus optional mapping/navigation and agent blueprints. It has no camera in the
keyboard blueprint. `go2.bat` and `dimos-go2.bat` can't run at the same time: the dog accepts one controller.

1. Power on the dog. Close the Unitree Go phone app (only one controller can connect at a time).
2. Join the dog's Wi-Fi: **Go2_61331_29d4be72**, password **88888888** (Unitree's default). The PC will have
   no internet while connected; that's expected.
3. Double-click **`dimos-go2.bat`**. It pings the dog at `192.168.12.1`, then starts the
   `unitree-go2-webrtc-keyboard-teleop` blueprint.
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
- **Nothing moves / keys ignored**: you're on `unitree-go2-basic` (view-only, no control), or the pygame window doesn't have focus.
  Also check no other DimOS run is still holding the dog's connection (`dimos status` / `dimos stop` in WSL); only one client can connect.
- **Ping works but WebRTC still fails**: WSL2 NAT can break WebRTC. Try mirrored networking
  (`networkingMode=mirrored` under `[wsl2]` in `%UserProfile%\.wslconfig`, then `wsl --shutdown`).

## Files

- `go2.bat` / `run_go2.sh` / `go2.py`: all-in-one FPV + driving + tricks + follow window (recommended)
- `follow.py`: object/person detector (YOLOX-tiny) and follow controller used by `go2.py`
- `voice.py`: microphone capture, local Whisper (default) and ElevenLabs speech-to-text, and the phrase matcher used by `go2.py`
- `set-elevenlabs-key.bat` / `set_elevenlabs_key.sh`: save an ElevenLabs API key (only for `--stt elevenlabs`; hidden prompt)
- `dimos-go2.bat` / `run_dimos.sh`: launcher for DimOS blueprints (Windows entry point / script that runs inside WSL)
- `fetch-aes-key.bat` / `fetch_aes_key.sh`: fetch and save the AES key
- `dog.py`, `dog.bat`, `connect_test.py`: an earlier standalone controller. **Obsolete**: it uses a library without
  AES-key support, so it can't connect to current firmware. Kept for reference.
