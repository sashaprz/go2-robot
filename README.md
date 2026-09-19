# go2-robot

Control a **Unitree Go2 Air**: live camera, keyboard driving, tricks, person-follow, object detection,
**voice commands (offline, wake word "ernest")**, and standing on the back legs (**"ernest, box step"**).
Built on [DimOS](https://github.com/dimensionalOS/dimos).

**Platform support:**
- **macOS**: Native support - run scripts directly
- **Windows**: Requires WSL2 (Ubuntu 24.04) - DimOS runs inside WSL, launched via `.bat` files
- **Linux**: Native support (Ubuntu 24.04+)

> **Read this first: what is and isn't tested.** Everything below was developed and tested **without the robot in the
> loop**: scripted window tests against a fake dog, generated speech through the real Whisper model, and a fake audio
> hub. Driving/tricks/follow have been tried on the real dog; the back-leg stand, voice on the
> real dog, and always-listening have **not**. Details in [Status and limits](#status-and-limits).

## One-time setup (per machine)

### macOS / Linux Setup

1. **Install DimOS** (Python 3.12 required):
   ```bash
   curl -fsSL https://raw.githubusercontent.com/dimensionalOS/dimos/main/scripts/install.sh | bash
   ```
   This creates `~/dimensional-applications` with a Python venv.

2. **Get the dog's AES key** (Go2 firmware >= 1.1.15 needs this). Run `./fetch-aes-key` **while online**:
   ```bash
   ./fetch-aes-key
   ```
   Or manually add to `~/.dimos.env`: `UNITREE_AES_128_KEY=<32 hex chars>`. **Never commit the key**.

3. **Download the models once, while online:** `./go2 --fetch-model` (object detector ~20 MB + Whisper `base.en`
   ~150 MB, cached under `~/.cache/`). After that, **no internet is needed**.

### Windows Setup

1. **WSL2 + Ubuntu 24.04**, then install DimOS inside it (creates `~/dimensional-applications` with a venv):
   ```bash
   curl -fsSL https://raw.githubusercontent.com/dimensionalOS/dimos/main/scripts/install.sh | bash
   ```

2. **The dog's AES key.** Run `fetch-aes-key.bat` **while online** (prompts for the Unitree account), or put
   `UNITREE_AES_128_KEY=<32 hex chars>` in `~/.dimos.env` inside WSL. **Never commit the key** (`*.env` is gitignored).

3. **Download the models once, while online:** `go2.bat --fetch-model` (object detector ~20 MB + Whisper `base.en`
   ~150 MB, cached under `~/.cache/`). After that, **no internet is needed**.

4. Clone this repo. The `.bat` files call scripts by absolute path (`/mnt/c/Users/Sasha/go2-robot/...`); edit them if
   your clone lives elsewhere or your Windows username differs.

## Running it

1. Power on the dog. **Close the Unitree Go phone app** (the dog accepts one controller at a time).
2. Join the dog's Wi-Fi: **Go2_61331_29d4be72**, password **88888888**. Your system will say "no internet"; that's fine.
3. Launch:
   - **macOS/Linux**: `./go2` then **click inside the window** for keyboard focus
   - **Windows**: Double-click **`go2.bat`** then **click inside the window** for keyboard focus

Preview mode (no dog needed):
- **macOS/Linux**: `./go2 --demo`
- **Windows**: `go2.bat --demo`

## Keys

| Key | Action |
|---|---|
| W / S | forward / back |
| Q / E | strafe left / right |
| A / D | turn left / right |
| Shift / Ctrl (held) | faster x1.5 / slower x0.5 (base 0.4 m/s, 0.8 rad/s) |
| Space | **emergency stop** (also ends box step, aborts routines) |
| 1 2 3 4 5 6 | stand up, balance, lie down, recovery stand, sit, rise from sit |
| 7 8 9 0 F | hello, stretch, content, wiggle hips, finger heart |
| N / M, R, then **Y** | dance 1 / dance 2 / greeting routine (each needs a confirming Y within 4 s) |
| T, then **Y** | follow the nearest person (T again / Space / any drive key stops it) |
| O | toggle object-detection overlay |
| U, then **Y** | stand on the **back legs** (U again = come down) |
| V (hold) | push-to-talk voice command |
| L | always-listening (the "ernest" ear) on/off |
| Esc | quit |

Driving sends BalanceStand first, is locked out while a trick runs, and stops if the window loses focus.

## Voice

Two ways to talk to it, both **offline** (local Whisper `base.en`; your audio never leaves the PC):

- **Always-listening ear (on by default).** Say the wake word first: **"ernest, sit down"**. Say just "ernest" and then
  a command within 6 s also works. Ordinary conversation is ignored. A bare **"stop"** (or "stop following") works
  **without** the wake word, and a bare **"yes"** answers a pending confirmation. `L` turns the ear off; `--no-listen`
  starts with it off; `--wake-word rex` changes the word.
- **Push-to-talk:** hold **V** and speak (no wake word needed).

The window shows what it heard and how it was handled ("ignored: no wake word", etc.).

**What you can say**

| Say | Does |
|---|---|
| "ready to dance" (or "stand up") | stand up |
| "sit", "lie down", "recover", "rise", "balance" | poses |
| "say hello" / "wave", "stretch", "wiggle your hips", "make a heart", "good boy", "dance", "dance two", "greeting" | tricks |
| "walk forward", "go back", "turn left/right", "turn around", "step left/right" | timed moves. Add an amount: "walk forward two seconds", "go forward one meter", "turn right 45 degrees", "a little" (half), "a lot" (double). Capped at 5 s walking / 8 s turning. |
| "follow me", "stop following" | person-follow (starts straight away; keyboard `T` still asks for `Y`) |
| "stand on your back legs" -> "yes" | back-leg stand (asks for a spoken "yes" or Y). "come down" / "four legs" returns |
| "box step" | **up on the back legs, then step in a square until you say "stop"** (see below). No "yes" needed: it can fall, so clear the space first |
| **"stop"** | emergency stop: halts the dog and ends box step. It stays on its back legs until "come down". |

"go ahead" is deliberately *not* a walk command.

## Box step

Saying **"ernest, box step"** (no confirmation, so clear the space first: the back-leg stand **can fall**) does, in order:
**StandUp -> BalanceStand -> back-leg stand -> steps in a box (forward, right, back, left, ~0.3 m per side, 1.2 s each)**.
It keeps going **until you say "stop"** (or press Space), then stays up on its back legs until "come down" / U. Saying a pose
or trick yourself ends it. *Whether the dog accepts walk commands while on its back legs is untested.*

## Standing on the back legs

Uses Unitree's back-leg stand. The WebRTC library lists it as **`BackStand` = 2050** (firmware 1.1.7+ numbering) and, in the
older numbering, id **1050** (mislabelled "Standup"; DimOS's helper sends that one). The app tries **2050** first and falls back
to **1050** if the dog refuses it (`--upright-api 1050` flips the order); the window prints the dog's reply to each attempt. It **asks for confirmation**, refuses tricks/follow while up (come down
first), and **can fall**: use a soft floor, clear space, and a spotter. A steady two-legged balance isn't something the
firmware exposes beyond this command. `Handstand` (front legs) is deliberately not included; neither are flips.

## Follow mode (`T` or "follow me")

Turns to keep the nearest person centred and walks toward them slowly (max 0.35 m/s), stopping when they fill ~60% of the
frame height. Gives up after ~1.2 s without seeing them. **No obstacle avoidance**: keep the path clear. The detector is
YOLOX-tiny on CPU (`follow.py`); no cloud. Tested on still photos and a fake robot only.

## Object detection (`O`)

Labelled boxes and a "seeing: ..." summary for 80 everyday object types, from the same model as follow mode (~25 ms/frame).
Detection only (no distance). It's a small model: expect missed small/far objects and the odd false positive.

## Other launchers

- **`dimos-go2.bat`**: the full DimOS stack (keyboard teleop, optional mapping/navigation and agent blueprints). It can't
  run at the same time as `go2.bat`: the dog accepts one controller.
- **`fetch-aes-key.bat`**: fetch and save the dog's AES key. **`set-elevenlabs-key.bat`**: only for the optional cloud
  speech backend (`go2.bat --stt elevenlabs`, needs internet, untested against the real API).

## Troubleshooting

- **`AesKeyRequiredError`**: the AES key is missing or not loaded (setup step 2).
- **`Robot at 192.168.12.1 is not exposing a signaling port`**: not on the dog's Wi-Fi, or the dog is off. Windows may hop back
  to a saved network with internet; re-select the dog's Wi-Fi.
- **Keys ignored**: click the window (it needs focus), and make sure no other controller (phone app, `dimos-go2.bat`) holds the dog.
- **"Voice model not downloaded"**: run `go2.bat --fetch-model` while online. **"Object/person detector not downloaded"**: same.
- **Voice mishears / does nothing**: check the "heard:" line under the status bar. No wake word = ignored (by design). Hold `V` to
  bypass the wake word. Whisper transcription of room chatter can take 1-4 s.
- **Back-leg stand does nothing**: the window prints the dog's reply to each attempt (2050 BackStand first, then 1050). "Code 0" only
  means the dog accepted the command; if it says "refused" for both, this model/firmware may not support it.

## Status and limits

Verified by tests (no dog): key logic, follow controller, phrase matcher (40+ phrases plus wake-word routing), and the whole voice pipeline on **generated** speech through the real
Whisper model (32/32 tricks, 40/40 moves, 44/44 wake-word cases, 9/9 utterances cut from a continuous stream).

**Not verified:** real people's voices, accents or noisy rooms; always-listening on the real mic for long periods; the
back-leg stand; whether the dog accepts walk commands while on its back legs. Room speech near the mic can trigger a Whisper "decoding loop"; guards discard those, and the wake word means
chatter can't move the dog, but expect to tune after real use. Stay near the dog and keep **Space** and the word **"stop"**
in mind: they are the emergency stop.

## Files

- `go2.bat` / `run_go2.sh` / `go2.py`: the app (window, keys, voice actions, box step)
- `follow.py`: object/person detector (YOLOX-tiny) and follow controller
- `voice.py`: microphone capture, wake word + always-listening, local Whisper (and optional ElevenLabs), phrase matcher
- `dimos-go2.bat` / `run_dimos.sh`, `fetch-aes-key.bat` / `fetch_aes_key.sh`, `set-elevenlabs-key.bat` / `set_elevenlabs_key.sh`
- `dog.py`, `dog.bat`, `connect_test.py`: an earlier standalone controller. **Obsolete** (no AES-key support); kept for reference.

## Developer tests

`run_go2.sh --selftest`, `--selftest-follow`, `--selftest-voice`, `--selftest-voicemove`, `--selftest-objects`,
`--selftest-listen` run headless scripted checks against the fake robot (inside WSL:
`wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/Sasha/go2-robot/run_go2.sh --selftest-listen`).
