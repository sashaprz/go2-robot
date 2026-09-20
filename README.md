# go2-robot

Control a **Unitree Go2 Air**: live camera, keyboard driving, tricks, person-follow, object detection,
**voice commands (offline, wake word "ernest")**, **heel** (walk at your side), and standing on the back legs (**"ernest, box step"**).
Built on [DimOS](https://github.com/dimensionalOS/dimos).

**Platform support:**
- **macOS**: Native support - run scripts directly
- **Windows**: Requires WSL2 (Ubuntu 24.04) - DimOS runs inside WSL, launched via `.bat` files
- **Linux**: Native support (Ubuntu 24.04+)

> **Read this first: what is and isn't tested.** Everything below was developed and tested **without the robot in the
> loop**: scripted window tests against a fake dog, generated speech through the real Whisper model, and a fake audio
> hub. Driving/tricks/follow have been tried on the real dog; **heel has only been run in simulation and against the fake dog**; the back-leg stand, voice on the
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

3. **Download the models once, while online:** `./go2 --fetch-model` (object detector ~20 MB + Whisper `small.en`
   ~490 MB, cached under `~/.cache/`). After that, **no internet is needed**.

### Windows Setup

1. **WSL2 + Ubuntu 24.04**, then install DimOS inside it (creates `~/dimensional-applications` with a venv):
   ```bash
   curl -fsSL https://raw.githubusercontent.com/dimensionalOS/dimos/main/scripts/install.sh | bash
   ```

2. **The dog's AES key.** Run `fetch-aes-key.bat` **while online** (prompts for the Unitree account), or put
   `UNITREE_AES_128_KEY=<32 hex chars>` in `~/.dimos.env` inside WSL. **Never commit the key** (`*.env` is gitignored).

3. **Download the models once, while online:** `go2.bat --fetch-model` (object detector ~20 MB + Whisper `small.en`
   ~490 MB, cached under `~/.cache/`). After that, **no internet is needed**.

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
| H, then **Y** | **heel**: walk at your side (H again / Space / any drive key stops it) |
| J | **calibrate heel's distance**: stand at three different distances in front of the dog, pressing J at each; uses the lidar, no tape (see Heel) |
| O | toggle object-detection overlay |
| U, then **Y** | stand on the **back legs** (U again = come down) |
| V (hold) | push-to-talk voice command |
| L | always-listening (the "ernest" ear) on/off |
| Esc | quit |

Driving sends BalanceStand first, is locked out while a trick runs, and stops if the window loses focus.

## Voice

Two ways to talk to it, both **offline** (local Whisper `small.en`; your audio never leaves the PC):

- **Always-listening ear (on by default).** Say the wake word first: **"ernest, sit down"**. Say just "ernest" and then
  a command within 6 s also works. Ordinary conversation is ignored. A bare **"stop"** (or "stop following") works
  **without** the wake word, and a bare **"yes"** answers a pending confirmation. `L` turns the ear off; `--no-listen`
  starts with it off; `--wake-word rex` changes the word.
- **Push-to-talk:** hold **V** and speak (no wake word needed).

The window shows what it heard and how it was handled ("ignored: no wake word", etc.).

**Loud places.** Tested on synthetic speech mixed into babble, hiss and rumble (so treat it as relative, not real-world accuracy):
`small.en` (now the default) got 71/108 commands right against 61/108 for the old `base.en`, and in chatter at +10 dB it got 11/12 against
4/12. It is about 2.4x slower per command (`--whisper-model base.en` goes back). It made **no wrong commands and no false triggers** in
those runs. Two things that did *not* help, so they aren't in: a rumble filter, and subtracting steady background noise (which made
things worse). **When the voices around you are as loud as yours, nothing in software recovers it** (0-1 of 12 right at that level, every
setting): put a microphone near your mouth (a headset or clip-on mic as the Windows default input) or use push-to-talk (`V`).
`--voice-log DIR` saves every utterance the always-on mic hears (wav + what it thought it heard) so real misses can be studied.

**The dog's own microphone (`--mic dog`).** The always-on ear can listen through the dog's built-in microphone instead of the computer's, over
the same link (the connection's audio channel; needs no computer microphone, so the laptop can stay in a backpack). Add `--mic dog` to the `OPTS`
line in `go2.bat`. About six seconds after it starts, the window says either "dog mic: receiving audio" (and the top bar shows the mic level) or "NO
audio arrived from the dog" and falls back to the computer's mic automatically. **Whether your dog streams its microphone, and how well it hears you
over its own fans and gait, is unknown until you try:** run `dogmic-probe.bat` first. It records ~44 s (silence, then you saying commands at 1 m
and 3 m), saves `dogmic_recording.wav`, and shows what the same Whisper model made of it and how far above the dog's own noise you were. Push-to-talk
(`V`) still uses the computer's mic. Tested here only with generated speech through the whole path (8/8 commands); gait noise while walking is the
big unknown, and the level in the top bar will show it. It is off by default.

**What you can say**

| Say | Does |
|---|---|
| "ready to dance" (or "stand up") | stand up |
| "sit", "lie down", "recover", "rise", "balance" | poses |
| "say hello" / "wave", "stretch", "wiggle your hips", "make a heart", "good boy", "dance", "dance two", "greeting" | tricks |
| "walk forward", "go back", "turn left/right", "turn around", "step left/right" | timed moves. Add an amount: "walk forward two seconds", "go forward one meter", "turn right 45 degrees", "a little" (half), "a lot" (double). Capped at 5 s walking / 8 s turning. |
| "follow me", "stop following" | person-follow (starts straight away; keyboard `T` still asks for `Y`) |
| "heel", "heel right", "walk beside me", "stop heeling" | walk at your side (default: dog on your left; starts straight away; `H` still asks for `Y`) |
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

Turns to keep the nearest person centred and walks toward them, stopping when they fill about **78% of the frame height** (about 1.3 m from the dog's
centre; it used to be 60%, about 2.5 m: `--follow-height`). It walks at up to **0.8 m/s** (`--follow-speed`, it used to be 0.35), with a stiffer speed response
and a small integral term so it keeps pace with a walking person instead of trailing 4-9 m behind, and it **backs away** if someone walks right up to it.
In simulation it stops about 1.3 m from you and trails about 2 m behind a walking person (it was 5.7 m and 6.7-8.9 m), never lost the person across
starts, stop-and-go and turns on a 15 or 10 fps detector, and kept a person walking straight at it at least 1.0 m away (the old settings touched them).
It uses the clothing lock described under Heel (it will only follow the person it locked onto, and says what that is). Gives up after ~1.2 s without
seeing them. **No obstacle avoidance**: keep the path clear, and remember it now moves faster. The detector is YOLOX-tiny on CPU (`follow.py`); no cloud.
Not yet tried on the real dog at the new settings.

## Heel (`H` or "ernest, heel")

The dog walks at your side and keeps pace as you walk, stop, and turn. `heel right` (or `--heel-side right`) puts it on your right.

> **What heel is now.** Heel uses the **steering of "follow me"** (which works well), holding you a little off-centre in the picture so the dog
> walks at your side, and it holds you **close, about a leash length** (`--heel-range 1.1`, metres from the dog's centre to you).
> - The **camera** decides who you are and which way to turn, and does the distance control (fast and smooth).
> - The **lidar** is coarse, so it doesn't drive the dog. Whenever it gets a reading of how far away you are, that teaches the camera how your feet's
>   position in the picture maps to distance (a small self-calibration; no tape, no `J`). If the lidar says you're *closer* than the camera thinks, the
>   dog believes the lidar. If the lidar drops out, the camera keeps going on what it learned, and where it has learned nothing it stops further off (~1.5 m).
> - The dog **backs away firmly** if you walk up to it. Below about 0.9 m from the dog's centre the camera loses your feet, so don't set the range much lower.
> - In simulation: on a 15 fps detector it never lost the person in starts, stop-and-go, 90-degree turns, a sharp turn or an S-bend; it holds about 1.1 m
>   standing and 1.25-1.4 m while you walk (including at 1.0 m/s), even with the camera badly mis-set, a wall behind you and a lidar that sees the front of your
>   body; a person walking straight at it stays at least 0.6 m away. **At 10 fps some turns lose you (3 of 24 runs), and at 8 fps many do**: the detector's speed
>   (in the banner) matters more than any setting. Not tried on the real dog. Your own lidar recording showed it sees a person in *front* of the dog but not
>   beside it, which is why it is used only for distance, along the camera's line of sight.
> - `--heel-offset` sets how far to the side, `--heel-range 0` ignores the lidar, `--no-heel-lidar` switches it off entirely, and `--heel-style geometric`
>   selects the older controller described below. Everything below this box (geometry, calibration with **J**) applies to that older style only. The clothing
>   lock, the on-screen TARGET label and the loss messages apply to both.

**How does it know it's beside you? It doesn't, exactly, and that is the honest limit.** The only sensor used is the front camera, and it
is about **78 degrees wide** (calibration file: 1280x720, fx 797), so a person directly beside the dog (90 degrees) is out of frame. What it
does instead: it keeps you near the edge of the picture on your side, works out how far away you are from where your feet meet the floor,
and holds that spot. So it walks at your side but about **1.3 m behind your hip** and 0.35 m to the side (`--heel-lead`, `--heel-gap`), where the camera can still see you.
**The camera is in charge; the lidar only helps.** The camera decides who you are and which direction you're in, and works out your
distance from your feet. If the dog is sending lidar (heel says after 3 seconds whether it is), the lidar may *sharpen that distance* along the
same line of sight, but only if it finds a compact, person-sized blob close to the camera's own estimate that does not continue sideways
the way a wall does. It cannot track you on its own: if the camera loses you the dog stops. `--no-heel-lidar` (in the `OPTS` line of `go2.bat`)
turns the lidar part off. **It is off by default in `go2.bat` now:** recordings show the lidar can't see a person closer than ~1 m and keeps a ghost of one for ~40 s, so it can't help heel.

> An earlier version let the lidar track you by itself so the dog could walk level with you. On the real dog it locked onto a wall instead of
> the person, so it was removed. Walking truly beside you (out of the camera's view) would need a lidar person-tracker that has been checked
> against real lidar data first; `lidar-probe.bat` is the first step toward that.

- **Why it can't walk right beside you.** The lens is only ~35 cm off the floor and sees 78 x 49 degrees. At 1 m it sees your legs and hips and
  nothing above; at 0.7 m your feet are out of the picture too, and someone level with the dog is out of frame sideways. `--heel-lead 1.0
  --heel-gap 0.5` (in the `OPTS` line of `go2.bat`) gets closer but more of you is cut off; when your feet are out of frame it judges distance
  from how wide you look (assumes ~0.5 m, so +-0.1 m), and the person detector still fired on legs-and-hips-only crops in a one-photo test.
  Genuinely level with you needs a sensor that sees sideways: the lidar. `lidar-probe.bat 60 save` records what it sees (`lidar_recording.npz`)
  so a leg tracker can be built against real data this time.
- **Once locked, it doesn't change person.** When heel or follow starts it looks for the person nearest the middle of the picture (not just the
  biggest), waits until the same person has been the clear candidate for a few frames, then takes a clothing fingerprint (colour of the top and of the
  trousers) and says what it locked onto ("locked onto: red top, blue trousers"). From then on:
  - **Colour:** it keeps a small bank of fingerprints from different views (whole body far away, only legs and hips close up). The first is never
    replaced. New views are learned only gradually and only while the person is the *only one in view*, so the bank cannot drift onto a stranger or
    be tainted by someone walking in front of them. Someone standing where the person just was is judged a little more gently; everyone else strictly.
  - **Plausibility:** a detection somewhere the person could not have got to (measured in the dog's own frame, so the dog turning doesn't matter) is
    refused unless the clothes match almost exactly.
  - **If nobody matches** it treats the person as not seen (stops, searches, gives up after a moment). It never picks up a stranger.
  - **The window shows its decisions:** a green box labelled `TARGET: <clothes>` on the person it follows; other people get a red box saying
    "NOT you (clothes) 0.62" or "NOT you (too far from where you were)", or an orange "also matches".
  Tested with synthetic people: crossing paths, a look-alike beside the target, the view changing gradually to legs-only, a stranger who appears
  far away, a stranger who takes the place of someone who left (0 switches). **Known limit:** someone dressed like the target *and* standing where they
  were can still be accepted (similar clothes scored 0.46 against the 0.50 limit). A big lighting change can make the right person stop matching (it
  then stops rather than guesses). Not yet tried on the dog.
- **When the camera loses you** (you end up level with the dog, out of its view) and the lidar still sees someone where you were, the dog
  **stands still and waits** up to 6 s instead of giving up. The lidar only answers "is someone there?"; it never steers the dog. A wall is
  not mistaken for a person. If the camera loses you for another reason, the window says why (for example "no person detected" or "1 person(s)
  seen but not matching the lock, closest colour match 0.52"), and the banner shows the detector's speed in fps.
- It follows your pace up to `--heel-speed` (default 0.8 m/s). Walk faster and it falls behind, then gives up ("lost the person") and stands.
- **Turning.** The camera is narrow, so when you turn you swing toward the edge of its picture. The dog now turns faster (up to 1.0 rad/s) and
  steers toward where you're heading in the picture, not where you were, which makes up for the delay between the camera, the detector and the dog.
  In simulation it kept hold of you through 90-degree turns either way, a sharp turn (90 degrees in one second) and an S-bend, including on a slower
  link (8 fps detector, 0.3 s lag). If the detector runs at only ~5 fps it loses you on turns anyway: the fps is in the banner, and the voice model
  competes with the detector for the CPU (`--whisper-model base.en` or `--no-listen` in the `OPTS` line of `go2.bat` frees some). A turn *toward*
  the dog still crowds it (personal-space rule backs it off). Not yet tried on the dog.
- **Calibrating the distance, no tape: press `J`** (in `go2.bat`, no flags). Stand about 2 m in front of the dog in the open (a metre from walls and
  furniture) with your feet in the picture and press J; the camera sees where your feet land and the **lidar measures how far away you are**. Then move to
  a clearly different distance (at least 0.5 m further or closer, between 1 and 3.5 m) and press J again; three spots in all. It works out the camera's
  real height and tilt, applies them straight away, and saves them in `heel_calibration.json`, which `go2.bat` loads automatically. It refuses a set of
  measurements that doesn't fit any camera and changes nothing. If the dog sends no lidar it says so and falls back to standing at 1.2 m and 2.4 m
  with a tape. The window's top bar shows `lidar N/s` once the lidar is on, so you can see whether it is streaming.
  *Accuracy, in simulation only:* about 10 cm worst case if the lidar's origin lines up with the camera lens, and it adds roughly the same again for any
  misalignment (unknown for this dog). The lidar sees the front of your body, likely ~10 cm nearer than your feet, which nothing corrects for.
  **Not yet tried with the real lidar.**
- Without calibrating, the distance estimate assumes the camera is 0.35 m off the floor and level (`--heel-cam-height`, `--heel-cam-pitch`). **Those two are guesses**:
  measure them once and set them in the `OPTS` line of `go2.bat` (e.g. `--heel-cam-height 0.31`), or the dog will hold the wrong gap. The simulation shows a 5 cm / 4 degree error costs ~15 cm of gap.
- **No obstacle avoidance**, same as follow. It locks onto the biggest person in view when it starts, and with several people close together
  it can switch to the wrong one.

## Corridor walking (`corridor.bat`, standalone, not in the app)

Walks the dog down a corridor using only its lidar. Each scan it heads for the most open direction (so a slanted corridor, a bend or a jog
sideways is just "open to one side": it turns and follows), keeps to the middle when heading straight (with one wall only it keeps ~0.6 m from it),
slows near things, and stops for good at a dead end or when boxed in. It does not plan or remember: it can't pick between two openings.
Bends need a corridor at least ~1 m wide (the safety strip is 0.6 m). Close the phone app / `go2.bat` first (one controller at a time).

1. `corridor-dry.bat`: with the dog standing in a corridor, prints the left / right distances, how much room there is in the direction it would steer, and the command it *would* send. Never moves the dog. Check the numbers against a tape measure.
2. `corridor.bat`: really walks (0.2 m/s, 40 s max; change `--speed` in its `OPTS` line). **Ctrl-C** stops it; it also stands still if the lidar goes quiet for half a second.

Tested against simulated corridors (`python corridor.py`) and briefly on the real dog. **What recordings of the dog's lidar (`lidar-probe.bat`) showed:**
each message is a persistent map (~13-40k points, ~7.7 per second), not one scan; nothing above ~1.2 m comes back; a person standing 2 m away shows up as a
solid blob, but one walking about barely registers, and a "ghost" of someone who left lingers for ~40 s; and nothing standing up was ever seen closer than
~1 m to the dog (a person 0.6 m beside it never appeared). So walls closer than ~1 m may be invisible to the corridor walker, and lidar can't track a person
at heel or guide-dog distance. The lidar also went silent once after 49 s and once never started: `lidar-probe.bat` and `corridor.bat` now reset it (off, then on) when that happens.

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

- `go2.bat` / `run_go2.sh` / `go2.py`: the app (window, keys, voice actions, box step, heel). Settings such as `--heel-speed` live in the `OPTS` line at the top of `go2.bat`, so you can just double-click it
- `follow.py`: object/person detector (YOLOX-tiny), follow controller, and the heel controller
- `heel_sim.py`: simulated walks that exercise the heel controller (no dog, no model): `python heel_sim.py`
- `obstacles.py`: lidar point cloud -> "how far is the nearest thing in the path / on each side" (self-test: `python obstacles.py`). Not wired into the app yet.
- `corridor.py` / `corridor.bat` / `corridor-dry.bat`: walk down a corridor by lidar (self-test: `python corridor.py`; the dry one never moves the dog)
- `pathplan.py`: lidar -> top-down room grid (saveable) -> A* path -> steering commands (self-test: `python pathplan.py`). Not used by anything yet: it is the next step (go to a place / round obstacles)
- `lidar_probe.py` / `lidar-probe.bat`: read-only check of what the dog's lidar delivers (never moves the dog)
- `dogmic_probe.py` / `dogmic-probe.bat`: read-only check of the dog's own microphone (never moves the dog); `dogmic_test.py`: checks the dog-mic conversion with fake frames
- `voice.py`: microphone capture, wake word + always-listening, local Whisper (and optional ElevenLabs), phrase matcher
- `dimos-go2.bat` / `run_dimos.sh`, `fetch-aes-key.bat` / `fetch_aes_key.sh`, `set-elevenlabs-key.bat` / `set_elevenlabs_key.sh`
- `dog.py`, `dog.bat`, `connect_test.py`: an earlier standalone controller. **Obsolete** (no AES-key support); kept for reference.

## Developer tests

`run_go2.sh --selftest`, `--selftest-follow`, `--selftest-voice`, `--selftest-voicemove`, `--selftest-objects`,
`--selftest-listen`, `--selftest-heel`, `--selftest-calib` run headless scripted checks (`python lock_test.py` checks the clothing lock, run inside WSL) against the fake robot (inside WSL:
`wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/Sasha/go2-robot/run_go2.sh --selftest-listen`).
