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
| G, then **Y** | **lead**: walk to a spot (`--lead-goal`) round obstacles, stopping at drop-offs (G again / Space / any drive key stops it) |
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

**The AirPods (or any named Windows microphone): `--mic win --mic-device AirPods` (off in `go2.bat` now: on this laptop the AirPods mic stays silent, and the phone link below replaces it; `set WINMIC=AirPods` turns it on).** WSL only ever sees whichever microphone
Windows has as its default, and can't start Windows programs here, so `go2.bat` starts a small Windows-side helper (`winmic.py`, using `sounddevice`) that
captures the microphone whose name contains "AirPods" and serves it to the app over a local socket. Both the always-on ear and push-to-talk (`V`) use it.
It listens only on the WSL virtual network address (not Wi-Fi or Ethernet), requires a secret token (`.winmic_token`), captures only while the app is
connected, and exits by itself when the app is gone. About 8 seconds after the ear starts the window says the mic is live (with its level in the top
bar) or says why not and **falls back to the laptop mic**: "SILENT" (the AirPods must be in your ears and connected to *this PC*, not the iPhone) or "can't
reach the Windows mic helper" (launch `go2.bat`). To always use the laptop mic, set `set WINMIC=` in `go2.bat`. On another PC:
`.venv\Scripts\python.exe -m pip install sounddevice`, then `.venv\Scripts\python.exe winmic.py --list`. Checked with a synthetic mic (15 checks), the real
Windows helper capturing the laptop mic through the real WSL link (level 0.019) and finding the AirPods by name; **when I tried it, the AirPods mic returned
digital silence (level 0.00001)**, i.e. it was not delivering audio, so I could not confirm it end to end with your voice.

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
| "lead", "lead me", "lead me five metres", "lead me forward 4 metres and left 2", "stop leading" | walk to a spot round obstacles, **stopping at drop-offs** (see Lead; starts straight away; `G` still asks for `Y`) |
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

## Lead (`G` or "ernest, lead")

Says "lead" and the dog walks from where it stands (A) to a point B, round obstacles, **and stops at drop-offs** (stairs down, a ledge, a hole). It is the guide mode
below (`guide.py`, `pathplan.py`, `dropoff.py`) running inside this app, on the dog's lidar, in a background thread so the window stays smooth.

- **Say it:** "ernest, lead" / "lead me" walks to the usual spot (`--lead-goal AHEAD,LEFT`, default `4,0` = 4 m ahead). **"ernest, lead me five metres"**, "lead me forward 4 metres and left 2",
  "guide me three metres to the right" and "lead me ten feet" choose the spot out loud (each number goes with the nearest direction word; a number with none is forward; capped at 15 m).
  "Led" and "leed" (how Whisper often hears it) work. **"stop"**, **"stop leading"**, Space, `G` again, any drive key, or the window losing focus ends it. No confirmation for the voice command
  (like "follow me"); `G` asks for `Y`.
- **B is measured from where the dog is standing and facing when you say it**, so saying "lead me 4 metres" a second time walks 4 m further along whatever way it is now facing.
- **What it does:** balances, waits for the dog's lidar, plans a path (A*) that keeps the dog's width off obstacles and a person's width off anything up to ~1.2 m high, and walks it at `--lead-speed` (0.3 m/s),
  re-planning twice a second. A drop-off is seen from about 2 m and is a no-go with a 0.75 m margin: **the dog stops about 1 m before it, says `DROP-OFF ahead` on screen, waits, and gives up after `--lead-patience`
  seconds (90) saying a drop-off is in the way.** With no way through it waits; if something is inside the lidar's ~1 m blind zone it backs away a little to look again (`--no-lead-recover` turns that off). It also stops
  if the lidar goes quiet or the dog is told to walk and isn't moving.
- **On screen:** the banner (`LEADING to 4.0 m ahead, +0.0 m left: going [2.3 m to go]`, or `waiting - drop-off 1.4 m ahead`), the messages, and a **top-down map** at the top right: red obstacle, orange something at
  person height, **magenta drop-off**, green path, yellow goal, cyan dog. Look at the magenta before trusting it.
- **Not people-aware, and it has never run on the real dog.** It does not check that you are following, and nothing shown on screen is spoken. The lidar returns nothing above ~1.2 m (a hanging sign, a low beam or ceiling is
  **not seen**), and the drop-off detector has only been tried on simulated ledges and your flat-floor recording. Test it on a real staircase from the top with someone at the bottom and a hand on Space. Wearing the phone/AirPods
  makes no difference to it. Tested headless (`run_go2.sh --selftest-lead` and `--selftest-lead-stairs`): a simulated room, the real lidar's quirks, "ernest, lead me" -> a box walked round and B reached; "stop" mid-walk holds the dog still;
  a staircase down across the room -> it stops ~1 m short, warns, gives up, and never goes near the edge.

## Follow mode (`T` or "follow me")

Turns to keep the nearest person centred and walks toward them, stopping when they fill about **78% of the frame height** (about 1.3 m from the dog's
centre; it used to be 60%, about 2.5 m: `--follow-height`). It walks at up to **0.8 m/s** (`--follow-speed`, it used to be 0.35), with a stiffer speed response
and a small integral term so it keeps pace with a walking person instead of trailing 4-9 m behind, and it **backs away** if someone walks right up to it.
In simulation it stops about 1.3 m from you and trails about 2 m behind a walking person (it was 5.7 m and 6.7-8.9 m), never lost the person across
starts, stop-and-go and turns on a 15 or 10 fps detector, and kept a person walking straight at it at least 1.0 m away (the old settings touched them).
It uses the clothing lock described under Heel (it will only follow the person it locked onto, and says what that is).
**When it loses you, it looks for you.** If the camera can't find you for over 0.6 s (you stepped out of the picture, walked past the dog, a fast turn),
the dog does not give up: it turns on the spot toward the side you were last seen on, about 75 degrees, then sweeps across to the other side and back,
for up to 8 s (`scan_for` in `follow.py`), and picks up again the moment your clothing colours reappear anywhere in the picture (it still never switches to
someone else). It never walks while searching. With a phone streaming and saying you've stopped, it keeps looking 8 s longer. In simulation on a 10 fps
link, in 9 runs of three losing situations (a quick step out of view, walking up level with the dog, a 90 degree turn) heel gave up in 5 without the scan and in
0 with it (at 5 fps it rescues sharp turns too; the walk-up-level case at 5 fps additionally needs the phone). If nothing is found after the search it stops and says so.
**No obstacle avoidance**: keep the path clear, and remember it now moves faster. The detector is YOLOX-tiny on CPU (`follow.py`); no cloud.
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

## Phone link (iPhone: microphone + walking/standing)

**Delay after you speak** is mostly the speech-to-text. On this laptop (measured, machine idle) `small.en` takes ~1.2-1.5 s for a short command with unpredictable
spikes to 7-11 s when something else is using the CPU, while `base.en` takes a steady ~0.45 s (Whisper always processes a fixed 30 s window, so only a smaller
model is faster). So while the phone's microphone is streaming (right by your mouth, clean sound) the app uses `base.en`, and the laptop mic keeps `small.en`
(the noisy-room tests: `small.en` 71/108 commands vs `base.en` 61/108, which a close mic mostly removes). Other changes: the sentence is cut ~0.2 s sooner, a
backlog of speech now drops the OLDEST sentence (it used to drop the newest) and a sentence that waited over 6 s is skipped rather than acted on late, the phone's
audio is kept within 0.25 s of real time, and the `heard:` line in the top bar ends with `[x.x s]`: the time from the end of your sentence to the text. To keep the
accurate model always, add `--fast-model none` to `OPTS` in `go2.bat`.

`go2.bat` starts `phone_server.py` (`set PHONE=1`, on by default; `set PHONE=` turns it off). Your iPhone opens a page from the PC and streams
(1) its **microphone**, which becomes the mic for the wake word and for `V` whenever it is streaming (the laptop mic before it connects and again if it
drops out, switching by itself both ways), and (2) its **motion sensors**, from which the page decides *walking or standing*. If the camera loses you and the
phone says you have stopped, the dog keeps looking (and waits up to 8 s longer) instead of giving up. Everything to do once, and every time, is in
**`phone-setup.txt`**: put the phone on the dog's Wi-Fi with the laptop, run `phone-firewall.bat` once as administrator, install and trust the small local
root certificate on the iPhone (iOS only gives a page the microphone and motion sensors over HTTPS, and only trusts certificates you install; this one is
restricted to local names and private addresses), then scan the QR code that `go2.bat` opens (`phone_qr.html`) and tap Start. The app says "phone
connected" and "the ear is now listening through the phone microphone", and the top bar shows the phone mic level.

Safety: what the phone streams can move the dog (a spoken command is a command), so the page and its WebSocket need a random secret that is only in your
link (`.phone_token`), the ports the app reads from listen only on the WSL virtual address and need a second secret (`.phone_wsl_token`), and the server exits
by itself once the app is gone. The iPhone's browser stops the page when the screen locks (the page asks to keep the screen awake); then the app says the phone
was lost and carries on with the laptop mic and the camera alone.

What was tested here: the server's TLS chain against our own root, the secrets, audio byte-for-byte through it (13 + 17 checks: `phone_test.py`,
`phonelink_test.py`), and the page itself in a real headless Edge with a fake beeping microphone (audio resampled to 16 kHz comes out as the right
400 Hz beep, motion arrives at 10 Hz; `phone_page_test.py`). **Not tested (needs your iPhone):** installing/trusting the certificate, Safari's permission
prompts, the real sensor readings, the screen lock, and whether the dog's Wi-Fi lets your phone talk to the laptop (some access points isolate clients).
The page also has a one-time "turning set-up" and can send the person's turning rate, but the follower ignores it: in simulation steering by it gave mixed
results and a backwards sign made the dog lose you, so `phone_ff` is off (only walking/standing is used). The phone is an aid, not a requirement: without it
everything works from the camera alone. (AirPods position is not available to apps, and an AirTag can't be read by the laptop or the dog either.)

## Corridor walking (`corridor.bat`, standalone, not in the app)

Walks the dog down a corridor using only its lidar. Each scan it heads for the most open direction (so a slanted corridor, a bend or a jog
sideways is just "open to one side": it turns and follows), keeps to the middle when heading straight (with one wall only it keeps ~0.6 m from it),
slows near things, and stops for good at a dead end or when boxed in. It does not plan or remember: it can't pick between two openings.
Bends need a corridor at least ~1 m wide (the safety strip is 0.6 m). Close the phone app / `go2.bat` first (one controller at a time).

1. `corridor-dry.bat`: with the dog standing in a corridor, prints the left / right distances, how much room there is in the direction it would steer, and the command it *would* send. Never moves the dog. Check the numbers against a tape measure.
2. `corridor.bat`: really walks (0.2 m/s, 40 s max; change `--speed` in its `OPTS` line). **Ctrl-C** stops it; it also stands still if the lidar goes quiet for half a second.

**Stairs and ledges:** each scan is also checked for a drop-off (see Guide mode). One within 1.3 m in the +-60 degree cone ahead is a **hard stop** (`drop-off ahead (stairs down?)`),
regardless of which way the steering wants to go, because a lip across the corridor leaves diagonals toward the side walls looking "open". Simulated: a corridor ending in a two-step staircase
stops 1.1 m short of the top step, and without the detector it walks over. Things up to ~1.2 m high count as obstacles now (a shelf across the corridor stops it); above that nothing is seen.
Like everything in the corridor walker it stops well short only of a drop-off: for an ordinary obstacle that fills the whole corridor it can end up ~0.3-0.7 m from it.

Tested against simulated corridors (`python corridor.py`) and briefly on the real dog. **What recordings of the dog's lidar (`lidar-probe.bat`) showed:**
each message is a persistent map (~13-40k points, ~7.7 per second), not one scan; nothing above ~1.2 m comes back; a person standing 2 m away shows up as a
solid blob, but one walking about barely registers, and a "ghost" of someone who left lingers for ~40 s; and nothing standing up was ever seen closer than
~1 m to the dog (a person 0.6 m beside it never appeared). So walls closer than ~1 m may be invisible to the corridor walker, and lidar can't track a person
at heel or guide-dog distance. The lidar also went silent once after 49 s and once never started: `lidar-probe.bat` and `corridor.bat` now reset it (off, then on) when that happens.

## Guide mode (`guide-dry.bat` / `guide.bat`, standalone, not in the app)

Leads the dog from where it stands (A) to a point B you give as **coordinates**, round whatever its lidar sees. B is in metres from the dog's
starting spot: `--goto 4,1` is 4 m **ahead** and 1 m to the **left** (`--goto 3,-2` is 3 ahead, 2 to the right). Several points (`--goto 4,0 4,3`) are
visited in order. `--odom` reads them as x,y in the dog's own odometry frame instead (it resets whenever the dog reboots). Close the phone app / `go2.bat` first.

How it works: each lidar message goes into a top-down grid (`pathplan.RoomMap`); A* finds the cheapest path that keeps clear of obstacles and prefers the
middle of a corridor; it follows that path and **re-plans twice a second**, so a box that turns up mid-walk is walked round. With no way through it **stands
still and waits**, and gives up after `--patience` seconds (default 90) saying why. It also stops if the lidar goes stale or nearly empty (and resets the
lidar), if it is told to walk but isn't moving (`stuck`), on Ctrl-C, and at `--seconds` (default 120).

1. `guide-dry.bat`: plans from the real lidar and prints the map (`#` obstacle, `+` too close to pass, `v` drop-off, `*` path, `D` dog, `G` goal) plus what it *would* send. Never moves the dog.
   Compare the picture with the room before trusting it.
2. `guide.bat`: really walks (`--speed 0.3` in its `OPTS` line). Ctrl-C stops it. `--no-recover` turns off backing away (see below).

**Drop-offs (stairs down, ledges, holes).** A lidar mounted 0.35 m up cannot see the ground just past a lip, so `dropoff.py` looks for what that leaves behind:
floor that was returned right up to a line and then stops in plain sight (with the floor just before it seen, and nothing standing there to cast a shadow), or ground
well below the floor (below -0.12 m: the recording's flat floor scatters within -0.08..+0.04 and never went lower). Those cells go in the map as no-go with a
**0.75 m margin** (the dog's centre stops about 0.6 m from the lip, its nose about 0.25 m), the dog will not walk toward one within 0.6 m even if the planner says go, and the map
remembers them after the dog gets too close to judge them (it judges 0.5-2.5 m ahead, front +-85 degrees only, because the recording showed floor returned 100% out to 1.5 m in
front, ~80% at 2-3 m, and poorly behind or beside a wall). A ledge is first noticed about 2 m away. Tested in simulation against a lidar that hides the ground past a lip
(a step, a 4-step staircase, a 50 cm drop, an angled ledge, a hole in the middle of a room, starting 0.4 m from an edge), and for false alarms: **0 in 95 frames of your real recording**,
0 across 10 runs with a lidar far sparser than the real one, and none in any of the flat-floor runs. It cannot see a drop with less than ~5 cm of shadow, a dark or mirrored floor looks like a
void (it stops: the safe way to be wrong), and a ramp down that shows lower ground is treated as a drop-off too.

**Things up high.** Anything the lidar returns up to ~1.2 m (`HEAD_TOP`; the recording tops out at 1.19 m) is an obstacle, up from 1.0 m before. Anything from 0.45 m up also keeps a
**person's width clear (0.35 m)**, not just the dog's (0.28 m): a table top, shelf or counter edge the dog could walk under still hits the person behind it. **Above ~1.2 m this lidar returns
nothing at all, so a hanging sign, a low beam, a low ceiling or an awning is not seen and not avoided** (there is a test that says so: the dog walks under a sign at 1.5 m). Seeing that would need
another sensor (a depth camera, or a monocular-depth model on the front camera).

**Back away and look again.** The lidar cannot see anything standing nearer than ~1 m, so an obstacle that gets that close is only *remembered*, and the map does not believe it has gone until the dog
can look again. Without a fix, someone who steps in front of the dog and leaves keeps blocking it until it gives up. So after 5 s of waiting, if something in its path (or a drop-off) is inside that blind
zone, it **backs straight away at 0.15 m/s just far enough to see past it** (at most 1.2 m), looks again, and carries on if the way is clear. Once per wait, and only into space the map knows is empty. **It
cannot see a person standing right behind it** (that is inside the blind zone too): use `--no-recover` if someone will be there.

Two properties of the real lidar shaped all of it (from `lidar_recording.npz`): each message is the dog's **own persistent map** (successive messages share 91-99% of their points), so the
grid is rebuilt from each message instead of accumulating, and **nothing at obstacle height is returned nearer than ~1 m**, so cells within 1.1 m of the dog keep what was seen before it got that
close (a new return there still counts; only "it's gone" isn't believed). *Which* of those two the dog's map really does near itself isn't known from the recordings, so the simulator tests both.
`python guide.py` runs 29 checks on simulated rooms (no dog): open floor, a box in the way, a box that appears mid-walk, a doorway across the room, a gap too narrow (waits, then gives up),
a corridor blocked and then cleared (waits out the 40 s ghost, then walks on), lidar dropout, a dog that won't move, two points in a row, a start that isn't at the origin or facing +x, stairs, a
hole, a table, a shelf, a sign, and someone stepping in and leaving (with and without the back-away).

**What it does not do (read before using it near a person):**
- **It does not lead a person.** It walks to B whether or not anyone is following. Nothing checks that the handler is still behind the dog; that is the next piece (turn and look, or a handle).
- **Nothing above ~1.2 m is seen** (see above), and it has no idea about stairs going *up* beyond the risers being obstacles, a wet or slippery floor, or glass.
- **A ghost is an obstacle.** Something that left lingers in the dog's map for ~40 s and the dog waits it out (patience 90 s covers that plus the time it stood there).
- **Untested on the real dog: none of this has moved it yet, and the drop-off detector has never seen a real ledge**, only your flat-floor recording and simulated ones. Try `guide-dry.bat` at the top of a
  staircase first (the map should show a `v` line at the lip) before trusting it to stop. Stay next to the dog with the window in reach: Ctrl-C stops it.

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
- `obstacles.py`: lidar point cloud -> "how far is the nearest thing in the path / on each side" (self-test: `python obstacles.py`). Used by the corridor walker, heel and guide; not wired into the app's other modes.
- `dropoff.py`: stairs down / ledge / hole detection from the lidar (self-test: `python dropoff.py`); used by `guide.py` (through `pathplan.RoomMap`) and `corridor.py`
- `corridor.py` / `corridor.bat` / `corridor-dry.bat`: walk down a corridor by lidar (self-test: `python corridor.py`; the dry one never moves the dog)
- `pathplan.py`: lidar -> top-down room grid (saveable) -> A* path -> steering commands (self-test: `python pathplan.py`). Used by `guide.py`
- `guide.py` / `guide.bat` / `guide-dry.bat`: go from A to a coordinate B round obstacles by lidar (self-test: `python guide.py`; the dry one never moves the dog)
- `lidar_probe.py` / `lidar-probe.bat`: read-only check of what the dog's lidar delivers (never moves the dog)
- `dogmic_probe.py` / `dogmic-probe.bat`: read-only check of the dog's own microphone (never moves the dog); `dogmic_test.py`: checks the dog-mic conversion with fake frames
- `winmic.py` (Windows-side, started by `go2.bat`) / `winmic_test.py`: capture a named Windows microphone such as the AirPods and serve it to the app
- `phone_server.py` + `phone.html` + `phone_tls.py` (Windows-side, started by `go2.bat`), `phonelink.py` (the app's side), `phone-setup.txt`, `phone-firewall.bat`: the iPhone link; tests `phone_test.py`, `phonelink_test.py`, `phone_page_test.py` (run with `.venv\Scripts\python.exe`)
- `voice.py`: microphone capture, wake word + always-listening, local Whisper (and optional ElevenLabs), phrase matcher
- `dimos-go2.bat` / `run_dimos.sh`, `fetch-aes-key.bat` / `fetch_aes_key.sh`, `set-elevenlabs-key.bat` / `set_elevenlabs_key.sh`
- `dog.py`, `dog.bat`, `connect_test.py`: an earlier standalone controller. **Obsolete** (no AES-key support); kept for reference.

## Developer tests

`run_go2.sh --selftest`, `--selftest-follow`, `--selftest-voice`, `--selftest-voicemove`, `--selftest-objects`,
`--selftest-listen`, `--selftest-heel`, `--selftest-calib`, `--selftest-lead`, `--selftest-lead-stairs` run headless scripted checks (`python lock_test.py` checks the clothing lock, run inside WSL) against the fake robot (inside WSL:
`wsl -d Ubuntu-24.04 -- bash /mnt/c/Users/Sasha/go2-robot/run_go2.sh --selftest-listen`).
