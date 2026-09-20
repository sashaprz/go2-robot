"""Wake-word matching: anything that sounds remotely like 'ernest' wakes it, ordinary words never do.   python wake_test.py"""
from __future__ import annotations

import voice

WAKES = ["Ernest, sit down", "Bernest, sit down.", "Burnest sit down", "Earnest, follow me", "Airnest, heel", "Urnest stop following", "Ernst, come here",
         "Burnett, lie down", "Ernie, dance", "Bernie, say hello", "Hey Bernest, turn around", "Okay earnest turn left", "Er nest, follow me", "Burn est, sit",
         "Earn est follow me", "Ernesto, stand up", "Ernestine, sit", "Irnest, sit", "Hernest, sit down", "Ernist, sit", "Ernest"]
NOT_WAKES = ["Turn left", "Turn around", "Sit down", "Stand up", "Stop", "Follow me", "Come here", "Nest of birds", "Rest a minute", "Burn the toast", "Earn some money",
             "The internet is down", "Burnt toast", "Harness the horse", "Yes", "Heel", "Lead me", "Go back", "Interest rates", "Learned a lot", "I turned around",
             "Walk forward", "Best of luck", "Honest question", "Concerned about it", "What time is it"]


def main() -> int:
    checks = {}
    miss = [t for t in WAKES if not voice.strip_wake(t)[0]]
    false = [t for t in NOT_WAKES if voice.strip_wake(t)[0]]
    print("  should wake but didn't:", miss or "none")
    print("  should NOT wake but did:", false or "none")
    checks[f"all {len(WAKES)} sound-alikes wake it (bernest, burnest, earnest, airnest, ernst, burnett, ernie, 'er nest', 'burn est', ...)"] = not miss
    checks[f"none of {len(NOT_WAKES)} ordinary phrases wake it (turn, nest, rest, burn, earn, internet, ...)"] = not false
    rest = {t: voice.strip_wake(t)[1] for t in ("Bernest, sit down", "Er nest, follow me", "Hey burnett lie down", "Earn est heel")}
    checks["what follows the wake word is passed on ('sit down', 'follow me', 'lie down', 'heel')"] = list(rest.values()) == ["sit down", "follow me", "lie down", "heel"]
    checks["the routing: 'Bernest, turn around' is a command, a bare 'Bernest' opens the window, 'turn around' alone is ignored"] = (
        voice.route_utterance("Bernest, turn around") == ("command", "turn around") and voice.route_utterance("Bernest.") == ("wake", "")
        and voice.route_utterance("turn around")[0] == "ignore")
    checks["a bare 'stop' still works without any wake word"] = voice.route_utterance("stop")[0] == "stop"
    checks["a custom wake word is still exact only ('rex, sit' wakes, 'bernest, sit' does not)"] = (
        voice.strip_wake("Rex, sit", "rex")[0] and not voice.strip_wake("Bernest, sit", "rex")[0])
    for name, ok in checks.items():
        print(("  PASS  " if ok else "  FAIL  ") + name)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
