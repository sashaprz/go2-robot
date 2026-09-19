"""Connect to a Go2 over its Wi-Fi hotspot and read status. Sends NO movement commands."""
import asyncio
import logging

from go2_webrtc_driver.webrtc_driver import Go2WebRTCConnection, WebRTCConnectionMethod
from go2_webrtc_driver.constants import RTC_TOPIC

logging.basicConfig(level=logging.WARNING)


async def main():
    # LocalAP = joined to the dog's own hotspot (dog is at 192.168.12.1)
    conn = Go2WebRTCConnection(WebRTCConnectionMethod.LocalAP)
    await conn.connect()
    print("Connected to Go2 over WebRTC.")

    got = asyncio.Event()

    def on_low_state(message):
        data = message.get("data", {})
        bms = data.get("bms_state", {})
        print(f"Battery: {bms.get('soc')}%")
        got.set()

    conn.datachannel.pub_sub.subscribe(RTC_TOPIC["LOW_STATE"], on_low_state)
    try:
        await asyncio.wait_for(got.wait(), timeout=10)
    except asyncio.TimeoutError:
        print("Connected, but no status message arrived within 10s.")

    await conn.disconnect()
    print("Disconnected.")


if __name__ == "__main__":
    asyncio.run(main())
