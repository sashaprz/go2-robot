#!/bin/bash
# Save your ElevenLabs API key for voice commands. Interactive: the key is typed hidden and stored only in
# ~/.dimos.env (owner-only permissions), which is NOT part of the repo.
echo "Get a key at https://elevenlabs.io (Profile -> API Keys). It needs the speech-to-text permission."
echo
read -rsp "Paste your ElevenLabs API key (hidden), then press Enter: " KEY
echo
KEY="${KEY//[[:space:]]/}"
[ -z "$KEY" ] && { echo "Nothing entered; nothing saved."; exit 1; }
touch ~/.dimos.env && chmod 600 ~/.dimos.env
sed -i '/^ELEVENLABS_API_KEY=/d' ~/.dimos.env
echo "ELEVENLABS_API_KEY=$KEY" >> ~/.dimos.env
echo "Saved to ~/.dimos.env (${#KEY} characters). Restart go2.bat to pick it up."
