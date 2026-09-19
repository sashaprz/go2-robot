#!/bin/bash
# Fetch the Go2's per-device AES key from the Unitree cloud and save it for DimOS.
# Run this while on a network WITH internet (not the dog's hotspot). Interactive: it prompts for your
# Unitree app account email + password; nothing is stored except the key it saves to ~/.dimos.env.
cd ~/dimensional-applications || { echo "DimOS install not found"; exit 1; }
source .venv/bin/activate

echo "Fetching the Go2 AES key from the Unitree cloud."
echo "Use the email + password of the account the dog is bound to in the Unitree Go app."
echo
read -rp "Unitree account email: " EMAIL
[ -z "$EMAIL" ] && { echo "no email given"; exit 1; }

# Prompts for the password itself (hidden). Without --sn it lists every device bound to the account.
unitree-fetch-aes-key --email "$EMAIL" --device-type Go2 || { echo; echo "Fetch failed (see message above)."; exit 1; }

echo
echo "Copy the 32-character key for your dog from the list above."
read -rp "Paste it here to save it (or press Enter to skip): " KEY
[ -z "$KEY" ] && { echo "Nothing saved."; exit 0; }
if ! [[ "$KEY" =~ ^[0-9a-fA-F]{32}$ ]]; then
  echo "That isn't 32 hex characters, so I didn't save it."; exit 1
fi

touch ~/.dimos.env && chmod 600 ~/.dimos.env
sed -i '/^UNITREE_AES_128_KEY=/d' ~/.dimos.env
echo "UNITREE_AES_128_KEY=$KEY" >> ~/.dimos.env
echo "Saved to ~/.dimos.env. Now switch to the dog's Wi-Fi and run dimos-go2.bat."
