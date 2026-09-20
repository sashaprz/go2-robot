cd ~/dimensional-applications && source .venv/bin/activate && set -a && . ~/.dimos.env 2>/dev/null; set +a; cd /mnt/c/Users/Sasha/go2-robot
bash -n run_go2.sh && echo "run_go2.sh syntax ok"
for t in selftest-lead selftest-lead-stairs selftest-listen selftest-voice; do
  out=$(timeout 200 python go2.py --$t 2>&1)
  echo "app --$t: $(echo "$out" | grep -c PASS) pass, $(echo "$out" | grep -c FAIL) fail"; echo "$out" | grep -E "FAIL|Traceback"
done
