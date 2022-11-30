import time
import sys

from Agents.LauncherAgent import LauncherAgent

port = 10000
for i in range(len(sys.argv)):
    if sys.argv[i] == "--interface-port" and len(sys.argv) > i:
        port = sys.argv[i+1]

n0 = LauncherAgent("my_launcher_agent@gtirouter.dsic.upv.es", "abcdefg", port)
f1 = n0.start(auto_register=True)
f1.result()


while n0.is_alive():
    try:
        time.sleep(1)
    except KeyboardInterrupt:
        n0.stop()
        break
