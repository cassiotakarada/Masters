import os
import subprocess
import sys
import time

PORT = 6010
LOGDIR = "runs"

def is_port_in_use(port):
    try:
        output = subprocess.check_output(f"lsof -i :{port}", shell=True, stderr=subprocess.DEVNULL)
        return output.decode().strip()
    except subprocess.CalledProcessError:
        return None

def kill_process_using_port(port):
    try:
        output = subprocess.check_output(f"lsof -t -i:{port}", shell=True)
        pids = output.decode().strip().split('\n')
        for pid in pids:
            subprocess.call(f"kill -9 {pid}", shell=True)
            print(f"🔪 Killed process {pid} using port {port}")
    except subprocess.CalledProcessError:
        print(f"⚠️ No process found using port {port}")

def run_tensorboard():
    print(f"🚀 Launching TensorBoard on port {PORT} with logdir='{LOGDIR}'...\n")
    os.system(f"tensorboard --logdir={LOGDIR} --host=0.0.0.0 --port={PORT}")

if __name__ == "__main__":
    print(f"🔍 Checking port {PORT}...")
    if is_port_in_use(PORT):
        print(f"❌ Port {PORT} is in use. Attempting to kill process...")
        kill_process_using_port(PORT)
        time.sleep(1)

    run_tensorboard()
