"""
Auto-sync local changes to remote server via rsync over SSH.
Usage: python sync_to_server.py --host root@your-server-ip

Watches for file changes and syncs automatically.
"""

import subprocess
import time
import argparse
import sys
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

parser = argparse.ArgumentParser()
parser.add_argument('--host', required=True, help='e.g. root@123.45.67.89')
parser.add_argument('--remote-dir', default='/root/autodl-tmp/motion-agent/')
parser.add_argument('--local-dir', default=str(Path(__file__).parent))
parser.add_argument('--port', default='22')
parser.add_argument('--debounce', type=float, default=2.0, help='seconds to wait before syncing after a change')
args = parser.parse_args()

EXCLUDE = [
    '--exclude=.git',
    '--exclude=venv_grpo',
    '--exclude=__pycache__',
    '--exclude=*.pyc',
    '--exclude=dataset/new_joint_vecs',
    '--exclude=dataset/new_joints',
    '--exclude=dataset/pose_data',
    '--exclude=*.pth',
    '--exclude=experiments_grpo',
    '--exclude=node-v*',
    '--exclude=.claude',
]

def sync():
    cmd = [
        'rsync', '-avz', '--progress',
        *EXCLUDE,
        '-e', f'ssh -p {args.port}',
        args.local_dir + '/',
        f'{args.host}:{args.remote_dir}',
    ]
    print(f'\n[SYNC] {time.strftime("%H:%M:%S")} Syncing...')
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        changed = [l for l in result.stdout.splitlines() if l and not l.startswith('sending') and not l.startswith('sent')]
        print(f'[OK] {len(changed)} files synced')
        for f in changed[:10]:
            print(f'  {f}')
    else:
        print(f'[ERROR] rsync failed:\n{result.stderr}')

class ChangeHandler(FileSystemEventHandler):
    def __init__(self):
        self._pending = False
        self._last_event = 0

    def on_any_event(self, event):
        if event.is_directory:
            return
        path = event.src_path
        # Skip irrelevant files
        skip = ['.git', 'venv_grpo', '__pycache__', '.pyc', '.pth',
                'experiments_grpo', '.claude', 'node-v']
        if any(s in path for s in skip):
            return
        self._last_event = time.time()
        self._pending = True

    def check_and_sync(self):
        if self._pending and (time.time() - self._last_event) >= args.debounce:
            self._pending = False
            sync()

# Initial sync
print(f'Syncing {args.local_dir} → {args.host}:{args.remote_dir}')
sync()

# Watch for changes
handler = ChangeHandler()
observer = Observer()
observer.schedule(handler, args.local_dir, recursive=True)
observer.start()
print(f'\nWatching for changes (debounce={args.debounce}s)... Ctrl+C to stop\n')

try:
    while True:
        handler.check_and_sync()
        time.sleep(0.5)
except KeyboardInterrupt:
    observer.stop()
    print('\nStopped.')
observer.join()
