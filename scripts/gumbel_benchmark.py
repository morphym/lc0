#!/usr/bin/env python3
"""Compare Gumbel batch sizes on a chosen lc0 backend; no external packages."""
import argparse
import json
import queue
import re
import statistics
import subprocess
import threading
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('engine')
parser.add_argument('--backend', default='random')
parser.add_argument('--weights', default='')
parser.add_argument('--backend-opts', default='')
parser.add_argument('--nodes', type=int, default=512)
parser.add_argument('--batches', default='1,8,32,64')
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--timeout', type=float, default=180)
parser.add_argument('--json', dest='json_path')
args = parser.parse_args()
if args.nodes < 1 or args.repeats < 1:
    parser.error('nodes and repeats must be positive')
results = []
for batch in map(int, args.batches.split(',')):
    if not 1 <= batch <= 256:
        parser.error('batch sizes must be in [1, 256]')
    engine = subprocess.Popen(
        [args.engine, '--backend=' + args.backend, '--weights=' + args.weights,
         '--backend-opts=' + args.backend_opts, '--gumbel-noise=0',
         '--gumbel-batch-size=' + str(batch)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1)
    lines, errors = queue.Queue(), []

    def read_output(stream, target):
        for line in stream:
            target(line.strip())
        target('ENGINE_EXITED')

    threading.Thread(target=read_output, args=(engine.stdout, lines.put), daemon=True).start()
    threading.Thread(target=read_output, args=(engine.stderr, errors.append), daemon=True).start()

    def send(command):
        engine.stdin.write(command + '\n')
        engine.stdin.flush()

    def search(nodes):
        send('ucinewgame')  # Clear the evaluation cache; keep the network warm.
        send('position startpos')
        send('go nodes ' + str(nodes))
        deadline, info = time.monotonic() + args.timeout, None
        while True:
            line = lines.get(timeout=max(0.001, deadline - time.monotonic()))
            if line == 'ENGINE_EXITED' or line.startswith('error ') or 'Gumbel search:' in line:
                raise RuntimeError('\n'.join([line] + errors[-15:]))
            if line.startswith('info depth '):
                info = line
            if line.startswith('bestmove '):
                break
        if info is None:
            raise RuntimeError('Search returned no statistics')
        parsed = {k: int(v) for k, v in re.findall(r'\b(nodes|time|nps|eps) (\d+)', info)}
        if parsed.get('nodes') != nodes:
            raise RuntimeError('Incorrect simulation count: ' + info)
        return parsed

    try:
        # Warm up loading, device initialization, and common batch shapes.
        search(args.nodes)
        runs = [search(args.nodes) for _ in range(args.repeats)]
        ms = statistics.median(max(1, run['time']) for run in runs)
        row = dict(backend=args.backend, batch=batch, nodes=args.nodes,
                   median_ms=ms, nodes_per_second=round(args.nodes * 1000 / ms), runs=runs)
        results.append(row)
        print(f"batch {batch:3d}: {ms:8.1f} ms, {row['nodes_per_second']:8d} nodes/s", flush=True)
    finally:
        if engine.poll() is None:
            send('quit')
            try:
                engine.wait(timeout=10)
            except subprocess.TimeoutExpired:
                engine.kill()
                engine.wait()
if args.json_path:
    with open(args.json_path, 'w') as output:
        json.dump(results, output, indent=2)
