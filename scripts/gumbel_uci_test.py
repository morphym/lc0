#!/usr/bin/env python3
"""Integration checks: python3 scripts/gumbel_uci_test.py /path/to/lc0."""
import queue
import re
import subprocess
import sys
import threading
import time

engine = subprocess.Popen(
    [sys.argv[1], '--backend=random', '--weights=',
     '--gumbel-noise=0', '--gumbel-simulations=64', *sys.argv[2:]],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    text=True, bufsize=1)
lines = queue.Queue()

def read():
    for line in engine.stdout:
        lines.put(line.strip())
    lines.put('ENGINE_EXITED')

threading.Thread(target=read, daemon=True).start()

def send(command):
    engine.stdin.write(command + '\n')
    engine.stdin.flush()

def until(prefix, timeout=10):
    seen = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = lines.get(timeout=max(0.001, deadline - time.monotonic()))
        seen.append(line)
        assert line != 'ENGINE_EXITED', seen
        assert 'Gumbel search:' not in line and not line.startswith('error '), seen
        if line.startswith(prefix):
            return line, seen
    raise AssertionError(seen)

def search(position, go, expected=None, nodes=None):
    send('position ' + position)
    send(go)
    best, seen = until('bestmove ')
    if expected is not None:
        assert best.split()[1] in expected, (best, seen)
    if nodes is not None:
        assert any(f'nodes {nodes} ' in s + ' ' for s in seen), seen
    return best

try:
    send('uci')
    _, seen = until('uciok')
    assert any('GumbelConsideredMoves' in line for line in seen)
    # Nibbler sends these options unconditionally to lc0 engines.
    for name, value in [('LogLiveStats', 'true'), ('MoveOverheadMs', '0'),
                        ('MultiPV', '500'), ('ScoreType', 'WDL_mu'),
                        ('SmartPruningFactor', '0'), ('UCI_ShowWDL', 'true'),
                        ('VerboseMoveStats', 'true')]:
        send(f'setoption name {name} value {value}')
    send('isready')
    until('readyok')
    send('position startpos')
    send('go nodes 1')
    _, seen = until('bestmove ')
    rows = [s for s in seen if re.match(r'info string [a-h][1-8][a-h][1-8]', s)]
    assert len(rows) >= 20, seen
    assert all(re.search(r'N: \d+ .*\(P: [0-9.]+%\).*\(D: [0-9.]+\).*\(Q: [-0-9.]+\)', s)
               for s in rows), rows
    priors = [float(re.search(r'\(P: ([0-9.]+)%', s)[1]) for s in rows[-20:]]
    assert abs(sum(priors) - 100) < 0.001, priors
    variants = [s for s in seen if ' pv ' in s]
    assert any('multipv 20 ' in s for s in variants), variants
    for line in variants:
        wdl = re.search(r'wdl (\d+) (\d+) (\d+)', line)
        assert wdl and sum(map(int, wdl.groups())) == 1000, line
    # Infinite searches must cross the initial budget while reusing the tree.
    send('go infinite')
    time.sleep(0.65)
    send('stop')
    _, seen = until('bestmove ')
    counts = [int(re.search(r'nodes (\d+)', s)[1]) for s in seen if ' pv ' in s]
    assert len(counts) >= 40 and max(counts) > 64, seen
    assert counts == sorted(counts), counts
    assert any(re.search(r'nps [1-9]\d*', s) for s in seen), seen
    send('setoption name ScoreType value win_percentage')
    send('go nodes 2 searchmoves e2e4')
    _, seen = until('bestmove ')
    assert all(0 <= int(re.search(r'score cp (-?\d+)', s)[1]) <= 10000
               for s in seen if ' pv ' in s), seen
    send('setoption name ScoreType value WDL_mu')
    search('startpos', 'go nodes 32 searchmoves e2e4 d2d4', {'e2e4', 'd2d4'}, 32)
    search('startpos moves e2e4', 'go nodes 17 searchmoves e7e5', {'e7e5'}, 17)
    # Terminal mate values must beat network values for both colors.
    search('fen 7k/8/5KQ1/8/8/8/8/8 w - - 0 1',
           'go nodes 64 searchmoves g6g7 g6g5', {'g6g7'}, 64)
    search('fen 8/8/8/8/8/5kq1/8/7K b - - 0 1',
           'go nodes 64 searchmoves g3g2 g3g4', {'g3g2'}, 64)
    for fen in ['7k/6Q1/5K2/8/8/8/8/8 b - - 0 1',
                '7k/5K2/6Q1/8/8/8/8/8 b - - 0 1',
                '7k/8/8/8/8/8/8/K7 w - - 0 1']:
        search('fen ' + fen, 'go nodes 10', {'0000'}, 0)
    search('fen 7k/P7/8/8/8/8/8/7K w - - 0 1',
           'go nodes 7 searchmoves a7a8n', {'a7a8n'}, 7)
    search('fen r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
           'go nodes 7 searchmoves e1g1', {'e1g1'}, 7)
    search('startpos', 'go nodes 0 searchmoves e2e4', {'e2e4'}, 0)
    search('startpos', 'go nodes 1000000 movetime 5')
    # Infinite and ponder must hold bestmove until stop, even after budget ends.
    for mode in ['infinite', 'ponder']:
        send('position startpos')
        send('go ' + mode + ' nodes 8')
        _, seen = until('info ')
        time.sleep(0.03)
        while not lines.empty():
            assert not lines.get_nowait().startswith('bestmove ')
        send('stop')
        until('bestmove ')
    send('position startpos moves e2e4')
    send('go ponder nodes 8')
    until('info ')
    send('ponderhit')
    best, seen = until('bestmove ')
    assert best.split()[1] != '0000', (best, seen)
    # Replacing an active search aborts silently; no stale response may leak.
    send('position startpos')
    send('go infinite nodes 1000000')
    search('startpos moves d2d4', 'go nodes 3 searchmoves d7d5', {'d7d5'}, 3)
    send('isready')
    _, seen = until('readyok')
    assert not any(s.startswith('bestmove ') for s in seen), seen
    print('Gumbel UCI integration tests passed')
finally:
    if engine.poll() is None:
        send('quit')
        try:
            code = engine.wait(timeout=5)
            assert code == 0, f'Engine exited with status {code}'
        except subprocess.TimeoutExpired:
            engine.kill()
            raise
