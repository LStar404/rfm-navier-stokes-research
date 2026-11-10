"""Reconstruct all tables/figures from the immutable saved trajectories.

This separate entry preserves the original execution driver's frozen source hash.
It does not rerun the CPU marches or create a new reference.
"""
from review_common import *
import run_bridge
import analyze_profiles
import subprocess

if __name__=='__main__':
    protocol=read(HERE/'bridge_protocol.json')
    for name,expected in protocol['sources'].items():
        assert sha(WORK/name)==expected, name
    for name,expected in protocol['frozen_protocols'].items():
        assert sha(WORK/name)==expected, name
    runs=[]
    for name in NAMES:
        record=read(HERE/(name+'_switch12.json'))
        assert sha(HERE/record['artifact'])==record['sha256']
        runs.append(record)
    run_bridge.analyze(runs)
    analyze_profiles.main()
    subprocess.run([sys.executable,str(HERE/'verify_profiles.py')],check=True)
