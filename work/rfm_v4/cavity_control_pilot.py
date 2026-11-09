"""Deterministic controls and matched test enrichment for the v4 cavity study."""
from cavity_pilot import ROOT,Score,evaluate,config,write_json,one_thread
import hashlib


def main():
    trials=[('leg595_p32',config(595,1,32,51,family='legendre',strategy='ladder')),
            ('leg595_p36',config(595,1,36,57,family='legendre',strategy='ladder')),
            ('leg666_p34',config(666,1,34,54,family='legendre',strategy='ladder')),
            ('leg666_p39',config(666,1,39,60,family='legendre',strategy='ladder')),
            ('rfm666_p39',config(666,16,39,60,strategy='ladder')),
            ('leg666_mild_p39',dict(config(666,1,39,58,family='legendre',strategy='guard'),
                                   wall=dict(warp='sinh',strength=[1.3,1.3])))]
    result=dict(rows=[],source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
                       for name in ('cavity_solver.py','cavity_pilot.py','cavity_control_pilot.py')})
    with one_thread():
        score=Score()
        for name,settings in trials:
            result['rows'].append(evaluate(name,settings,17,score));write_json(ROOT/'cavity_control_pilot.json',result)


if __name__=='__main__':main()
