"""Regenerate the two publication plots from their CSV data only."""
from review_common import *
from analyze_profiles import make_plots
import csv

if __name__=='__main__':
    arrays={};r=[]
    for filename,slot,column in [('radial_profiles.csv',0,'rms_error_over_global_reference_rms'),
                                 ('angular_error_spectra.csv',1,'sqrt_relative_error_energy')]:
        with (HERE/filename).open(encoding='utf-8',newline='') as f:
            for row in csv.DictReader(f):
                key=(row['name'],row['arm'],row['reference'],float(row['time']),row['field'])
                arrays.setdefault(key,[[],[]])[slot].append(float(row[column]))
                if slot==0 and key==('rfm41_k80','combined','angular_fine',.5,'velocity'):
                    r.append(float(row['radius']))
    curves={key:tuple(np.array(a) for a in value) for key,value in arrays.items()}
    assert len(r)==144 and all(len(a)==144 and len(b)==145 for a,b in curves.values())
    make_plots(curves,np.array(r))
