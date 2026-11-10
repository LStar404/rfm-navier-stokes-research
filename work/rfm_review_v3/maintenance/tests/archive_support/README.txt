Exact archival support for the isolated comparison process
=========================================================

certify_rfm.py is an exact-byte copy of work/rfm_ns/certify_rfm.py from the
research workspace. Its SHA-256 and original location are in ../../provenance.json.
It has not been rewritten, simplified, stubbed, or substituted for any solver.

Reason: the published legacy operator_opt.py imports certify_rfm.raw_fields
for its audit driver, but the older public distribution omitted that module.
Importing the original Gram assembly therefore failed at that unrelated import.
The maintenance test child process adds this directory to sys.path and executes
the original archived modules with their original arithmetic. The support file
is never imported by the maintained solver, and its command-line audit is not run.
