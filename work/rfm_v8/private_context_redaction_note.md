# Private-context failure log provenance

Real private-context device failures from the initial health run and the teardown callback probe. Their untouched original stderr logs stay in the workspace. Public copies preserve the trace text with local path prefixes replaced. The original and public digests identify different byte sequences. Frozen protocol, phase, result and watchdog records retain their original bytes. The public derivatives do not stand in for original hashes in any verification. No experiment was rerun to produce them.

The [redaction mapping](private_context_stderr_redaction_map.json) records both byte identities. Include the public traces and map together when staging the two failure directories. Keep the original stderr paths out of the release allowlist.
