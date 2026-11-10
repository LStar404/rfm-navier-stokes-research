# v8 evidence and fixture provenance

Numerical trajectories and scientific failure records are kept as saved. Their frozen hashes continue to identify the original bytes.

Two synthetic invalid-parent-birth stderr records contain local path prefixes. Their originals remain in the workspace and are omitted here. The two separately named public copies substitute path placeholders. The original frozen summaries still bind their private originals and their recorded hashes do not match the public copies. The [redaction mapping](synthetic_fixture_stderr_redaction_map.json) records both original and public hashes. Run the synthetic fixture again to generate local original-form stderr.

The inventory may identify exact historical source capsules at a different path. For historical replay use those bytes at the original logical path in a disposable worktree. The released current files keep their own hashes.

The [vendor recipe](cuda129_vendor_public_manifest.json) identifies the official Windows wheels, license sources and hashes. Retrieve the tools locally. This archive carries no modern vendor binaries.
