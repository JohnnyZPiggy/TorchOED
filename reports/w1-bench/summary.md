# W1 D-path paired benchmark — FAIL / BLOCKED

Base `91244d841befbbe5d0a9f80faf6e19f269970d69`; head/source `10fdb697b93efb9be23b2ba9896349ca7730f163`; original bench_b0.py, base then head once, no rerun/tuning.
Quality PASS: 40/40 feasible rows have identical base/head IDs and logdet (max paired logdet difference0); grid2n12 remains infeasible.
Informational cross-platform max |logdet_head-logdet_b8| = 3.5527136788005009e-15.
Time FAIL: maximum head/base case-median ratio 1.180738159 >=1.10 on grid3 n=10 (base0.094809320000s,head0.111944982000s).
Case ratios (grid2n6/n7,grid3n10/n20,grid4n15/n30,random500n10/n20): 0.778079, 0.719132, 1.180738, 0.997353, 0.932959, 0.855725, 0.893903, 0.946982.
Observed Tesla T4; accelerator allocated only because the pinned controller has no CPU runtime; unused; no GPU claim.
Environment: {'python': '3.13.16', 'platform': 'Linux-6.6.159+-x86_64-with-glibc2.39', 'cpu': 'x86_64', 'torch': '2.11.0+cu130', 'numpy': '2.4.6', 'scipy': '1.16.3', 'pydoe': '1.6.0', 'threads': 1}; CUDA_VISIBLE_DEVICES='' in each benchmark child. PyDOE rows retained, not judged; wheel hash assertion passed.
Pinned controller5fb26304e4c85183aa215e6ad4db9ced928cab12; --device T4 --task-seconds2400 --allocation-seconds2300 --work-timeout1800; one allocation.
Cleanup confirmed: stop PASS, authenticated assignments0 before/0 after; no active task session. Existing entitlement only.
Preflight first returned LOCAL_PREFLIGHT_FAILED before allocation; a read-only diagnostic then passed; no source/controller change. Only one paired experiment ran.
External download SHA256/byte size in results.json; JSON compacted whitespace only. No tracked file hashes. Prior code CI5/5 PASS at10fdb69; required local tests3 PASS.
Next: Architect ruling on the failing time gate; no rerun permitted by current dispatch.
