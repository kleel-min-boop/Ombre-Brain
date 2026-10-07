# Zer Home Ombre production baseline

This independent branch preserves official Ombre 3.6.14 at
`P0luz/Ombre-Brain@115d831d979bf67efc7150f976e2cb34c333c935`, plus the already
deployed Zer `dream_archive` overlay. It does not replace the old fork main.

The overlay is exactly the four-file result of the version-pinned
`external/integrations/ombre-dream-archive/apply_overlay.py` in
`kleel-min-boop/zer-home@8da815304d2e71351898b005196b73330c678a45`:
`src/server.py`, `src/bucket_manager.py`, `src/tools/hold/__init__.py`, and
`src/tools/hold/core.py`. Explicit `hold(dream_archive=True)` creates a separate
bucket with `dont_surface=True` at first write and returns its real ID.
Ordinary hold/grow rules are unchanged. No provider data or credentials are here.

Verified image: `zer-home/ombre-brain:3.6.14-dream-archive-v1`,
`sha256:ef37d8301e1304f4e65b147c588a9a7d55e6df6d5182d36a2b1513def44b6a18`.
Public runtime source must match this baseline before enabling the lane.
Future official upgrades must port/retest this explicit extension; do not use
old Compose, lose the overlay, or replay it onto already-patched source.
