# Release candidate status

- Runtime published to coolbho3k's GHCR, tag `20260917-rc2`; immutable digest
  pinned in `recipe-lock.json`. 15.40 GiB compressed, 39 bounded layers.
- Anonymous GHCR access and both public HF manifest pins are verified.
- One-pass attention and exact length-aware radix top-k match the v106-tested
  serving overlay byte-for-byte. The original cooperative MoE remains enabled.
- 64 offline tests pass. All 39 image layers were rehashed and scanned, including
  superseded content and 14 nested archives. No current credentials matched;
  all 160 pattern/path findings were reviewed as upstream fixtures or nonsecrets.
- The cache contains 15,341 verified files, including 2,384 additions. Runtime
  asset versions are isolated so upgrades preserve old caches and reuse weights.
- All target/drafter HF revisions and hashes are unchanged.
- Running server, original image/kit and configuration untouched; no GPU test,
  live kernel replacement, systemd/network configuration or serving restart.
- Full fresh-clone two-node GPU boot is still deferred until explicitly allowed.
  CPU checks and old serving success do not substitute for that qualification.

See `release/GHCR.md` and `docs/release-validation.md`. No runtime was uploaded
to Hugging Face, and no model weights were changed.
