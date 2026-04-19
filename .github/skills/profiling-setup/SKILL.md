---
name: profiling-setup
description: 'Profile Rust example binaries using samply to capture CPU flamegraphs. Use when the user wants to find performance bottlenecks, identify hot paths, investigate slow code, or set up profiling for this crate.'
---

# Profiling Setup

Set up and run CPU profiling for example binaries in this crate using `samply`, a sampling profiler that opens results in the Firefox Profiler UI.

## When to Use

- The user wants to profile an example binary from this crate
- Investigating slow performance or looking for hot paths
- Setting up a new machine for profiling Rust code
- The user mentions flamegraphs, CPU profiling, or `samply`

## Procedure

### 1. Install `samply`

```bash
cargo install --locked samply
```

### 2. Ensure the `profiling` cargo profile exists

Check whether `~/.cargo/config.toml` already contains a `[profile.profiling]` section. If it does not, **append** the following block (do not overwrite existing content):

```toml
[profile.profiling]
inherits = "release"
debug = true
```

This keeps release-level optimizations while preserving debug symbols so `samply` can resolve function names in stack traces.

### 3. Compile the example binary

Build the target example with the `profiling` profile. Adjust the example name and features as needed:

```bash
cargo build --profile profiling --example simple
```

> Add `--features serde` only if the example requires the `serde` feature.

### 4. Run & record with `samply`

```bash
samply record target/profiling/examples/simple
```

`samply` will launch the binary, collect samples, and open the Firefox Profiler in your default browser when the process exits.

### 5. Verify

- A browser tab should open showing the Firefox Profiler UI with a flamegraph.
- Confirm that function names are resolved (not just hex addresses). If they are not, the `profiling` profile may be missing `debug = true`.

## Troubleshooting

| Symptom | Likely Cause | Fix |
|---|---|---|
| `samply: command not found` | Not installed or not in `$PATH` | Run `cargo install --locked samply` and ensure `~/.cargo/bin` is in `$PATH` |
| Function names show as hex addresses | Missing debug info | Verify `[profile.profiling]` has `debug = true` in `~/.cargo/config.toml` |
| `samply` fails with permission error (macOS) | SIP or missing entitlements | Run with `sudo` or see the [samply README](https://github.com/mstange/samply/blob/main/README.md) for macOS setup |
| Browser does not open | Headless environment or no default browser set | Copy the localhost URL from terminal output and open it manually |

## Background Information

`samply` is a command-line CPU profiler for macOS and Linux. It collects stack samples and uploads them to the Firefox Profiler for interactive analysis (flamegraphs, call trees, timelines).

- Official repository: https://github.com/mstange/samply