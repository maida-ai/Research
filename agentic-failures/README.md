# Fable paper-writer — setup

A Claude Code configuration where **Fable** plans, delegates, and evaluates,
while cheaper subagents do the token-heavy work.

## What's in here

```
.claude/
  settings.json          # Fable as default model + auto-approved permissions
  settings.sandbox.json  # hardened drop-in for running experiment code (mac/Linux/WSL2)
  agents/
    code-writer.md       # Sonnet  — designs/runs experiments, dumps results (code + bash + web)
    researcher.md        # Sonnet  — finds & synthesizes sources (read-only + web)
    section-writer.md    # Sonnet  — drafts/revises sections (write)
    citation-checker.md  # Haiku   — verifies claims & citations (read-only)
    copy-editor.md       # Haiku   — line edits for clarity/style
CLAUDE.md                # tells Fable to orchestrate & evaluate, not produce
```

## Install (one time)

1. Have Claude Code installed and updated:
   ```
   npm install -g @anthropic-ai/claude-code
   claude --version        # need a recent version (subagent `fable`/`effort` support)
   ```
2. Put this folder's contents at the **root of your paper project** — i.e. copy
   the `.claude/` directory and `CLAUDE.md` into your project so the paths are
   `your-project/.claude/...` and `your-project/CLAUDE.md`.
3. From the project root, start it:
   ```
   cd your-project
   claude
   ```
   The startup header should show the model as Fable. If not, run `/model` and
   pick Fable (or `/model claude-fable-5`). Run `/agents` to confirm the four
   subagents are listed.

## How it works

- **Fable is the main session** (it must be — subagents can't spawn subagents,
  so the orchestrator lives at the top level). It plans, hands tasks to
  subagents, and judges every result before accepting it. See `CLAUDE.md`.
- **Subagents carry their own model + effort** in their frontmatter, so heavy
  reading/searching/drafting runs on Sonnet/Haiku at low/medium effort while
  Fable only sees their summaries. That's where the token savings come from.
- A subagent's model can't exceed the main model's cost tier. Because Fable is
  top-tier, all cheaper models are available — but if you accidentally run the
  session on Sonnet, the subagents get capped to Sonnet. Keep the session on
  Fable (that's why it's pinned in `settings.json`).

## Permissions (auto-approved)

`settings.json` sets `defaultMode: acceptEdits`, so file edits and common
filesystem commands run without prompting. Subagents **inherit** this mode and
can't downgrade it. Web search/fetch, git/build, and experiment commands
(`python`, `pip`, `uv`, `pytest`, ...) are added to the `allow` list explicitly,
because `acceptEdits` alone would still prompt on those (and background subagents
auto-deny anything that would prompt). The `deny` list blocks reads of secret
files (`.env`, credentials, `~/.ssh`, `~/.aws`) and the few genuinely
destructive commands (`sudo`, `git push`). Note: `curl`/`wget` are *not* denied,
since denying them while allowing `python` buys nothing — see the isolation
section for the boundary that actually holds.

Permission tiers, from safest to most permissive — change `defaultMode` if you
want a different level:
- `acceptEdits` (current) — auto file edits, web/git allow-listed. Recommended.
- `dontAsk` + explicit `allow` rules — hands-off but anything not allow-listed
  is auto-denied rather than run.
- `bypassPermissions` / `claude --dangerously-skip-permissions` — skips all
  prompts. Only in a disposable/sandboxed environment; it can run destructive
  commands without asking.

## Running experiments (code-writer)

The `code-writer` agent can write code, run Python, install packages, launch
experiment processes, and reach the internet — then dump structured results to
`results/` for the writing agents. Internet already works: Claude Code's sandbox
is **off by default**, so executed code has normal network access with no extra
config. The expanded `allow` list in `settings.json` adds `python`, `pip`, `uv`,
`pytest`, `jupyter`, and `git clone` so experiments run without prompts (and so
backgrounded runs don't auto-deny).

Two limits worth knowing:

- **A subagent can't start other subagents.** `code-writer` can launch its own
  processes (including agent experiments) via Bash, but it can't spawn Claude
  Code agents. When you want experiments run in parallel, **Fable** launches
  several `code-writer` instances — that orchestration stays at the top level.
- **Allow-lists don't actually contain code execution.** Once an agent can run
  arbitrary `python`, it can read your keys, hit any URL, or delete files
  regardless of what `curl`/`wget` rules say. Command rules are a speed bump,
  not a boundary. So isolate instead of relying on them (next section).

## Isolating the experiment runner (recommended)

Pick the boundary that fits where you're running:

1. **Claude Code sandbox** (macOS / Linux / WSL2). Use the hardened drop-in:
   ```
   cp .claude/settings.json .claude/settings.nosandbox.json   # keep a copy
   cp .claude/settings.sandbox.json .claude/settings.json
   ```
   It enables OS-level isolation (`sandbox.enabled`), auto-approves bash *inside*
   the sandbox (`autoAllowBashIfSandboxed`), blocks escaping it
   (`allowUnsandboxedCommands: false`), restricts the network to an allowlist
   (PyPI, GitHub, HuggingFace — edit as needed), and denies reads of
   `~/.ssh` / `~/.aws`. Requires `bubblewrap` on Linux (`apt install bubblewrap`);
   uses Seatbelt on macOS. Not supported on native Windows or WSL1.
2. **Container / VM / cloud box.** Run the whole project inside a disposable
   container or remote machine, mount only this project, and let execution be
   fully automatic there. This is the safest option if experiments are heavy or
   you want zero prompts. Inside a throwaway container you can also use
   `claude --dangerously-skip-permissions` safely.

If you're on plain `settings.json` without the sandbox, treat the machine you
run on as one the agent effectively has full user-level access to.

Just talk to Fable in plain language; it delegates automatically:

```
Outline a literature review on <topic>, then draft the introduction.
```

Or steer explicitly:

```
Use the researcher subagent to gather sources on <X>, then have section-writer
draft section 2 from those notes, then run citation-checker on it.
```

Tune cost by editing the `model:` and `effort:` lines in each agent file. If you
add or edit an agent file by hand, restart the session (or use `/agents`, which
takes effect immediately).
