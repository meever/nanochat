# Copilot Instructions for Git Workflow (Fork + Upstream)

## Branching policy

- Treat `main` as a **read-only mirror branch**.
- Do **all development** on `learn` (or short-lived feature branches created from `learn`).
- Never commit custom code directly on `main`.
- Never merge `learn` into `main`.

## Remote policy

- `origin` = personal fork (push target)
- `upstream` = original repository (sync source)

## Push policy

- Push development work only to `origin/learn`.
- Open PRs from `origin/learn` into upstream default branch.

## Sync policy (strict routine)

When syncing with upstream:

1. `git fetch upstream --prune`
2. `git switch main`
3. `git merge --ff-only upstream/master`
4. `git switch learn`
5. `git rebase main`
6. Resolve conflicts if any, then continue rebase.
7. `git push --force-with-lease origin learn` (only after successful rebase)

## Important repository-specific note

- Upstream default branch is currently `master`, so `main` must mirror `upstream/master`.
- If upstream changes default branch to `main` in the future, update step 3 accordingly.

## Safety rules for assistants

- Before any sync/rebase, check `git status --short --branch`.
- If working tree is dirty, stash changes (`git stash -u`) before switching branches, then restore after sync.
- Prefer `--ff-only` merges for mirror branch updates.
- Never rewrite `main` history.
- Use `--force-with-lease` (not `--force`) when pushing rebased `learn`.

## Local hygiene and defaults

- Keep local branch list minimal to avoid mistakes; remove stale local branches like `master` and `original` when they are not needed.
- Configure local default start branch to `learn` for this repository:
	- `git config --local init.defaultBranch learn`
- Keep local branch tracking explicit:
	- `learn` tracks `origin/learn`
	- `main` tracks `upstream/master` (read-only mirror)
