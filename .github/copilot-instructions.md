# Copilot Instructions for Git Workflow (Fork + Upstream)

## Branching policy

- Treat `master` as a **read-only mirror branch**.
- Do **all development** on `learn` (or short-lived feature branches created from `learn`).
- Never commit custom code directly on `master`.
- Never merge `learn` into `master`.

## Remote policy

- `origin` = personal fork (push target)
- `upstream` = original repository (sync source)

## Push policy

- Push development work only to `origin/learn`.
- Open PRs from `origin/learn` into upstream default branch.

## Sync policy (strict routine)

When syncing with upstream:

1. `git fetch upstream`
2. `git checkout master`
3. `git merge --ff-only upstream/master`
4. `git push origin master`
5. `git checkout learn`
6. `git rebase master`
7. Resolve conflicts if any, then continue rebase.
8. `git push origin learn --force-with-lease`

## Important repository-specific note

- Upstream default branch is currently `master`, so local `master` mirrors `upstream/master`.
- If upstream changes default branch to `main` in the future, update sync commands accordingly.

## Safety rules for assistants

- Before any sync/rebase, check `git status --short --branch`.
- If working tree is dirty, stash changes (`git stash -u`) before switching branches, then restore after sync.
- Prefer `--ff-only` merges for mirror branch updates.
- Never rewrite `master` history.

## Local hygiene and defaults

- Keep local branch list minimal to avoid mistakes; remove stale local branches like `main` and `original` when they are not needed.
- Configure local default start branch to `learn` for this repository:
	- `git config --local init.defaultBranch learn`
- Keep local branch tracking explicit:
	- `learn` tracks `origin/learn`
	- `master` tracks `origin/master` (fork mirror for push)
