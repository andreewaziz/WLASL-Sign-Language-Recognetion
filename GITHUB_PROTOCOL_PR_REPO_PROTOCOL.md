# GitHub / Repo / PR Protocol

This file defines required team workflow rules for Smart Interview Simulator.

## 1) Commit Message Style (Required)
Use clear, conventional prefixes in uppercase:
- `FEAT: add interview session creation endpoint`
- `FIX: handle null transcript in analysis pipeline`
- `DOCS: update retention policy section`
- `REFACTOR: split scoring logic into service module`
- `TEST: add API tests for media upload URL`
- `CHORE: update dependencies`

Rules:
1. Keep messages specific and action-oriented.
2. One logical change per commit where possible.
3. Avoid generic messages like "update", "changes", or "final".

## 2) Branch Strategy (Required)
Do not work directly on `main`.

Use branch names like:
- `feat/interview-room-ui`
- `fix/analysis-timeout`
- `docs/readme-architecture`
- `refactor/session-service`

Rules:
1. Create a new branch from latest `main`.
2. Keep branches focused on a single task or feature.
3. Rebase or merge `main` regularly to reduce conflicts.

## 3) Pull Request Rules (Required)
Before opening a PR:
1. Pull latest `main` and resolve conflicts in your branch.
2. Run tests locally.
3. Ensure no secrets or private keys are committed.
4. Update docs if behavior or APIs changed.

PR title format:
- `FEAT: add per-question feedback endpoint`
- `FIX: prevent unauthorized media access`

PR description must include:
1. What changed.
2. Why it changed.
3. How it was tested.
4. Any risks or follow-up work.

## 4) Review and Merge Policy (Required)
1. No direct pushes to `main`.
2. No merge to `main` before:
   - tests pass
   - at least one teammate review/approval
3. Address review comments before merge.
4. Prefer squash merge for clean history unless multi-commit history is needed.

## 5) Testing Gate (Required)
At minimum before merge:
1. Backend tests pass.
2. Frontend tests/build pass.
3. Critical interview flow manually verified:
   - create session
   - submit answer
   - run analysis
   - view report

## 6) Conflict and Stability Best Practices
1. Pull/rebase often to reduce merge conflicts.
2. Keep PRs small and focused.
3. Do not mix unrelated refactors with feature work.
4. If a breaking API change is introduced, document it in README and PR.

## 7) Security and Data Safety
1. Never commit `.env` files or credentials.
2. Use environment variables and secret managers.
3. Validate media and input payloads before processing.
4. Treat user recordings/transcripts as sensitive data.

## 8) Definition of Done (DoD)
A task is done only when:
1. Code is implemented.
2. Tests are added/updated and passing.
3. Documentation is updated.
4. PR is reviewed and approved.
5. Branch is merged according to policy.
