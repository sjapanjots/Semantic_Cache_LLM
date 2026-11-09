# Releasing semcacher

Releases are published to PyPI by GitHub Actions (`.github/workflows/release.yml`)
using [Trusted Publishing](https://docs.pypi.org/trusted-publishers/). No API token is
stored anywhere: PyPI trusts this repository's release workflow directly.

## One-time setup

1. **PyPI.** Sign in at <https://pypi.org> (create an account and enable 2FA if needed).
   Go to *Your account → Publishing → Add a new pending publisher* and enter:

   | Field | Value |
   |---|---|
   | PyPI project name | `semcacher` |
   | Owner | `sjapanjots` |
   | Repository name | `Semantic_Cache_LLM` |
   | Workflow name | `release.yml` |
   | Environment name | `pypi` |

   A *pending* publisher reserves the name for this workflow until the first upload
   creates the project.

2. **TestPyPI (optional, for dry runs).** Do the same at <https://test.pypi.org>
   (it uses a separate account), with environment name `testpypi`.

3. **GitHub environments.** In the repository, open *Settings → Environments* and
   create `pypi` (and `testpypi` if you did step 2). Adding yourself under
   *Required reviewers* on `pypi` makes every upload wait for your approval.

## Dry run on TestPyPI (optional)

*Actions → Release → Run workflow*. It builds, checks and uploads to TestPyPI.
Then try it:

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ semcacher
```

TestPyPI never accepts the same version twice either, so use a dev version
(for example `0.1.0.dev1`) for repeat dry runs.

## Each release

1. In a PR: set `__version__` in `src/semcache/__init__.py` to the new version
   (for example `0.2.0`), and in `CHANGELOG.md` rename `## [Unreleased]` to
   `## [0.2.0] - YYYY-MM-DD` and add a fresh empty `## [Unreleased]` above it.
2. Merge the PR once CI is green.
3. On GitHub: *Releases → Draft a new release*, create the tag `v0.2.0` with
   **target `main`**, paste the changelog section as the notes, and click
   *Publish release*.
4. The *Release* workflow checks that the tag matches `__version__` and that the
   tagged commit is on `main`, builds, runs `twine check`, and uploads to PyPI.
   A minute later:

   ```bash
   pip install --upgrade semcacher
   ```

A version number can be uploaded only once, even if you delete it later, so
always bump the version for a new upload.

**If the release workflow fails its checks**, nothing was uploaded. Fix the cause
(merge the version bump, or pick `main` as the target), then delete the failed
release *and* its tag (*Releases* → the release → *Delete*, then *Tags* → the tag →
*Delete*) and publish the release again. Re-running the failed workflow does not
help: it re-uses the old tag, which still points at the old commit.

## Versioning

[Semantic Versioning](https://semver.org/): while the version is `0.x`, bump the minor
number (`0.2.0`) for new features or breaking changes and the patch number (`0.1.1`)
for fixes only. Day-to-day development happens on `main`; new work goes under
`## [Unreleased]` in the changelog until the next release.
