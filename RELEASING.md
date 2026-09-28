# Releasing ForkScope

ForkScope releases are published to PyPI from GitHub Actions using PyPI trusted publishing. No PyPI API token is stored in GitHub.

## One-time setup

1. Confirm the project name `forkscope` is available on PyPI and create the project with its first trusted publisher. For the first release, PyPI's pending publisher flow can be used before the project exists.
2. Configure the trusted publisher with:
   - Owner: `alonso130r`
   - Repository: `forkscope`
   - Workflow: `publish.yml`
   - Environment: `pypi`
3. In the GitHub repository, create the `pypi` deployment environment under **Settings > Environments**. Configure any desired approval rules before publishing.

## Each release

1. Update `version` in `pyproject.toml` and commit the change to `main`.
2. Create and push a tag that matches that version with a `v` prefix, for example `v0.1.0` for version `0.1.0`.
3. Create a GitHub Release from that tag and publish it. The `Publish to PyPI` workflow builds the sdist and wheel, then publishes both to PyPI.
4. Confirm the workflow succeeds and the new version appears on the [ForkScope PyPI project page](https://pypi.org/project/forkscope/).

PyPI does not allow re-uploading an existing version. If a release workflow fails after a version has already been published, fix the issue and release a new patch version.
