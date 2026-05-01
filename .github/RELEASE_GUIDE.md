# Creating a FoldTree2 Release

Quick reference for creating releases with automated conda package builds.

## Quick Release Workflow

```bash
# 1. Update version in pyproject.toml
vim pyproject.toml  # Change version = "0.1.0" to "0.2.0"

# 2. Update conda recipe version
vim conda-recipe/meta.yaml  # Update version under {% set version = "..." %}

# 3. Commit version changes
git add pyproject.toml conda-recipe/meta.yaml
git commit -m "Bump version to 0.2.0"
git push

# 4. Create and push tag
git tag -a v0.2.0 -m "Release v0.2.0

Major features:
- Feature 1
- Feature 2
- Bug fixes
"
git push origin v0.2.0

# 5. GitHub Actions will automatically:
#    ✅ Build conda package
#    ✅ Run verification checks
#    ✅ Create GitHub release
#    ✅ Upload package as release asset
#    ✅ (Optional) Upload to Anaconda Cloud

# 6. Monitor progress
# Go to: https://github.com/DessimozLab/foldtree2/actions
```

## Version Numbering

Follow [Semantic Versioning](https://semver.org/):
- **Major** (1.0.0): Breaking changes
- **Minor** (0.1.0): New features, backwards compatible
- **Patch** (0.0.1): Bug fixes, backwards compatible

Examples:
- `v0.1.0` - Initial release
- `v0.1.1` - Bug fix
- `v0.2.0` - New feature added
- `v1.0.0` - First stable release

## Pre-release Tags

For beta/alpha releases:
```bash
git tag -a v0.2.0-beta.1 -m "Beta release for testing"
git push origin v0.2.0-beta.1
```

## Checklist Before Release

- [ ] All tests passing locally
- [ ] Version updated in `pyproject.toml`
- [ ] Version updated in `conda-recipe/meta.yaml`
- [ ] CHANGELOG.md updated with release notes
- [ ] Documentation updated
- [ ] README.md reflects new features
- [ ] All commits pushed to main branch
- [ ] GitHub Actions workflows passing

## After Release

1. **Verify package:**
   - Check GitHub Actions completed successfully
   - Download `.conda` file from release assets
   - Test installation in clean environment

2. **Test installation:**
   ```bash
   conda create -n test_v0.2.0 python=3.10
   conda activate test_v0.2.0
   conda install ./foldtree2-0.2.0-py_0.conda -c conda-forge -c pytorch -c bioconda
   foldtree2 --about
   ```

3. **Announce:**
   - GitHub Discussions
   - Project documentation
   - Update README with new installation instructions

## Troubleshooting

### Tag already exists
```bash
# Delete local tag
git tag -d v0.2.0

# Delete remote tag
git push --delete origin v0.2.0

# Create new tag
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
```

### Workflow failed
- Check Actions tab for error logs
- Common issues:
  - Version mismatch in meta.yaml
  - Unwanted files in package (check .conda_build_ignore)
  - Conda build errors (check dependencies)

### Need to re-run workflow
- Go to Actions tab
- Select failed workflow run
- Click "Re-run all jobs"

## Advanced: Manual Package Build

If you need to build locally before releasing:

```bash
# Clean build
rm -rf /tmp/cb
conda build conda-recipe --croot /tmp/cb --prefix-length 80 --no-test

# Verify package
PACKAGE=/tmp/cb/noarch/foldtree2-*.conda
du -h $PACKAGE  # Should be ~24MB
tar -tzf $PACKAGE | grep -E '\.h5$|\.ipynb$'  # Should be empty

# Test locally
conda create -n local_test python=3.10
conda install -n local_test $PACKAGE -c conda-forge -c pytorch -c bioconda
conda run -n local_test foldtree2 --about
```

## Release Notes Template

```markdown
## What's New in v0.2.0

### Features
- Added support for X
- Improved Y performance by Z%
- New command: `foldtree2 command`

### Bug Fixes
- Fixed issue with A
- Corrected B behavior

### Documentation
- Updated installation guide
- Added tutorial for X

### Breaking Changes
- Changed API for function Y (see migration guide)

### Installation

**Conda (recommended):**
\`\`\`bash
# Download from release assets
conda install ./foldtree2-0.2.0-py_0.conda -c conda-forge -c pytorch -c bioconda
\`\`\`

**Full dependencies:**
\`\`\`bash
conda create -n foldtree2 python=3.10
conda activate foldtree2
conda install pytorch pytorch-geometric -c pytorch -c pyg
conda install biopython pandas numpy -c conda-forge
conda install ./foldtree2-0.2.0-py_0.conda
\`\`\`
```
