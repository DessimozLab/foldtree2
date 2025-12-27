# GitHub Pages Setup Instructions

## Enabling GitHub Pages for Documentation

To enable automatic documentation publishing, follow these steps:

1. **Go to your repository settings**:
   - Navigate to `https://github.com/DessimozLab/foldtree2/settings/pages`

2. **Configure GitHub Pages**:
   - Under "Source", select "GitHub Actions"
   - Save the settings

3. **Verify workflow permissions**:
   - Go to `https://github.com/DessimozLab/foldtree2/settings/actions`
   - Under "Workflow permissions", ensure "Read and write permissions" is selected
   - Check "Allow GitHub Actions to create and approve pull requests"

4. **Trigger the workflow**:
   - Push any commit to the `main` branch
   - Or manually trigger the workflow from the Actions tab

5. **Access your documentation**:
   - Once the workflow completes, your docs will be available at:
   - `https://dessimozlab.github.io/foldtree2/`

## Workflow Features

The documentation workflow will:

- ✅ Automatically run on pushes to `main`
- ✅ Generate API documentation from docstrings
- ✅ Support Google/NumPy style docstrings
- ✅ Use the ReadTheDocs theme
- ✅ Include project overview and installation instructions
- ✅ Cache dependencies for faster builds

## Customization

To customize the documentation:

- Edit files in the `docs/` directory
- Modify `docs/conf.py` for Sphinx configuration
- Update `docs/index.rst` for the main page content
- Add new `.rst` files for additional pages