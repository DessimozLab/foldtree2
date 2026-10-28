"""Summarize a completed preliminary contact objective screen."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def report(root):
    root = Path(root)
    summary = json.loads((root / 'summary.json').read_text())
    provenance = json.loads((root / 'provenance.json').read_text())
    args = provenance['args']
    lines = [
        '# Preliminary contact objective screen', '',
        f"Fixed final epoch {args['epochs']}; seed {args['seed']}; "
        f"{args['train_count']} training / {args['val_count']} development-validation chains.", '',
        f"All variants share a hidden-{args['hidden']}, two-layer, eight-head "
        'transformer with dropout 0.1, AdamW at 0.001, and weight decay 0.05. '
        'The production encoder/decoder remain frozen. '
        'Distance cross-entropy uses uniformly sampled pairs. Auxiliary contact supervision '
        'uses equally weighted contact and noncontact means on extra stratified samples. '
        'The spherical head sets contact mass within the distance distribution; '
        'the pair MLP predicts conditional distance bins.', '',
        '| Variant | Contact AP | Top-L precision | Precision @0.5 | Recall @0.5 | Predicted contacts | Distance MAE (A) | Distance NLL |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    def percent(value):
        return 'undefined' if value is None else f'{100*value:.2f}%'
    for name, row in summary.items():
        v = row['validation']
        lines.append(f"| {name} | {v['contact_average_precision']:.4f} | "
                     f"{percent(v['top_L_precision'])} | {percent(v['contact_precision'])} | "
                     f"{percent(v['contact_recall'])} | {v['contacts_predicted']} | "
                     f"{v['capped_distance_mae_angstrom']:.3f} | {v['nll']:.4f} |")
    lines += ['', 'Contacts mean CA distance <8 A, with residue row separation >=12. '
              'Top-L precision uses a pooled ranking budget equal to the sum of valid chain lengths. '
              'Validation includes every eligible pair at its natural prevalence. '
              'The 0.5 threshold applies to the raw scores; balanced objectives alter calibration.', '',
              'This is a small, single-seed development screen, not a full-data result or an '
              'independent test. Production pretraining independence of the holdout is unverified. '
              'Checkpoint reloads reproduce the saved final validation metrics exactly.', '',
              '![Validation learning curves](learning_curves.png)', '']
    (root / 'report.md').write_text('\n'.join(lines))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    for name in summary:
        history = json.loads((root / name / 'history.json').read_text())
        for ax, key in zip(axes, ['contact_average_precision', 'contact_recall', 'capped_distance_mae_angstrom']):
            ax.plot([r['epoch'] for r in history], [r['validation'][key] for r in history], label=name)
    for ax, title in zip(axes, ['Contact average precision', 'Contact recall at 0.5', 'Capped distance MAE (A)']):
        ax.set(xlabel='Epoch', title=title)
        ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(root / 'learning_curves.png', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    report(parser.parse_args().root)
