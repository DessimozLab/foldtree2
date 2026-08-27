import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def load_module():
    module_path = Path(__file__).resolve().parents[1] / ".." / "scripts" / "optuna_phylo_information_search.py"
    spec = spec_from_file_location("optuna_phylo_information_search", module_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_build_trial_command_includes_training_submat_and_eval_steps(tmp_path):
    module = load_module()
    cfg = module.OptunaSearchConfig(
        study_name="test-study",
        output_dir=str(tmp_path),
        dataset=str(tmp_path / "dataset.h5"),
        structures_dir=str(tmp_path / "structures"),
        benchmark_alignment=str(tmp_path / "alignment.fasta"),
        benchmark_tree=str(tmp_path / "tree.nwk"),
        training_epochs=2,
        batch_size=4,
        num_embeddings=8,
        embedding_dim=16,
        hidden_size=32,
        seed=7,
    )

    command = module.build_trial_command(cfg, trial_number=3)

    assert command[0].endswith("python")
    assert any("learn_lightning.py" in part for part in command)
    assert any("makesubmat.py" in part for part in command)
    assert any("phylogenetic_information_gain.py" in part for part in command)
    assert any("trial_0003" in part for part in command)
