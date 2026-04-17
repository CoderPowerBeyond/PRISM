from __future__ import annotations

from pathlib import Path

import yaml


_KEY_ALIASES = {
    "NUM_EPOCHS": ("epochs",),
    "LR": ("lr",),
    "batch_size": ("batch_size", "batch"),
    "num_layers": ("layers",),
    "select_dataset": ("dataset", "dataset_name"),
    "model_select": ("model",),
}


def _load_yaml_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data if isinstance(data, dict) else {}


def parse_args_with_gat_config(parser, default_config: str = "configs/gat_path.yaml"):
    """Parse args after applying defaults from gat_path.yaml."""
    if not any(action.dest == "config" for action in parser._actions):
        parser.add_argument(
            "--config",
            type=str,
            default=default_config,
            help="Path to YAML config file (default: configs/gat_path.yaml)",
        )

    pre_args, _ = parser.parse_known_args()
    yaml_cfg = _load_yaml_config(pre_args.config)

    valid_dests = {action.dest for action in parser._actions}
    overrides = {}
    for key, value in yaml_cfg.items():
        if key in valid_dests:
            overrides[key] = value
        for alias in _KEY_ALIASES.get(key, ()):
            if alias in valid_dests:
                overrides[alias] = value

    if overrides:
        parser.set_defaults(**overrides)

    return parser.parse_args()
