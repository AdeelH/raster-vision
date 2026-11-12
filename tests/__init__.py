from collections.abc import Callable
import os

# Never use the developer's real AWS credentials or config in tests (e.g. an
# expired SSO session would make some tests fail). Tests that need AWS use moto.
for _var in ('AWS_PROFILE', 'AWS_DEFAULT_PROFILE', 'AWS_SESSION_TOKEN'):
    os.environ.pop(_var, None)
os.environ['AWS_CONFIG_FILE'] = os.devnull
os.environ['AWS_SHARED_CREDENTIALS_FILE'] = os.devnull
os.environ['AWS_ACCESS_KEY_ID'] = 'testing'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'testing'
os.environ.setdefault('AWS_DEFAULT_REGION', 'us-east-1')


def data_file_path(rel_path: str) -> str:
    data_dir = os.path.join(os.path.dirname(__file__), 'data_files')
    return os.path.join(data_dir, rel_path)


def check_config_upgrader(
    cfg_class: type,
    old_cfg_dict: dict,
    upgrader: Callable[[dict, int], dict],
    curr_version: int,
) -> None:
    """Try to use upgrader to update cfg dict to curr_version."""
    from rastervision.pipeline.config import build_config

    for i in range(curr_version):
        old_cfg_dict = upgrader(old_cfg_dict, i)
    new_cfg = build_config(old_cfg_dict)
    assert isinstance(new_cfg, cfg_class)
