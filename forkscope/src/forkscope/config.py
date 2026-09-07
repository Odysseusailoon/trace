from pathlib import Path

import yaml
from pydantic import BaseModel


class ServerCfg(BaseModel):
    port: int = 30000
    mem_fraction_static: float = 0.85
    host: str = "127.0.0.1"


class FpaCfg(BaseModel):
    branch_prob_threshold: float = 0.05
    samples_per_branch: int = 20
    samples_t0: int = 100
    spacing: int = 4
    spacing_mode: str = "token"  # token | sentence | tool_call
    max_continuation_tokens: int = 512
    top_logprobs_num: int = 10


class ReferenceCfg(BaseModel):
    samples_per_branch: int = 200
    spacing: int = 1


class BranchStatCfg(BaseModel):
    """Between-branch dispersion (src/forkscope/branchstat.py)."""
    alpha: float = 0.05
    persist_max: float = 0.9
    sims: int = 20000
    screen_sims: int = 2000


class SmoothingCfg(BaseModel):
    pelt_cost: str = "l2"
    penalty_grid: list[float] = [1, 4, 16, 64, 256]
    bandwidth_grid: list[float] = [2, 4, 8, 16]
    cv_folds: int = 5


class Settings(BaseModel):
    model: str = "Qwen/Qwen3-8B"
    server: ServerCfg = ServerCfg()
    fpa: FpaCfg = FpaCfg()
    reference: ReferenceCfg = ReferenceCfg()
    branchstat: BranchStatCfg = BranchStatCfg()
    smoothing: SmoothingCfg = SmoothingCfg()
    fork_threshold: float = 0.10
    concurrency: int = 32
    data_dir: Path = Path("data")

    @property
    def base_url(self) -> str:
        return f"http://{self.server.host}:{self.server.port}"


def load_settings(path: str | Path) -> Settings:
    with open(path) as f:
        return Settings(**yaml.safe_load(f))
