import os
import numpy as np
import jax
import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils import make_env, evaluate_env, final_evaluation
from src.WCSAC_IQN import WCSAC_IQN

# https://jax.readthedocs.io/en/latest/gpu_performance_tips.html
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true "
    "--xla_gpu_triton_gemm_any=True "
    "--xla_gpu_enable_async_collectives=true "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_highest_priority_async_stream=true "
)


@hydra.main(version_base=None, config_path="cfg", config_name="wcsac.yaml")
def main(cfg: DictConfig) -> None:
    if jax.default_backend() == "cpu":
        raise RuntimeError(
            "Not able to run on GPU. Aborting as CPU would be used instead..."
        )

    if "seed" not in cfg:
        OmegaConf.update(cfg, "seed", np.random.randint(2**32 - 1), force_add=True)

    env = make_env(cfg)
    eval_env = make_env(cfg, eval=True)
    algo = WCSAC_IQN(cfg, env, eval_env)
    algo.learn(evaluate_env, final_evaluation)


if __name__ == "__main__":
    main()
