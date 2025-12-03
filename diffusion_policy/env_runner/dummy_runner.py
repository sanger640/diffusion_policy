from diffusion_policy.env_runner.base_image_runner import BaseImageRunner

class DummyRunner(BaseImageRunner):
    def __init__(self, output_dir, **kwargs):
        super().__init__(output_dir)
        # We swallow **kwargs so Hydra can pass whatever it wants 
        # (n_train, n_test, fps, etc.) without crashing.

    def run(self, policy):
        # The training loop expects a dictionary of metrics.
        # We return a placeholder so WandB doesn't complain.
        return {
            'dummy_eval/success_rate': 0.0,
            'dummy_eval/mean_score': 0.0
        }