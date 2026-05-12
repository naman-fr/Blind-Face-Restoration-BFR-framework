import os
import sys
import cv2
import torch
import numpy as np
import gradio as gr
import wandb
import sentry_sdk
from pathlib import Path
from loguru import logger

# Initialize Sentry for Shadow Trace
sentry_sdk.init(dsn=os.getenv("SENTRY_DSN", ""))

# Add src to path for package discovery
sys.path.append(str(Path(__file__).parent / "src"))

from bfr_framework.sampler import DifFaceSampler
from bfr_framework.degradation_estimator import DegradationEstimator
from bfr_framework.ensemble_selector import EnsembleSelector
from bfr_framework.ensemble import ensemble_restore, weighted_ensemble_restore, best_of_n_restore
from bfr_framework.utils import util_image
import tempfile

# --- NeoForge Aesthetic Matrix ---
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=JetBrains+Mono:wght@400;700&display=swap');

body, .gradio-container {
    background-color: #0a0a0a !important;
    color: #00ff88 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

#header {
    text-align: center;
    padding: 3rem 0;
    background: linear-gradient(180deg, #000000 0%, #0a0a0a 100%);
    border-bottom: 2px solid #ff00ff;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}

#header h1 {
    font-family: 'Orbitron', sans-serif;
    font-size: 3.5rem;
    text-transform: uppercase;
    letter-spacing: 5px;
    color: #00ff88;
    text-shadow: 0 0 10px #00ff88, 0 0 20px #00ff88;
    animation: glitch 1s infinite;
}

@keyframes glitch {
    0% { transform: translate(0); }
    20% { transform: translate(-2px, 2px); }
    40% { transform: translate(-2px, -2px); }
    60% { transform: translate(2px, 2px); }
    80% { transform: translate(2px, -2px); }
    100% { transform: translate(0); }
}

.premium-card {
    border: 1px solid #ff00ff !important;
    border-radius: 0px !important;
    background: rgba(255, 0, 255, 0.05) !important;
    box-shadow: 0 0 15px rgba(255, 0, 255, 0.2);
}

.gr-button-primary {
    background: linear-gradient(90deg, #ff00ff 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    font-family: 'Orbitron', sans-serif !important;
    font-weight: bold !important;
    text-transform: uppercase !important;
    letter-spacing: 2px !important;
}

.gr-button-primary:hover {
    box-shadow: 0 0 20px #ff00ff !important;
    transform: scale(1.02);
}

footer { visibility: hidden; }
"""

class NeoForgeBFR:
    def __init__(self):
        self.estimator = DegradationEstimator()
        self.selector = EnsembleSelector()
        self.samplers = {}
        # Shadow Init W&B
        if os.getenv("WANDB_API_KEY"):
            wandb.init(project="bfr-singularity-2027")

    def get_sampler(self, task: str, aligned: bool):
        key = f"{task}_{aligned}"
        if key not in self.samplers:
            from omegaconf import OmegaConf
            cfg_path = f"configs/sample/{'iddpm_ffhq512_swinir.yaml' if task=='restoration' else 'difface_inpainting_lama256.yaml'}"
            configs = OmegaConf.load(cfg_path)
            configs.aligned = aligned
            # Use FP16 only if CUDA is available
            use_fp16 = torch.cuda.is_available()
            self.samplers[key] = DifFaceSampler(configs, use_fp16=use_fp16)
        return self.samplers[key]

    async def nexus_predict(self, image, task, aligned, eta, use_adaptive, use_ensemble, seeds):
        try:
            if image is None: return None, "⚠️ ACCESS DENIED: Image Missing."
            
            # 1. Initialize Sampler
            sampler = self.get_sampler(task, aligned)
            
            # 2. Neural Analysis & Adaptive N
            n_step, severity, info = self.estimator.select_n_adaptive(image, n_min=300, n_max=700)
            if not use_adaptive:
                n_step = 150 # Stronger restoration
            
            # W&B Logging
            if wandb.run:
                wandb.log({"severity": severity, "n_step": n_step})

            # 3. Preparation: Save Gradio image to temp file
            with tempfile.TemporaryDirectory() as tmpdir:
                in_path = Path(tmpdir) / "input.png"
                out_dir = Path(tmpdir) / "output"
                out_dir.mkdir()
                cv2.imwrite(str(in_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                # 4. Perform Restoration
                if use_ensemble:
                    logger.info(f"Initiating {seeds}-seed ensemble restoration (N={n_step})...")
                    restored_bgr, individual_results, best_idx = best_of_n_restore(
                        sampler=sampler,
                        in_path=str(in_path),
                        out_dir=str(out_dir),
                        num_seeds=int(seeds),
                        start_timesteps=n_step,
                        task=task,
                        eta=1.0, # Maximize stochastic diversity for ensemble
                        aligned=aligned
                    )
                else:
                    logger.info(f"Initiating single-pass restoration (N={n_step})...")
                    sampler.inference(
                        in_path=str(in_path),
                        out_path=str(out_dir),
                        bs=1,
                        start_timesteps=n_step,
                        task=task,
                        need_restoration=True,
                        eta=eta
                    )
                    # Robust search for the restored image
                    res_files = list(out_dir.rglob("input.png"))
                    if not res_files:
                        res_files = list(out_dir.rglob("*.png"))
                    
                    if not res_files:
                        raise FileNotFoundError("Reconstruction engine failed to materialize the output matrix.")
                    
                    restored_bgr = cv2.imread(str(res_files[0]))

            # 5. Post-process
            restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
            
            status = f"⚡ NEXUS STATUS: Restoration Complete | Severity {severity:.3f} | Optimal N: {n_step}\n"
            status += f"🔮 ENSEMBLE: Active ({seeds} seeds)" if use_ensemble else "🔮 ENSEMBLE: Single Pass"
            
            return restored_rgb, status
        except Exception as e:
            logger.error(f"Nexus Error: {str(e)}")
            sentry_sdk.capture_exception(e)
            return None, f"❌ CRITICAL FAILURE: {str(e)}"

# Initialize Neural Core
core = NeoForgeBFR()

with gr.Blocks(theme=gr.themes.Base(), css=CSS) as singularity:
    with gr.Column(elem_id="header"):
        gr.Markdown("# 👾 BFR NEXUS v2.0")
        gr.Markdown("Initializing 2027 Neural Architecture... [NeoForge Rogue AI Phase]")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group(elem_classes="premium-card"):
                gr.Markdown("### 📡 NEURAL INPUT")
                input_img = gr.Image(label="Source Matrix", type="numpy")
                with gr.Row():
                    task = gr.Radio(["restoration", "inpainting"], label="Mode", value="restoration")
                    aligned = gr.Checkbox(label="Aligned", value=True)
                eta = gr.Slider(0, 1, value=0.5, label="Fidelity Leak (Eta)")
            
            with gr.Accordion("Quantum Overrides", open=False):
                use_adaptive = gr.Checkbox(label="Dynamic Timestep (Delta 1)", value=True)
                use_ensemble = gr.Checkbox(label="Best-of-N Ensemble (Delta 2)", value=True)
                num_seeds = gr.Slider(2, 8, value=4, step=1, label="Seeds")
            
            submit = gr.Button("RESTORE REALITY ✨", variant="primary")

        with gr.Column(scale=1):
            output_img = gr.Image(label="Neural Reconstruction")
            info = gr.Markdown("Waiting for uplink...")

    submit.click(
        core.nexus_predict,
        inputs=[input_img, task, aligned, eta, use_adaptive, use_ensemble, num_seeds],
        outputs=[output_img, info]
    )

if __name__ == "__main__":
    singularity.queue().launch()
