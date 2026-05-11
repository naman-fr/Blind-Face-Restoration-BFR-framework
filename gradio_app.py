import os
import cv2
import torch
import random
import numpy as np
import tempfile
from pathlib import Path
from omegaconf import OmegaConf
from basicsr.utils.download_util import load_file_from_url
from sampler import DifFaceSampler
from utils import util_image
from adaptive_n import estimate_degradation_severity, select_N_adaptive
from ensemble import (ensemble_restore, weighted_ensemble_restore,
                      best_of_n_restore, compute_sharpness)

# Pre-load samplers for fast inference
samplers = {}

def get_configs(task, aligned):
    if task == 'restoration':
        cfg_path = 'configs/sample/iddpm_ffhq512_swinir.yaml'
    elif task == 'inpainting':
        cfg_path = 'configs/sample/difface_inpainting_lama256.yaml'
    
    configs = OmegaConf.load(cfg_path)
    configs.seed = 12345
    configs.diffusion.params.timestep_respacing = 'ddim250'
    configs.aligned = aligned
    return configs

def download_models(task, configs):
    if task == 'restoration':
        if not Path(configs.model_ir.ckpt_path).exists():
            load_file_from_url(
                url="https://github.com/zsyOAOA/DifFace/releases/download/V1.0/swinir_restoration512_L1.pth",
                model_dir=str(Path(configs.model_ir.ckpt_path).parent),
                progress=True,
                file_name=Path(configs.model_ir.ckpt_path).name,
            )
        if not Path(configs.model.ckpt_path).exists():
            load_file_from_url(
                url="https://github.com/zsyOAOA/DifFace/releases/download/V1.0/iddpm_ffhq512_ema500000.pth",
                model_dir=str(Path(configs.model.ckpt_path).parent),
                progress=True,
                file_name=Path(configs.model.ckpt_path).name,
            )
    elif task == 'inpainting':
        if not Path(configs.model_ir.ckpt_path).exists():
            load_file_from_url(
                url="https://github.com/zsyOAOA/DifFace/releases/download/V1.0/lama_inpainting256.pth",
                model_dir=str(Path(configs.model_ir.ckpt_path).parent),
                progress=True,
                file_name=Path(configs.model_ir.ckpt_path).name,
            )
        if not Path(configs.model.ckpt_path).exists():
            load_file_from_url(
                url="https://github.com/zsyOAOA/DifFace/releases/download/V1.0/iddpm_ffhq256_ema750000.pth",
                model_dir=str(Path(configs.model.ckpt_path).parent),
                progress=True,
                file_name=Path(configs.model.ckpt_path).name,
            )

def load_sampler(task, aligned, use_fp16=True):
    key = f"{task}_{aligned}"
    if key in samplers:
        return samplers[key]
    
    print(f"Loading models for {task} (aligned={aligned})...")
    configs = get_configs(task, aligned)
    download_models(task, configs)
    
    sampler = DifFaceSampler(
        configs,
        im_size=configs.model.params.image_size,
        use_fp16=use_fp16,
    )
    samplers[key] = sampler
    return sampler

import gradio as gr

def _run_inference(sampler, in_dir, out_dir, task, start_timesteps, eta):
    """Run DifFace inference with given start_timesteps."""
    _GAMMA = {'restoration': 0.0, 'inpainting': 0.5}
    sampler.inference(
        in_path=in_dir,
        out_path=out_dir,
        bs=1,
        start_timesteps=start_timesteps,
        task=task,
        need_restoration=True,
        gamma=_GAMMA[task],
        num_update=1,
        draw_box=False,
        suffix=None,
        eta=eta if task == 'restoration' else 1.0,
        mask_back=True,
    )


def _read_result(out_dir, aligned):
    """Read the result image from the output directory."""
    if aligned:
        res_dir = os.path.join(out_dir, "restored_faces")
    else:
        res_dir = os.path.join(out_dir, "restored_image")

    res_files = list(Path(res_dir).glob("*.png"))
    if not res_files:
        return None

    out_img = cv2.imread(str(res_files[0]))
    return cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)


def process_image(image, task, aligned, eta,
                  use_adaptive_n, n_min, n_max,
                  use_ensemble, num_seeds, ensemble_mode):
    if image is None:
        return None, None, ""
    
    # Inpainting only supports aligned currently in DifFace
    if task == 'inpainting':
        aligned = True
        
    sampler = load_sampler(task, aligned, use_fp16=False)
    
    _START_TIMESTEPS = {'restoration': 100, 'inpainting': 120}
    _GAMMA = {'restoration': 0.0, 'inpainting': 0.5}
    fixed_N = _START_TIMESTEPS[task]

    status_lines = []

    # ---- Step 1: Determine N ----
    if use_adaptive_n:
        img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        adaptive_N, severity, info = select_N_adaptive(img_bgr, N_min=n_min, N_max=n_max)
        start_N = adaptive_N
        status_lines.append("**Adaptive N Analysis:**")
        status_lines.append(f"- Laplacian var: {info['laplacian_var']:.1f}")
        status_lines.append(f"- Blur: {info['blur_score']:.3f} | Noise: {info['noise_score']:.3f}")
        status_lines.append(f"- Severity: **{severity:.3f}** -> N={adaptive_N} (fixed would be {fixed_N})")
    else:
        start_N = fixed_N
        status_lines.append(f"Using fixed N={fixed_N}")

    # ---- Step 2: Run inference ----
    if use_ensemble:
        num_seeds = int(num_seeds)
        status_lines.append(f"\n**Ensemble ({ensemble_mode}, {num_seeds} seeds):**")

        with tempfile.TemporaryDirectory() as in_dir:
            in_path = os.path.join(in_dir, "input.png")
            cv2.imwrite(in_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            seeds = list(range(num_seeds))
            common_kwargs = dict(
                sampler=sampler,
                in_path=in_path,
                out_dir=in_dir,
                num_seeds=num_seeds,
                seeds=seeds,
                start_timesteps=start_N,
                task=task,
                eta=eta,
                gamma=_GAMMA[task],
                bs=1,
                draw_box=False,
                aligned=aligned,
            )

            weights = None
            best_idx = None

            if ensemble_mode == 'best':
                ensemble_result_bgr, individual_results, best_idx = best_of_n_restore(**common_kwargs)
                best_seed, _, best_sharp = individual_results[best_idx]
                status_lines.append(f"- **Winner: Seed {best_seed}** (sharpness={best_sharp:.0f})")
                for i, (seed, _, sharpness) in enumerate(individual_results):
                    marker = " << BEST" if i == best_idx else ""
                    status_lines.append(f"- Seed {seed}: sharpness={sharpness:.0f}{marker}")
            elif ensemble_mode == 'weighted':
                ensemble_result_bgr, individual_results, weights = weighted_ensemble_restore(**common_kwargs)
                for (seed, _, sharpness), w in zip(individual_results, weights):
                    status_lines.append(f"- Seed {seed}: sharp={sharpness:.0f}, weight={w:.3f}")
            else:  # mean
                ensemble_result_bgr, individual_results = ensemble_restore(**common_kwargs)
                for seed, _, sharpness in individual_results:
                    status_lines.append(f"- Seed {seed}: sharpness={sharpness:.0f}")

            # Ensemble sharpness
            ens_sharp = compute_sharpness(ensemble_result_bgr)
            avg_sharp = np.mean([s for _, _, s in individual_results])
            status_lines.append(f"- **Ensemble sharpness: {ens_sharp:.0f}** (avg individual: {avg_sharp:.0f})")

            result_img = cv2.cvtColor(ensemble_result_bgr, cv2.COLOR_BGR2RGB)

    else:
        # ---- Single-seed inference ----
        with tempfile.TemporaryDirectory() as in_dir:
            with tempfile.TemporaryDirectory() as out_dir:
                in_path = os.path.join(in_dir, "input.png")
                cv2.imwrite(in_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

                _run_inference(sampler, in_dir, out_dir, task, start_N, eta)
                result_img = _read_result(out_dir, aligned)

    status_text = "\n".join(status_lines)
    return result_img, status_text

# Setup Gradio Interface
css = """
.container { max-width: 1200px; margin: auto; }
#header { text-align: center; margin-bottom: 2rem; }
#header h1 { font-size: 2.5rem; color: #fdfdfd; font-weight: 800; margin-bottom: 0.5rem; }
#header p { font-size: 1.1rem; color: #a0aec0; }
.gr-button-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important; border: none !important; }
.gr-button-primary:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(118, 75, 162, 0.4); }
.status-box { background: #1a202c; padding: 1rem; border-radius: 0.5rem; border: 1px solid #2d3748; }
"""

with gr.Blocks(title="AI Face Restoration", theme=gr.themes.Soft(primary_hue="purple", secondary_hue="slate")) as app:
    with gr.Div(elem_id="header"):
        gr.Markdown("# ✨ AI Blind Face Restoration (BFR)")
        gr.Markdown("Restore low-quality, blurry, or noisy faces using Diffusion Models with Adaptive Timestep Selection and Best-of-N Ensemble.")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 📸 Input & Basic Settings")
                input_img = gr.Image(label="Upload Face", type="numpy")
                task = gr.Radio(["restoration", "inpainting"], label="Mode", value="restoration")
                aligned = gr.Checkbox(label="Aligned Face (Cropped)", value=True)
                eta = gr.Slider(0.0, 1.0, value=0.5, step=0.1, label="Fidelity vs Realness (Eta)")

            with gr.Accordion("🚀 Advanced Enhancements (Delta 1 & 2)", open=True):
                with gr.Group():
                    gr.Markdown("#### Delta 1: Adaptive N Selection")
                    use_adaptive_n = gr.Checkbox(label="Enable Dynamic N (Auto-detect degradation)", value=True)
                    with gr.Row():
                        n_min = gr.Slider(100, 500, value=250, step=10, label="Min Timestep")
                        n_max = gr.Slider(200, 800, value=500, step=10, label="Max Timestep")

                with gr.Group():
                    gr.Markdown("#### Delta 2: Best-of-N Ensemble")
                    use_ensemble = gr.Checkbox(label="Enable Ensemble (Multi-seed selection)", value=True)
                    with gr.Row():
                        num_seeds = gr.Slider(2, 10, value=5, step=1, label="Num Seeds")
                        ensemble_mode = gr.Radio(["best", "mean", "weighted"], label="Mode", value="best")

            submit_btn = gr.Button("🚀 Restore Image", variant="primary")
            
        with gr.Column(scale=1):
            output_img = gr.Image(label="Restored Result")
            with gr.Div(elem_classes="status-box"):
                gr.Markdown("### 📊 Diagnostics & Analysis")
                status_text = gr.Markdown("Ready for processing...")
            
    submit_btn.click(
        fn=process_image,
        inputs=[input_img, task, aligned, eta,
                use_adaptive_n, n_min, n_max,
                use_ensemble, num_seeds, ensemble_mode],
        outputs=[output_img, status_text]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, css=css)
