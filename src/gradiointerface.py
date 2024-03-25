import gradio as gr
from pathlib import Path
from main import *

def sampler_builder(timesteps, stop, stop_after_timesteps, n_sample, noise_type, sampler_type, model_filename, embeddings, gpu):
    context = [0.0, 0.0, 0.0, 0.0, 0.0]
    for index in embeddings:
        context[index] = 1.0
    if not stop:
        stop_after_timesteps = None
    device, gpu_perf = initialize()

    output_filename = './data/grid.png'
    gpu_perf_filename = './data/gpu_perf.png'

    sample(device, gpu_perf, timesteps, noise_type, n_sample, sampler_type, model_filename, context, grid_filename = output_filename, stop_after_timesteps=stop_after_timesteps)
    gpu_perf.save_plot(gpu_perf_filename)
    return f"timesteps: {timesteps}, stop_before: {stop}, stop_after_timesteps: {stop_after_timesteps}, batch_size: {n_sample}, noise_type: {noise_type}, sampler_type: {sampler_type}, embeddings: {context}, gpu: {gpu}", \
        output_filename, \
        gpu_perf.info, \
        gpu_perf_filename

models = list(Path('./weights').glob('*.pth'))
demo = gr.Interface(
    sampler_builder,
    [
        gr.Slider(10, 1000, value=500, step=10, label="Timesteps", info=""),
        gr.Checkbox(label="Stop before", value=False, info="Stop after a number of timesteps"),
        gr.Slider(10, 1000, value=500, step=10, label="timesteps stop", info=""),
        gr.Slider(1, 81, value=16, step=1, label="Batch size", info=""),
        gr.Radio(["LINEAR", "QUADRATIC", "SIGMOID", "COSINE"], value="LINEAR", label="Noise Scheduler", info=""),
        gr.Radio(["DDPM", "DDIM"], value="DDPM", label="Sampler", info=""),
        gr.Radio(models, value=models[0], label="Model", info=""),
        gr.CheckboxGroup(["hero", "non-hero", "food", "spell", "side-facing"], value=["hero"], type="index", label="Embeddings", info=""),
        gr.Checkbox(label="GPU", value=True, info="GPU-Enabled?"),
    ],
    ["text", "image", "text", "image"],
    examples=[
        [500, 4, "LINEAR", "DDPM", models[0],  ["hero", "food"], True],
        [500, 4, "LINEAR", "DDIM", models[0], ["hero"], True],
    ]
)

if __name__ == "__main__":
    demo.launch()