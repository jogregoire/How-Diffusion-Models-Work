import gradio as gr
from pathlib import Path
from main import *

def sampler_builder(timesteps, n_sample, noise_type, sampler_type, model_filename, embeddings, gpu):
    context = [0.0, 0.0, 0.0, 0.0, 0.0]
    for index in embeddings:
        context[index] = 1.0
    device, gpu_perf = initialize()
    sample(device, gpu_perf, timesteps, noise_type, n_sample, sampler_type, model_filename, context, grid_filename = '../data/grid.png')
    return f"timesteps: {timesteps}, batch_size: {n_sample}, noise_type: {noise_type}, sampler_type: {sampler_type}, embeddings: {context}, gpu: {gpu}"

models = list(Path('../weights').glob('*.pth'))
demo = gr.Interface(
    sampler_builder,
    [
        gr.Slider(300, 1000, value=500, label="Timesteps", info="Choose between 300 and 1000"),
        gr.Slider(1, 81, value=4, label="Batch size", info=""),
        gr.Radio(["LINEAR", "QUADRATIC", "SIGMOID", "COSINE"], label="Noise Scheduler", info=""),
        gr.Radio(["DDPM", "DDIM"], label="Sampler", info=""),
        gr.Radio(models, value=models[0], label="Model", info=""),
        gr.CheckboxGroup(["hero", "non-hero", "food", "spell", "side-facing"], value=["hero"], type="index", label="Embeddings", info=""),
        gr.Checkbox(label="GPU", value=True, info="GPU-Enabled?"),
    ],
    "text",
    examples=[
        [500, 4, "LINEAR", "DDPM", models[0],  ["hero", "food"], True],
        [500, 4, "LINEAR", "DDIM", models[0], ["hero"], True],
    ]
)

if __name__ == "__main__":
    demo.launch()