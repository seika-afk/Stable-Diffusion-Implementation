# 🧠 Stable Diffusion Implementation (from Scratch)

This project demonstrates a basic implementation of **Stable Diffusion**, a state-of-the-art deep generative model for image synthesis.

## 📘 What is Stable Diffusion?

Stable Diffusion is a **latent diffusion model (LDM)** that generates high-quality images from text prompts. It learns to denoise random noise into meaningful images through a process called **denoising diffusion**.

---

## 📂 Project Structure

- `Stable_diffusion_Implementation.ipynb`:  
  The main Jupyter notebook implementing each step from loading the model to generating images from prompts.

---

## 📦 Dependencies

Make sure you have the following installed:

```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate scipy safetensors
```

Also, ensure you're using a GPU-enabled runtime (like Colab with GPU).

---

## 🚀 Running the Notebook

1. Clone or download this repository.
2. Open the `.ipynb` file in Jupyter/Colab.
3. Make sure you're using a GPU runtime.
4. Run all cells — the notebook will:
   - Load the pretrained Stable Diffusion model
   - Disable the safety checker for full control
   - Generate stunning images from your text prompts ✨

---

## ⚠️ Disclaimer

- This implementation disables safety checks for academic freedom. Please use responsibly.
- Generated content depends entirely on your prompt. Avoid harmful, unethical, or illegal usage.

---

## 🧪 Example Prompt

```python
prompt = "a futuristic cityscape at sunset, highly detailed"
```

Yields an image like:

![example-image](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image.png)

---

## ❤️ Credits

- [Hugging Face Diffusers](https://github.com/huggingface/diffusers)
- [CompVis](https://github.com/CompVis/stable-diffusion)
- Community contributors for open-sourcing the tools we love!

---

## 📜 License

MIT License. Feel free to fork, experiment, and learn
