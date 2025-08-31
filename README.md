# Pixel Enhancer – Patch-Based 10× Image Upscaler with Self-Learning

Pixel Enhancer is a **CPU-friendly, self-learning image upscaler** that progressively enhances images up to **10× resolution**. It uses a **patch-based SRCNN model** to improve details over time, learning incrementally from each image you provide.  

This project is designed for **easy interactive use**: select an image via Windows Explorer, get an immediate 10× upscaled output, and let the model learn in the background for future improvements.

---

## Features

- **Progressive 10× upscaling** using bicubic interpolation  
- **Patch-based SRCNN refinement** for faster, memory-efficient training  
- **Incremental self-learning**: model improves with every new image  
- **Immediate output**: the upscaled image is saved instantly  
- **Background training**: the SRCNN model trains without blocking user interaction  
- **Interactive Windows Explorer file picker**  
- **Automatic output and model directories**  

---

## Folder Structure

```
pixel-enhancer/
│── app.py                # Main script
│── output/               # Auto-saves enhanced images
│── dataset/              # Stores SRCNN model weights (srcnn_patch.pth)
```

Optional:

```
│── requirements.txt      # Dependencies for easy setup
│── README.md             # This documentation
│── sample_images/        # Example images for testing
```

---

## Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/pixel-enhancer.git
cd pixel-enhancer
```

2. **Install Python dependencies**:
```bash
pip install torch torchvision opencv-python pillow numpy
```

3. Ensure `output/` and `dataset/` folders exist (the script will auto-create them if not).  

---

## Usage

1. Run the main script:
```bash
python app.py
```

2. A Windows Explorer dialog will open. Select the image you want to enhance.  

3. The script will immediately save the **10× upscaled image** in the `output/` folder.  

4. The **SRCNN model will start background training** on the selected image to improve future enhancements.  

5. The model is stored in `dataset/srcnn_patch.pth` and is updated with every new image processed.  

---

## How It Works

1. **Progressive Upscaling** – The image is upscaled step by step (e.g., 2× → 2× → 2.5× ≈ 10×) for better visual quality.  
2. **Patch-Based Learning** – The SRCNN model trains on small patches instead of the full image, allowing faster and memory-efficient incremental learning.  
3. **Incremental Self-Learning** – Each new image adds more patches, gradually improving the SRCNN’s ability to restore details over time.  

---

## Tips & Notes

- **Patch Size**: Default is 32×32 for CPU efficiency; you can increase to 64×64 for faster training with less granularity.  
- **CPU-Only**: Large images may take several minutes for background training; smaller images are processed faster.  
- **Future Improvement**: GPU support can drastically speed up patch-based training.  

---

## License

This project is **MIT licensed**. Feel free to use, modify, and distribute freely.  

---

## Screenshots

*(Optional: Add before/after images here for demonstration.)*  

---

## Acknowledgements

- Inspired by **SRCNN** for super-resolution  
- Uses **PyTorch**, **OpenCV**, **Pillow**, and **Tkinter** for core functionality  

---

**Author:** Imam Ud Doula  
**Email:** imamshadin004@gmail.com  
**Location:** Dhaka, Bangladesh
