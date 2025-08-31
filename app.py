import os
import torch
import torch.nn as nn
import torch.optim as optim
import cv2
from PIL import Image
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from torchvision import transforms

# --- Tiny SRCNN model ---
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# --- Paths ---
output_dir = "output"
dataset_dir = "dataset"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(dataset_dir, exist_ok=True)

# --- Pick file ---
Tk().withdraw()
input_path = askopenfilename(title="Select an image to enhance",
                             filetypes=[("Image files", "*.png *.jpg *.jpeg")])
if not input_path:
    print("❌ No file selected. Exiting.")
    exit()

# --- Load image ---
img = cv2.imread(input_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
height, width = img_rgb.shape[:2]

# --- Progressive 10x upscaling ---
scale_factors = [2, 2, 2.5]  # progressive: 2*2*2.5 ≈ 10x
upscaled = img_rgb.copy()
for factor in scale_factors:
    new_size = (int(upscaled.shape[1]*factor), int(upscaled.shape[0]*factor))
    upscaled = cv2.resize(upscaled, new_size, interpolation=cv2.INTER_CUBIC)

# --- Save initial progressive upscaled image ---
filename = os.path.basename(input_path)
output_path = os.path.join(output_dir, f"enhanced_{filename}")
cv2.imwrite(output_path, cv2.cvtColor(upscaled, cv2.COLOR_RGB2BGR))

# --- Patch-based learning preparation ---
patch_size = 32
scale_total = int(upscaled.shape[0]/height)  # total scale factor

# Extract patches
patches_lr, patches_hr = [], []
for i in range(0, height-patch_size+1, patch_size):
    for j in range(0, width-patch_size+1, patch_size):
        lr_patch = img_rgb[i:i+patch_size, j:j+patch_size]
        hr_patch = upscaled[i*scale_total:(i+patch_size)*scale_total,
                            j*scale_total:(j+patch_size)*scale_total]
        # Upscale LR patch to match HR size
        lr_patch_up = cv2.resize(lr_patch, (hr_patch.shape[1], hr_patch.shape[0]), interpolation=cv2.INTER_CUBIC)
        patches_lr.append(lr_patch_up)
        patches_hr.append(hr_patch)

# Convert to tensors
transform = transforms.ToTensor()
patches_lr = torch.stack([transform(p).unsqueeze(0) for p in patches_lr]).squeeze(1)
patches_hr = torch.stack([transform(p).unsqueeze(0) for p in patches_hr]).squeeze(1)

# --- Load or initialize model ---
model_path = os.path.join(dataset_dir, "srcnn_patch.pth")
device = torch.device("cpu")
model = SRCNN().to(device)
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))

# --- Patch-based incremental training ---
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
model.train()

for epoch in range(5):  # small number of epochs per image
    for lr_patch, hr_patch in zip(patches_lr, patches_hr):
        lr_patch = lr_patch.unsqueeze(0).to(device)
        hr_patch = hr_patch.unsqueeze(0).to(device)
        optimizer.zero_grad()
        output = model(lr_patch)
        loss = criterion(output, hr_patch)
        loss.backward()
        optimizer.step()

# --- Save updated model ---
torch.save(model.state_dict(), model_path)
print(f"✅ Image saved as '{output_path}' and model updated with patch-based learning.")
