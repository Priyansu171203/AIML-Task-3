import torch
from torchvision import transforms, models
from PIL import Image, ImageTk
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading

device = ("cuda" if torch.cuda.is_available() else "cpu")

model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
for p in model.parameters():
    p.requires_grad = False
model.to(device)

def model_activations(input, model):
    layers = {
        '0': 'conv1_1',
        '5': 'conv2_1',
        '10': 'conv3_1',
        '19': 'conv4_1',
        '21': 'conv4_2',
        '28': 'conv5_1'
    }
    features = {}
    x = input.unsqueeze(0)
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
    return features

transform = transforms.Compose([
    transforms.Resize(300),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

def imcnvt(image):
    x = image.to("cpu").clone().detach().numpy().squeeze()
    x = x.transpose(1,2,0)
    x = x * np.array((0.5,0.5,0.5)) + np.array((0.5,0.5,0.5))
    return np.clip(x, 0, 1)

def gram_matrix(imgfeature):
    _, d, h, w = imgfeature.size()
    imgfeature = imgfeature.view(d, h * w)
    gram_mat = torch.mm(imgfeature, imgfeature.t())
    return gram_mat


class StyleTransferApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("🎨 Neural Style Transfer")
        self.geometry("900x700")
        self.configure(bg="#2c3e50")  # Dark blue background

        self.content_path = None
        self.style_path = None

        # Custom fonts
        self.title_font = ("Segoe UI", 22, "bold")
        self.label_font = ("Segoe UI", 12, "bold")
        self.button_font = ("Segoe UI", 14, "bold")

        self.create_styles()
        self.create_widgets()

        self.content_img_tensor = None
        self.style_img_tensor = None
        self.output_img_tensor = None

    def create_styles(self):
        style = ttk.Style(self)
        style.theme_use('clam')

        style.configure('TLabel', background="#34495e", foreground="white", font=self.label_font, padding=10)
        style.configure('TButton', font=self.button_font, background="#e67e22", foreground="white", padding=10)
        style.map('TButton',
                  foreground=[('active', 'white')],
                  background=[('active', '#d35400')])

    def create_widgets(self):
        # Header Label
        header = tk.Label(self, text="Neural Style Transfer", font=self.title_font, fg="white", bg="#2c3e50")
        header.pack(pady=20)

        # Frame for content and style images
        frame = tk.Frame(self, bg="#2c3e50")
        frame.pack(pady=10)

        # Content Image Frame
        self.content_frame = ttk.LabelFrame(frame, text="Content Image", width=320, height=320)
        self.content_frame.grid(row=0, column=0, padx=20)
        self.content_frame.pack_propagate(False)  # Fix frame size

        self.content_label = tk.Label(self.content_frame, text="Click to select image", bg="#34495e", fg="white", font=self.label_font, width=35, height=15)
        self.content_label.pack(fill='both', expand=True)
        self.content_label.bind("<Button-1>", self.load_content_image)

        # Style Image Frame
        self.style_frame = ttk.LabelFrame(frame, text="Style Image", width=320, height=320)
        self.style_frame.grid(row=0, column=1, padx=20)
        self.style_frame.pack_propagate(False)

        self.style_label = tk.Label(self.style_frame, text="Click to select image", bg="#34495e", fg="white", font=self.label_font, width=35, height=15)
        self.style_label.pack(fill='both', expand=True)
        self.style_label.bind("<Button-1>", self.load_style_image)

        # Run button with custom style
        self.run_button = ttk.Button(self, text="Run Style Transfer", command=self.run_style_transfer)
        self.run_button.pack(pady=30, ipadx=20, ipady=10)

        # Progress Bar
        self.progress = ttk.Progressbar(self, orient="horizontal", length=600, mode="determinate")
        self.progress.pack(pady=10)

        # Output Frame
        self.output_frame = ttk.LabelFrame(self, text="Output Image", width=640, height=480)
        self.output_frame.pack(pady=10)
        self.output_frame.pack_propagate(False)

        self.output_label = tk.Label(self.output_frame, bg="#34495e")
        self.output_label.pack(fill='both', expand=True)

    def load_content_image(self, event=None):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if path:
            self.content_path = path
            img = Image.open(path).resize((300, 300))
            self.content_img = ImageTk.PhotoImage(img)
            self.content_label.config(image=self.content_img, text="")

            self.content_img_tensor = transform(Image.open(path).convert("RGB")).to(device)

    def load_style_image(self, event=None):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if path:
            self.style_path = path
            img = Image.open(path).resize((300, 300))
            self.style_img = ImageTk.PhotoImage(img)
            self.style_label.config(image=self.style_img, text="")

            self.style_img_tensor = transform(Image.open(path).convert("RGB")).to(device)

    def run_style_transfer(self):
        if self.content_img_tensor is None or self.style_img_tensor is None:
            messagebox.showerror("Error", "Please load both content and style images!")
            return
        self.run_button.config(state="disabled")
        threading.Thread(target=self.style_transfer_thread, daemon=True).start()

    def style_transfer_thread(self):
        target = self.content_img_tensor.clone().requires_grad_(True).to(device)

        style_features = model_activations(self.style_img_tensor, model)
        content_features = model_activations(self.content_img_tensor, model)

        style_wt_meas = {"conv1_1": 1.0, "conv2_1": 0.8, "conv3_1": 0.4, "conv4_1": 0.2, "conv5_1": 0.1}
        style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

        content_wt = 100
        style_wt = 1e8
        epochs = 100
        optimizer = torch.optim.Adam([target], lr=0.007)

        for i in range(1, epochs + 1):
            target_features = model_activations(target, model)
            content_loss = torch.mean((content_features['conv4_2'] - target_features['conv4_2']) ** 2)

            style_loss = 0
            for layer in style_wt_meas:
                style_gram = style_grams[layer]
                target_gram = target_features[layer]
                _, d, w, h = target_gram.shape
                target_gram = gram_matrix(target_gram)

                style_loss += (style_wt_meas[layer] * torch.mean((target_gram - style_gram) ** 2)) / (d * w * h)

            total_loss = content_wt * content_loss + style_wt * style_loss

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Update progress bar correctly using lambda
            progress_val = (i / epochs) * 100
            self.progress.after(0, lambda val=progress_val: self.progress.configure(value=val))
            self.update_idletasks()

            if i % 20 == 0:
                print(f"Epoch {i}/{epochs} Loss: {total_loss.item():.4f}")

        self.output_img_tensor = target.detach()

        # Update UI on main thread
        self.progress.after(0, self.show_output_image)
        self.progress.after(0, lambda: self.run_button.config(state="normal"))
        self.progress.after(0, lambda: self.progress.configure(value=0))

    def show_output_image(self):
        img_np = imcnvt(self.output_img_tensor)
        img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
        img_pil = img_pil.resize((600, 450))
        self.output_img = ImageTk.PhotoImage(img_pil)
        self.output_label.config(image=self.output_img)
        self.output_label.image = self.output_img


if __name__ == "__main__":
    app = StyleTransferApp()
    app.mainloop()
