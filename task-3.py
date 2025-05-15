import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import torchvision.models as models

# Set image size
IMG_SIZE = 256
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load VGG19 features (for placeholder use)
model = models.vgg19(pretrained=True).features.eval().to(device)
for param in model.parameters():
    param.requires_grad = False

# Load and preprocess image
def load_image(image_path):
    image = Image.open(image_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

# Dummy style transfer using ratio
def dummy_style_transfer(content_tensor, style_tensor, ratio):
    return (content_tensor * (1 - ratio) + style_tensor * ratio).clamp(0, 1)

# GUI Application
class StyleTransferGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🎨 Neural Style Transfer")
        self.root.geometry("800x750")
        self.root.configure(bg="#f0f8ff")

        self.content_img_path = None
        self.style_img_path = None

        tk.Label(root, text="Neural Style Transfer", font=("Helvetica", 24, "bold"),
                 bg="#f0f8ff", fg="#333").pack(pady=10)

        # Canvas Frame
        canvas_frame = tk.Frame(root, bg="#f0f8ff")
        canvas_frame.pack(pady=5)

        self.content_canvas = tk.Canvas(canvas_frame, width=IMG_SIZE, height=IMG_SIZE, bg="#e3f2fd")
        self.content_canvas.grid(row=0, column=0, padx=10)

        self.style_canvas = tk.Canvas(canvas_frame, width=IMG_SIZE, height=IMG_SIZE, bg="#fce4ec")
        self.style_canvas.grid(row=0, column=1, padx=10)

        # Control Buttons
        tk.Button(root, text="📷 Load Content Image", font=("Arial", 14), bg="#90caf9",
                  command=self.load_content).pack(pady=5)

        tk.Button(root, text="🎨 Load Style Image", font=("Arial", 14), bg="#f48fb1",
                  command=self.load_style).pack(pady=5)

        # Ratio Slider
        tk.Label(root, text="Style Ratio", font=("Arial", 12, "bold"), bg="#f0f8ff").pack()
        self.ratio_slider = tk.Scale(root, from_=0.0, to=1.0, resolution=0.05,
                                     orient=tk.HORIZONTAL, length=300, bg="#f0f8ff")
        self.ratio_slider.set(0.4)
        self.ratio_slider.pack(pady=5)

        # Style Button
        tk.Button(root, text="✨ Apply Style", font=("Arial", 16, "bold"), bg="#aed581",
                  command=self.apply_style).pack(pady=10)

        # Output canvas
        self.output_canvas = tk.Canvas(root, width=IMG_SIZE, height=IMG_SIZE, bg="#fffde7")
        self.output_canvas.pack(pady=10)

    def load_content(self):
        self.content_img_path = filedialog.askopenfilename()
        if self.content_img_path:
            self.content_image = Image.open(self.content_img_path).resize((IMG_SIZE, IMG_SIZE))
            self.tk_content_img = ImageTk.PhotoImage(self.content_image)
            self.content_canvas.create_image(0, 0, anchor='nw', image=self.tk_content_img)

    def load_style(self):
        self.style_img_path = filedialog.askopenfilename()
        if self.style_img_path:
            self.style_image = Image.open(self.style_img_path).resize((IMG_SIZE, IMG_SIZE))
            self.tk_style_img = ImageTk.PhotoImage(self.style_image)
            self.style_canvas.create_image(0, 0, anchor='nw', image=self.tk_style_img)

    def apply_style(self):
        if not self.content_img_path or not self.style_img_path:
            messagebox.showerror("Error", "Please load both content and style images.")
            return

        try:
            content = load_image(self.content_img_path)
            style = load_image(self.style_img_path)
            ratio = self.ratio_slider.get()

            output = dummy_style_transfer(content, style, ratio)
            output_img = transforms.ToPILImage()(output.squeeze().cpu())

            # Display and save output
            self.tk_output_img = ImageTk.PhotoImage(output_img.resize((IMG_SIZE, IMG_SIZE)))
            self.output_canvas.create_image(0, 0, anchor='nw', image=self.tk_output_img)
            output_img.save("styled_output.png")

            messagebox.showinfo("Success", f"Style applied with ratio {ratio:.2f}. Saved as styled_output.png")

        except Exception as e:
            messagebox.showerror("Style Transfer Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = StyleTransferGUI(root)
    root.mainloop()
