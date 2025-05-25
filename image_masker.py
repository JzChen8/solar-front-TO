import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
class ImageMaskerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Mask Generator")
        self.root.geometry("1000x800")
        
        #variables
        self.original_image = None
        self.mask_image = None
        self.current_mask = None
        self.gray_image = None
        self.blurred_image = None
        #frame setup
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        #upload and Save buttons
        self.upload_btn = tk.Button(self.main_frame, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(pady=10)
        self.save_btn = tk.Button(self.main_frame, text="Save Mask", command=self.save_mask, state=tk.DISABLED)
        self.save_btn.pack(pady=10)
        #image frame
        self.image_frame = tk.Frame(self.main_frame)
        self.image_frame.pack(fill=tk.BOTH, expand=True)
        self.original_image_label = tk.Label(self.image_frame)
        self.original_image_label.grid(row=1, column=0, padx=10, pady=10)
        self.mask_image_label = tk.Label(self.image_frame)
        self.mask_image_label.grid(row=1, column=1, padx=10, pady=10)
        tk.Label(self.image_frame, text="Original Image").grid(row=0, column=0)
        tk.Label(self.image_frame, text="Generated Mask").grid(row=0, column=1)
        #control sliders
        self.controls_frame = ttk.Frame(self.main_frame)
        self.controls_frame.pack(pady=10)
        #fill option
        self.fill_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.controls_frame, text="Fill Mask", variable=self.fill_var, command=self.update_mask).grid(row=0, column=0, sticky='w', pady=5)
        #kernel size slider for filling
        ttk.Label(self.controls_frame, text="Kernel Size:").grid(row=1, column=0, sticky='e')
        self.kernel_size_var = tk.DoubleVar(value=5)
        self.kernel_size_slider = ttk.Scale(
            self.controls_frame, from_=1, to=20,
            orient=tk.HORIZONTAL, variable=self.kernel_size_var,
            command=lambda x: self.update_mask()
        )
        self.kernel_size_slider.grid(row=1, column=1, padx=10)
        self.kernel_size_value = ttk.Label(self.controls_frame, text="5")
        self.kernel_size_value.grid(row=1, column=2)
        # Block size slider
        ttk.Label(self.controls_frame, text="Block Size:").grid(row=2, column=0, sticky='e')
        self.block_size_var = tk.DoubleVar(value=11)
        self.block_size_slider = ttk.Scale(
            self.controls_frame, from_=3, to=51,
            orient=tk.HORIZONTAL, variable=self.block_size_var,
            command=lambda x: self.update_mask()
        )
        self.block_size_slider.grid(row=2, column=1, padx=10)
        self.block_size_value = ttk.Label(self.controls_frame, text="11")
        self.block_size_value.grid(row=2, column=2)
        # C constant slider
        ttk.Label(self.controls_frame, text="C Constant:").grid(row=3, column=0, sticky='e')
        self.c_constant_var = tk.DoubleVar(value=2)
        self.c_constant_slider = ttk.Scale(
            self.controls_frame, from_=-10, to=10,
            orient=tk.HORIZONTAL, variable=self.c_constant_var,
            command=lambda x: self.update_mask()
        )
        self.c_constant_slider.grid(row=3, column=1, padx=10)
        self.c_constant_value = ttk.Label(self.controls_frame, text="2")
        self.c_constant_value.grid(row=3, column=2)
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.gif")]
        )
        if file_path:
            try:
                self.original_image = cv2.imread(file_path)
                if self.original_image is None:
                    raise ValueError("Could not load image")
                self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
                self.blurred_image = cv2.GaussianBlur(self.gray_image, (5, 5), 0)
                self.create_mask()
                self.display_images()
                self.save_btn.config(state=tk.NORMAL)
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    def create_mask(self):
        block_size = int(round(self.block_size_var.get()))
        c_constant = int(round(self.c_constant_var.get()))
        kernel_size = int(round(self.kernel_size_var.get()))
        if block_size % 2 == 0:
            block_size += 1
        self.block_size_value.config(text=str(block_size))
        self.c_constant_value.config(text=str(c_constant))
        self.kernel_size_value.config(text=str(kernel_size))
        #create initial mask using adaptive thresholding
        mask = cv2.adaptiveThreshold(
            self.blurred_image,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block_size,
            c_constant
        )
        #if fill option is selected, apply morphological operations
        if self.fill_var.get():
            #create kernel for morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            #apply dilation to fill gaps
            mask = cv2.dilate(mask, kernel, iterations=1)
            
            # apply morphological closing to smooth the mask
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        self.current_mask = mask
        self.mask_image = cv2.cvtColor(self.current_mask, cv2.COLOR_GRAY2RGB)
    def update_mask(self, _=None):
        if self.original_image is not None:
            self.create_mask()
            self.display_images()
    def display_images(self):
        size = (400, 400)
        #og image
        original_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        original_pil = Image.fromarray(original_rgb)
        original_pil.thumbnail(size)
        original_tk = ImageTk.PhotoImage(original_pil)
        self.original_image_label.config(image=original_tk)
        self.original_image_label.image = original_tk
        #mask image
        mask_pil = Image.fromarray(self.mask_image)
        mask_pil.thumbnail(size)
        mask_tk = ImageTk.PhotoImage(mask_pil)
        self.mask_image_label.config(image=mask_tk)
        self.mask_image_label.image = mask_tk
    def save_mask(self):
        if self.current_mask is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png")])
            if file_path:
                cv2.imwrite(file_path, self.current_mask)
                messagebox.showinfo("Saved", "Mask saved successfully!")
def main():
    root = tk.Tk()
    app = ImageMaskerApp(root)
    root.mainloop()
if __name__ == "__main__":
    main()
