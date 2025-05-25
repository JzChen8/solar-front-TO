import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['toolbar'] = 'None'  # Hide matplotlib toolbar
from skimage import io

def mask_to_mesh(mask):
    density = mask.astype(float)
    designable = mask == 1  # Only white pixels are designable
    return density, designable

def visualize_mesh(mask, designable):
    plt.figure(figsize=(6,6))
    plt.imshow(mask, cmap='gray', origin='upper')
    # Stronger red overlay for voids
    void_overlay = np.zeros((*mask.shape, 4), dtype=np.float32)
    void_overlay[..., 0] = 1.0  # Red channel
    void_overlay[..., 3] = (~designable) * 0.7  # Alpha channel, 0.7 for voids
    plt.imshow(void_overlay, origin='upper')
    plt.title('Mesh: White=Designable, Red=Fixed Void')
    plt.axis('off')
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='white', edgecolor='k', label='Designable'),
                      Patch(facecolor='red', edgecolor='k', label='Fixed Void')]
    plt.legend(handles=legend_elements, loc='lower left', framealpha=1)
    plt.show()

def upload_and_show_mesh():
    file_path = filedialog.askopenfilename(filetypes=[("PNG Images", "*.png")])
    if not file_path:
        return
    try:
        img = io.imread(file_path, as_gray=True)
        mask = (img > 0.5).astype(np.uint8)
        density, designable = mask_to_mesh(mask)
        n_voids = np.size(designable) - np.sum(designable)
        stats = f"Mask shape: {mask.shape}\nDesignable elements: {np.sum(designable)}\nFixed voids: {n_voids}"
        print(stats)
        if n_voids == 0:
            messagebox.showwarning("Mesh Info", stats + "\n\n(No fixed voids detected! Your mask may be all white.)")
        else:
            messagebox.showinfo("Mesh Info", stats)
        visualize_mesh(mask, designable)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to process image: {str(e)}")

root = tk.Tk()
root.title("Mesh Viewer")
root.geometry("300x100")

upload_btn = tk.Button(root, text="Upload Mask PNG", command=upload_and_show_mesh)
upload_btn.pack(pady=30)

root.mainloop()
