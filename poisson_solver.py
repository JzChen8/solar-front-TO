import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['toolbar'] = 'None'
from skimage import io
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def mask_to_conductivity(mask, r=3, sigma0=1.0, void_eps=1e-6):
    #SIMP penalization: sigma(x) = x^r * sigma0, prevents the problem from becoming discrete, allows for gradient biasing
    density = mask.astype(float)
    sigma = (density ** r) * sigma0
    sigma[density == 0] = void_eps  #voids get near-zero conductivity (not abs zero or else it could cause more trouble at the border eg. dividing by zero)
    return sigma

def solve_poisson(mask, sigma, electrode_mask, electrode_potential=1.0):
    ny, nx = mask.shape
    material = mask == 1
    electrodes = electrode_mask == 1
    #maps (i, j) for material pixels to equation indices
    ij_to_eq = -np.ones((ny, nx), dtype=int)
    eq_to_ij = []
    eq = 0
    for i in range(ny):
        for j in range(nx):
            if material[i, j]:
                ij_to_eq[i, j] = eq
                eq_to_ij.append((i, j))
                eq += 1
    N = eq  #number of unknowns (material pixels only)
    A = lil_matrix((N, N))
    b = np.zeros(N)
    for eq, (i, j) in enumerate(eq_to_ij):
        if electrodes[i, j]:
            #Dirichlet: fixed potential at electrode
            A[eq, eq] = 1
            b[eq] = electrode_potential
        else:
            #this is the interior (material, not electrode)
            s_center = 0.0
            #left neighbor
            if j > 0 and material[i, j - 1]:
                sxm = sigma[i, j - 1]
                s_center += sxm
                A[eq, ij_to_eq[i, j - 1]] = sxm
            #right neighbor
            if j < nx - 1 and material[i, j + 1]:
                sxp = sigma[i, j + 1]
                s_center += sxp
                A[eq, ij_to_eq[i, j + 1]] = sxp
            #up neighbor
            if i > 0 and material[i - 1, j]:
                sym = sigma[i - 1, j]
                s_center += sym
                A[eq, ij_to_eq[i - 1, j]] = sym
            #down neighbor
            if i < ny - 1 and material[i + 1, j]:
                syp = sigma[i + 1, j]
                s_center += syp
                A[eq, ij_to_eq[i + 1, j]] = syp
            A[eq, eq] = -s_center
    V_sol = spsolve(A.tocsr(), b)
    V = np.full((ny, nx), np.nan)
    for eq, (i, j) in enumerate(eq_to_ij):
        V[i, j] = V_sol[eq]
    return V

def visualize_potential(V, mask):
    fig, ax = plt.subplots(figsize=(6,6))
    im = ax.imshow(V, cmap='plasma', origin='upper')
    ax.imshow(mask == 0, cmap='gray', alpha=0.4, origin='upper')
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Potential (V)')
    ax.set_title('Potential Field')
    ax.axis('off')
    #description below the colorbar
    description = (
        "Plasma colormap: yellow/white = high potential (left, 1V),\n"
        "purple = low potential (right, 0V).\n"
        "Gray overlays are voids (non-conductive regions).\n"
        "Boundary conditions: left edge = 1V, right edge = 0V."
    )
    plt.gcf().text(0.5, 0.01, description, ha='center', va='bottom', fontsize=9)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.show()

def interactive_solve():
    file_path = filedialog.askopenfilename(title="Select Mask PNG", filetypes=[("PNG Images", "*.png")])
    if not file_path:
        return
    try:
        img = io.imread(file_path, as_gray=True)
        mask = (img > 0.5).astype(np.uint8)
        sigma = mask_to_conductivity(mask)
        fig, ax = plt.subplots(figsize=(6,6))
        ax.imshow(mask, cmap='gray', origin='upper')
        ax.set_title('Click Anode (1V, red) then Cathode (0V, blue)')
        electrode_points = []
        markers = []
        def onclick(event):
            if event.inaxes != ax:
                return
            x, y = int(round(event.xdata)), int(round(event.ydata))
            if 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x] == 1:
                electrode_points.append((y, x))
                color = 'red' if len(electrode_points) == 1 else 'blue'
                marker = ax.plot(x, y, marker='o', color=color, markersize=12, markeredgewidth=2)[0]
                markers.append(marker)
                fig.canvas.draw()
                if len(electrode_points) == 2:
                    fig.canvas.mpl_disconnect(cid)
                    plt.close(fig)
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show()
        if len(electrode_points) != 2:
            messagebox.showwarning("Electrode Selection", "You must select two points inside the mask.")
            return
        #builds electrode mask and potentials
        electrode_mask = np.zeros_like(mask)
        electrode_potential = np.full_like(mask, np.nan, dtype=float)
        (y1, x1), (y2, x2) = electrode_points
        electrode_mask[y1, x1] = 1
        electrode_mask[y2, x2] = 1
        electrode_potential[y1, x1] = 1.0  # Anode
        electrode_potential[y2, x2] = 0.0  # Cathode
        #modifies solver to accept per-pixel electrode_potential
        V = solve_poisson_custom_electrodes(mask, sigma, electrode_mask, electrode_potential)
        messagebox.showinfo("Poisson Solver", f"Solved for potential.\nShape: {V.shape}")
        visualize_potential(V, mask)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to solve Poisson: {str(e)}")

def solve_poisson_custom_electrodes(mask, sigma, electrode_mask, electrode_potential):
    ny, nx = mask.shape
    material = mask == 1
    electrodes = electrode_mask == 1
    ij_to_eq = -np.ones((ny, nx), dtype=int)
    eq_to_ij = []
    eq = 0
    for i in range(ny):
        for j in range(nx):
            if material[i, j]:
                ij_to_eq[i, j] = eq
                eq_to_ij.append((i, j))
                eq += 1
    N = eq
    A = lil_matrix((N, N))
    b = np.zeros(N)
    for eq, (i, j) in enumerate(eq_to_ij):
        if electrodes[i, j]:
            A[eq, eq] = 1
            b[eq] = electrode_potential[i, j]
        else:
            s_center = 0.0
            if j > 0 and material[i, j - 1]:
                sxm = sigma[i, j - 1]
                s_center += sxm
                A[eq, ij_to_eq[i, j - 1]] = sxm
            if j < nx - 1 and material[i, j + 1]:
                sxp = sigma[i, j + 1]
                s_center += sxp
                A[eq, ij_to_eq[i, j + 1]] = sxp
            if i > 0 and material[i - 1, j]:
                sym = sigma[i - 1, j]
                s_center += sym
                A[eq, ij_to_eq[i - 1, j]] = sym
            if i < ny - 1 and material[i + 1, j]:
                syp = sigma[i + 1, j]
                s_center += syp
                A[eq, ij_to_eq[i + 1, j]] = syp
            A[eq, eq] = -s_center
    V_sol = spsolve(A.tocsr(), b)
    V = np.full((ny, nx), np.nan)
    for eq, (i, j) in enumerate(eq_to_ij):
        V[i, j] = V_sol[eq]
    return V

root = tk.Tk()
root.title("Poisson Solver")
root.geometry("300x100")

upload_btn = tk.Button(root, text="Upload Mask & Select Electrodes", command=interactive_solve)
upload_btn.pack(pady=30)

root.mainloop()
