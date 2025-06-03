import numpy as np
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#these are the main libraries: numpy for math, tkinter for gui, matplotlib for plots
#think of tkinter as the control panel, matplotlib as the window to see the electrode growth

#constants for the physics and grid
q = 1.602e-19 #electron charge
kB = 1.38e-23 #boltzmann constant
T = 300 #room temp (kelvin)
beta = q / (kB * T) #thermal voltage factor
sigma_TCO = 1e5 #conductivity of tco (transparent conductor)
sigma_metal = 1e7 #conductivity of metal
t_hickness_TCO = 200e-9 #tco thickness
thickness_metal = 10e-6 #metal thickness
Nx, Ny = 40, 40 #grid size (elements in x/y)
Lx, Ly = 0.1, 0.1 #physical size in meters
dx, dy = Lx / Nx, Ly / Ny #element size
V_bus = 0.6 #busbar voltage (voltage applied to left)
penal = 3.0 #simp penalization (encourages 0/1 material)
r_min = 0.01 #filter radius (for smoothing design)
vol_frac_target = 0.05 #target fraction of metal
j0 = 0.0001 #reverse current density
jL = 40 #photocurrent density
max_newton_iter = 20 #max iterations for newton solver
newton_tol = 1e-6 #tolerance for newton solver
max_TO_iter = 50 #max topology optimization iterations


# New functions based on solar cell paper equations (Eq. 15, 17, 18, 19, 20, 30, 31, 33)

def calculate_local_voltage(shape_function_vector: np.ndarray, nodal_voltage_vector: np.ndarray) -> float:
    #eq 15: local voltage is just a weighted sum of the voltages at the corners
    #analogy: like blending four corner heights to get the height at a point on a trampoline
    if shape_function_vector.shape != nodal_voltage_vector.shape:
        raise ValueError("Shape function vector and nodal voltage vector must have the same dimensions.")
    return np.dot(shape_function_vector, nodal_voltage_vector)

def calculate_corrected_photocurrent_density(jL: float, xe: float, r_val: float) -> float:
    #eq 18: as metal covers the surface, less light gets in, so jL drops
    #analogy: like putting more umbrellas over a field, less rain reaches the ground
    return jL * ((1 - xe)**r_val)

def calculate_element_current_density(j_star_L: float, j0_val: float, beta_val: float, local_voltage: float) -> float:
    #eq 17: total current is the sum of light current and diode current
    #analogy: like water flow = rain (j*_L) + flow leaking out a pipe (j0 * exp(...))
    return j_star_L + j0_val * (np.exp(beta_val * local_voltage) - 1)

def calculate_power_output(Vbus: float, Ae: float, J_elements: np.ndarray) -> float:
    #eq 19: total power is voltage times total current (sum of all element currents)
    #analogy: like total water delivered = pressure * total pipe area * average flow rate
    return Vbus * Ae * np.sum(J_elements)

def objective_function(Pout: float) -> float:
    #eq 20: we want to maximize power, but optimizers minimize, so we take -Pout
    #analogy: like running backwards to win a race because the finish line is behind you
    return -Pout

def calculate_weight_factor_Hei(rmin: float, distance_e_i: float) -> float:
    #eq 31: weight for filtering drops off with distance, like how much a neighbor influences you
    #analogy: friends close to you have more say in your decisions
    return max(0, rmin - distance_e_i)

def calculate_filtered_density_single_element(densities_in_neighborhood: np.ndarray, weights_in_neighborhood: np.ndarray) -> float:
    #eq 30: filtered density is a weighted average of neighbors
    #analogy: like your opinion is shaped by your closest friends (weighted by how close)
    sum_weights = np.sum(weights_in_neighborhood)
    if sum_weights == 0:
        return 0.0
    return np.sum(weights_in_neighborhood * densities_in_neighborhood) / sum_weights

def calculate_efficiency(Pout: float, Ac: float, pinp: float) -> float:
    #eq 33: efficiency is how much output you get vs what you put in
    #analogy: like miles per gallon for a car, but for sunlight
    if Ac <= 0 or pinp <= 0:
        raise ValueError("Cell area (Ac) and input power density (pinp) must be positive.")
    return (Pout / (Ac * pinp)) * 100.0

# End of new functions

def element_conductivity_matrix(dx, dy):
    #returns the local conductivity matrix for a single element
    #analogy: think of this as the stiffness of a little tile in a floor
    k = 1 / (dx * dy)
    G0 = k * np.array([[2, -1, -1, 0],
                       [-1, 2, 0, -1],
                       [-1, 0, 2, -1],
                       [0, -1, -1, 2]])
    return G0

G0 = element_conductivity_matrix(dx, dy)

def simp_interpolation(xe, penal, sigma_min, sigma_max):
    #simp: smoothly blends between tco and metal conductivity based on density
    #analogy: like a dimmer switch between two light bulbs
    return sigma_min + (sigma_max - sigma_min) * (xe ** penal)

def assemble_global_conductivity(x, penal):
    #builds the big conductivity matrix for the whole grid
    #analogy: like assembling a giant lego wall out of little blocks (elements)
    n_nodes_x = Nx + 1
    n_nodes_y = Ny + 1
    n_nodes = n_nodes_x * n_nodes_y

    G = np.zeros((n_nodes, n_nodes))

    for ey in range(Ny):
        for ex in range(Nx):
            e = ey * Nx + ex
            sigma_e = simp_interpolation(x[e], penal, sigma_TCO, sigma_metal)
            ke = sigma_e * (thickness_TCO + thickness_metal) * G0
            n1 = ey * n_nodes_x + ex
            n2 = n1 + 1
            n3 = n1 + n_nodes_x
            n4 = n3 + 1
            nodes = [n1, n2, n4, n3]
            for i_local, i_global in enumerate(nodes):
                for j_local, j_global in enumerate(nodes):
                    G[i_global, j_global] += ke[i_local, j_local]
    return G

def apply_boundary_conditions(G, b, V_bus):
    #sets the left edge to the busbar voltage
    #analogy: like clamping one edge of a trampoline to a fixed height
    n_nodes_x = Nx + 1
    n_nodes_y = Ny + 1
    for j in range(n_nodes_y):
        node = j * n_nodes_x
        G[node, :] = 0
        G[:, node] = 0
        G[node, node] = 1
        b[node] = V_bus
    return G, b

def j_from_V(Ve, xe, penal, r=3):
    #converts voltage at an element to current density
    #analogy: like how much water flows out depending on the pressure (voltage) and blockage (xe)
    jL_star = jL * (1 - xe ** r)
    j0_const = j0
    j = jL_star - j0_const * (np.exp(beta * Ve) - 1)
    return j

def build_nodal_to_element_mapping():
    #makes a lookup table for which nodes belong to each element
    #analogy: like a seating chart for a classroom
    n_nodes_x = Nx + 1
    nodemap = np.zeros((Ny, Nx, 4), dtype=int)
    for ey in range(Ny):
        for ex in range(Nx):
            n1 = ey * n_nodes_x + ex
            n2 = n1 + 1
            n3 = n1 + n_nodes_x
            n4 = n3 + 1
            nodemap[ey, ex, :] = [n1, n2, n4, n3]
    return nodemap

nodemap = build_nodal_to_element_mapping()

def compute_current_density(U, xe, penal):
    #computes current density for each element
    #analogy: like measuring how much water flows through each pipe in a network
    j_e = np.zeros(Nx * Ny)
    for ey in range(Ny):
        for ex in range(Nx):
            e = ey * Nx + ex
            nodes = nodemap[ey, ex, :]
            Ve = np.mean(U[nodes])
            j_e[e] = j_from_V(Ve, xe[e], penal)
    return j_e

def power_output(U, xe, penal):
    #computes total power output from all elements
    #analogy: like adding up all the electricity generated by each solar panel tile
    j_e = compute_current_density(U, xe, penal)
    A_e = dx * dy
    Pout = np.sum(j_e * U[:- (Nx + 1)]) * A_e
    return Pout

def residual(U, xe, penal):
    #calculates the difference between current and what the grid expects
    #analogy: like checking if the water pressure everywhere matches what the pipes should deliver
    G = assemble_global_conductivity(xe, penal)
    b = np.zeros_like(U)
    G, b = apply_boundary_conditions(G, b, V_bus)

    I = np.zeros_like(U)
    for ey in range(Ny):
        for ex in range(Nx):
            e = ey * Nx + ex
            nodes = nodemap[ey, ex, :]
            Ve = np.mean(U[nodes])
            je = j_from_V(Ve, xe[e], penal)
            for n in nodes:
                I[n] += je / 4
    R = G @ U - I
    return R, G, I

def newton_solver(xe, penal):
    #solves for voltages using newton's method
    #analogy: like tweaking the heights of a trampoline until all springs are balanced
    n_nodes = (Nx + 1) * (Ny + 1)
    U = np.zeros(n_nodes)
    for j in range(Ny + 1):
        node = j * (Nx + 1)
        U[node] = V_bus
    for iter in range(max_newton_iter):
        R, G, I = residual(U, xe, penal)
        normR = np.linalg.norm(R)
        if normR < newton_tol:
            break
        try:
            delta_U = np.linalg.solve(G, -R)
        except np.linalg.LinAlgError:
            print("Singular matrix in Newton solver")
            break
        U += delta_U
    return U

def density_filter(x, r_min):
    #smooths the design by averaging with neighbors (removes checkerboarding)
    #analogy: like blurring an image so sharp spots get smoothed out
    x_filtered = np.zeros_like(x)
    n = Nx * Ny
    for i in range(n):
        ix, iy = i % Nx, i // Nx
        sum_w = 0
        val = 0
        for j in range(n):
            jx, jy = j % Nx, j // Nx
            dist = np.sqrt((ix - jx) ** 2 + (iy - jy) ** 2)
            if dist <= r_min * Nx:
                w = r_min * Nx - dist
                val += w * x[j]
                sum_w += w
        x_filtered[i] = val / sum_w if sum_w > 0 else x[i]
    return x_filtered

def sensitivity_analysis(U, x, penal):
    dPdx = np.zeros_like(x)
    n_nodes_x = Nx + 1
    
    # Calculate voltage gradients for all elements
    gradV = np.zeros((Nx * Ny, 2))
    for ey in range(Ny):
        for ex in range(Nx):
            e = ey * Nx + ex
            nodes = nodemap[ey, ex, :]
            Vn = U[nodes]
            gradV[e, 0] = (Vn[1] - Vn[0]) / dx  # x-gradient
            gradV[e, 1] = (Vn[3] - Vn[0]) / dy  # y-gradient
    
    # Calculate sensitivity with enhanced dendritic growth
    for ey in range(Ny):
        for ex in range(Nx):
            e = ey * Nx + ex
            if x[e] > 0.01:  # Already metal (lower threshold for debug)
                # Stronger sensitivity for existing metal to promote growth
                sigma_deriv = penal * (sigma_metal - sigma_TCO) * x[e] ** (penal - 1)
            else:
                # Enhanced dendritic growth logic
                neighbors = []
                if (ex > 0): neighbors.append((e-1, 'left'))
                if (ex < Nx-1): neighbors.append((e+1, 'right'))
                if (ey > 0): neighbors.append((e-Nx, 'down'))
                if (ey < Ny-1): neighbors.append((e+Nx, 'up'))
                
                # Calculate dendritic potential
                dendritic_potential = 0
                metal_neighbors = []
                for nbr, direction in neighbors:
                    if x[nbr] > 0.01:  # If neighbor is metal (lower threshold for debug)
                        metal_neighbors.append((nbr, direction))
                
                if metal_neighbors:
                    # Calculate branching potential based on neighbor gradients and positions
                    for nbr, direction in metal_neighbors:
                        # Calculate angle between gradients
                        grad_dot = np.dot(gradV[e], gradV[nbr])
                        grad_norm = np.linalg.norm(gradV[e]) * np.linalg.norm(gradV[nbr])
                        if grad_norm > 0:
                            cos_angle = grad_dot / grad_norm
                            # Strongly promote growth at angles between 30-150 degrees
                            if abs(cos_angle) < 0.866:  # cos(30 degrees)
                                dendritic_potential += 2.0  # Increased from 1.0
                    
                    # Additional dendritic growth promotion
                    if len(metal_neighbors) == 1:  # Single metal neighbor
                        dendritic_potential *= 1.5  # Encourage extension
                    elif len(metal_neighbors) == 2:  # Two metal neighbors
                        dendritic_potential *= 2.0  # Strongly encourage branching
                
                # Enhanced sensitivity for potential dendritic points
                sigma_deriv = penal * (sigma_metal - sigma_TCO) * (0.1 + 0.9 * dendritic_potential)
            
            gradVsq = np.sum(gradV[e]**2)
            dPdx[e] = -sigma_deriv * (thickness_TCO + thickness_metal) * gradVsq * dx * dy
    
    return dPdx

def update_design(x, dPdx, vol_frac_target, move=0.05):
    x_new = np.copy(x)
    n = Nx * Ny
    
    # First pass: identify potential dendritic growth points
    growth_potential = np.zeros_like(x)
    for i in range(n):
        if x[i] < 0.01:  # Not metal (lower threshold for debug)
            # Check neighbors for existing metal
            neighbors = []
            if (i % Nx) > 0: neighbors.append((i-1, 'left'))
            if (i % Nx) < Nx-1: neighbors.append((i+1, 'right'))
            if (i // Nx) > 0: neighbors.append((i-Nx, 'down'))
            if (i // Nx) < Ny-1: neighbors.append((i+Nx, 'up'))
            
            metal_neighbors = [(nbr, dir) for nbr, dir in neighbors if x[nbr] > 0.01] # Lower threshold
            
            if metal_neighbors:
                # Simplified: Moderate bonus if any metal neighbor exists to encourage general growth
                growth_potential[i] = 1.5 # Moderate, uniform bonus
    
    # Second pass: update design with enhanced dendritic growth
    for i in range(n):
        if x[i] < 0.01:  # Not metal (lower threshold for debug)
            if growth_potential[i] > 0:
                # Enhanced growth at dendritic points
                x_new[i] = min(1.0, x[i] + 0.5)  # Aggressive debug growth
        else:
            # Existing metal: update normally
            x_new[i] = min(1.0, x[i] + 0.1)   # Aggressive debug reinforcement
    
    # Apply stronger filtering to maintain dendritic structure
    x_new = np.clip(x_new, 0, 1)
    
    # Enforce volume constraint with gradual adjustment
    vol = np.mean(x_new)
    if vol > vol_frac_target:
        # Gradual volume adjustment to maintain dendritic structure
        adjustment_factor = 0.95 + 0.05 * (vol_frac_target / vol)
        x_new = x_new * adjustment_factor
    
    return x_new

class TO_GUI:
    def __init__(self, root):
        #main gui class: sets up everything you see and click
        self.root = root
        self.flicker_state = False  #for flicker effect during optimization
        self.root.title("Solar Cell Front Electrode Topology Optimization - Dendrite Growth")
        #no fixed minsize: everything stretches with window
        self.root.columnconfigure(0, weight=1, minsize=150) #left for controls
        self.root.columnconfigure(1, weight=3) #right for plot
        for i in range(12):
            self.root.rowconfigure(i, weight=1) #make all rows stretch

        self.Nx = Nx
        self.Ny = Ny
        #these are tkinter variables so sliders and entries update automatically
        self.vol_frac_target = tk.DoubleVar(value=0.03)  #volume fraction slider
        self.penal = tk.DoubleVar(value=4.0)  #simp penalization slider
        self.r_min = tk.DoubleVar(value=0.0075)  #filter radius slider

        #design variable: 1d array for electrode density, starts empty
        self.x = np.zeros(self.Nx * self.Ny)
        self.mask = None #mask will be loaded from png

        #matplotlib figure for showing the electrode pattern
        self.fig, self.ax = plt.subplots(figsize=(5,5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas.get_tk_widget().grid(row=0, column=1, rowspan=12, padx=10, pady=10, sticky='nsew') #plot fills right side

        #all the widgets below are controls for the optimization
        #load mask button lets you pick a png file
        self.load_button = ttk.Button(root, text="Load Mask PNG", command=self.load_mask)
        self.load_button.grid(row=0, column=0, pady=0, sticky='nsew')

        #volume fraction slider
        ttk.Label(root, text="Volume Fraction Target").grid(row=1, column=0, sticky='nsew')
        self.vol_slider = ttk.Scale(root, from_=0.01, to=0.1, variable=self.vol_frac_target, orient='horizontal')
        self.vol_slider.grid(row=2, column=0, pady=0, sticky='nsew')

        #simp penalization slider
        ttk.Label(root, text="SIMP Penalization (p)").grid(row=3, column=0, sticky='nsew')
        self.penal_slider = ttk.Scale(root, from_=2.0, to=6.0, variable=self.penal, orient='horizontal')
        self.penal_slider.grid(row=4, column=0, pady=0, sticky='nsew')

        #filter radius slider
        ttk.Label(root, text="Filter Radius (fraction of Nx)").grid(row=5, column=0, sticky='nsew')
        self.rmin_slider = ttk.Scale(root, from_=0.0025, to=0.05, variable=self.r_min, orient='horizontal')
        self.rmin_slider.grid(row=6, column=0, pady=0, sticky='nsew')

        #max iterations entry
        ttk.Label(root, text="Max Iterations").grid(row=7, column=0, sticky='nsew')
        self.max_iter_entry = ttk.Entry(root)
        self.max_iter_entry.insert(0, "100")
        self.max_iter_entry.grid(row=8, column=0, pady=0, sticky='nsew')

        #run button starts the optimization
        self.run_button = ttk.Button(root, text="Run Optimization", command=self.run_optimization)
        self.run_button.grid(row=9, column=0, pady=0, sticky='nsew')

        #reset button resets design to initial state
        self.reset_button = ttk.Button(root, text="Reset", command=self.reset)
        self.reset_button.grid(row=10, column=0, pady=0, sticky='nsew')

        #status label shows progress and errors
        self.status_label = ttk.Label(root, text="")
        self.status_label.grid(row=11, column=0, columnspan=2, pady=0, sticky='nsew')

        self.update_plot() #draw the initial (empty) plot

    def load_mask(self):
        #lets user pick a png file as a mask
        #analogy: like putting painter's tape on a canvas to block certain areas
        from tkinter import filedialog
        from PIL import Image
        
        file_path = filedialog.askopenfilename(
            title="Select Mask PNG",
            filetypes=[("PNG files", "*.png")]
        )
        
        if file_path:
            try:
                #load and resize image to match grid size
                img = Image.open(file_path).convert('L')
                img = img.resize((self.Nx, self.Ny))
                self.mask = np.array(img) / 255.0
                
                #initialize design variable with mask (all zeros except mask)
                self.x = np.zeros_like(self.mask.flatten())
                #add seed points where mask is white (could be more clever)
                seed_points = np.where(self.mask.flatten() > 0.5)[0]
                if len(seed_points) > 0:
                    self.x[seed_points] = 1.0
                
                #show the mask on the plot
                self.update_plot()
                self.status_label.config(text=f"Loaded mask from {file_path}")
            except Exception as e:
                self.status_label.config(text=f"Error loading mask: {str(e)}")

    def update_plot(self):
        #draws the current design and mask on the matplotlib plot
        #analogy: like painting the current state of the canvas after each brush stroke
        if not hasattr(self, 'ax') or self.ax is None:
            return #plot components not initialized (shouldn't happen)
        self.ax.clear()
        x_reshaped = self.x.reshape((self.Ny, self.Nx))
        self.ax.imshow(x_reshaped, cmap='inferno', origin='lower', vmin=0, vmax=1)
        if self.mask is not None:
            #overlay mask in semi-transparent white
            self.ax.imshow(self.mask, cmap='gray', alpha=0.3, origin='lower')
        self.ax.set_title("Electrode Density")
        self.canvas.draw()

    def reset(self):
        #resets the design: if mask is loaded, seeds from mask; otherwise, single center seed
        #analogy: like wiping the canvas and starting with a single paint dot or a stencil
        if self.mask is not None:
            #reset to seed points in mask
            self.x = np.zeros_like(self.mask.flatten())
            seed_points = np.where(self.mask.flatten() > 0.5)[0]
            if len(seed_points) > 0:
                self.x[seed_points] = 1.0
        else:
            #reset to single center point
            self.x = np.zeros(self.Nx * self.Ny)
            center_idx = (self.Ny // 2) * self.Nx + (self.Nx // 2)
            self.x[center_idx] = 1.0
        self.update_plot()

    def run_optimization(self):
        #main optimization loop: runs when you click 'run optimization'
        #analogy: like growing a crystal, step by step, while showing progress
        try:
            max_iter = int(self.max_iter_entry.get())
        except ValueError:
            max_iter = 100 #if entry is invalid, use default
            self.max_iter_entry.delete(0, tk.END)
            self.max_iter_entry.insert(0, str(max_iter))
            print("Warning: Invalid max_iter input, using default 100")

        penal_val = self.penal.get()
        vol_target = self.vol_frac_target.get()
        rmin_val = self.r_min.get()
        print(f"Using GUI params: max_iter={max_iter}, penal={penal_val:.2f}, vol_target={vol_target:.3f}, rmin={rmin_val:.4f}")

        x = self.x.copy() #start with the current design (from reset or mask)
        
        self.status_label.config(text="Starting optimization...")
        self.root.update_idletasks() #show status right away

        for it in range(max_iter):
            move_limit = 0.015 #step size for design update
            
            U = newton_solver(x, penal_val) #solve for voltages
            dPdx = sensitivity_analysis(U, x, penal_val) #get sensitivities
            x_filtered = density_filter(x, rmin_val) #smooth the design
            x_updated = update_design(x_filtered, dPdx, vol_target, move=move_limit) #update electrode pattern
            
            #apply mask if present (blocks growth in masked regions)
            if self.mask is not None:
                x = x_updated * self.mask.flatten()
            else:
                x = x_updated

            #update plot every 5 steps (and last step)
            if it % 5 == 0 or it == max_iter - 1:
                self.x = x #update gui's design variable
                #flicker effect: toggles a 5x5 square in the corner so you can see it updating
                sz = 5
                if self.flicker_state:
                    self.x[:sz*sz] = 1.0
                else:
                    self.x[:sz*sz] = 0.0
                self.flicker_state = not self.flicker_state
                self.status_label.config(text=f"Iteration {it+1}/{max_iter}")
                self.update_plot()
                self.root.update() #process gui events
                self.root.update_idletasks() #extra update for smoothness

        self.status_label.config(text="Optimization Complete")

root = tk.Tk()
app = TO_GUI(root)
root.mainloop()
