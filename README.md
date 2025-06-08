# solar-front-TO
The goal: generate optimized front electrode topology for any free-from shape
  Currently... (1)is able to convert a user-inputed image into a mask, (2)have the basic toolkit for simulating, solving and optimizing the electrical properties of a 2D solar cell using Poisson's equation (which describes the electrical field and potential within a region)

  *when running, you can work on something else, please be patient! it takes a while to generate and run its iterations

**Repo Contents:**
  image_masker.py
    prompts user to input an image and gives parameters for the user to convert the image into a mask. I've tried a few, though some work better than others. For trial, you can save this image to your laptop then upload to the GUI: 
<img width="383" alt="Screenshot 2025-04-25 at 11 20 45 PM" src="https://github.com/user-attachments/assets/14bd6193-1f66-40d5-8a5f-c7b08717630c" /> or copy into your browser:
https://www.google.com/search?vsrid=CMWRiduP377u1gEQAhgBIiQzYzM0NjU2ZC1kZTYyLTRiNGYtYjczOS1kMThlNGQ1ZTMzMGY4qYKp3LTgjQM&vsint=CAIqDAoCCAcSAggKGAEgATojChYNAAAAPxUAAAA_HQAAgD8lAACAPzABEJ4HGJQHJQAAgD8&udm=26&lns_mode=un&source=lns.web.gisbubb&vsdim=926,916&gsessionid=lb4dtr3lyAgkRL3zdOG4ekAnA28W9YsgTrbxs4XTpzUei0tbyO5l1w&lsessionid=XpwS3ytQvlmUC9hQ11kRNNXdzF-6jGTzigPY0BS05tXWCW_dmb86BA&lns_surface=26&authuser=0&lns_vfs=e&qsubts=1749337151647&biw=751&bih=732&hl=en#vhid=Mlsq6r6gjeYq9M&vssid=mosaic 

  you should click on the fill option. then SAVE the output/mask to your laptop where you will be able to find it again

  mesh_viewer.py
    INPUT: Mask (generated from image_masker.py)
    is there in case you want to check how it will interpret the mask. It will give you the designable and void regions.

  poisson_solver.py
    INPUT: Mask (generated from image_masker.py)
    Using the Poisson equation, visualizes the potential field with a gradient map. 
    Note: the reason why you have to select an anode and a cathode (2 electrodes), is because typically the cell's anode and cathode are stacked, that's how it collects sunlight and convert it into. Eg. the anode is the bottom metal contact, however, for the simulation using the Poisson equation, if I don't define the two points, the map that it generates will be meaningless becuase potential will be 0/1 across the entire region. This was an earlier iteration in Phase II so it is not accurate, later on (the next file) it automatically sets the busbar to the edge/border of the mask which is what allows the dendrites(electrodes)to generate

  mask_selector.py
    INPUT: Mask (generated from image_masker.py)
    *the code itself has some explanations. The equations used are based off of this research paper: https://arxiv.org/pdf/2104.04017 
    it does not work. You cannot actually see the optimization but I think all of the necessary structures are in there or at least the majority of the mathematical models are.
    There is another version coming... 



**In the future**
  1. busbars need to be at mask border
  2. complete optimization
  3. optimize vertical component/light-trapping area https://www.nature.com/articles/srep01025
  4. add CNN for faster generation and better generation
