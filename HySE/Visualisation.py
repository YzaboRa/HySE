import numpy as np
import matplotlib.pyplot as plt
import cv2  # New dependency for robust Sobel
from skimage import feature, color, exposure
from skimage.restoration import inpaint
from scipy.ndimage import binary_dilation, gaussian_filter, median_filter

### N.A. Functions written with Gemini and Claude


def visualize_edge_density_overlay(hypercube, method='sobel', 
								   overlay_color=(1.0, 0.0, 0.0), # RGB for Red
								   sobel_ksize=5, 
								   canny_sigma=2.0,
								   cap_intensity=0.5, raw_threshold=0.1, 
								   reflection_mask=None, nframe=0, 
								   mask_dilation_kernel=0, 
								   denoise_sigma=0.0, 
								   denoise_method='gaussian',
								   mask_inpaint_method='zero_fill', 
								   min_gradient_threshold=0.0,
								   opacity_gain=1.0): # New param to boost visibility
	"""
	Overlays accumulated edges from all frames onto a single monochrome context frame.
	Stable edges appear solid; transient/moving edges appear faint.

	Parameters:
	-----------
	overlay_color : tuple
		RGB tuple (0-1) for the edge color. Default is Red (1.0, 0.0, 0.0).
	opacity_gain : float
		Multiplier for the overlay opacity. 
		- 1.0: Linear (50% overlap = 50% opacity).
		- >1.0: Makes faint edges more visible.
	(Other parameters are identical to the previous function)
	"""
	
	n_frames, height, width = hypercube.shape
	hypercube = np.nan_to_num(hypercube, nan=0.0)

	# --- 1. Prepare Background (Context Frame) ---
	bg_frame = hypercube[nframe].copy()
	bg_norm = (bg_frame - bg_frame.min()) / (bg_frame.max() - bg_frame.min() + 1e-8)
	
	# Process the background frame (Inpaint/Mask) for display cleanliness
	if reflection_mask is not None:
		bg_mask = reflection_mask[nframe].astype(bool) if len(reflection_mask.shape) == 3 else reflection_mask.astype(bool)
		if mask_dilation_kernel > 0:
			bg_mask = binary_dilation(bg_mask, structure=np.ones((mask_dilation_kernel, mask_dilation_kernel)))

		if mask_inpaint_method == 'inpaint':
			bg_norm = inpaint.inpaint_biharmonic(bg_norm, bg_mask, channel_axis=None)
		else:
			bg_norm[bg_mask] = 0.0 # Just zero out for background display

	# Convert grayscale background to RGB so we can overlay color
	background_rgb = np.stack([bg_norm]*3, axis=-1)

	# --- 2. Accumulate Edges ---
	edge_accumulator = np.zeros((height, width), dtype=np.float32)

	# Helper for denoising
	def apply_denoise(img, strength, method):
		if strength <= 0: return img
		if method == 'gaussian': return gaussian_filter(img, sigma=strength)
		elif method == 'median': return median_filter(img, size=int(max(2, strength)))
		return img

	for i in range(n_frames):
		frame = hypercube[i]
		frame_norm = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
		working_frame = frame_norm.copy()
		
		# Masking / Inpainting
		if reflection_mask is not None:
			mask = reflection_mask[i].astype(bool) if len(reflection_mask.shape) == 3 else reflection_mask.astype(bool)
			if mask_dilation_kernel > 0:
				mask = binary_dilation(mask, structure=np.ones((mask_dilation_kernel, mask_dilation_kernel)))
			
			if method in ['canny', 'sobel']:
				if mask_inpaint_method == 'inpaint':
					working_frame = inpaint.inpaint_biharmonic(working_frame, mask, channel_axis=None)
				else: 
					working_frame[mask] = 0.0
		
		# Filtering
		if method in ['canny', 'sobel']:
			working_frame = apply_denoise(working_frame, denoise_sigma, denoise_method)
			
			if method == 'canny':
				feature_map = feature.canny(working_frame, sigma=canny_sigma).astype(np.float32)
			elif method == 'sobel':
				grad_x = cv2.Sobel(working_frame, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
				grad_y = cv2.Sobel(working_frame, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
				grad_mag = cv2.magnitude(grad_x, grad_y)
				feature_map = exposure.rescale_intensity(grad_mag, in_range='image')

			if min_gradient_threshold > 0.0:
				feature_map[feature_map < min_gradient_threshold] = 0.0
			
			# Cap and Normalize per frame
			feature_map = np.clip(feature_map, 0.0, cap_intensity)
			if cap_intensity < 1.0 and feature_map.max() > 0:
				feature_map = feature_map / cap_intensity  
				
		elif method == 'raw_thresholded':
			feature_map = (working_frame < raw_threshold).astype(np.float32)
			if reflection_mask is not None:
				# Re-mask logic omitted for brevity, similar to previous func
				pass 

		# Accumulate
		edge_accumulator += feature_map

	# --- 3. Create Overlay ---
	
	# Normalize accumulator: 
	# 0.0 means no edge ever. 
	# 1.0 means an edge was present in EVERY frame (if we divide by n_frames).
	# Using n_frames ensures that "Solid Red" = "Totally Stable".
	alpha_map = edge_accumulator / n_frames
	
	# Apply Gain (make faint edges more visible if desired)
	alpha_map = alpha_map * opacity_gain
	alpha_map = np.clip(alpha_map, 0.0, 1.0)
	
	# Create the solid color layer
	color_layer = np.zeros((height, width, 3), dtype=np.float32)
	color_layer[:] = overlay_color
	
	# Expand alpha to 3 channels for broadcasting
	alpha_3ch = alpha_map[..., np.newaxis]

	# --- 4. Alpha Blending ---
	# Formula: Result = Background * (1 - Alpha) + Overlay * Alpha
	final_composite = background_rgb * (1.0 - alpha_3ch) + color_layer * alpha_3ch
	final_composite = np.clip(final_composite, 0, 1)

	# --- Plotting ---
#     plt.close()
	fig, ax = plt.subplots(figsize=(6, 4))
	ax.imshow(final_composite)
#     ax.set_title(f"Edge Density Overlay (Method: {method})\nSolid Color = Stable Feature across {n_frames} frames")
	ax.axis('off')
	
	plt.tight_layout()
	plt.show()
	return final_composite



def visualize_hypercube_movement(hypercube, method='sobel', 
								sobel_ksize=5,  # Integer kernel size (3, 5, 7, etc.)
								canny_sigma=2.0, # Separated Canny sigma
								cap_intensity=0.5, raw_threshold=0.1, 
								reflection_mask=None, nframe=0, 
								mask_dilation_kernel=0, 
								denoise_sigma=0.0, 
								denoise_method='gaussian',
								mask_inpaint_method='zero_fill', 
								display_power_gamma=1.0,
								min_gradient_threshold=0.0,
								white_overlay=10):
	"""
	Visualizes movement in a hypercube using OpenCV for robust gradient calculation.

	Parameters:
	-----------
	hypercube : numpy.ndarray
		Input array (N, Y, X).
	method : str
		'sobel', 'canny', or 'raw_thresholded'.
	sobel_ksize : int
		Kernel size for OpenCV Sobel. Must be an odd integer (1, 3, 5, 7, ...).
		- 3: Standard, sharp edges, susceptible to noise.
		- 5: Smooths noise, highlights medium features. (RECOMMENDED for your data)
		- 7: Very smooth, ignores fine texture completely.
	canny_sigma : float
		Sigma for Canny edge detection (if method='canny').
	cap_intensity : float
		Clips high-intensity features (0.0 to 1.0).
	raw_threshold : float
		Threshold for 'raw_thresholded'.
	reflection_mask : numpy.ndarray
		Binary mask (2D or 3D).
	nframe : int
		Context frame index.
	mask_dilation_kernel : int
		Size of dilation for reflection mask.
	denoise_sigma : float
		Pre-filtering strength.
	denoise_method : str
		'gaussian' or 'median'.
	mask_inpaint_method : str
		'zero_fill' or 'inpaint'.
	display_power_gamma : float
		Power law (val^gamma) to darken background noise.
	min_gradient_threshold : float
		Hard cutoff (0.0 to 1.0). Signals below this are set to 0.

	Returns:
	--------
	fig : matplotlib.figure.Figure
	"""
	
	n_frames, height, width = hypercube.shape
	hypercube = np.nan_to_num(hypercube, nan=0.0)

	# --- Validation ---
	if reflection_mask is not None:
		if len(reflection_mask.shape) == 3:
			if reflection_mask.shape != hypercube.shape: raise ValueError("Mask shape mismatch.")
		elif len(reflection_mask.shape) == 2:
			 if reflection_mask.shape != (height, width): raise ValueError("Mask shape mismatch.")
	
	# Ensure sobel_ksize is odd
	if sobel_ksize % 2 == 0:
		sobel_ksize += 1
		print(f"Warning: sobel_ksize must be odd. Automatically adjusted to {sobel_ksize}.")

	composite_image = np.zeros((height, width, 3), dtype=np.float32)
#     hues = np.linspace(0, 1, n_frames + 1)[:-1]
	golden_ratio_conjugate = 0.618033988749895
	hues = (np.arange(n_frames) * golden_ratio_conjugate) % 1.0
	
	def apply_denoise(img, strength, method):
		if strength <= 0: return img
		if method == 'gaussian': return gaussian_filter(img, sigma=strength)
		elif method == 'median': return median_filter(img, size=int(max(2, strength)))
		return img

	# --- Context Frame (Left Subplot) ---
	ref_frame = hypercube[nframe].copy()
	ref_frame_norm = (ref_frame - ref_frame.min()) / (ref_frame.max() - ref_frame.min() + 1e-8)
	ref_frame_processed = ref_frame_norm.copy()
	mask_label = "Raw Normalized"
	
	if reflection_mask is not None and method in ['canny', 'sobel']:
		ref_mask = reflection_mask[nframe].astype(bool) if len(reflection_mask.shape) == 3 else reflection_mask.astype(bool)
		if mask_dilation_kernel > 0:
			ref_mask = binary_dilation(ref_mask, structure=np.ones((mask_dilation_kernel, mask_dilation_kernel)))

		if mask_inpaint_method == 'inpaint':
			ref_frame_processed = inpaint.inpaint_biharmonic(ref_frame_norm, ref_mask, channel_axis=None)
			mask_label = "Inpainted"
		else: 
			ref_frame_processed[ref_mask] = 0.0
			mask_label = "Zero-Fill"
		
		if denoise_sigma > 0.0:
			ref_frame_processed = apply_denoise(ref_frame_processed, denoise_sigma, denoise_method)

	overlap_count = np.zeros((height, width), dtype=np.int32)
	# --- Main Loop ---
	for i in range(n_frames):
		frame = hypercube[i]
		frame_norm = (frame - frame.min()) / (frame.max() - frame.min() + 1e-8)
		working_frame = frame_norm.copy()
		
		# 1. Masking / Inpainting
		if reflection_mask is not None:
			mask = reflection_mask[i].astype(bool) if len(reflection_mask.shape) == 3 else reflection_mask.astype(bool)
			if mask_dilation_kernel > 0:
				mask = binary_dilation(mask, structure=np.ones((mask_dilation_kernel, mask_dilation_kernel)))
			
			if method in ['canny', 'sobel']:
				if mask_inpaint_method == 'inpaint':
					working_frame = inpaint.inpaint_biharmonic(working_frame, mask, channel_axis=None)
				else: 
					working_frame[mask] = 0.0
		
		# 2. Filtering
		if method in ['canny', 'sobel']:
			# A. Pre-Denoise
			working_frame = apply_denoise(working_frame, denoise_sigma, denoise_method)
			
			# B. Edge Detection
			if method == 'canny':
				feature_map = feature.canny(working_frame, sigma=canny_sigma).astype(np.float32)
			
			elif method == 'sobel':

				# CV_64F handles negative gradients correctly (important!)
				grad_x = cv2.Sobel(working_frame, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
				grad_y = cv2.Sobel(working_frame, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
				
				# Calculate magnitude
				grad_mag = cv2.magnitude(grad_x, grad_y)
				
				# Rescale to 0-1 for display
				feature_map = exposure.rescale_intensity(grad_mag, in_range='image')

			# C. Post-Processing
			if min_gradient_threshold > 0.0:
				feature_map[feature_map < min_gradient_threshold] = 0.0
			
			feature_map = np.clip(feature_map, 0.0, cap_intensity)
			if cap_intensity < 1.0 and feature_map.max() > 0:
				feature_map = feature_map / cap_intensity  
			
			if display_power_gamma != 1.0:
				feature_map = feature_map ** display_power_gamma
				
		elif method == 'raw_thresholded':
			feature_map = (working_frame < raw_threshold).astype(np.float32)
			if reflection_mask is not None:
				if mask_dilation_kernel > 0:
					mask = binary_dilation(mask, structure=np.ones((mask_dilation_kernel, mask_dilation_kernel)))
				feature_map[mask] = 0.0 

		# 3. Coloring
		current_color_hsv = np.array([[[hues[i], 1.0, 1.0]]])
		current_color_rgb = color.hsv2rgb(current_color_hsv)
		
		colored_layer = feature_map[..., np.newaxis] * current_color_rgb
		overlap_count += (feature_map > 0).astype(np.int32)
		composite_image += colored_layer

	composite_image[overlap_count >= white_overlay] = [1.0, 1.0, 1.0] 
	final_display = np.clip(composite_image, 0, 1)
#     final_display = np.clip(composite_image, 0, 1)
	
	# --- Plotting ---
#     plt.close()
	fig, axes = plt.subplots(1, 2, figsize=(8, 5))

	axes[0].imshow(ref_frame_processed, cmap='gray')
	axes[0].set_title(f"Context Frame (Index: {nframe})\n{mask_label}")
	axes[0].axis('off')

	axes[1].imshow(final_display)
	axes[1].set_title(f"Movement Vis\nSobel Kernel: {sobel_ksize}x{sobel_ksize}, Thresh: {min_gradient_threshold}")
	axes[1].axis('off')
	
	plt.tight_layout()
	return final_display, ref_frame_processed








#### ------------------------------------------------
#### ------------------------------------------------
#### ------------------------------------------------
#### ------------------------------------------------


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RadioButtons, Slider
from matplotlib.path import Path
import io
import matplotlib as mpl
import glob


def GetPath(DataPath, Name, extension='.npz'):
	String = f'{DataPath}*{Name}*{extension}'
	paths = glob.glob(String)
	if len(paths)==1:
		output = paths[0]
	else:
		print(f'ATTENTION: There are {len(paths)} files containing {Name} at this location:')
		for i in range(0,len(paths)):
			print(f'   {paths[i]}')
			print(f'{String}')
		output = None
	return output


def LoadBiopsiesData(SavingPath, Keys=None):
	"""
	Simplified function to load biospies data

	Handles different keys.

		Default:
		['Coordinates', 'AvgSpectra_Green', 'AllSpectra_Green', 'AvgSpectra_Blue', 'AllSpectra_Blue', 'Pathology']

		Accepts:
		['Coordinates', 'AvgSpectra', 'AllSpectra']

		or any key input through "Keys=" input. It must be a list of strings.

	Outputs:
		Output, 
			List of length len(Keys), containing the data for each key (array)

		KeysOutput
			List of length len(Keys) containing the labels for each key (string)



	"""
	loaded_archive = np.load(SavingPath, allow_pickle=True)
	BiopsiesLocation = loaded_archive['ROIs'].tolist()
	Output = []
	if Keys is not None:
		for k in Keys:
			values = [roi[k] for roi in BiopsiesLocation]
			Output.append(values)
			KeysOutput = Keys
		
	else: ## If not proving keys, use standard
		Coordinates = [roi['Coordinates'] for roi in BiopsiesLocation]
		Output.append(Coordinates)
		try:
			Keys_All = ['AvgSpectra_Green', 'AllSpectra_Green', 'AvgSpectra_Blue', 'AllSpectra_Blue', 'Pathology']
			print(f'Looking for keys: \n   \'Coordinates\', and {Keys_All}')
			for k in Keys_All:
				values = [roi[k] for roi in BiopsiesLocation]
				Output.append(values)
				KeysOutput = Keys_All

		except KeyError:
			Keys_Old = ['AvgSpectra', 'AllSpectra']
			print(f'New keys not existing. Using old ones: \n   \'Coordinates\', and {Keys_Old}')
			for k in Keys_Old:
				values = [roi[k] for roi in BiopsiesLocation]
				Output.append(values)
				KeysOutput = Keys_Old
	return Output, KeysOutput


def SaveBiopsiesData(SavingPath, Coordinates, AvgSpectra_Green, AllSpectra_Green, AvgSpectra_Blue, AllSpectra_Blue, Pathology):
	"""
	Takes ROI coordinates and spectra information to save.

	To open:

	loaded_archive = np.load(SavingPath, allow_pickle=True)
	BiopsiesLocation = loaded_archive['ROIs'].tolist()
	# Extract coordinates 
	SavedCoords = [roi['Coordinates'] for roi in BiopsiesLocation]

	etc.


	"""
	ROIs = [
		dict(Coordinates=c, AvgSpectra_Green=ag, AllSpectra_Green=sg, AvgSpectra_Blue=ab, AllSpectra_Blue=sb, Pathology=p)
		for c, ag, sg, ab, sb, p in zip(
			Coordinates,
			AvgSpectra_Green,
			AllSpectra_Green,
			AvgSpectra_Blue,
			AllSpectra_Blue,
			Pathology,
		)
	]
	np.savez_compressed(SavingPath, ROIs=np.array(ROIs, dtype=object))
	print(f'Saved data at:\n{SavingPath}')


def GetBiopsyLocations(RegisteredHypercube, initial_rois=None, **kwargs):
	"""
	GUI to draw ROIs on a hypercube frame (Shape: Frames, Y, X).
	
	Parameters:
	-----------
	RegisteredHypercube : numpy.ndarray
		3D array of shape (Frames, Height, Width).
	initial_rois : list, optional
		List of (N, 2) arrays containing (x, y) vertices for pre-existing ROIs.
	**kwargs :
		display_frame_idx (int): Index of the frame to display initially.
		figsize (tuple): Optional figure size.
		colours (list, optional): List of matplotlib colours (e.g. 'tab:green', 'tab:red'),
			one per entry in `initial_rois`, used to draw/fill each pre-loaded ROI instead of
			the default auto-cycled colour. Must be the same length as `initial_rois`. Has no
			effect on ROIs drawn interactively after the GUI opens (those still auto-cycle).

	Returns:
	--------
	ROI_coordinates : list
		List of (N, 2) arrays containing (x, y) vertices for each ROI.
	ROI_AvgSpectra : list
		List of 1D arrays containing the average spectrum (across frames) for each ROI.
	ROI_AllSpectra : list
		List of 2D arrays (M_pixels, Frames) containing spectra for all pixels.
	Preview_Image : numpy.ndarray
		RGB array of the clean image (current frame) with ROIs and labels.
	"""

	class BiopsyPicker:
		def __init__(self, cube, initial_rois=None, display_frame_idx=None, user_figsize=None, colours=None):

			if 'left' in mpl.rcParams['keymap.back']:
				mpl.rcParams['keymap.back'].remove('left')
			if 'right' in mpl.rcParams['keymap.forward']:
				mpl.rcParams['keymap.forward'].remove('right')
				
			# --- FIX 2: Unbind default 'k' and 'l' log-scaling shortcuts ---
			if 'k' in mpl.rcParams['keymap.xscale']:
				mpl.rcParams['keymap.xscale'].remove('k')
			if 'l' in mpl.rcParams['keymap.yscale']:
				mpl.rcParams['keymap.yscale'].remove('l')

			# --- FIX 3: Unbind default 'q' quit shortcut to prevent accidental closes ---
			if 'q' in mpl.rcParams['keymap.quit']:
				mpl.rcParams['keymap.quit'].remove('q')


			self.cube = cube
			self.frames, self.h, self.w = cube.shape
			
			# --- AUTO-SIZE FIGURE ---
			if user_figsize is None:
				img_ar = self.w / self.h
				fig_h = 9
				fig_w = max(6, min(14, fig_h * img_ar))
				figsize = (fig_w + 1.3, fig_h + 1.5) 
			else:
				figsize = user_figsize

			# Frame selection
			if display_frame_idx is None:
				self.display_frame_idx = 0  # Default to first frame as requested
			else:
				self.display_frame_idx = int(display_frame_idx)
			
			self.image_data = self.cube[self.display_frame_idx, :, :]
			
			# ROI Storage
			self.rois = [] 
			self.current_roi_verts = []
			
			# State
			self.state = 'IDLE' 
			self.active_roi_idx = -1
			self.active_vertex_idx = -1
			self.drag_active = False
			self.show_rois = True
			
			# Colors
			self.cmap_cycle = plt.cm.Set1.colors #plt.cm.tab10.colors 
			self.alpha_draw = 0.25
			self.alpha_done = 0.5

			# Figure Setup
			self.fig = plt.figure(figsize=figsize)
			gs = self.fig.add_gridspec(1, 1, left=0.05, right=0.9, bottom=0.2, top=0.89)
			self.ax = self.fig.add_subplot(gs[0, 0])
			self.ax.set_facecolor('xkcd:light rose')
			
			# Plot Image
			self.img_norm = self._normalize(self.image_data)
			self.im_obj = self.ax.imshow(self.img_norm, cmap='gray', vmin=0)
			self.ax.set_title(f"Frame {self.display_frame_idx} / {self.frames}\n"
							  f"'r': New ROI | 'z': Undo | 'q': Delete | 'p': Show/Hide\n"
							  f"Double-click ROI to Edit | 'e' Exit drawing\n")

			# Temporary artist for drawing active lines
			self.line_active, = self.ax.plot([], [], linestyle='--', marker='o', color='white', lw=1.5, animated=False)
			
			# --- WIDGETS ---
			ax_color = 'lightgoldenrodyellow'
			
			# Sliders
			ax_min = self.fig.add_axes([0.18, 0.1, 0.23, 0.03], facecolor=ax_color)
			ax_max = self.fig.add_axes([0.18, 0.06, 0.23, 0.03], facecolor=ax_color)
			self.slider_min = Slider(ax_min, 'Min', 0.0, 1.0, valinit=0.0)
			self.slider_max = Slider(ax_max, 'Max', 0.0, 1.0, valinit=1.0)
			self.slider_min.on_changed(self.update_clim)
			self.slider_max.on_changed(self.update_clim)

			ax_frame = self.fig.add_axes([0.50, 0.08, 0.23, 0.03], facecolor=ax_color)
			self.slider_frame = Slider(ax_frame, 'Frame', 0, self.frames - 1, valinit=self.display_frame_idx, valstep=1)
			self.slider_frame.on_changed(self.update_frame)

			ax_radio = self.fig.add_axes([0.8, 0.4, 0.08, 0.15], facecolor=ax_color)
			self.radio = RadioButtons(ax_radio, ('gray', 'magma', 'viridis'))
			self.radio.on_clicked(self.update_cmap)
			
			# Event Connections
			self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
			self.cid_move = self.fig.canvas.mpl_connect('motion_notify_event', self.on_move)
			self.cid_release = self.fig.canvas.mpl_connect('button_release_event', self.on_release)
			self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
			self.cid_close = self.fig.canvas.mpl_connect('close_event', self.on_close)


			if initial_rois is not None:
				if colours is not None and len(colours) != len(initial_rois):
					raise ValueError(
						f"'colours' has {len(colours)} entries but 'initial_rois' has {len(initial_rois)} "
						f"entries; they must be the same length."
					)
				for i, verts in enumerate(initial_rois):
					self.current_roi_verts = list(verts)
					roi_color = colours[i] if colours is not None else None
					self._finish_roi(color=roi_color)

		def _normalize(self, img):
			p2, p98 = np.nanpercentile(img, (0, 100))
			denom = p98 - p2
			return np.clip((img - p2) / (denom if denom != 0 else 1), 0, 1)

		def update_clim(self, val):
			vmin = self.slider_min.val
			vmax = self.slider_max.val
			if vmin >= vmax: vmax = vmin + 0.01
			self.im_obj.set_clim(vmin, vmax)
			self.fig.canvas.draw_idle()

		def update_frame(self, val):
			idx = int(self.slider_frame.val)
			self.display_frame_idx = idx
			self.image_data = self.cube[idx, :, :]
			self.img_norm = self._normalize(self.image_data)
			self.im_obj.set_data(self.img_norm)
			self.ax.set_title(f"Frame {self.display_frame_idx} / {self.frames}\n"
							  f"'r': New | 'z': Undo | 'q': Delete | 'p': Show/Hide\n"
							  f"Double-click ROI to Edit | 'e': to exit ROI")
			self.fig.canvas.draw_idle()

		def update_cmap(self, label):
			self.im_obj.set_cmap(label)
			self.fig.canvas.draw_idle()

		def _get_next_color(self):
			idx = len(self.rois) % len(self.cmap_cycle)
			return self.cmap_cycle[idx]

		def _redraw_active_line(self):
			if not self.current_roi_verts or not self.show_rois:
				self.line_active.set_data([], [])
			else:
				xs, ys = zip(*self.current_roi_verts)
				self.line_active.set_data(xs, ys)
				self.line_active.set_color(self._get_next_color())
			self.fig.canvas.draw_idle()

		def _finish_roi(self, color=None):
			if len(self.current_roi_verts) < 3:
				print("ROI needs at least 3 points.")
				self.current_roi_verts = []
				self._redraw_active_line()
				self.state = 'IDLE'
				return

			if color is None:
				color = self._get_next_color()
			poly = plt.Polygon(self.current_roi_verts, closed=True, 
							   facecolor=color, edgecolor=color, alpha=self.alpha_done, label=f"ROI {len(self.rois)+1}")
			self.ax.add_patch(poly)
			
			cx = np.nanmean([v[0] for v in self.current_roi_verts])
			cy = np.nanmean([v[1] for v in self.current_roi_verts])
			text = self.ax.text(cx, cy, str(len(self.rois) + 1), color='white', 
								weight='bold', ha='center', va='center', fontsize=10)
			
			poly.set_visible(self.show_rois)
			text.set_visible(self.show_rois)

			self.rois.append({
				'vertices': list(self.current_roi_verts),
				'color': color,
				'artist_poly': poly,
				'artist_text': text,
				'artist_markers': None 
			})
			
			self.current_roi_verts = []
			self._redraw_active_line()
			self.state = 'IDLE'
			self.fig.canvas.draw_idle()

		def _update_roi_visuals(self, roi_idx):
			if roi_idx < 0 or roi_idx >= len(self.rois): return
			roi = self.rois[roi_idx]
			roi['artist_poly'].set_xy(roi['vertices'])
			cx = np.nanmean([v[0] for v in roi['vertices']])
			cy = np.nanmean([v[1] for v in roi['vertices']])
			roi['artist_text'].set_position((cx, cy))
			if roi['artist_markers']:
				xs, ys = zip(*roi['vertices'])
				roi['artist_markers'].set_data(xs, ys)

		def _delete_roi(self, idx):
			# 1. Reset active index first to stop on_move from crashing
			self.active_roi_idx = -1 
			self.drag_active = False
			self.state = 'IDLE'

			if 0 <= idx < len(self.rois):
				try:
					roi = self.rois.pop(idx)
					roi['artist_poly'].remove()
					roi['artist_text'].remove()
					if roi['artist_markers']:
						roi['artist_markers'].remove()
					
					# Renumber
					for i, r in enumerate(self.rois):
						r['artist_text'].set_text(str(i+1))
					
					print(f"Deleted ROI {idx+1}")
					self.fig.canvas.draw_idle()
				except Exception as e:
					print(f"Error deleting ROI: {e}")
			else:
				print("No valid ROI to delete.")

		def on_key(self, event):
			if event.key == 'r':
				if self.state == 'IDLE':
					self.state = 'DRAWING'
					self.current_roi_verts = []
					print("Mode: DRAWING")
				elif self.state == 'EDITING':
					self._exit_edit_mode()
					self.state = 'DRAWING'
					self.current_roi_verts = []

			elif event.key == 'z':
				if self.state == 'DRAWING' and self.current_roi_verts:
					self.current_roi_verts.pop()
					self._redraw_active_line()
				elif self.state == 'IDLE' and self.rois:
					self._delete_roi(len(self.rois)-1)

			elif event.key == 'q':
				if self.state == 'EDITING':
					# We have a specific target
					idx_to_del = self.active_roi_idx
					# self._exit_edit_mode() # Clear markers first
					self._delete_roi(idx_to_del)
				else:
					# Search for ROI under mouse
					if event.inaxes == self.ax and event.xdata is not None:
						found = -1
						for i, roi in enumerate(self.rois):
							path = Path(roi['vertices'])
							if path.contains_point((event.xdata, event.ydata)):
								found = i
								break
						if found != -1:
							self._delete_roi(found)

			elif event.key == 'p':
				self.show_rois = not self.show_rois
				for roi in self.rois:
					roi['artist_poly'].set_visible(self.show_rois)
					roi['artist_text'].set_visible(self.show_rois)
					if roi['artist_markers']:
						roi['artist_markers'].set_visible(self.show_rois)
				self.line_active.set_visible(self.show_rois)
				self.fig.canvas.draw_idle()

			elif event.key == 'e':
				self._exit_edit_mode()

		def on_click(self, event):
			if event.inaxes != self.ax or event.button != 1: return
			if self.fig.canvas.manager.toolbar.mode != '': return

			if event.dblclick:
				if self.state == 'IDLE':
					for i, roi in enumerate(self.rois):
						path = Path(roi['vertices'])
						if path.contains_point((event.xdata, event.ydata)):
							self._enter_edit_mode(i)
							return
							
				elif self.state == 'EDITING':
					# Editing logic (Delete point or Add Point)
					roi = self.rois[self.active_roi_idx]
					verts = roi['vertices']
					curr_pt = np.array([event.xdata, event.ydata])
					
					dist = np.linalg.norm(np.array(verts) - curr_pt, axis=1)
					if np.min(dist) < 10: 
						idx_to_remove = np.argmin(dist)
						if len(verts) > 3:
							verts.pop(idx_to_remove)
							self._update_roi_visuals(self.active_roi_idx)
							self.fig.canvas.draw_idle()
						return

					# Add point logic
					best_dist = float('inf')
					insert_idx = -1
					for i in range(len(verts)):
						p1 = np.array(verts[i])
						p2 = np.array(verts[(i + 1) % len(verts)])
						vec_line = p2 - p1
						vec_pt = curr_pt - p1
						line_len_sq = np.dot(vec_line, vec_line)
						if line_len_sq == 0: continue
						t = np.clip(np.dot(vec_pt, vec_line) / line_len_sq, 0, 1)
						proj = p1 + t * vec_line
						d = np.linalg.norm(curr_pt - proj)
						if d < best_dist:
							best_dist = d
							insert_idx = i + 1
					
					verts.insert(insert_idx, (event.xdata, event.ydata))
					self._update_roi_visuals(self.active_roi_idx)
					self.fig.canvas.draw_idle()
				return

			if self.state == 'DRAWING':
				if len(self.current_roi_verts) > 2:
					start_pt = np.array(self.current_roi_verts[0])
					curr_pt = np.array([event.xdata, event.ydata])
					xlim = self.ax.get_xlim()
					tol = (xlim[1] - xlim[0]) * 0.02 
					if np.linalg.norm(start_pt - curr_pt) < tol:
						self._finish_roi()
						return

				self.current_roi_verts.append((event.xdata, event.ydata))
				self._redraw_active_line()

			elif self.state == 'EDITING':
				roi = self.rois[self.active_roi_idx]
				verts = roi['vertices']
				dist = np.linalg.norm(np.array(verts) - np.array([event.xdata, event.ydata]), axis=1)
				xlim = self.ax.get_xlim()
				tol = (xlim[1] - xlim[0]) * 0.02
				if np.min(dist) < tol:
					self.active_vertex_idx = np.argmin(dist)
					self.drag_active = True

		def on_move(self, event):
			if event.inaxes != self.ax: return
			# --- FIX: Check active_roi_idx validity ---
			if self.state == 'EDITING' and self.drag_active and self.active_roi_idx != -1:
				try:
					roi = self.rois[self.active_roi_idx]
					roi['vertices'][self.active_vertex_idx] = (event.xdata, event.ydata)
					self._update_roi_visuals(self.active_roi_idx)
					self.fig.canvas.draw_idle()
				except IndexError:
					# If ROI disappeared, reset state
					self.state = 'IDLE'
					self.drag_active = False

		def on_release(self, event):
			if self.state == 'EDITING':
				self.drag_active = False
				self.active_vertex_idx = -1

		def _enter_edit_mode(self, idx):
			if self.state == 'EDITING' and self.active_roi_idx != idx:
				self._exit_edit_mode()
			self.state = 'EDITING'
			self.active_roi_idx = idx
			roi = self.rois[idx]
			roi['artist_poly'].set_alpha(self.alpha_draw)
			xs, ys = zip(*roi['vertices'])
			line, = self.ax.plot(xs, ys, 'o', color='white', markeredgecolor=roi['color'], markersize=8, zorder=10)
			roi['artist_markers'] = line
			self.fig.canvas.draw_idle()

		def _exit_edit_mode(self):
			if self.active_roi_idx == -1: return
			try:
				roi = self.rois[self.active_roi_idx]
				roi['artist_poly'].set_alpha(self.alpha_done)
				if roi['artist_markers']:
					roi['artist_markers'].remove()
					roi['artist_markers'] = None
			except IndexError:
				pass
			self.state = 'IDLE'
			self.active_roi_idx = -1
			self.drag_active = False
			self.fig.canvas.draw_idle()

		def disconnect(self):
			self.fig.canvas.mpl_disconnect(self.cid_click)
			self.fig.canvas.mpl_disconnect(self.cid_move)
			self.fig.canvas.mpl_disconnect(self.cid_release)
			self.fig.canvas.mpl_disconnect(self.cid_key)
			self.fig.canvas.mpl_disconnect(self.cid_close)

		def on_close(self, event):
			self.disconnect()

		def get_results(self):
			all_coords = [np.array(r['vertices']) for r in self.rois]
			avg_spectra = []
			all_spectra = []

			# --- PREVIEW GENERATION: CLEAN FIGURE ---
			# Create a completely new figure to ensure no GUI elements are present
			# We use the current display settings (frame, normalization)
			h, w = self.h, self.w
			preview_fig = plt.figure(figsize=(10, 10 * (h/w)))
			p_ax = preview_fig.add_axes([0, 0, 1, 1])
			p_ax.axis('off')
			
			# Plot the CURRENTLY DISPLAYED frame (so it matches what the user saw)
			# Use same normalization as GUI
			img_data = self.cube[self.display_frame_idx, :, :]
			p_norm = self._normalize(img_data)
			p_ax.imshow(p_norm, cmap='gray', vmin=self.slider_min.val, vmax=self.slider_max.val)
			
			# Draw ROIs
			for i, r in enumerate(self.rois):
				poly = plt.Polygon(r['vertices'], closed=True, 
								   facecolor=r['color'], edgecolor=r['color'], alpha=0.7)
				p_ax.add_patch(poly)
				cx = np.nanmean([v[0] for v in r['vertices']])
				cy = np.nanmean([v[1] for v in r['vertices']])
				p_ax.text(cx, cy, str(i+1), color='white', weight='bold', ha='center', va='center', fontsize=12)
			
			# Save to buffer
			buf = io.BytesIO()
			preview_fig.savefig(buf, format='png', dpi=100)
			plt.close(preview_fig) # Clean up memory
			buf.seek(0)
			preview_img = plt.imread(buf)

			# --- SPECTRA EXTRACTION ---
			y_indices, x_indices = np.mgrid[:self.h, :self.w]
			points = np.vstack((x_indices.ravel(), y_indices.ravel())).T
			
			for roi in self.rois:
				path = Path(roi['vertices'])
				mask_flat = path.contains_points(points)
				mask = mask_flat.reshape((self.h, self.w))
				pixels = self.cube[:, mask]
				
				if pixels.shape[1] > 0:
					pixels = pixels.T 
					all_spectra.append(pixels)
					avg_spectra.append(np.nanmean(pixels, axis=0))
				else:
					all_spectra.append(np.empty((0, self.frames)))
					avg_spectra.append(np.zeros(self.frames))
					
			return all_coords, avg_spectra, all_spectra, preview_img

	figsize = kwargs.pop('figsize', None)
	picker = BiopsyPicker(RegisteredHypercube, initial_rois, user_figsize=figsize, **kwargs)
	plt.show(block=True)
	return picker.get_results()



