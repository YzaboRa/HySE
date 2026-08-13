"""

Manual Registration (indicating fixed points manually)

N.B. Functions and GUI written with Gemini


"""

import numpy as np
import cv2
import os
from datetime import datetime
from scipy.signal import savgol_filter, find_peaks
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.pyplot as plt
import imageio
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import RectangleSelector
# import SimpleITK as sitk
import time
from tqdm import trange
import inspect
import matplotlib.patheffects as PathEffects

from matplotlib.widgets import Slider, RadioButtons, Button
import scipy.ndimage as ndimage

from PyQt5 import QtWidgets, QtCore, QtGui

matplotlib.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "arial"


import HySE.UserTools
import HySE.Import
import HySE.ManipulateHypercube
import HySE.CoRegistrationTools


PythonEnvironment = get_ipython().__class__.__name__

from ._optional import sitk as _sitk
from skimage.metrics import normalized_mutual_information as nmi 
from scipy.ndimage import gaussian_filter

from PIL import Image
from natsort import natsorted
import glob
import copy
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import tempfile
import pickle
import re
import json
import traceback

import SimpleITK as _sitk





# OriginPosition = 'lower'
OriginPosition = 'upper' ## standard python




class LandmarkPicker:
	"""
	An advanced interactive class to select corresponding points in two images
	with undo, color-coding, enforced point selection, and image controls.
	"""
	def __init__(self, fixed_image, moving_image=None, fixed_points_to_display=None,
				 frame_info=None, warning_message=None, deviation_threshold=150, colourmap='magma'):
		self.fixed_image = fixed_image
		self.moving_image = moving_image
		self.fixed_points = []
		self.moving_points = []
		self.plotted_artists = []
		self.show_text_labels = False 
		self.deviation_threshold = deviation_threshold
		self.warning_messages = []
		
		# --- FIX: Store frame info string for persistent labeling ---
		self.frame_info = str(frame_info) if frame_info is not None else ""

		self.fig = plt.figure(figsize=(16, 9))
		
		# Adjust layout to make room for widgets at the bottom
		gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.1)
		self.fig.subplots_adjust(bottom=0.25, right=0.88)
		
		self.ax_fixed = self.fig.add_subplot(gs[0, 0])
		self.ax_moving = self.fig.add_subplot(gs[0, 1])
		self.axes = [self.ax_fixed, self.ax_moving]
		self.cbar_ax = self.fig.add_axes([0.9, 0.25, 0.02, 0.6])

		self.phase = 'moving' if moving_image is not None else 'fixed'

		# --- DISPLAY IMAGES & STORE REFERENCES ---
		if self.phase == 'fixed':
			self.num_total_points = None
			self.fig.suptitle("PHASE 1: Select landmarks. Use toolbar to Zoom/Pan.\n'z': undo, 'p': labels. Close to finish.", fontsize=14)
			self.fixed_img_norm = self._normalize(self.fixed_image)
			self.im_fixed_obj = self.ax_fixed.imshow(self.fixed_img_norm, cmap=colourmap, origin=OriginPosition)
			
			# Add label to fixed frame title
			lbl = f" | {self.frame_info}" if self.frame_info else ""
			self.ax_fixed.set_title(f'Click to select FIXED points{lbl}')
			self.ax_moving.axis('off')
			self.im_moving_obj = None
		else: # moving phase
			self.fixed_points_to_display = fixed_points_to_display
			self.num_total_points = len(self.fixed_points_to_display)
			
			self.fig.suptitle(f"PHASE 2: Registering Frame\nUse toolbar to Zoom/Pan. 'z': undo, 'p': labels, 'w': warnings. Select {self.num_total_points} points.", fontsize=14)
			
			self.fixed_img_norm = self._normalize(self.fixed_image)
			self.im_fixed_obj = self.ax_fixed.imshow(self.fixed_img_norm, cmap=colourmap, origin=OriginPosition)
			self.ax_fixed.set_title('FIXED points (reference)')
			
			self.moving_img_norm = self._normalize(self.moving_image)
			self.im_moving_obj = self.ax_moving.imshow(self.moving_img_norm, cmap=colourmap, origin=OriginPosition)
		
		if warning_message:
			self.fig.text(0.5, 0.95, warning_message, color='red', ha='center', fontsize=12, weight='bold',
						  bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

		# --- WIDGETS ---
		ax_color = 'lightgoldenrodyellow'
		
		# Sliders for Fixed Image (Left side)
		ax_fix_min = self.fig.add_axes([0.15, 0.1, 0.25, 0.03], facecolor=ax_color)
		ax_fix_max = self.fig.add_axes([0.15, 0.06, 0.25, 0.03], facecolor=ax_color)
		self.slider_fix_min = Slider(ax_fix_min, 'Fixed Min', 0.0, 1.0, valinit=0.0)
		self.slider_fix_max = Slider(ax_fix_max, 'Fixed Max', 0.0, 1.0, valinit=1.0)
		
		self.slider_fix_min.on_changed(self.update_fixed_clim)
		self.slider_fix_max.on_changed(self.update_fixed_clim)

		self._move_to_active_screen()
		self._redraw()

		# Sliders for Moving Image (Right side) - Only if in moving phase
		if self.phase == 'moving':
			ax_mov_min = self.fig.add_axes([0.55, 0.1, 0.25, 0.03], facecolor=ax_color)
			ax_mov_max = self.fig.add_axes([0.55, 0.06, 0.25, 0.03], facecolor=ax_color)
			self.slider_mov_min = Slider(ax_mov_min, 'Moving Min', 0.0, 1.0, valinit=0.0)
			self.slider_mov_max = Slider(ax_mov_max, 'Moving Max', 0.0, 1.0, valinit=1.0)
			
			self.slider_mov_min.on_changed(self.update_moving_clim)
			self.slider_mov_max.on_changed(self.update_moving_clim)

		# Radio Buttons for Colormap
		ax_radio = self.fig.add_axes([0.92, 0.05, 0.07, 0.15], facecolor=ax_color)
		self.radio = RadioButtons(ax_radio, ('gray', 'magma', 'viridis'))
		self.radio.on_clicked(self.update_cmap)

		self.cid_click = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
		self.cid_key = self.fig.canvas.mpl_connect('key_press_event', self.on_key)
		self.cid_close = self.fig.canvas.mpl_connect('close_event', self.on_close)
		self._redraw()

	def _move_to_active_screen(self):
		try:
			mngr = plt.get_current_fig_manager()
			app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
			screen = app.screenAt(QtGui.QCursor.pos())
			if screen:
				geom = screen.availableGeometry()
				win_w, win_h = 1600, 900
				x = geom.left() + (geom.width() - win_w) // 2
				y = geom.top() + (geom.height() - win_h) // 2
				mngr.window.setGeometry(x, y, win_w, win_h)
		except Exception as e:
			print(f"Window positioning failed: {e}")

	def update_fixed_clim(self, val):
		if self.im_fixed_obj:
			vmin = self.slider_fix_min.val
			vmax = self.slider_fix_max.val
			if vmin >= vmax: vmax = vmin + 0.01
			self.im_fixed_obj.set_clim(vmin, vmax)
			self.fig.canvas.draw_idle()

	def update_moving_clim(self, val):
		if self.im_moving_obj:
			vmin = self.slider_mov_min.val
			vmax = self.slider_mov_max.val
			if vmin >= vmax: vmax = vmin + 0.01
			self.im_moving_obj.set_clim(vmin, vmax)
			self.fig.canvas.draw_idle()

	def update_cmap(self, label):
		if self.im_fixed_obj: self.im_fixed_obj.set_cmap(label)
		if self.im_moving_obj: self.im_moving_obj.set_cmap(label)
		self.fig.canvas.draw_idle()

	def on_click(self, event):
		if event.inaxes is None or event.button != 1: return
		toolbar = self.fig.canvas.manager.toolbar
		if toolbar is not None and toolbar.mode != '': return
		if event.inaxes not in [self.ax_fixed, self.ax_moving]: return

		x, y = event.xdata, event.ydata
		if self.phase == 'fixed' and event.inaxes == self.ax_fixed:
			self.fixed_points.append((x, y))
		elif self.phase == 'moving' and event.inaxes == self.ax_moving:
			if len(self.moving_points) < self.num_total_points:
				self.moving_points.append((x, y))
				idx = len(self.moving_points) - 1
				p_fixed = self.fixed_points_to_display[idx]
				p_moving = self.moving_points[idx]
				dist = np.linalg.norm(np.array(p_fixed) - np.array(p_moving))

				if dist > self.deviation_threshold:
					warning = f"Point #{idx + 1} deviation: {dist:.1f} px (>{self.deviation_threshold} px)"
					self.warning_messages.append(warning)
					print(f"    /!\\ WARNING: {warning}")
			else:
				print("All fixed points have a corresponding moving point.")
		self._redraw()

	def disconnect(self):
		self.fig.canvas.mpl_disconnect(self.cid_click)
		self.fig.canvas.mpl_disconnect(self.cid_key)
		self.fig.canvas.mpl_disconnect(self.cid_close)

	def on_close(self, event):
		if self.phase == 'moving' and len(self.moving_points) < self.num_total_points:
			print(f"ACTION BLOCKED: Please select all {self.num_total_points} points.")
		else:
			self.disconnect()
			plt.close(self.fig)
			
	# --- FIX: Replaced harsh 2/98 percentile clipping with full range normalization ---
	def _normalize(self, img):
		"""Normalized using absolute min/max to preserve high-intensity details."""
		imin, imax = np.min(img), np.max(img)
		if imax == imin:
			return np.zeros_like(img, dtype=np.float32)
		return (img - imin) / (imax - imin)

	def on_key(self, event):
		if event.key == 'z':
			if self.phase == 'fixed' and self.fixed_points:
				self.fixed_points.pop()
			elif self.phase == 'moving' and self.moving_points:
				point_idx_to_remove = len(self.moving_points)
				self.warning_messages = [msg for msg in self.warning_messages if not msg.startswith(f"Point #{point_idx_to_remove}")]
				self.moving_points.pop()
			self._redraw()
		elif event.key == 'w':
			self.warning_messages = ['']
			self._redraw()
		elif event.key == 'p':
			self.show_text_labels = not self.show_text_labels
			self._redraw()
	
	def _redraw(self):
		for artist in self.plotted_artists: artist.remove()
		self.plotted_artists.clear()
		self.cbar_ax.clear()
		
		# --- FIX: Keep frame metadata visible dynamically in title text ---
		lbl = f" | {self.frame_info}" if self.frame_info else ""
		if self.phase == 'moving':
			self.ax_moving.set_title(f'Click to select MOVING point {len(self.moving_points)+1} of {self.num_total_points}{lbl}')
			if len(self.moving_points) == self.num_total_points:
				self.ax_moving.set_title(f'All {self.num_total_points} points selected. Ready to close.{lbl}')

		if self.warning_messages and self.warning_messages != ['']:
			full_warning_text = "DEVIATION WARNINGS:\n" + "\n".join(self.warning_messages)
			warning_artist = self.fig.text(0.5, 0.05, full_warning_text, color='red',   ha='center', fontsize=10, weight='bold',
										   bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
			self.plotted_artists.append(warning_artist)
		
		points_to_color = self.fixed_points if self.phase == 'fixed' else self.fixed_points_to_display
		num_colors = len(points_to_color) if len(points_to_color) > 1 else 2
		cmap = plt.colormaps['jet']
		colors = cmap(np.linspace(0, 1, num_colors))
		
		for i, (x, y) in enumerate(points_to_color):
			p = self.ax_fixed.plot(x, y, 'o', markersize=7, mfc=colors[i], mec='white', mew=0.5, alpha=0.6)
			self.plotted_artists.extend(p)
			if self.show_text_labels:
				t = self.ax_fixed.text(x + 5, y, str(i+1), color='white', fontsize=10, weight='bold')
				self.plotted_artists.append(t)
				
		for i, (x, y) in enumerate(self.moving_points):
			p = self.ax_moving.plot(x, y, 'o', markersize=7, mfc=colors[i], mec='white', mew=0.5, alpha=0.6)
			self.plotted_artists.extend(p)
			if self.show_text_labels:
				t = self.ax_moving.text(x + 5, y, str(i+1), color='white', fontsize=10, weight='bold')
				self.plotted_artists.append(t)
				
		if len(points_to_color) > 0:
			num_ticks = len(points_to_color)
			norm = mcolors.Normalize(vmin=1, vmax=num_ticks)
			sm = cm.ScalarMappable(cmap=cmap, norm=norm)
			sm.set_array([])
			ticks = np.linspace(1, num_ticks, num_ticks) if num_ticks > 1 else [1]
			cbar = self.fig.colorbar(sm, cax=self.cbar_ax, ticks=ticks)
			cbar.set_label('Point Index')
		self.fig.canvas.draw()
		
	def get_points(self):
		plt.show(block=True)
		self.disconnect()
		return self.fixed_points if self.phase == 'fixed' else self.moving_points


	

def _compute_landmark_transform(fixed_points, moving_points, transform_type='affine'):
	"""Takes lists of points and computes the SimpleITK transform."""
	min_points = {'rigid': 2, 'similarity': 2, 'affine': 3}.get(transform_type, 3)
	if len(fixed_points) < min_points or len(moving_points) < min_points:
		print(f"Warning: Not enough points for {transform_type} transform. Skipping.")
		return None
	fixed_flat = [p for point in fixed_points for p in point]
	moving_flat = [p for point in moving_points for p in point]
	if transform_type == 'affine': transform = _sitk.AffineTransform(2)
	elif transform_type == 'similarity': transform = _sitk.Similarity2DTransform()
	elif transform_type == 'rigid': transform = _sitk.Euler2DTransform()
	else: raise ValueError("Unsupported TransformType.")
	final_transform = _sitk.LandmarkBasedTransformInitializer(transform, fixed_flat, moving_flat)
	return final_transform



from scipy.interpolate import Rbf #RBFInterpolator #


import SimpleITK as _sitk
import numpy as np
from scipy.interpolate import RBFInterpolator

def compute_robust_tps_transform(fixed_points, moving_points, reference_image_array, smoothing=0.0, anchor_corners=True):
	"""
	Computes a regularized Thin Plate Spline (TPS) transform using the modern
	RBFInterpolator. Includes options for smoothing and corner pinning to 
	prevent distortion in difficult registration scenarios.

	Args:
		fixed_points: List of tuples (x, y) in physical coordinates.
		moving_points: List of tuples (x, y) in physical coordinates.
		reference_image_array: Numpy array of the reference image.
		smoothing: Float >= 0. Controls the stiffness. 
				   0.0 forces exact matches (susceptible to noise/folding).
				   Values > 0 (e.g., 10.0, 100.0) allow points to misalign slightly 
				   to preserve grid smoothness.
		anchor_corners: Boolean. If True, adds fixed points at the image corners 
						to prevent background extrapolation artifacts.

	Returns:
		sitk.DisplacementFieldTransform
	"""
	
	# Convert input array to SimpleITK image to get physical space info
	reference_image = _sitk.GetImageFromArray(reference_image_array)
	
	# 1. Prepare Points
	# Convert lists to numpy arrays
	fx = np.array([p[0] for p in fixed_points])
	fy = np.array([p[1] for p in fixed_points])
	mx = np.array([p[0] for p in moving_points])
	my = np.array([p[1] for p in moving_points])

	# 2. Optional: Anchor Corners
	# This "pins" the corners of the image so the transform decays to identity 
	# at the edges, preventing wild extrapolation outside the ROI.
	if anchor_corners:
		origin = reference_image.GetOrigin()
		size = reference_image.GetSize()
		spacing = reference_image.GetSpacing()
		
		# Calculate physical coordinates of the 4 corners
		# (Assuming 2D image)
		width_phys = size[0] * spacing[0]
		height_phys = size[1] * spacing[1]
		
		# Corners: TL, TR, BL, BR
		corners = np.array([
			[origin[0], origin[1]],
			[origin[0] + width_phys, origin[1]],
			[origin[0], origin[1] + height_phys],
			[origin[0] + width_phys, origin[1] + height_phys]
		])
		
		# Separate into x and y components
		cx = corners[:, 0]
		cy = corners[:, 1]
		
		# Append corners to both Fixed and Moving (Identity transform at corners)
		fx = np.concatenate([fx, cx])
		fy = np.concatenate([fy, cy])
		mx = np.concatenate([mx, cx])
		my = np.concatenate([my, cy])

	# 3. Calculate Displacements (Target - Source)
	# RBFInterpolator expects shape (N_samples, N_features)
	# We stack X and Y to fit them simultaneously
	source_coords = np.stack([fx, fy], axis=1) # Shape (N, 2)
	displacements = np.stack([mx - fx, my - fy], axis=1) # Shape (N, 2)

	# 4. Fit RBF Interpolator
	# kernel='thin_plate_spline' is standard TPS.
	# smoothing > 0 performs Ridge Regression instead of exact interpolation
	rbf = RBFInterpolator(source_coords, displacements, 
						  kernel='thin_plate_spline', 
						  smoothing=smoothing)

	# 5. Generate Grid
	size = reference_image.GetSize()
	spacing = reference_image.GetSpacing()
	origin = reference_image.GetOrigin()
	direction = np.array(reference_image.GetDirection()).reshape(2, 2)

	# Create index grid
	x_idx = np.arange(0, size[0])
	y_idx = np.arange(0, size[1])
	xv_idx, yv_idx = np.meshgrid(x_idx, y_idx, indexing='xy')

	# Convert index grid to physical coordinates
	# Physical = Origin + Direction * (Index * Spacing)
	phys_x = origin[0] + direction[0, 0] * xv_idx * spacing[0] + direction[0, 1] * yv_idx * spacing[1]
	phys_y = origin[1] + direction[1, 0] * xv_idx * spacing[0] + direction[1, 1] * yv_idx * spacing[1]

	# Flatten for prediction
	flat_phys_coords = np.stack([phys_x.ravel(), phys_y.ravel()], axis=1)

	# 6. Predict Displacements
	predicted_disp = rbf(flat_phys_coords)
	
	# Reshape back to image grid (Height, Width, 2)
	# Note: numpy order is (y, x), but sitk expects vector image components
	disp_x_grid = predicted_disp[:, 0].reshape(size[1], size[0])
	disp_y_grid = predicted_disp[:, 1].reshape(size[1], size[0])

	# 7. Create SimpleITK Displacement Field
	# Stack x and y displacements. 
	# IMPORTANT: SimpleITK expects the array index to be (y, x) but the 
	# vector value to be (displacement_x, displacement_y).
	disp_np = np.stack((disp_x_grid, disp_y_grid), axis=-1)
	
	displacement_img = _sitk.GetImageFromArray(disp_np, isVector=True)
	displacement_img.CopyInformation(reference_image)

	return _sitk.DisplacementFieldTransform(displacement_img)



	
def CoRegisterImages_Manual(im_static, im_shifted, **kwargs):
	"""
	Co-registers a shifted image to a defined static image using SimpleITK (elastix).
	Can be initialized with a pre-existing transform for two-stage registration.
	"""
	InitialTransformFileName = kwargs.get('InitialTransformFileName')
	# This is a placeholder for your full CoRegisterImages_Manual function.
	# Please use the full version provided in the previous response.
	# For this example to run, we'll create a simplified version:
	print(f"--- Running Elastix Registration (B-Spline) ---")
	if InitialTransformFileName:
		print(f"  -> Initialized with transform from: {InitialTransformFileName}")
	
	elastixImageFilter = _sitk.ElastixImageFilter()
	elastixImageFilter.SetFixedImage(_sitk.GetImageFromArray(im_static))
	elastixImageFilter.SetMovingImage(_sitk.GetImageFromArray(im_shifted))
	
	parameterMap = _sitk.GetDefaultParameterMap('bspline')
	parameterMap['FinalGridSpacingInPhysicalUnits'] = [str(kwargs.get('GridSpacing', 20))]
	if InitialTransformFileName:
		parameterMap['InitialTransformParametersFileName'] = [InitialTransformFileName]

	elastixImageFilter.SetParameterMap(parameterMap)
	elastixImageFilter.LogToConsoleOff()
	elastixImageFilter.Execute()
	
	transformParameterMap = elastixImageFilter.GetTransformParameterMap()
	
	transformixImageFilter = _sitk.TransformixImageFilter()
	transformixImageFilter.LogToConsoleOff()
	transformixImageFilter.SetTransformParameterMap(transformParameterMap)
	transformixImageFilter.SetMovingImage(_sitk.GetImageFromArray(im_shifted.astype(np.float32)))
	result_orig_se = transformixImageFilter.Execute()
	
	return _sitk.GetArrayFromImage(result_orig_se), transformParameterMap



def ManualRegistration(RawHypercube, Wavelengths_list, **kwargs):
	"""
	Manual Registration

	Inputs:
		- RawHypercube
		- Wavelengths_list
		- kwargs:
			- TransformSmoothing (100)
			- AnchorCorners (True)
			- Static_Index (0)
			- Cropping (0)
			- Exact (True)
			- Blurring (True)
			- Sigma (2)
			- HideReflections (True)
			- AllReflectionsMasks (None)
			- deviation_threshold (200)
			- 




	"""

	# --- Standard Parameters ---
	TransformSmoothing = kwargs.get('TransformSmoothing', 100)
	AnchorCorners = kwargs.get('AnchorCorners', True)
	Static_Index = kwargs.get('StaticIndex', 0)
	Cropping = kwargs.get('Cropping', 0)
	Exact = kwargs.get('Exact', True)
	Blurring = kwargs.get('Blurring', True)
	Sigma = kwargs.get('Sigma', 2)
	HideReflections = kwargs.get('HideReflections', True)
	AllReflectionsMasks = kwargs.get('AllReflectionsMasks')
	deviation_threshold = kwargs.get('DeviationThreshold', 200)
	GoodFramesLabels = kwargs.get('GoodFramesLabels', None)
	
	backup_path = 'registration_backup.json'

	t0 = time.time()
	(NN, YY, XX) = RawHypercube.shape

	# --- Cropping Logic ---
	if Cropping != 0:
		RawHypercube = RawHypercube[:, Cropping:-Cropping, Cropping:-Cropping]
		if AllReflectionsMasks is not None: 
			AllReflectionsMasks = AllReflectionsMasks[:, Cropping:-Cropping, Cropping:-Cropping]
	
	(NN_crop, YY_crop, XX_crop) = RawHypercube.shape
	CombinedMask = np.zeros((YY_crop, XX_crop), dtype=bool)
	im_static = RawHypercube[Static_Index, :, :]
	sitk_im_static = _sitk.GetImageFromArray(im_static.astype(np.float32))

	# --- Recovery & Input Handling ---
	AllLandmarkPoints_input = kwargs.get('AllLandmarkPoints')
	
	# Auto-load from backup file if exists and no input provided
	if AllLandmarkPoints_input is None and os.path.exists(backup_path):
		print(f"Found existing backup: {backup_path}. Loading progress...")
		with open(backup_path, 'r') as f:
			AllLandmarkPoints_input = json.load(f)

	PromptUser = AllLandmarkPoints_input is None
	Hypercube = []
	AllTransforms = []
	
	fixed_landmarks = AllLandmarkPoints_input['fixed_points'] if not PromptUser else []
	all_moving_landmarks_input = AllLandmarkPoints_input['moving_points'] if not PromptUser else []
	all_moving_landmarks_output = []

	if PromptUser:
		print(f'Setting frame {Static_Index} as static frame')

		# Construct a clean label for the static frame
		if GoodFramesLabels is not None:
			static_info = f"Static Frame: {GoodFramesLabels[Static_Index]}"
		else:
			static_info = f"Static Frame index {Static_Index} ({Wavelengths_list[Static_Index]})"

		picker = LandmarkPicker(im_static, frame_info=static_info)
		fixed_landmarks = picker.get_points()
		if len(fixed_landmarks) < 3:
			raise ValueError(f"Not enough fixed points ({len(fixed_landmarks)}). At least 3 required.")

	try:
		for c in range(0, NN):
			# If we are resuming from a backup, skip frames already in the backup
			if not PromptUser and c < len(all_moving_landmarks_input) and all_moving_landmarks_input[c] is not None:
				moving_landmarks_for_frame = all_moving_landmarks_input[c]
				skip_gui = True
			else:
				skip_gui = False

			if c == Static_Index:
				im_static_processed = copy.deepcopy(im_static).astype(np.float32)
				if HideReflections and AllReflectionsMasks is not None:
					Ref_Mask = AllReflectionsMasks[Static_Index, :, :]
					if Ref_Mask is not None: im_static_processed[Ref_Mask > 0.5] = np.nan
				
				if Blurring:
					try: im_static_processed = HySE.CoRegistrationTools.gaussian_blur_nan(im_static_processed, sigma=Sigma)
					except: im_static_processed = _sitk.GetArrayFromImage(_sitk.DiscreteGaussian(_sitk.GetImageFromArray(im_static_processed), Sigma))

				Hypercube.append(im_static_processed)
				AllTransforms.append(0)
				all_moving_landmarks_output.append(fixed_landmarks)
				CombinedMask |= (np.isnan(im_static_processed) | (im_static_processed == 0))
				continue

			print(f'\nWorking on Frame: {c+1} / {NN}')
			im_shifted = RawHypercube[c, :, :]
			
			if not skip_gui:
				# Construct a descriptive string matching your FrameSelector's style
				if GoodFramesLabels is not None:
					moving_info = f"Frame {c+1}/{NN} [{GoodFramesLabels[c]}]"
				else:
					moving_info = f"Frame {c+1}/{NN} (Wav: {Wavelengths_list[c]})"

				picker = LandmarkPicker(im_static, im_shifted, 
										fixed_points_to_display=fixed_landmarks,
										frame_info=moving_info) # Pass the clean string!
				moving_landmarks_for_frame = picker.get_points()
	
				# picker = LandmarkPicker(im_static, im_shifted, 
				# 						fixed_points_to_display=fixed_landmarks,
				# 						frame_info=(c + 1, NN))
				# moving_landmarks_for_frame = picker.get_points()
			
			all_moving_landmarks_output.append(moving_landmarks_for_frame)

			# --- Registration Calculation ---
			if Exact:
				initial_transform = compute_robust_tps_transform(fixed_landmarks, moving_landmarks_for_frame, im_shifted, smoothing=TransformSmoothing, anchor_corners=AnchorCorners)
			else:
				initial_transform = _compute_landmark_transform(fixed_landmarks, moving_landmarks_for_frame, kwargs.get('TransformType', 'Affine').lower())
			
			if initial_transform:
				sitk_im_shifted = _sitk.GetImageFromArray(im_shifted.astype(np.float32))
				sitk_im_shifted.CopyInformation(sitk_im_static)
				sitk_im_coregistered = _sitk.Resample(sitk_im_shifted, sitk_im_static, initial_transform, _sitk.sitkLinear, 0.0)
				im_coregistered_final = _sitk.GetArrayFromImage(sitk_im_coregistered)
				transform_to_store = initial_transform
			else:
				im_coregistered_final = im_shifted.astype(np.float32)
				transform_to_store = None

			# --- Masking & Blurring ---
			if HideReflections and AllReflectionsMasks is not None and transform_to_store is not None:
				Ref_Mask_Shift = AllReflectionsMasks[c, :, :]
				if Ref_Mask_Shift is not None:
					hole_punch = (Ref_Mask_Shift > 0.5).astype(np.uint8)
					sitk_m = _sitk.GetImageFromArray(hole_punch)
					sitk_m.CopyInformation(sitk_im_static)
					reg_m_sitk = _sitk.Resample(sitk_m, sitk_im_static, transform_to_store, _sitk.sitkNearestNeighbor, 0.0)
					im_coregistered_final[_sitk.GetArrayFromImage(reg_m_sitk).astype(bool)] = np.nan
			
			if Blurring:
				try: im_coregistered_final = HySE.CoRegistrationTools.gaussian_blur_nan(im_coregistered_final, sigma=Sigma)
				except: im_coregistered_final = _sitk.GetArrayFromImage(_sitk.DiscreteGaussian(_sitk.GetImageFromArray(im_coregistered_final), Sigma))

			Hypercube.append(im_coregistered_final)
			AllTransforms.append(transform_to_store)
			CombinedMask |= (np.isnan(im_coregistered_final) | (im_coregistered_final == 0))

			# --- Autosave checkpoint ---
			checkpoint = {'fixed_points': fixed_landmarks, 'moving_points': all_moving_landmarks_output}
			with open(backup_path, 'w') as f:
				json.dump(checkpoint, f)

	except Exception as e:
		print(f"\n[CRITICAL ERROR] Frame {len(all_moving_landmarks_output)}: {e}")

	finally:
		# Final processing and return
		tf = time.time(); time_total = tf - t0
		print(f'\nCo-registration process finished in {time_total/60:.1f} min')
		
		AllPoints = {'fixed_points': fixed_landmarks, 'moving_points': all_moving_landmarks_output}
		Order = kwargs.get('Order', False)
		order_list = np.argsort(Wavelengths_list) if Order else np.arange(len(Hypercube))
		
		return np.array(Hypercube)[order_list], [AllTransforms[i] for i in order_list], CombinedMask, AllPoints



def MakeCombinedHypercubeForRegistration(hypercube, selector_results, original_wavelengths=None):
	"""
	Flattens a 4D Hypercube into a 3D stack based on valid frames selected in the GUI.
	
	Parameters:
	-----------
	hypercube : np.ndarray
		Input 4D array of shape [Nsweeps, Nwavelengths, Y, X].
	selector_results : tuple
		The output from selector.get_results(), containing (mask, indices).
	original_wavelengths : list, optional
		List of strings describing the wavelengths (length Nwavelengths).
		Used to generate unique labels for the flattened stack (e.g., "Sweep0_Wav2").
		
	Returns:
	--------
	filtered_stack : np.ndarray
		3D array [N_valid, Y, X] ready for ManualRegistration.
	new_labels : list
		List of strings describing each frame in the filtered stack, 
		matching the length of filtered_stack.
	"""
	# Unpack the results from the GUI
	_, good_indices = selector_results
	
	# good_indices is shape [N_valid, 2] where cols are [sweep_idx, wav_idx]
	# We use advanced indexing to extract all valid frames at once
	# Resulting shape: [N_valid, Y, X]
	filtered_stack = hypercube[good_indices[:, 0], good_indices[:, 1]]
	
	# Generate updated Wavelengths_list for ManualRegistration
	new_labels = []
	
	# Default labels if none provided
	if original_wavelengths is None:
		num_wavs = hypercube.shape[1]
		original_wavelengths = [f"Frame{i}" for i in range(num_wavs)]
		
	for i in range(len(good_indices)):
		s_idx, w_idx = good_indices[i]
		# Create a label that preserves the sweep and wavelength identity
		# e.g., "S0_Frame0"
		label = f"S{s_idx}_{original_wavelengths[w_idx]}"
		new_labels.append(label)
		
	return filtered_stack, new_labels



def SaveAllTransforms(transforms_list, labels_list=None, filename="RegistrationTransforms.pkl"):
	"""
	Saves a list of SimpleITK transforms and their associated labels to a single file.
	
	Backwards compatible with Automatic Registration output (tuples of ParameterMaps)
	and Manual Registration output (sitk.Transform objects).

	Parameters:
	-----------
	transforms_list : list
		List of transform objects. Can be:
		- sitk.Transform (Manual Registration)
		- int (0 for fixed frames)
		- tuple of sitk.ParameterMap (Automatic Registration)
	labels_list : list, optional
		List of strings identifying each transform. If None, sequential labels are generated.
	filename : str
		Path to save the output file.
	"""
	# If no labels are provided, auto-generate them based on the number of transforms
	if labels_list is None:
		labels_list = [f"Frame_{i}" for i in range(len(transforms_list))]

	if len(transforms_list) != len(labels_list):
		raise ValueError("Error: The number of transforms must match the number of labels.")

	def _make_serializable(item):
		"""Recursively converts SwigPyObjects (like ParameterMaps) to pickle-able Python types."""
		# Pass integers through (handles the '0' for fixed frames)
		if isinstance(item, int):
			return item
		
		# Handle tuples recursively (handles the tuple of 2 ParameterMaps)
		if isinstance(item, tuple):
			return tuple(_make_serializable(x) for x in item)
		
		# Handle lists recursively
		if isinstance(item, list):
			return [_make_serializable(x) for x in item]

		# Handle SimpleITK ParameterMaps (The source of the TypeError)
		# converting them to dicts makes them serializable.
		if "ParameterMap" in item.__class__.__name__:
			return dict(item)

		# Default: Return the item as is (e.g., sitk.Transform objects from Manual Registration)
		return item

	# Clean the input list to ensure all items are serializable
	clean_transforms_list = [_make_serializable(t) for t in transforms_list]

	# Create a dictionary mapping Label -> Transform
	transform_dict = dict(zip(labels_list, clean_transforms_list))

	# Save to a single pickle file
	with open(filename, 'wb') as f:
		pickle.dump(transform_dict, f)
	
	print(f"Successfully saved {len(transforms_list)} transforms to {filename}")



def ApplyAllTransforms(reduced_stack, LoadedOutcome, transforms_file_path, n_selected_frames=3, original_wavelengths=None, labels_list=None):
	"""
	Applies loaded transforms to a 3D stack of valid frames.
	NaN values (from spatial normalisation masking) are preserved through resampling.
	
	Parameters:
	-----------
	reduced_stack : np.ndarray
		3D array [N_valid, Y, X] containing the spatially normalized data (may contain NaNs).
	LoadedOutcome : tuple, list/array, or None
		The output from the FrameSelector. If None, assumes sequential frames.
	transforms_file_path : str
		Path to the .pkl file created by SaveAllTransforms.
	n_selected_frames : int (3)
		Number of frames kept for each plateau/wavelength combination. Standard 3 or 1, adjust if needed
	original_wavelengths : list, optional
		List of original frame names (used when LoadedOutcome is provided).
	labels_list : list, optional
		List of custom transform labels. Used if LoadedOutcome is None but custom labels were saved.
	"""
	# 1. Load Transforms
	with open(transforms_file_path, 'rb') as f:
		transform_dict = pickle.load(f)

	n_valid, h, w = reduced_stack.shape

	# 2. Unpack indices and handle labels based on LoadedOutcome presence
	if LoadedOutcome is not None and len(LoadedOutcome) > 0:
		if isinstance(LoadedOutcome, tuple) and len(LoadedOutcome) == 2:
			good_indices = LoadedOutcome[1]
		else:
			good_indices = LoadedOutcome

		# Handle default labels for standard multiplexed mode
		if original_wavelengths is None:
			max_w_idx = max(idx[1] for idx in good_indices)
			original_wavelengths = [f"Frame{i}" for i in range(max_w_idx + 1)]
	else:
		good_indices = None

	# --- NEW: AUTO-DETECT CROPPING ---
	target_h, target_w = h, w
	
	# Scan for the first Elastix parameter map to find expected target dimensions
	for tform in transform_dict.values():
		if isinstance(tform, (list, tuple)) and len(tform) > 0 and isinstance(tform[0], dict):
			if 'Size' in tform[0]:
				# Elastix Size is typically (Width, Height)
				size_str = tform[0]['Size']
				target_w, target_h = int(size_str[0]), int(size_str[1])
				break
	
	# If the expected dimensions differ from the input stack, verify and apply the crop
	if target_h != h or target_w != w:
		dh = h - target_h
		dw = w - target_w
		
		# Validate that the size mismatch is due to a uniform, symmetric crop
		if dh != dw or dh < 0 or dw < 0 or dh % 2 != 0:
			raise ValueError(
				f"Inconsistent size mismatch detected. Original shape: {(h, w)}, "
				f"Target shape from transform: {(target_h, target_w)}. "
				"Mismatch cannot be explained by a symmetric cropping."
			)
		
		crop = dh // 2
		print(f"Detected uniform cropping of {crop} pixels. Applying to input stack...")
		
		# Apply the exact crop to all sides
		reduced_stack = reduced_stack[:, crop:-crop, crop:-crop]
		
		# Update h and w for the rest of the function (pre-allocation and reference maps)
		h, w = target_h, target_w

	# 3. Prepare Output Array (float32 to support NaN)
	transformed_stack = np.full((n_valid, h, w), np.nan, dtype=np.float32)
	valid_labels = []

	print(f"Applying transforms to {n_valid} frames...")

	def _apply_nan_safe_resample(image_np, resample_fn):
		"""
		Applies a resampling function while preserving NaN regions.
		"""
		nan_mask = np.isnan(image_np)
		has_nans = nan_mask.any()

		if has_nans:
			clean = np.where(nan_mask, 0.0, image_np).astype(np.float32)
			valid_mask = (~nan_mask).astype(np.float32)
		else:
			clean = image_np.astype(np.float32)

		warped = resample_fn(clean)

		if has_nans:
			warped_mask = resample_fn(valid_mask)
			warped[warped_mask < 0.9] = np.nan

		return warped

	# Iterate strictly over the number of frames in the stack
	for i in range(n_valid):
		image_np = reduced_stack[i, :, :].astype(np.float32)
		
		# # Determine the correct dictionary key based on our operating mode
		# if good_indices is not None:
		# 	s_idx, w_idx = good_indices[i]
		# 	label_key = f"S{s_idx}_{original_wavelengths[w_idx]}"
		# else:
		# 	if labels_list is not None:
		# 		label_key = labels_list[i]
		# 	else:
		# 		label_key = f"Frame_{i}"


		# Determine the correct dictionary key based on our operating mode
		if labels_list is not None:
			# FIX 1: Prioritize explicit custom labels if you provide them
			label_key = labels_list[i]
		elif good_indices is not None:
			s_idx, w_idx = good_indices[i]
			label_key = f"S{s_idx}_{original_wavelengths[w_idx]}"
			
			# FIX 2: Auto-detect flattened pseudo-sweeps
			if label_key not in transform_dict:
				# Map the flattened sweep index back to the original (e.g., 69 // 3 = 23)
				original_s_idx = s_idx // n_selected_frames
				fallback_key = f"S{original_s_idx}_{original_wavelengths[w_idx]}"
				if fallback_key in transform_dict:
					label_key = fallback_key
		else:
			label_key = f"Frame_{i}"


		if label_key not in transform_dict:
			print(f"WARNING: Key '{label_key}' not found in saved transforms. Skipping!")
			transformed_stack[i, :, :] = image_np
			valid_labels.append(label_key)
			continue

		tform_data = transform_dict[label_key]

		# --- CASE 1: FIXED FRAME (No Transform) ---
		if isinstance(tform_data, int) and tform_data == 0:
			transformed_stack[i, :, :] = image_np
			valid_labels.append(label_key)
			continue

		# --- CASE 2: AUTOMATIC REGISTRATION (Elastix) ---
		elif isinstance(tform_data, (list, tuple)) and len(tform_data) > 0 and isinstance(tform_data[0], dict):
			try:
				pm_vector = _sitk.VectorOfParameterMap()
				for p_dict in tform_data:
					pm = _sitk.ParameterMap()
					for k, v in p_dict.items():
						pm[k] = v
					pm_vector.append(pm)

				def _elastix_resample(arr_np):
					sitk_img = _sitk.GetImageFromArray(arr_np)
					transformix = _sitk.TransformixImageFilter()
					transformix.SetMovingImage(sitk_img)
					transformix.SetTransformParameterMap(pm_vector)
					transformix.LogToConsoleOff()
					return _sitk.GetArrayFromImage(transformix.Execute())

				transformed_stack[i, :, :] = _apply_nan_safe_resample(image_np, _elastix_resample)
				valid_labels.append(label_key)
			except Exception as e:
				print(f"Error applying Elastix map for {label_key}: {e}")
			continue

		# --- CASE 3: MANUAL REGISTRATION ---
		elif isinstance(tform_data, _sitk.Transform):
			final_transform = tform_data

		# --- CASE 4: LEGACY LANDMARKS ---
		elif isinstance(tform_data, (tuple, list)) and len(tform_data) == 2 and not isinstance(tform_data[0], dict):
			src_pts, dst_pts = tform_data
			src_flat = [coord for point in src_pts for coord in point]
			dst_flat = [coord for point in dst_pts for coord in point]
			final_transform = _sitk.LandmarkBasedTransformInitializer(
				_sitk.BSplineTransform(2, 3),
				src_flat,
				dst_flat,
				_sitk.BSplineTransformInitializerFilter.BSPLINE_ORDER_3
			)
		else:
			print(f"Unknown transform format for {label_key}. Skipping.")
			continue

		# Apply Standard SimpleITK Resampling (Cases 3 & 4)
		reference_sitk = _sitk.GetImageFromArray(np.zeros((h, w), dtype=np.float32))

		def _sitk_resample(arr_np, transform=final_transform, ref=reference_sitk):
			sitk_img = _sitk.GetImageFromArray(arr_np)
			resampler = _sitk.ResampleImageFilter()
			resampler.SetReferenceImage(ref)
			resampler.SetTransform(transform)
			resampler.SetInterpolator(_sitk.sitkLinear)
			resampler.SetDefaultPixelValue(0)
			return _sitk.GetArrayFromImage(resampler.Execute(sitk_img))

		try:
			transformed_stack[i, :, :] = _apply_nan_safe_resample(image_np, _sitk_resample)
			valid_labels.append(label_key)
		except Exception as e:
			print(f"Failed to apply transform for {label_key}: {e}")

	return transformed_stack, valid_labels
	



# def CombineFrames(transformed_stack, labels_list, n_wavelengths=16):
# 	"""
# 	Averages registered frames belonging to the same wavelength index.

# 	NB: Current implementation returns NaN when any of the averaged frames contain a 
# 	NaN at this pixel. A NaN-aware version is commented out, which allows to return
# 	the average of the non-NaN pixels instead.

# 	Parameters:
# 	-----------
# 	transformed_stack : np.ndarray
# 		3D array [N_valid, Y, X] output from ApplyAllTransforms.
# 	labels_list : list
# 		List of labels matching transformed_stack (e.g., ["S0_Frame0", "S1_Frame0"...]).
# 	n_wavelengths : int (16)
# 		The expected number of unique wavelengths (frames per sweep).

# 	Returns:
# 	--------
# 	combined_cube : np.ndarray
# 		3D array [Nwavelengths, Y, X] containing the averaged data.
# 	"""
# 	N, h, w = transformed_stack.shape
	
# 	# Initialize accumulators
# 	# sum_cube stores the pixel sums
# 	sum_cube = np.zeros((n_wavelengths, h, w), dtype=np.float32)
# 	# count_cube stores how many frames contributed to each pixel
# 	count_cube = np.zeros((n_wavelengths, h, w), dtype=np.float32)

# 	print("Combining frames by averaging...")

# 	for i, label in enumerate(labels_list):
	# 	# Parse the label to get the frame index
	# 	# Assuming label format "S{sweep}_Frame{wav}" or similar
	# 	# We look for the part after the underscore or explicitly find the Wav index
		
	# 	# Regex to find the index associated with the "Frame" part
	# 	# Looks for "Frame" followed by digits
	# 	match = re.search(r'Frame(\d+)', label)
	# 	if match:
	# 		wav_idx = int(match.group(1))
	# 	else:
	# 		# Fallback: if you used a custom list like ['Blue', 'Red']
	# 		# We have to map string -> index manually. 
	# 		# For now, assuming "FrameX" format as per previous steps.
	# 		print(f"Error parsing label {label}. Skipping.")
	# 		continue
			
	# 	# print(f'label: {label} - wav_idx: {wav_idx}')
	# 	if wav_idx >= n_wavelengths:
	# 		continue

	# 	# Add current frame to accumulator
	# 	# Convert to float to avoid rounding errors
	# 	sum_cube[wav_idx] += transformed_stack[i].astype('float32')
	# 	count_cube[wav_idx] += 1

	# 	# ## NaN aware version:
	# 	# valid = ~np.isnan(frame)
	# 	# sum_cube[wav_idx][valid] += frame[valid]
	# 	# count_cube[wav_idx][valid] += 1

	# # Divide by count to get average
	# # Avoid division by zero where no frames were present
	# with np.errstate(divide='ignore', invalid='ignore'):
	# 	combined_cube = sum_cube / count_cube
	# 	combined_cube[count_cube == 0] = 0 # Handle empty frames

	# return combined_cube


def CombineFrames(transformed_stack, labels_list, n_wavelengths=16, NaN_aware=False):
	"""
	Averages registered frames belonging to the same wavelength index.

	Parameters:
	-----------
	transformed_stack : np.ndarray
		3D array [N_valid, Y, X] output from ApplyAllTransforms.
	labels_list : list
		List of labels matching transformed_stack (e.g., ["S0_Frame0", "S1_Frame0"...]).
	n_wavelengths : int
		The expected number of unique wavelengths.
	NaN_aware : bool
		If True, ignore NaN pixels when averaging.
		If False, use standard averaging (current behaviour).

	Returns:
	--------
	combined_cube : np.ndarray
		3D array [Nwavelengths, Y, X] containing the averaged data.
	"""

	N, h, w = transformed_stack.shape

	print("Combining frames by averaging...")

	if not NaN_aware:
		sum_cube = np.zeros((n_wavelengths, h, w), dtype=np.float32)
		count_cube = np.zeros((n_wavelengths, h, w), dtype=np.float32)

		for i, label in enumerate(labels_list):

			match = re.search(r'Frame(\d+)', label)
			if match:
				wav_idx = int(match.group(1))
			else:
				print(f"Error parsing label {label}. Skipping.")
				continue

			if wav_idx >= n_wavelengths:
				continue

			sum_cube[wav_idx] += transformed_stack[i].astype(np.float32)
			count_cube[wav_idx] += 1

		with np.errstate(divide='ignore', invalid='ignore'):
			combined_cube = sum_cube / count_cube
			combined_cube[count_cube == 0] = 0

		return combined_cube

	else:
		# NaN-aware implementation
		sum_cube = np.zeros((n_wavelengths, h, w), dtype=np.float32)
		count_cube = np.zeros((n_wavelengths, h, w), dtype=np.float32)

		for i, label in enumerate(labels_list):

			match = re.search(r'Frame(\d+)', label)
			if match:
				wav_idx = int(match.group(1))
			else:
				print(f"Error parsing label {label}. Skipping.")
				continue

			if wav_idx >= n_wavelengths:
				continue

			frame = transformed_stack[i].astype(np.float32)

			valid_mask = ~np.isnan(frame)

			sum_cube[wav_idx][valid_mask] += frame[valid_mask]
			count_cube[wav_idx][valid_mask] += 1

		with np.errstate(divide='ignore', invalid='ignore'):
			combined_cube = sum_cube / count_cube
			combined_cube[count_cube == 0] = np.nan

		return combined_cube





def PlotFixedPoints(HypercubeForRegistration, StaticIndex, AllPoints, SavingPath="StaticFramePoints.png", Labels=False):
	"""
	Plots and saves the static frame from a hypercube with the identified registration points.

	Args:
		HypercubeForRegistration (numpy.ndarray): The unregistered hypercube (H, W, Bands).
		StaticIndex (int): The index of the static frame.
		AllPoints (list or dict): Structure containing control points. Assumes AllPoints[StaticIndex] 
								  returns an array of shape (N, 2) representing (x, y) coordinates.
		filename (str): Path to save the resulting figure.
		Labels (bool): If True, plots the numerical index (1-based) of each point.
	"""
	# Extract the static frame image
	static_image = HypercubeForRegistration[StaticIndex, :, :]
	
	# Extract fixed points and convert list of tuples to numpy array for slicing
	points = np.array(AllPoints['fixed_points'])
	num_points = len(points)

	# Generate colors from the jet colormap
	colors = cm.jet(np.linspace(0, 1, num_points))

	plt.figure(figsize=(10, 10))
	plt.imshow(static_image, cmap='gray')
	
	# Plot each point (assuming points are x, y)
	plt.scatter(points[:, 0], points[:, 1], c=colors, s=50, edgecolors='white', linewidths=0.5)

	if Labels:
		for i, (x, y) in enumerate(points):
			plt.text(x, y, str(i + 1), color='yellow', fontsize=12, fontweight='bold', 
					 path_effects=[PathEffects.withStroke(linewidth=2, foreground="black")])

	plt.title(f"Static Frame (Index {StaticIndex}) with Fixed Points")
	plt.axis('off')
	plt.tight_layout()
	plt.savefig(SavingPath, bbox_inches='tight', dpi=300)
	plt.close()


