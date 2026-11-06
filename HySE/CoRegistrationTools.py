"""

Functions that relate to co-registration


"""


import numpy as np
import cv2
import os
from datetime import datetime
from scipy.signal import savgol_filter, find_peaks
import matplotlib
from matplotlib import pyplot as plt
import imageio
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import RectangleSelector
# import SimpleITK as sitk
import time
from tqdm import trange
import inspect

matplotlib.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "arial"


import HySE.UserTools
import HySE.Import
import HySE.ManipulateHypercube


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
from natsort import natsorted
from astropy.convolution import convolve, Gaussian2DKernel


####  -------   HELPER FUNCTIONS   -------
####  ------------------------------------

def GetGlobalMask(**kwargs):
	'''
	Function that inputs optional arguments only, to output a global mask 
	(for static or shifted images) that can be either a combination of the edge
	and reflections mask, only one of those, or none, depending on what masks are available.

	Inputs:
		- kwargs:
			- Help
			- EdgeMask
			- ReflectionsMask
			- PrintInfo = True


	Outputs:
		- GlobalMask

	'''
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(GetGlobalMask))
		return 
	EdgeMask = kwargs.get('EdgeMask')
	ReflectionsMask = kwargs.get('ReflectionsMask')
	PrintInfo = kwargs.get('PrintInfo', True)

	if EdgeMask is not None:
		if ReflectionsMask is not None: 
			## If there is both an EdgeMask and a StaticMask, combine them
			GlobalMask = np.logical_or(EdgeMask > 0, ReflectionsMask > 0).astype(np.uint8)
			if PrintInfo:
				print(f'Global Mask = EdgeMask + ReflectionsMask')
		else:
			## Otherwise the global mask will be whaterver mask is given
			GlobalMask = EdgeMask
			if PrintInfo:
				print(f'Global Mask = EdgeMask only')
	elif ReflectionsMask is not None:
		GlobalMask = ReflectionsMask
		if PrintInfo:
			print(f'Global Mask = ReflectionsMask only')
	else:
		## If no mask is give, return None
		GlobalMask = None
		if PrintInfo:
			print(f'Global Mask = None')
	# if GlobalMask is not None:
	# 	## If there is a mask, invert it to be used with SimpleITK
	# 	GlobalMask = np.invert(GlobalMask) ## SimpleITK uses invert logic

	return GlobalMask


def GetNMI(Data, **kwargs):
	"""
	Computes the normalised mutual information of a hypercube.

	Inputs:
		- Data : Hypercube, shape (N, Y, X), may contain NaNs
		- kwargs:
			- Help : if True, print docstring
			- ReferenceIndex = 1 : Which image to reference to

	Outputs:
		- NMI_average : float, mean NMI vs reference
		- NMI_vs_reference : array of shape (N,), NMI each slice vs reference
		- NMI_pairwise : array of shape (N-1,), pairwise NMI between consecutive slices
	"""
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(GetNMI))
		return 0, 0, 0


	cube = Data
	N, H, W = cube.shape

	# Contrast-normalise each slice, ignoring NaNs
	cube_norm = cube.astype(np.float32).copy()
	for i in range(N):
		sl = cube_norm[i]
		min_val = np.nanmin(sl)
		rng = np.nanmax(sl) - min_val
		if rng == 0 or np.isnan(rng):
			cube_norm[i] = np.zeros_like(sl, dtype=np.float32)
		else:
			cube_norm[i] = (sl - min_val) / (rng + 1e-7)

	# Reference slice (e.g. index 1)
	# ref_idx = 1
	ref_idx = kwargs.get('ReferenceIndex', 1)
	ref = cube_norm[ref_idx]
	bins = 128

	# Helper to mask NaNs before NMI call
	def safe_nmi(a, b, bins=128):
		mask = ~np.isnan(a) & ~np.isnan(b)
		if mask.sum() == 0:   # no overlap
			return np.nan
		return nmi(a[mask], b[mask], bins=bins)

	# NMI vs reference
	nmi_vs_ref = np.empty(N, dtype=np.float32)
	for i in range(N):
		nmi_vs_ref[i] = safe_nmi(cube_norm[i], ref, bins=bins)
	mean_nmi = np.nanmean(nmi_vs_ref)

	# Pairwise NMI
	pairwise_nmi = np.empty(N - 1, dtype=np.float32)
	for i in range(N - 1):
		pairwise_nmi[i] = safe_nmi(cube_norm[i], cube_norm[i + 1], bins=bins)

	return mean_nmi, nmi_vs_ref, pairwise_nmi



def get_interactive_roi(image, title='Select Registration ROI'):
	"""
	Displays an image and allows the user to select a rectangular ROI.
	
	Returns:
		A tuple of (y_start, y_end, x_start, x_end) or None if window is closed.
	"""
	roi_coords = {}

	def line_select_callback(eclick, erelease):
		"""Callback for processing the rectangle selector's click and release events."""
		x1, y1 = int(eclick.xdata), int(eclick.ydata)
		x2, y2 = int(erelease.xdata), int(erelease.ydata)
		roi_coords['x_start'] = min(x1, x2)
		roi_coords['x_end'] = max(x1, x2)
		roi_coords['y_start'] = min(y1, y2)
		roi_coords['y_end'] = max(y1, y2)

	fig, ax = plt.subplots(figsize=(10, 8))
	
	# Normalize image for better visualization
	p2, p98 = np.percentile(image, (2, 98))
	img_display = np.clip((image - p2) / (p98 - p2), 0, 1)
	
	ax.imshow(img_display, cmap='gray')
	ax.set_title(f"{title}\nDraw a rectangle and close the window to continue.")
	
	rs = RectangleSelector(ax, line_select_callback,
						   useblit=True,
						   button=[1],  # Left mouse button
						   minspanx=5, minspany=5,
						   spancoords='pixels',
						   interactive=True,
						   props=dict(facecolor='cyan', edgecolor='cyan', alpha=0.3, fill=True))

	plt.show(block=True) # This will pause execution until you close the plot window

	if 'x_start' in roi_coords:
		return (roi_coords['y_start'], roi_coords['y_end'], 
				roi_coords['x_start'], roi_coords['x_end'])
	else:
		print("No ROI selected. Proceeding without interactive crop.")
		return None


							

def SaveFramesSweeps(Hypercube_all, SavingPath, Name, NameSub, **kwargs):
	"""
	Function that takes all hypercubes (where hypercube computed from specific sweepw were kept individually 
	instead of averaged) and saves frames as individual images. 
	All images are saved in folder called {Name}_{NameSub}_RawFrames inside SavingPath, and images from sweep i are
	saved in the folder Sweep{s}.

	Modified to allow expanded hypercubes that include individual (non-averaged) frames within plateaus.

	*Update*: Includes robust contrast adjustment to handle specular reflections (clips top 0.1% intensity).
	*Update*: Added MP4 Video option compatible with QuickTime.

	Input:
		- Hypercube_all : All the hypercubes computed from all sweeps. Shape (Nsweeps, Nwav, Y, X) or (Nsweeps, Nwav, Nframes, Y, X)
		- RescalePercentile = 95 : Rescale image to this percentile of the image's values. To remove impact of specular reflection.
		- SavingPath : Where to save all results (generic) (expected to end with '/')
		- Name : General name of the data (i.e. patient)
		- NameSub : Specific name for the hypercubes (i.e. lesion)
		- kwargs:
			- Sweeps [int, int, ...] : If indicated, which sweeps to save
			- Frames [int] : If indicated, which frames inside a given sweep to save
			- Video [bool] : If True, saves the sequence of frames as an mp4 file
			- FPS [int] : Frames per second for the output video (default 10)
			- Help


	Outputs:
		- All frames saved as png images in individual folders
		- (Optional) MP4 video file for each sweep
	"""

	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(SaveFramesSweeps))
		return 

	hypercube_shape = Hypercube_all.shape
	if len(hypercube_shape) == 4:
		Nsweeps, Nwav, YY, XX = Hypercube_all.shape
		Nframes = 1
	else:
		Nsweeps, Nwav, Nframes, YY, XX = Hypercube_all.shape

	Sweeps = kwargs.get('Sweeps')
	if Sweeps is None:
		Sweeps = [i for i in range(0, Nsweeps)]
	else:
		print(f'Only saving sweeps {Sweeps}')

	Frames = kwargs.get('Frames')
	if Frames is not None:
		print(f'Only saving frames {Frames}')

	RescalePercentile = kwargs.get('RescalePercentile', 95)
	SaveVideo = kwargs.get('Video', False)
	FPS = kwargs.get('FPS', 10)

	print(f'Rescaling images to {RescalePercentile} percentile')

	GeneralDirectory = f'{Name}_{NameSub}_RawFrames'
	if not os.path.exists(f'{SavingPath}{GeneralDirectory}'):
		os.mkdir(f'{SavingPath}{GeneralDirectory}')

	for s in range(0, Nsweeps):
		if s in Sweeps:
			DirectoryName = f'{SavingPath}{GeneralDirectory}/Sweep{s}'
			if not os.path.exists(DirectoryName):
				os.mkdir(DirectoryName)

			video_writer = None
			if SaveVideo:
				video_path = f'{DirectoryName}/Sweep{s}_video.mp4'
				# 'mp4v' is widely compatible with QuickTime/macOS
				fourcc = cv2.VideoWriter_fourcc(*'mp4v')
				video_writer = cv2.VideoWriter(video_path, fourcc, FPS, (XX, YY), False)

			if len(hypercube_shape) == 4:
				hypercube = Hypercube_all[s, :, :, :]
				for w in range(0, Nwav):
					raw_frame = hypercube[w, :, :]

					v_max = np.percentile(raw_frame, 99.9)
					if v_max == 0: v_max = 1

					frame = np.clip(raw_frame, 0, v_max)
					frame = (frame / v_max) * 255
					frame_uint8 = frame.astype(np.uint8)

					if SaveVideo:
						video_writer.write(frame_uint8)
					else:
						im = Image.fromarray(frame_uint8)
						im = im.convert("L")
						im.save(f'{DirectoryName}/Im{w}.png')
			else:
				hypercube = Hypercube_all[s, :, :, :, :]
				CurrentFrames = Frames if Frames is not None else range(0, Nframes)

				for w in range(0, Nwav):
					for f in range(0, Nframes):
						if f in CurrentFrames:
							raw_frame = hypercube[w, f, :, :]

							v_max = np.percentile(raw_frame, RescalePercentile)
							if v_max == 0: v_max = 1

							frame = np.clip(raw_frame, 0, v_max)
							frame = (frame / v_max) * 255
							frame_uint8 = frame.astype(np.uint8)

							if SaveVideo:
								video_writer.write(frame_uint8)
							else:
								im = Image.fromarray(frame_uint8)
								im = im.convert("L")
								im.save(f'{DirectoryName}/Im{w}_frame{f}.png')

			if video_writer is not None:
				video_writer.release()



####  -------   DEAL WITH TRANSFORMS   -------
####  ----------------------------------------




def ApplyTransform(Frames, Transforms, **kwargs):
	'''
	Functions that inputs the list of all transforms generated by HySE.CoRegisterHypercube() and applies 
	it to new frames.
	Note that the transforms can be saved as .txt files as well

	Inputs:
		- Frames : Array, shape (N, Y, X)
		- Transforms : List, length N (one transform per 2D frame)
		- kwargs :
			- Help
			- Verbose = False : If true, prints default console output from SimpleITK

	Outputs:
		- TransformedFrames
	'''

	Verbose = kwargs.get('Verbose', False)
	(Nwav, Y, X) = Frames.shape
	
	if Nwav != len(Transforms):
		print(f'Error: Frames ({Nwav}) and Transforms ({len(Transforms)}) count mismatch.')
		return 0
	
	TransformedFrames = []
	for i in range(Nwav):
		im_raw = Frames[i,:,:]
		transform_set = Transforms[i]
		
		# 1. Handle Static Frame
		if transform_set == 0:
			TransformedFrames.append(im_raw)
			continue
			
		# Ensure transform_set is a list for uniform processing
		if not isinstance(transform_set, list):
			transform_list = [transform_set]
		else:
			transform_list = transform_set

		# Convert moving image to SITK
		sitk_moving = _sitk.GetImageFromArray(im_raw)

		# 2. Check the type of the first sub-transform to determine the engine
		# We assume one frame uses one engine (either SITK or Elastix)
		first_t = transform_list[0]

		if isinstance(first_t, _sitk.Transform):
			# Option A: SimpleITK Resampler (Manual/Affine/TPS)
			# Combine multiple SITK transforms if necessary
			composite = _sitk.CompositeTransform(first_t.GetDimension())
			for t in transform_list:
				composite.AddTransform(t)
			
			resampler = _sitk.ResampleImageFilter()
			resampler.SetReferenceImage(sitk_moving) # Use input as grid ref
			resampler.SetTransform(composite)
			resampler.SetInterpolator(_sitk.sitkLinear)
			result_img = resampler.Execute(sitk_moving)
			
		else:
			# Option B: Transformix (Elastix ParameterMaps)
			transformix = _sitk.TransformixImageFilter()
			transformix.SetLogToConsole(Verbose)
			transformix.SetMovingImage(sitk_moving)
			
			# Transformix handles a list of ParameterMaps natively
			if isinstance(transform_set, list):
				# Ensure they are all ParameterMaps
				p_maps = [t for t in transform_set if not isinstance(t, _sitk.Transform)]
				transformix.SetTransformParameterMaps(p_maps)
			else:
				transformix.SetTransformParameterMap(transform_set)
				
			result_img = transformix.Execute()

		TransformedFrames.append(_sitk.GetArrayFromImage(result_img))
		
	return np.array(TransformedFrames)

def SaveTransforms(AllTransforms, TransformsSavingPath, **kwargs):
	'''
	Function that saves all the transforms output from the coregistration 
	pipeline (AllTransforms) as txt files.
	
	This version is robust and can save both Elastix-style ParameterMaps 
	(from B-spline) and SimpleITK Transform Objects (from LandmarkOnly).
	
	Inputs:
		- AllTransforms (Output from CoRegisterHypercube())
		- TransformsSavingPath : Where to save files (do not add extension)
		- kwargs
			- Help
	Ouput:
	
	'''
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(SaveTransforms))
		return 
	
	N = len(AllTransforms)
	for n in range(0,N):
		transform = AllTransforms[n]
		if transform==0:
			# Static image, do nothing
			pass
	
		
		# CASE 1: This is a SimpleITK Transform Object (e.g., Affine, BSpline)
		elif isinstance(transform, _sitk.Transform):
			# Use WriteTransform for transform objects
			_sitk.WriteTransform(transform, f'{TransformsSavingPath}_Frame{n}.txt')

		# CASE 2: This is an Elastix ParameterMap (a list of dicts)
		elif isinstance(transform, list):
			# Use WriteParameterFile for each parameter map (dict) in the list
			for i in range(0,len(transform)):
				_sitk.WriteParameterFile(transform[i], f'{TransformsSavingPath}_Frame{n}_{i}.txt')
		
		# CASE 3: Fallback for a single ParameterMap that isn't in a list
		elif isinstance(transform, dict):
			_sitk.WriteParameterFile(transform, f'{TransformsSavingPath}_Frame{n}.txt')

		# # CASE 4: Tuple (Likely a composite transform)
		# elif isinstance(transform, tuple):
		# 	# Iterate over elements of the tuple and save each one
		# 	for i, sub_transform in enumerate(transform):
		# 		if isinstance(sub_transform, _sitk.Transform):
		# 			_sitk.WriteTransform(sub_transform, f'{TransformsSavingPath}_Frame{n}_Tuple{i}.txt')
		# 		elif isinstance(sub_transform, dict):
		# 			_sitk.WriteParameterFile(sub_transform, f'{TransformsSavingPath}_Frame{n}_Tuple{i}.txt')
		# 		elif sub_transform == 0:
		# 			pass
		# 		else:
		# 			print(f"Frame {n}, Sub-transform {i}: Unknown type in tuple ({type(sub_transform)}). Skipping.")

		# CASE 4: Tuple
		elif isinstance(transform, tuple):
			# print(f"Frame {n}: Handling Composite Transform (tuple).")
			for i, sub_transform in enumerate(transform):
				
				# Check if the sub-transform is a SimpleITK Transform Object
				if isinstance(sub_transform, _sitk.Transform):
					_sitk.WriteTransform(sub_transform, f'{TransformsSavingPath}_Frame{n}_Comp{i}.txt')
					
				# Check if the sub-transform is a ParameterMap (dict or SimpleITK ParameterMap object)
				# We assume any dictionary-like object that isn't a simple transform is a ParameterMap
				elif sub_transform == 0:
					pass # Skip
					
				else:
					# Try to write it as a ParameterFile. This is often the safest bet
					# for SimpleITK's ParameterMap objects, as they may not inherit from dict.
					try:
						_sitk.WriteParameterFile(sub_transform, f'{TransformsSavingPath}_Frame{n}_Comp{i}.txt')
					except AttributeError:
						print(f"Frame {n}, Sub-transform {i}: Unknown type in tuple ({type(sub_transform)}). Skipping.")
					except Exception as e:
						 print(f"Frame {n}, Sub-transform {i}: Failed to write transform ({e}). Skipping.")

		# CASE 5: Fallback for unknown type
		else:
			print(f"Frame {n}: Unknown transform type ({type(transform)}). Skipping.")
				
			





def LoadTransforms(TransformsPath, Nframes=16, **kwargs):
	Files = natsorted(glob.glob(TransformsPath + '*.txt'))
	organized_data = {}

	for file in Files:
		name = os.path.basename(file)
		info = name.split('Frame')[-1].replace('.txt', '')
		try:
			if '_' in info:
				parts = info.split('_')
				frame_idx = int(''.join(filter(str.isdigit, parts[0])))
				sub_idx = int(''.join(filter(str.isdigit, parts[1])))
			else:
				frame_idx = int(info)
				sub_idx = 0
			
			with open(file, 'r') as f:
				first_line = f.readline()

			if first_line.startswith("#Insight Transform"):
				transform = _sitk.ReadTransform(file)
			else:
				transform = _sitk.ReadParameterFile(file)
			
			if frame_idx not in organized_data:
				organized_data[frame_idx] = {}
			organized_data[frame_idx][sub_idx] = transform
		except Exception:
			continue

	AllTransforms = []
	for n in range(Nframes):
		if n in organized_data:
			# Sort sub-transforms (e.g., 0=Affine, 1=BSpline)
			sorted_keys = sorted(organized_data[n].keys())
			subs = [organized_data[n][k] for k in sorted_keys]
			# If there's only one sub-transform, we can keep it as a single object or list
			AllTransforms.append(subs[0] if len(subs) == 1 else subs)
		else:
			# This is our static frame
			AllTransforms.append(0)
	return AllTransforms



####  -------   MAIN COREGISTRATION FUNCTIONS   -------
####  -------------------------------------------------


def CoRegisterImages(im_static, im_shifted, **kwargs):
	"""
	Co-registers a shifted image to a defined static image using SimpleITK (v3)

	Inputs:
		- im_static
		- im_shifted
		- kwargs
			- Affine = False : use affine transform only (True/False)
			- TwoStage = False : perform an affine registration, then a non-rigid (BSpline) registration (True/False).
			- Verbose = False : print logs (True/False)
			- MaximumNumberOfIterations: int
			- Metric = 'AdvancedMattesMutualInformation' : e.g. 'AdvancedMattesMutualInformation', 'AdvancedMeanSquares', 'AdvancedNormalizedCorrelation', 
				'AdvancedKappaStatistic', '', 'NormalizedMutualInformation', 'DisplacementMagnitudePenalty'
			- Optimizer = 'AdaptiveStochasticGradientDescent' : e.g. 'AdaptiveStochasticGradientDescent', 'RegularStepGradientDescent', 'CMAES'
			- Transform = 'BSplineTransform' : e.g. 'BSplineTransform', 'Euler2DTransform', 'Similarity2DTransform', 'AffineTransform', 'TranslationTransform'
			- GradientMagnitude = False: if True, register based on gradient images
			- HistogramMatch = False: if True, match moving image histogram to fixed
			- IntensityNorm = False: if True, z-score normalize both images
			- Blurring = False: if True, apply Gaussian blur
			- Sigma = 2: If blurring images, blur by sigma (Gaussian)
			- GridSpacing : Spacing of the B-spline control point grid. A larger value produces a stiffer, smoother transform and reduces artifacts.
			- StaticMask: Usually a combination of EdgeMask and ReflectionsMask, either, or None. Applied to the static image
			- ShiftedMask: Usually a combination of EdgeMask and ReflectionsMask, either, or None. Applied to the shifted image

	Outputs:
		- Registered Image
		- transform parameter map (to be applied to other data)
			To be used:
				transformixImageFilter = sitk.TransformixImageFilter()
				transformixImageFilter.SetMovingImage(sitk.ReadImage("other_image.tif", sitk.sitkFloat32))
				transformixImageFilter.SetTransformParameterMap(sitk.ReadParameterFile("TransformParameters.0.0.txt"))
				transformixImageFilter.Execute()

	"""
	if _sitk is None:
		raise ImportError("SimpleITK is required. Install it with `pip install SimpleITK`.")

	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(CoRegisterImages))
		return 0, 0, 0

	Affine = kwargs.get('Affine', False)
	TwoStage = kwargs.get('TwoStage', False)
	Verbose = kwargs.get('Verbose', False)
	Metric = kwargs.get('Metric', 'AdvancedMattesMutualInformation')
	Transform = kwargs.get('Transform', 'BSplineTransform')
	Optimizer = kwargs.get('Optimizer', 'AdaptiveStochasticGradientDescent')
	GridSpacing = kwargs.get('GridSpacing')

	HistogramMatch = kwargs.get('HistogramMatch', False)
	IntensityNorm = kwargs.get('IntensityNorm', False)
	GradientMagnitude = kwargs.get('GradientMagnitude', False)
	Blurring = kwargs.get('Blurring', False)
	Sigma = kwargs.get('Sigma', 2)
	StaticMask = kwargs.get('StaticMask')
	ShiftedMask = kwargs.get('ShiftedMask')

	# Print the configuration
	if TwoStage:
		print(f'SimpleITK: Two-Stage Registration (Affine -> BSpline) with GridSpacing = {GridSpacing}')
	else:
		print(f'SimpleITK: Transform = {Transform}, Optimizer = {Optimizer}, Metric = {Metric}')

	t0 = time.time()
	
	# New keyword argument for this specific debugging purpose
	return_all_maps = kwargs.get('return_all_maps', False)

	# Convert images to float32
	im_static_orig = im_static.astype(np.float32)
	im_shifted_orig = im_shifted.astype(np.float32)

	# Create copies for preprocessing
	im_static_proc = np.copy(im_static_orig)
	im_shifted_proc = np.copy(im_shifted_orig)

	# Optional preprocessing
	if IntensityNorm:
		# Z-score normalization
		im_static_proc = (im_static_proc - np.mean(im_static_proc)) / np.std(im_static_proc)
		im_shifted_proc = (im_shifted_proc - np.mean(im_shifted_proc)) / np.std(im_shifted_proc)

	if Blurring:
		im_static_proc = gaussian_filter(im_static_proc, sigma=Sigma)
		im_shifted_proc = gaussian_filter(im_shifted_proc, sigma=Sigma)

	# Convert to SimpleITK
	im_static_se = _sitk.GetImageFromArray(im_static_proc)
	im_shifted_se = _sitk.GetImageFromArray(im_shifted_proc)

	if GradientMagnitude:
		# Apply gradient magnitude to the preprocessed images
		im_static_se = _sitk.GradientMagnitude(im_static_se)
		im_shifted_se = _sitk.GradientMagnitude(im_shifted_se)

	if HistogramMatch:
		matcher = _sitk.HistogramMatchingImageFilter()
		matcher.SetNumberOfHistogramLevels(256)
		matcher.SetNumberOfMatchPoints(50)
		matcher.ThresholdAtMeanIntensityOn()
		im_shifted_se = matcher.Execute(im_shifted_se, im_static_se)

	elastixImageFilter = _sitk.ElastixImageFilter()
	if not Verbose:
		elastixImageFilter.LogToConsoleOff()
		

	elastixImageFilter.SetFixedImage(im_static_se)
	elastixImageFilter.SetMovingImage(im_shifted_se)

	if StaticMask is not None:
		# Ensure strictly binary (0 or 1)
		StaticMask[StaticMask > 0] = 1

		# Invert Mask (SimpleITK expects 1 to include, vs usually 1 to exclude)
		sitk_static_mask_array = (1 - StaticMask).astype(np.uint8)

		# Ensure same shape
		if sitk_static_mask_array.shape != im_static_orig.shape:
			raise ValueError(f"StaticMask shape {sitk_static_mask_array.shape} does not match image shape {im_static_orig.shape}")

		# Create SITK mask with same geometry
		staticmask_se = _sitk.GetImageFromArray(sitk_static_mask_array) # Use the inverted mask
		staticmask_se.CopyInformation(im_static_se)

		# print(f'    Setting Static Mask')
		elastixImageFilter.SetFixedMask(staticmask_se)

	if ShiftedMask is not None:
		ShiftedMask[ShiftedMask > 0] = 1
		sitk_shifted_mask_array = (1 - ShiftedMask).astype(np.uint8)
		# Ensure same shape
		if sitk_shifted_mask_array.shape != im_shifted_orig.shape:
			raise ValueError(f"ShiftedMask shape {sitk_shifted_mask_array.shape} does not match image shape {im_shifted_orig.shape}")

		# Create SITK mask with same geometry
		shiftedmask_se = _sitk.GetImageFromArray(sitk_shifted_mask_array) # Use the inverted mask
		shiftedmask_se.CopyInformation(im_shifted_se)

		# print(f'    Setting Shifted Mask')
		elastixImageFilter.SetMovingMask(shiftedmask_se)


	# Set up the parameter map(s) based on the registration mode
	if TwoStage:
		# Create a list of parameter maps for a two-stage registration
		parameterMap_affine = _sitk.GetDefaultParameterMap('affine')
		parameterMap_bspline = _sitk.GetDefaultParameterMap('bspline')

		# Set the new grid spacing parameter
		if GridSpacing is not None:
			parameterMap_bspline['FinalGridSpacingInPhysicalUnits'] = [str(GridSpacing)]

		# Set the user-defined parameters for the second stage
		parameterMap_bspline['Metric'] = [Metric]
		parameterMap_bspline['Optimizer'] = [Optimizer]
		MaximumNumberOfIterations = kwargs.get('MaximumNumberOfIterations')
		if MaximumNumberOfIterations is not None:
			parameterMap_affine['MaximumNumberOfIterations'] = [str(MaximumNumberOfIterations)]
			parameterMap_bspline['MaximumNumberOfIterations'] = [str(MaximumNumberOfIterations)]

		parameterMap = [parameterMap_affine, parameterMap_bspline]
	elif Affine:
		parameterMap = _sitk.GetDefaultParameterMap('affine')
	else:
		parameterMap = _sitk.GetDefaultParameterMap('translation')
		parameterMap['Metric'] = [Metric]
		parameterMap['Transform'] = [Transform]
		parameterMap['Optimizer'] = [Optimizer]
		# Set the new grid spacing if it's a non-affine transform
		if GridSpacing is not None:
			if 'BSplineTransform' in Transform or 'SplineKernelTransform' in Transform:
				parameterMap['FinalGridSpacingInPhysicalUnits'] = [str(GridSpacing)]


	if not TwoStage and not Affine:
		# These are set by default for TwoStage and Affine
		MaximumNumberOfIterations = kwargs.get('MaximumNumberOfIterations')
		if MaximumNumberOfIterations is not None:
			parameterMap['MaximumNumberOfIterations'] = [str(MaximumNumberOfIterations)]

	elastixImageFilter.SetParameterMap(parameterMap)
	result = elastixImageFilter.Execute()

	# Apply the transform to the original, un-preprocessed image
	transformixImageFilter = _sitk.TransformixImageFilter()
	# This line was added to suppress log output from transformix
	if not Verbose:
		transformixImageFilter.LogToConsoleOff()

	transformixImageFilter.SetTransformParameterMap(elastixImageFilter.GetTransformParameterMap())
	transformixImageFilter.SetMovingImage(_sitk.GetImageFromArray(im_shifted_orig))
	result_orig = transformixImageFilter.Execute()

	# Save the transform
	transformParameterMap = elastixImageFilter.GetTransformParameterMap()

	im_coregistered = _sitk.GetArrayFromImage(result_orig)

	if Blurring:
		im_coregistered = gaussian_filter(im_coregistered, sigma=Sigma)

	return im_coregistered, transformParameterMap


#### ----------------------------------------------------------------------

def gaussian_blur_nan(image, sigma, **kwargs):
	"""
	Applies a 2D Gaussian blur to an image while correctly handling np.nan values.

	This function uses astropy.convolution.convolve to interpolate the blur
	over NaN pixels, preventing them from spreading.

	Args:
		image (np.ndarray): The 2D input image, which may contain np.nan values.
		sigma (float): The standard deviation of the Gaussian kernel.
		**kwargs: Additional keyword arguments to pass to astropy.convolution.convolve.
				  Common ones include:
				  - boundary (str): How to handle boundaries. Defaults to 'extend'.
									Options: 'extend', 'fill', 'wrap'.
				  - preserve_nan (bool): If True, pixels that were NaN in the input
										 will be set to NaN in the output. 
										 Defaults to True.

	Returns:
		np.ndarray: The blurred 2D image.
	"""
	
	# Set default behaviors
	boundary = kwargs.get('boundary', 'extend')
	preserve_nan = kwargs.get('preserve_nan', True)
	
	# 1. Create the 2D blur kernel from the sigma
	kernel = Gaussian2DKernel(x_stddev=sigma, y_stddev=sigma)
	
	# 2. Store original NaN locations if we need to preserve them
	original_nan_mask = None
	if preserve_nan:
		original_nan_mask = np.isnan(image)

	# 3. Convolve, interpolating over NaN values
	blurred_image = convolve(
		image, 
		kernel, 
		nan_treatment='interpolate', 
		boundary=boundary,
		**kwargs # Pass any other user-defined kwargs
	)
	
	# 4. (Optional) Re-apply the original NaNs
	if preserve_nan and original_nan_mask is not None:
		blurred_image[original_nan_mask] = np.nan
		
	return blurred_image

## Original basic function:
def CoRegisterHypercube(RawHypercube, Wavelengths_list, **kwargs):
	"""

	Apply Simple Elastix co-registration to all sweep

	Uses CoRegisterImages

	Input:
		- RawHypercube : To co-registrate. Shape [N, Y, X]
		- Wavelengths_list
		- kwargs:
			- Help
			- Cropping = 0
			- Order = False: Whether to order the coregistered image (based on Wavelenghts_list)
			- StaticIndex = 0: Which image is set as the static one (others are registered to it)
			- SaveHypercube
			- PlotDiff = False. If True, plots differences between static, moving and registered images
			- SavingPath. If PlotDiff or SaveHypercbybe is True, where to save the data/figure
			- EdgeMask -> Static mask
			- AllReflectionsMasks -> Moving mask
			- HideReflections = True


	Outputs:
		- Hypercube_Coregistered
		- Coregistration_Transforms

	"""
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(CoRegisterHypercube))
		return 0, 0

	PlotDiff = kwargs.get('PlotDiff', False)

	SavingPath = kwargs.get('SavingPath')
	if SavingPath is None:
		SavingPath = ''
		if PlotDiff:
			print(f'PlotDiff has been set to True. Indicate a SavingPath.')

	Static_Index = kwargs.get('StaticIndex')
	if Static_Index is None:
		Static_Index = 0
		print(f'Static index set to default {Static_Index}')

	## Handle cropping
	Cropping = kwargs.get('Cropping', 0)

	EdgeMask = kwargs.get('EdgeMask')
	AllReflectionsMasks = kwargs.get('AllReflectionsMasks')

	HideReflections = kwargs.get('HideReflections', True)

	SaveHypercube = kwargs.get('SaveHypercube', False)
	if SaveHypercube:
		print(f'Saving Hypercube')

	Order = kwargs.get('Order', False)
	Blurring = kwargs.get('Blurring', False)
	Sigma = kwargs.get('Sigma', 2)


	t0 = time.time()
	(NN, YY, XX) = RawHypercube.shape


	## Sort Wavelengths
	order_list = np.argsort(Wavelengths_list)
	Wavelengths_sorted = Wavelengths_list[order_list]

	Hypercube = []
	AllTransforms = []

	if Cropping!=0:
		print(f'Image will be cropped by {Cropping} on all sides.')
		RawHypercube = RawHypercube[:,Cropping:(-1*Cropping),Cropping:(-1*Cropping)]
		if EdgeMask is not None:
			EdgeMask=EdgeMask[Cropping:(-1*Cropping),Cropping:(-1*Cropping)]
		if AllReflectionsMasks is not None:
			AllReflectionsMasks = AllReflectionsMasks[:,Cropping:(-1*Cropping),Cropping:(-1*Cropping)]

	## Define static image
	im_static = RawHypercube[Static_Index, :,:]

	## Define static mask:
	if AllReflectionsMasks is not None:
		ReflectionsMask_Static = AllReflectionsMasks[Static_Index, :,:]
	else:
		ReflectionsMask_Static = None

	# print(f'PreRegistration -- RawHypercube.shape:{RawHypercube.shape}, AllReflectionsMasks.shape:{AllReflectionsMasks.shape}')

	StaticMask = GetGlobalMask(EdgeMask=EdgeMask, ReflectionsMask=ReflectionsMask_Static)

	for c in range(0, NN):
		if c==Static_Index:
			# pass
			im0 = copy.deepcopy(RawHypercube[c,:,:])
			print(f'  Static image')
			# # print(f'c={c}: static_im.shape = {im.shape}')

			if (HideReflections is not None) and (AllReflectionsMasks is not None):
				# if Blurring:
				# 	im0 = gaussian_filter(im0, sigma=Sigma)
				im0[ReflectionsMask_Static>0.5] = np.nan

			if Blurring:
				print(f' - Blurring -')
				# im0_blurred = gaussian_filter(im0, sigma=Sigma)
				im0_blurred = gaussian_blur_nan(im0, sigma=Sigma)
			else:
				im0_blurred = im0

			Hypercube.append(im0_blurred)
			AllTransforms.append(0)
			ReflectionsMask_Shifted = None
		else:
			print(f'Working on: {c+1} /{NN}')
			im_shifted = RawHypercube[c, :,:]
			if AllReflectionsMasks is not None:
				# print(f'   Masking Reflections')
				ReflectionsMask_Shifted = AllReflectionsMasks[c,:,:]
			else:
				ReflectionsMask_Shifted = None
			
			# print(f'Applying to mask -- ReflectionsMask_Shifted.shape:{ReflectionsMask_Shifted.shape}, EdgeMask.shape:{EdgeMask.shape}')
			ShiftedMask = GetGlobalMask(EdgeMask=EdgeMask, ReflectionsMask=ReflectionsMask_Shifted, PrintInfo=False)
			
			### PREVIOUS:
			# im_coregistered, coregister_transform = CoRegisterImages(im_static, im_shifted, StaticMask=StaticMask, ShiftedMask=ShiftedMask, **kwargs)

			### NOW
			# print(f'Starting Registration')
			# --- STEP 1 & 2: Perform Registration to get Transform Map ---
			im_coregistered_smeared, transform_map = CoRegisterImages(im_static, im_shifted, StaticMask=StaticMask, ShiftedMask=ShiftedMask, **kwargs)

			# --- STEP 3 & 4: Transform the Mask and Punch Holes ---
			im_coregistered_final = im_coregistered_smeared.astype(np.float32)
			

			if HideReflections:
				if ReflectionsMask_Shifted is not None:
					# print(f'     Transforming mask to create holes...')
					
					# Prepare the Transformix filter for the mask
					transformixImageFilter = _sitk.TransformixImageFilter()
					transformixImageFilter.LogToConsoleOff() # Keep it quiet
					
					# Set the transform map from the image registration
					transformixImageFilter.SetTransformParameterMap(transform_map)

					# Modify the parameter map to use nearest neighbor interpolation
					transform_map[0]['FinalBSplineInterpolationOrder'] = ['0']
					if len(transform_map) > 1: # Handle TwoStage registration
						transform_map[1]['FinalBSplineInterpolationOrder'] = ['0']
					
					transformixImageFilter.SetTransformParameterMap(transform_map)

					print(f"Frame {c}: transform = {transform_map[0]['TransformParameters'][:10]}")
					
					moving_mask_sitk = _sitk.GetImageFromArray(ReflectionsMask_Shifted.astype(np.uint8))
					transformixImageFilter.SetMovingImage(moving_mask_sitk)

					# Execute and get the warped mask
					registered_mask_sitk = transformixImageFilter.Execute()
					registered_mask = _sitk.GetArrayFromImage(registered_mask_sitk).astype(bool)

					# Punch holes in the final registered image
					im_coregistered_final[registered_mask] = np.nan


			if Blurring:
				# im_coregistered_final_blurred = gaussian_filter(im_coregistered_final, sigma=Sigma)
				im_coregistered_final_blurred = gaussian_blur_nan(im_coregistered_final, sigma=Sigma)
			else:
				im_coregistered_final_blurred = im_coregistered_final

			Hypercube.append(im_coregistered_final_blurred)
			AllTransforms.append(transform_map)

			if PlotDiff:
				if '.png' in SavingPath:
					NameTot = SavingPath.split('/')[-1]
					Name = NameTot.replace('.png', '')+f'_{c}.png'
					SavingPathWithName = SavingPath.replace(NameTot, Name)
				else:
					Name = f'_{c}_CoRegistration.png'
					SavingPathWithName = SavingPath+Name
				HySE.UserTools.PlotCoRegistered(im_static, im_shifted, im_coregistered, SavePlot=True, SavingPathWithName=SavingPathWithName)


	tf = time.time()
	# print(f'Hypercube size: len = {len(Hypercube)}')
	Hypercube = np.array(Hypercube)
	## Calculate time taken
	time_total = tf-t0
	minutes = int(time_total/60)
	seconds = time_total - minutes*60
	print(f'\n\n Co-registration took {minutes} min and {seconds:.0f} s in total\n')

	## Sort hypercube according to the order_list
	## Ensures wavelenghts are ordered from blue to red
	if Order:
		Hypercube_sorted = []
		AllTransforms_sorted = []
		for k in range(0,Hypercube.shape[0]):
			Hypercube_sorted.append(Hypercube[order_list[k]])
			AllTransforms_sorted.append(AllTransforms[order_list[k]])
		Hypercube_sorted = np.array(Hypercube_sorted)
	else:
		Hypercube_sorted = np.array(Hypercube)
		AllTransforms_sorted = AllTransforms


	if SaveHypercube:
		if '.png' in SavingPath:
			NameTot = SavingPath.split('/')[-1]
			Name = NameTot.replace('.png', '')+f'_CoregisteredHypercube.npz'
			Name_wav = NameTot.replace('.png', '')+f'_CoregisteredHypercube_wavelengths.npz'
			SavingPathHypercube = SavingPath.replace(NameTot, Name)
			SavingPathWavelengths = SavingPath.replace(NameTot, Name_wav)
		else:
			Name = f'_CoregisteredHypercube.npz'
			Name_wav = f'_CoregisteredHypercube_wavelengths.npz'
			SavingPathHypercube = SavingPath+Name
			SavingPathWavelengths = SavingPath+Name_wav

		np.savez(f'{SavingPathHypercube}', Hypercube)
		np.savez(f'{SavingPathWavelengths}', Wavelengths_sorted)

	return Hypercube_sorted, AllTransforms_sorted




import numpy as np
import SimpleITK as _sitk
import copy
import time
import inspect


def CoRegisterHypercubeAndMask(RawHypercube, Wavelengths_list, **kwargs):
	"""
	Apply Simple Elastix co-registration to all sweep
	Uses CoRegisterImages

	Input:
		- RawHypercube : To co-registrate. Shape [N, Y, X]
		- Wavelengths_list
		- kwargs:
			- Help
			- Cropping = 0
			- Order = False: Whether to order the coregistered image (based on Wavelenghts_list)
			- StaticIndex = 0: Which image is set as the static one (others are registered to it)
			- SaveHypercube
			- PlotDiff = False. If True, plots differences between static, moving and registered images
			- SavingPath. If PlotDiff or SaveHypercbybe is True, where to save the data/figure
			- EdgeMask -> Static mask
			- AllReflectionsMasks -> Moving mask
			- HideReflections = True
			- InteractiveMasks (False): If True, opens a window for the user to select
				a mask on the static image, and then a corresponding mask on each
				moving image to guide the registration.
			- AllROICoordinates : If indicated, the code will use the provided coordinates instead of
				prompting the user.
			- MinVal = 0 (float) : minimum value (every pixel < MinVal will be masked/set to NaN)

	Outputs:
		- Hypercube_Coregistered: The final co-registered hypercube.
		- Coregistration_Transforms: The list of transformation maps.
		- CombinedMask: A single 2D boolean mask where True indicates a pixel that
			is invalid in at least one of the co-registered frames.
		- AllROICoordinates: List of all the coordinates (y_start, y_end, x_start, x_end)
	"""
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(CoRegisterHypercubeAndMask))
		return 0, 0, 0, 0

	InteractiveMasks = kwargs.get('InteractiveMasks', False)

	# Other parameters
	PlotDiff = kwargs.get('PlotDiff', False)

	SavingPath = kwargs.get('SavingPath')
	if SavingPath is None:
		SavingPath = ''
		if PlotDiff:
			print(f'PlotDiff has been set to True. Indicate a SavingPath.')

	Static_Index = kwargs.get('StaticIndex')
	if Static_Index is None:
		Static_Index = 0
		print(f'Static index set to default {Static_Index}')

	## Handle cropping
	Cropping = kwargs.get('Cropping', 0)

	EdgeMask = kwargs.get('EdgeMask')
	AllReflectionsMasks = kwargs.get('AllReflectionsMasks')

	HideReflections = kwargs.get('HideReflections', True)

	SaveHypercube = kwargs.get('SaveHypercube', False)
	if SaveHypercube:
		print(f'Saving Hypercube')

	Order = kwargs.get('Order', False)
	Blurring = kwargs.get('Blurring', False)
	Sigma = kwargs.get('Sigma', 2)

	t0 = time.time()
	(NN, YY, XX) = RawHypercube.shape

	MinVal = kwargs.get('MinVal', 0)

	Cropping = kwargs.get('Cropping', 0)
	if Cropping != 0 and not InteractiveMasks:
		RawHypercube = RawHypercube[:, Cropping:(-1*Cropping), Cropping:(-1*Cropping)]
		if EdgeMask is not None:
			EdgeMask = EdgeMask[Cropping:(-1*Cropping), Cropping:(-1*Cropping)]
		if AllReflectionsMasks is not None:
			AllReflectionsMasks = AllReflectionsMasks[:, Cropping:(-1*Cropping), Cropping:(-1*Cropping)]

	(NN_crop, YY_crop, XX_crop) = RawHypercube.shape
	CombinedMask = np.zeros((YY_crop, XX_crop), dtype=bool)
	im_static = RawHypercube[Static_Index, :, :]

	Hypercube = []
	AllTransforms = []
	AllROICoordinates = kwargs.get('AllROICoordinates')
	if AllROICoordinates is None:
		PromptUser = True
		AllROICoordinates = []
	else:
		PromptUser = False

	StaticMask_interactive = None
	static_roi = None
	if InteractiveMasks:
		print("--- Interactive Mask Selection ---")

		# 1. Get the mask for the static image (the target)
		if PromptUser:
			# Assuming get_interactive_roi is defined externally
			static_roi = get_interactive_roi(im_static, title='Select ROI on STATIC image (this is the target)')
		else:
			static_roi = AllROICoordinates[Static_Index]
			
		if static_roi is not None:
			y_start, y_end, x_start, x_end = static_roi
			StaticMask_interactive = np.zeros_like(im_static, dtype=np.uint8)
			StaticMask_interactive[y_start:y_end, x_start:x_end] = 1 # 1 means "focus here"

	if AllReflectionsMasks is not None:
		ReflectionsMask_Static = AllReflectionsMasks[Static_Index, :, :]
	else:
		ReflectionsMask_Static = None
	
	# Assuming GetGlobalMask is available or defined
	StaticMask = HySE.GetGlobalMask(EdgeMask=EdgeMask, ReflectionsMask=ReflectionsMask_Static, PrintInfo=False)

	for c in range(0, NN):
		if c == Static_Index:
			im0 = copy.deepcopy(im_static).astype(np.float32)
			print(f'Static Image')
			if PromptUser:
				AllROICoordinates.append(static_roi)

			if StaticMask_interactive is not None:
				im0[StaticMask_interactive] = np.nan
				
			# if HideReflections is not None:
			# 	if StaticMask is not None:
			# 		# print(masking)
			# 		im0[StaticMask > 0.5] = np.nan

			# --- FIX: Apply MinVal to Static Image ---
			im0[im0 < MinVal] = np.nan
			# -----------------------------------------

			if Blurring:
				im0 = gaussian_blur_nan(im0, sigma=Sigma)
		
			Hypercube.append(im0)
			AllTransforms.append(0)

		else:
			print(f'Working on: {c+1} /{NN}')
			im_shifted = RawHypercube[c, :, :]

			StaticMask_for_reg = None
			ShiftedMask_for_reg = None

			if InteractiveMasks and StaticMask_interactive is not None:
				# 2. For each moving image, get its corresponding mask
				if PromptUser:
					shifted_roi = get_interactive_roi(im_shifted, title=f'Select ROI on MOVING image {c+1}/{NN}')
					AllROICoordinates.append(shifted_roi)
				else:
					shifted_roi = AllROICoordinates[c]
				if shifted_roi is not None:
					y_start, y_end, x_start, x_end = shifted_roi
			
					if AllReflectionsMasks is not None:
						ReflectionsMask_Shifted = AllReflectionsMasks[c, :, :]
					else:
						ReflectionsMask_Shifted = None
					
					ShiftedMask_for_reg_orig = HySE.GetGlobalMask(EdgeMask=EdgeMask, ReflectionsMask=ReflectionsMask_Shifted, PrintInfo=False)

					ShiftedMask_interactive = np.zeros_like(im_shifted, dtype=np.uint8)
					ShiftedMask_interactive[y_start:y_end, x_start:x_end] = 1

					# Use these interactive masks for the registration
					StaticMask_for_reg = (1 - StaticMask_interactive)
					ShiftedMask_for_reg = (1 - ShiftedMask_interactive)

			else:
				if AllReflectionsMasks is not None:
					ReflectionsMask_Shifted = AllReflectionsMasks[c, :, :]
				else:
					ReflectionsMask_Shifted = None
				ShiftedMask_for_reg = HySE.GetGlobalMask(EdgeMask=EdgeMask, ReflectionsMask=ReflectionsMask_Shifted, PrintInfo=False)

			# 3. Perform the masked registration
			im_coregistered_smeared, transform_map = HySE.CoRegisterImages(im_static, im_shifted, StaticMask=StaticMask_for_reg, ShiftedMask=ShiftedMask_for_reg, **kwargs)

			im_coregistered_final = im_coregistered_smeared.astype(np.float32)

			if HideReflections and ShiftedMask_for_reg is not None:
				## a. Create "Hole-Punch Mask".
				if InteractiveMasks:
					# Start with a mask that is True everywhere (remove all)
					hole_punch_mask_moving = np.ones_like(im_shifted, dtype=np.uint8)
					# Then set the ROI interior to False (keep ROI)
					hole_punch_mask_moving[y_start:y_end, x_start:x_end] = 0 
				else:
					# Start with a mask that is False everywhere (keep all)
					hole_punch_mask_moving = np.zeros_like(im_shifted, dtype=np.uint8)

				## b. Add reflections to the mask.
				if ReflectionsMask_Shifted is not None:
					hole_punch_mask_moving[ReflectionsMask_Shifted > 0.5] = 1

				## c. Now warp this combined hole-punch mask.
				transformixImageFilter = _sitk.TransformixImageFilter()
				transformixImageFilter.LogToConsoleOff()

				## Set the transform map from the image registration
				transformixImageFilter.SetTransformParameterMap(transform_map)

				## Modify the parameter map to use nearest neighbor interpolation
				transform_map[0]['FinalBSplineInterpolationOrder'] = ['0']
				if len(transform_map) > 1: # Handle TwoStage registration
					transform_map[1]['FinalBSplineInterpolationOrder'] = ['0']

				transformixImageFilter.SetTransformParameterMap(transform_map)

				moving_mask_sitk = _sitk.GetImageFromArray(hole_punch_mask_moving)
				transformixImageFilter.SetMovingImage(moving_mask_sitk)

				registered_mask_sitk = transformixImageFilter.Execute()
				registered_mask = _sitk.GetArrayFromImage(registered_mask_sitk).astype(bool)

				## e. Punch holes in the final registered image
				im_coregistered_final[registered_mask] = np.nan

			# --- FIX: Apply MinVal to Registered Image ---
			im_coregistered_final[im_coregistered_final < MinVal] = np.nan
			# ---------------------------------------------

			if Blurring:
				im_coregistered_final = gaussian_blur_nan(im_coregistered_final, sigma=Sigma)

			Hypercube.append(im_coregistered_final)
			AllTransforms.append(transform_map)

		# After each frame (static or moving) is processed, update the combined mask.
		current_frame = Hypercube[-1]
		
		# An invalid pixel is one that is NaN (from reflections OR MinVal)
		# Note: Since we set < MinVal to NaN above, np.isnan catches both cases now.
		invalid_pixels_mask = np.isnan(current_frame)
		CombinedMask = CombinedMask | invalid_pixels_mask

		# Optional Plotting
		if PlotDiff and c != Static_Index:
			if '.png' in SavingPath:
				NameTot = SavingPath.split('/')[-1]
				Name = NameTot.replace('.png', '')+f'_{c}.png'
				SavingPathWithName = SavingPath.replace(NameTot, Name)
			else:
				Name = f'_{c}_CoRegistration.png'
				SavingPathWithName = SavingPath+Name
			
			# Assuming im_coregistered was intended to be im_coregistered_final or im_coregistered_smeared
			HySE.UserTools.PlotCoRegistered(im_static, im_shifted, im_coregistered_final, SavePlot=True, SavingPathWithName=SavingPathWithName)

	tf = time.time()
	Hypercube = np.array(Hypercube)
	time_total = tf - t0
	minutes = int(time_total / 60)
	seconds = time_total - minutes * 60
	print(f'\n\n Co-registration took {minutes} min and {seconds:.0f} s in total\n')

	# Sort hypercube, transforms, and the combined mask according to the order_list
	if Order:
		order_list = np.argsort(Wavelengths_list) # Defined here to ensure variable exists
		Hypercube_sorted = Hypercube[order_list]
		# Reorder transforms list
		AllTransforms_sorted = [AllTransforms[i] for i in order_list]
		Wavelengths_sorted = np.array(Wavelengths_list)[order_list]
	else:
		Hypercube_sorted = Hypercube
		AllTransforms_sorted = AllTransforms
		Wavelengths_sorted = Wavelengths_list

	if SaveHypercube:
		if '.png' in SavingPath:
			NameTot = SavingPath.split('/')[-1]
			Name = NameTot.replace('.png', '')+f'_CoregisteredHypercube.npz'
			Name_wav = NameTot.replace('.png', '')+f'_CoregisteredHypercube_wavelengths.npz'
			SavingPathHypercube = SavingPath.replace(NameTot, Name)
			SavingPathWavelengths = SavingPath.replace(NameTot, Name_wav)
		else:
			Name = f'_CoregisteredHypercube.npz'
			Name_wav = f'_CoregisteredHypercube_wavelengths.npz'
			SavingPathHypercube = SavingPath+Name
			SavingPathWavelengths = SavingPath+Name_wav

		np.savez(f'{SavingPathHypercube}', Hypercube_sorted)
		np.savez(f'{SavingPathWavelengths}', Wavelengths_sorted)

	return Hypercube_sorted, AllTransforms_sorted, CombinedMask, AllROICoordinates

#### ----------------------------------------------------------------------


def GetHypercubeForRegistration(Nsweep, Nframe, Path, EdgePos, Wavelengths_list, **kwargs):
	"""
	Function that generates a hypercube for co-registration. 
	Specifically, it allows to select a single specific frame to run the co-registration.

	Inputs:
		- Nsweep (int) : Which sweep in the dataset to use
		- AllSweeps = False: Whether to keep all sweeps 
		- Nframe (int or [int, int, ...]) : Which frame(s), within this sweep, to use (clearest one).
			If more than one frame indicated, the code will compute a hypercube for each integer indicated and then
			concatenate all hypercubes
		- Path : Where the data is located
		- EdgePos : Positions of the start of each sweep
		- OnlySweep = True: If True, uses ComputeHypercube_RGB() to only load identified sweep. Otherwise
			loads all sweeps.
		- Wavelengths_list
		- CropImDimensions
		- kwargs:
			- Help
			- Buffer = 3 : How many frames to skip at the start and end of a sweep
			- RGB = True : To make function backwards compatible with previous methods
				before distinct RGB channels were treated differently
			- Channel = 'G' : If RGB is True, indicates which of the RGB channels to take the data from


	Outputs:
		- Hypercube for registration

	"""
	Help = kwargs.get('Help', False)
	if Help:
		print(inspect.getdoc(GetHypercubeForRegistration))
		return 0

	Buffer = kwargs.get('Buffer')
	CropImDimensions = kwargs.get('CropImDimensions')
	AllSweeps = kwargs.get('AllSweeps', False)
	OnlySweep = kwargs.get('OnlySweep', True)
	if Buffer is None:
		Buffer = 3
		print(f'Setting Buffer to default {Buffer}')
	RGB = kwargs.get('RGB', True)
	Channel = kwargs.get('Channel')
	if RGB:
#         Frames, RGB_Dark = HySE.ComputeHypercube_RGB_orig(DataPath, EdgePos_Data, Buffer=Buffer, BlueShift=1, SaveArray=False)

		if OnlySweep:
			print(f'Only loading single sweep:')
			if CropImDimensions:
				Hypercube_all, Dark_all = HySE.ComputeHypercube_RGB(Path, EdgePos, Buffer=Buffer, NsweepOnly=Nsweep, BlueShift=1, CropImDimensions=CropImDimensions, SaveArray=False)
			else:
				Hypercube_all, Dark_all = HySE.ComputeHypercube_RGB(Path, EdgePos, Buffer=Buffer, NsweepOnly=Nsweep, BlueShift=1, SaveArray=False)
		else:
			if CropImDimensions:
				Hypercube_all, Dark_all = HySE.ComputeHypercube_RGB(Path, EdgePos, Buffer=Buffer, BlueShift=1, CropImDimensions=CropImDimensions, SaveArray=False)
			else:
				print(f'Loading all sweeps (frame interator for memory efficiency):')
				Hypercube_all, Dark_all = HySE.ComputeHypercube_RGB(Path, EdgePos, Buffer=Buffer, BlueShift=1, SaveArray=False)

		print(f' -> Hypercube_all.shape: {Hypercube_all.shape}')
		
		if Channel=='G' or Channel==1:
			Channel=1
		elif Channel=='B' or Channel==0:
			Channel=0
		elif Channel=='R' or Channel==2:
			Channel=2
		else:
			print('When using the RGB option, you need to define a channel (\'R\', \'G\', \'B\' or 2, 1, 0)')
			print('  Setting to defaul green channel')
			Channel=1
		
		if OnlySweep==False:
			if AllSweeps:
				if isinstance(Nframe, int):
					print(Hypercube_all.shape)
					HypercubeForRegistration = Hypercube_all[:,:,Nframe,:,:,Channel]
				elif isinstance(Nframe, list):
					HypercubeForRegistration = []
					for i in range(0,len(Nframe)):
						HypercubeForRegistration_sub = Hypercube_all[:,:,Nframe[i],:,:,Channel]
						if i==0:
							HypercubeForRegistration = HypercubeForRegistration_sub
						else:
							HypercubeForRegistration = np.concatenate((HypercubeForRegistration, HypercubeForRegistration_sub), axis=0)
				else:
					print(f'Nframe format not accepted.')
					HypercubeForRegistration=0

			else:
				if isinstance(Nframe, int):
					print(Hypercube_all.shape)
					HypercubeForRegistration = Hypercube_all[Nsweep,:,Nframe,:,:,Channel]
				elif isinstance(Nframe, list):
					HypercubeForRegistration = []
					for i in range(0,len(Nframe)):
						HypercubeForRegistration_sub = Hypercube_all[Nsweep,:,Nframe[i],:,:,Channel]
						if i==0:
							HypercubeForRegistration = HypercubeForRegistration_sub
						else:
							HypercubeForRegistration = np.concatenate((HypercubeForRegistration, HypercubeForRegistration_sub), axis=0)
				else:
					print(f'Nframe format not accepted.')
					HypercubeForRegistration=0
		else:
			HypercubeForRegistration = Hypercube_all[0,:,Nframe,:,:,Channel]
		
		
	else:
		if AllSweeps==False:
			print('RGB mode is set to False')
			print(f'  Note that you might need to adjust the buffer size to account for this. It is currently set to {Buffer}')
			if CropImDimensions:
				Hypercube_all, Dark_all = HySE.ComputeHypercube(Path, EdgePos, Buffer=Buffer, Average=False, Order=False, Help=False, SaveFig=False, 
					CropImDimensions=CropImDimensions, SaveArray=False, Plot=False, ForCoRegistration=True)
			else:
				Hypercube_all, Dark_all = HySE.ComputeHypercube(Path, EdgePos, Buffer=Buffer, Average=False, Order=False, Help=False, SaveFig=False, SaveArray=False, Plot=False, ForCoRegistration=True)
			
			if isinstance(Nframe, int):
				HypercubeForRegistration = Hypercube_all[Nsweep,:,Nframe,:,:]
			elif isinstance(Nframe, list):
				HypercubeForRegistration = []
				for i in range(0,len(Nframe)):
					HypercubeForRegistration_sub = Hypercube_all[Nsweep,:,Nframe[i],:,:]
					if i==0:
						HypercubeForRegistration = HypercubeForRegistration_sub
					else:
						HypercubeForRegistration = np.concatenate((HypercubeForRegistration, HypercubeForRegistration_sub), axis=0)
			else:
				print(f'Nframe format not accepted.')
				HypercubeForRegistration=0
		
	
	return HypercubeForRegistration








			

