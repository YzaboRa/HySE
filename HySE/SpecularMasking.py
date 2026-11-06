"""
SpecularMasking.py

Detect and mask specular reflections (bright glints on wet mucosa) in a REGISTERED
hypercube, with a GUI to check that the right pixels - and only those - are masked.

Where it goes in the pre-processing notebook
--------------------------------------------
	1. Detect on the registered red hypercube (CoregisteredHypercube), i.e. after
	   "Review Points" / "Remove Bad Frames" and before "Unmixing Parameters".
	2. Apply the resulting per-frame masks (as NaNs) to RegGreen / RegBlue, right after
	   "Apply All Transforms" and before "Combine Corresponding Frames", then combine with
	   NaN_aware=True. Pixels that are specular in only some of the repeated frames of a
	   wavelength are averaged from the remaining ones; pixels specular in ALL repeats
	   become NaN, which UnmixDataSmoothNNLS already skips and RestoreNaNs restores.

How detection works (per frame)
-------------------------------
	A specular highlight is a small, bright feature sitting on a smoother tissue background.
	1. Background = morphological opening of the frame (removes anything narrower than
	   OpeningSize px), computed only inside the frame's support so the registration borders
	   do not bias it, then smoothed.
	2. Residual = frame - background. Tissue (even bright tissue) has a residual close to 0;
	   glints stand out.
	3. Hysteresis threshold, in units of the frame's median tissue intensity:
	   a connected blob is kept if its peak is above SeedThreshold; it then extends to every
	   connected pixel above GrowThreshold (this captures the halo around a glint).
	4. The kept blobs are dilated by Dilation px to also cover the bloom / blur left by
	   registration interpolation.

With Claude Sonnet 5.5 from Anthropic.
"""

import re
import json
import traceback

import numpy as np
import cv2
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons


############################################################
# Core detection
############################################################

def _MaskedOpening(Frame, Domain, Size):
	"""
	Morphological opening restricted to `Domain` (pixels outside the domain neither
	constrain the erosion nor contribute to the dilation), so the zero-padded borders left by
	registration do not create fake highlights - and highlights touching the border
	are still removed from the background estimate.
	"""
	Kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (int(Size), int(Size)))
	Big = np.float32(Frame[Domain].max() * 2 + 1) if Domain.any() else np.float32(1)
	A = np.ascontiguousarray(Frame.copy())
	A[~Domain] = Big
	E = cv2.erode(A, Kernel)
	E[~Domain] = -Big
	return cv2.dilate(E, Kernel)


def _MedianBackground(Frame, Domain, Size, Factor=3):
	"""
	Centre-surround background: large-window median of the frame, computed at reduced
	resolution (fast). Unlike an opening it is not biased on smooth shoulders / ramps
	(e.g. the vignetting towards the registration footprint edge), so a feature only counts as
	'bright' if it is brighter than its surroundings in (almost) all directions.
	"""
	Idx = ndi.distance_transform_edt(~Domain, return_distances=False, return_indices=True)
	Filled = Frame[tuple(Idx)]
	H, W = Frame.shape
	Small = cv2.resize(Filled, (max(W // Factor, 1), max(H // Factor, 1)), interpolation=cv2.INTER_AREA)
	K = max(int(round(Size / Factor)), 3) | 1
	Med = ndi.median_filter(Small, size=K, mode='nearest')
	return cv2.resize(Med, (W, H), interpolation=cv2.INTER_CUBIC)


def ComputeSpecularResidual(Frame, ValidMask, OpeningSize=25):
	"""
	Compute the 'bright-feature' residual of a single frame.

	Parameters:
	-----------
	Frame : 2D array (Y, X). May contain NaNs / zeros outside the registered footprint.
	ValidMask : 2D bool (Y, X). True = valid pixel (i.e. ~CombinedMask).
	OpeningSize : int. Diameter (px) of the structuring element. Features narrower than this
		are considered 'specular'. Should be a bit larger than the widest glint you want to catch
		(but much smaller than the structures you want to keep).

	Returns:
	--------
	Residual : 2D float32 array, Frame - smooth background (0 outside the frame's support)
	Scale : float, median intensity of the valid pixels (used to express thresholds
		relative to the tissue brightness, so they transfer across wavelengths / patients)
	"""
	Frame = np.nan_to_num(np.asarray(Frame, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
	ValidMask = np.asarray(ValidMask, dtype=bool)
	if not ValidMask.any():
		return np.zeros_like(Frame), 0.0

	Domain = (Frame > 0) | ValidMask

	# (a) opening-based background (smoothed). Reference for 'narrower than OpeningSize'
	Background = _MaskedOpening(Frame, Domain, OpeningSize)
	Sigma = max(OpeningSize / 4.0, 1.0)
	W = Domain.astype(np.float32)
	Smooth = cv2.GaussianBlur(Background * W, (0, 0), Sigma) / np.maximum(cv2.GaussianBlur(W, (0, 0), Sigma), 1e-3)
	ResidualOpen = Frame - Smooth

	# (b) median-based background. Rejects smooth shoulders / texture that an opening mistakes for glints
	ResidualMedian = Frame - _MedianBackground(Frame, Domain, int(round(1.5 * OpeningSize)))

	# A pixel is a glint only if it is bright relative to BOTH backgrounds
	Residual = np.minimum(ResidualOpen, ResidualMedian).astype(np.float32)
	Residual[~Domain] = 0
	Scale = float(np.median(Frame[ValidMask]))
	return Residual, Scale


def ThresholdSpecularResidual(Residual, Scale, ValidMask, SeedThreshold=0.3, GrowThreshold=0.12,
							  Dilation=2, MinArea=1, ExcludeMask=None):
	"""
	Turn a residual into a boolean specular mask (True = specular).

	Parameters:
	-----------
	Residual, Scale : output of ComputeSpecularResidual
	ValidMask : 2D bool, True = valid pixel
	SeedThreshold : a blob is kept only if its peak residual > SeedThreshold * Scale
	GrowThreshold : the blob extends to all connected pixels with residual > GrowThreshold * Scale
		(must be <= SeedThreshold)
	Dilation : px radius of the final dilation (covers bloom and registration blur). 0 = none.
	MinArea : minimum size (px, before dilation) of a blob to keep. Raise it to ignore isolated single-pixel hot spots.
	ExcludeMask : optional 2D bool, True = never flag (e.g. the rigid endoscope rim, which is not tissue)
	"""
	if GrowThreshold > SeedThreshold:
		raise ValueError(f'GrowThreshold ({GrowThreshold}) must be <= SeedThreshold ({SeedThreshold}).')

	Allowed = np.asarray(ValidMask, dtype=bool).copy()
	if ExcludeMask is not None:
		Allowed &= ~np.asarray(ExcludeMask, dtype=bool)
	if Scale <= 0 or not Allowed.any():
		return np.zeros(Residual.shape, dtype=bool)

	Low = (Residual > GrowThreshold * Scale) & Allowed
	Labels, N = ndi.label(Low, structure=np.ones((3, 3)))
	if N == 0:
		return np.zeros(Residual.shape, dtype=bool)

	Index = np.arange(1, N + 1)
	Peak = np.asarray(ndi.maximum(Residual, Labels, Index))
	Area = np.bincount(Labels.ravel(), minlength=N + 1)[1:]
	Keep = np.zeros(N + 1, dtype=bool)
	Keep[1:] = (Peak > SeedThreshold * Scale) & (Area >= MinArea)
	Mask = Keep[Labels]

	if Dilation > 0:
		K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(Dilation) + 1, 2 * int(Dilation) + 1))
		Mask = cv2.dilate(Mask.astype(np.uint8), K).astype(bool)
	return Mask & Allowed


def DetectSpecularReflections(Hypercube, CombinedMask, GoodFramesLabels=None, OpeningSize=25,
							  SeedThreshold=0.3, GrowThreshold=0.12, Dilation=2, MinArea=1,
							  ExcludeMask=None, MaxMaskedFraction=0.10, Verbose=True):
	"""
	Detect specular reflections in every frame of a registered hypercube.

	Parameters:
	-----------
	Hypercube : 3D array (Nframes, Y, X). Use the registered red hypercube (CoregisteredHypercube).
	CombinedMask : 2D bool (Y, X). HySE convention: True = INVALID pixel.
	GoodFramesLabels : list of str, one per frame (e.g. 'S3_Frame7'). Used for reporting only.
	OpeningSize, SeedThreshold, GrowThreshold, Dilation, MinArea, ExcludeMask :
		see ComputeSpecularResidual / ThresholdSpecularResidual.
	MaxMaskedFraction : frames where more than this fraction of valid pixels is flagged
		are marked '<<< CHECK' (a sign the thresholds are too low, or the frame is mostly saturated).

	Returns:
	--------
	SpecularMasks : 3D bool array (Nframes, Y, X). True = specular. One mask PER FRAME,
		because a glint only corrupts the frame (wavelength) it appears in.
	"""
	Hypercube = np.asarray(Hypercube) if not isinstance(Hypercube, np.ndarray) else Hypercube
	if Hypercube.ndim != 3:
		raise ValueError(f'Hypercube must be 3D (Nframes, Y, X), got shape {Hypercube.shape}.')
	N, H, W = Hypercube.shape
	CombinedMask = np.asarray(CombinedMask, dtype=bool)
	if CombinedMask.shape != (H, W):
		raise ValueError(f'CombinedMask shape {CombinedMask.shape} does not match the hypercube frames {(H, W)}.')
	if GoodFramesLabels is not None and len(GoodFramesLabels) != N:
		raise ValueError(f'{len(GoodFramesLabels)} labels for {N} frames - they must match.')
	if ExcludeMask is not None and np.asarray(ExcludeMask).shape != (H, W):
		raise ValueError(f'ExcludeMask shape {np.asarray(ExcludeMask).shape} does not match the frames {(H, W)}.')

	Labels = list(GoodFramesLabels) if GoodFramesLabels is not None else [f'Frame {i}' for i in range(N)]
	Valid = ~CombinedMask

	# Sanity check on the mask convention (valid tissue should be brighter than the discarded border)
	if CombinedMask.any() and Valid.any():
		F0 = np.nan_to_num(Hypercube[0].astype(np.float32))
		if F0[Valid].mean() < F0[CombinedMask].mean():
			print('WARNING: the valid region is darker than the masked region. Is CombinedMask inverted? '
				  '(HySE convention: True = INVALID.)')

	SpecularMasks = np.zeros((N, H, W), dtype=bool)
	Fractions = np.zeros(N)
	if Verbose:
		print(f'Detecting specular reflections in {N} frames '
			  f'(OpeningSize={OpeningSize}, Seed={SeedThreshold}, Grow={GrowThreshold}, Dilation={Dilation}, MinArea={MinArea})')

	for i in range(N):
		Residual, Scale = ComputeSpecularResidual(Hypercube[i], Valid, OpeningSize=OpeningSize)
		M = ThresholdSpecularResidual(Residual, Scale, Valid, SeedThreshold=SeedThreshold,
									  GrowThreshold=GrowThreshold, Dilation=Dilation,
									  MinArea=MinArea, ExcludeMask=ExcludeMask)
		SpecularMasks[i] = M
		Fractions[i] = M.sum() / max(Valid.sum(), 1)
		if Verbose:
			NBlobs = ndi.label(M)[1]
			Flag = '   <<< CHECK' if Fractions[i] > MaxMaskedFraction else ''
			print(f'  {Labels[i]:>14s} : {100 * Fractions[i]:5.2f}% of valid pixels, {NBlobs:3d} blobs{Flag}')

	if Verbose:
		print(f'Mean masked fraction: {100 * Fractions.mean():.2f}%  (max {100 * Fractions.max():.2f}% in {Labels[int(Fractions.argmax())]})')
	return SpecularMasks


############################################################
# Applying the masks and quantifying the data loss
############################################################

def ApplySpecularMask(Stack, SpecularMasks, InPlace=False):
	"""
	Set the specular pixels of each frame to NaN.

	Stack : 3D array (Nframes, Y, X), frame-aligned with SpecularMasks
		(e.g. RegGreen or RegBlue, which share the order of GoodFramesLabels).
	SpecularMasks : 3D bool (Nframes, Y, X) from DetectSpecularReflections.
	InPlace : modify Stack directly (saves memory on large datasets). Stack must be float.

	Returns the NaN-masked float32 stack. Combine afterwards with HySE.CombineFrames(..., NaN_aware=True).
	"""
	if Stack.shape != SpecularMasks.shape:
		raise ValueError(f'Stack {Stack.shape} and SpecularMasks {SpecularMasks.shape} are not aligned. '
						 f'Both must be (Nframes, Y, X) in the same frame order (GoodFramesLabels).')
	if InPlace:
		if not np.issubdtype(Stack.dtype, np.floating):
			raise TypeError('InPlace=True needs a float stack (NaN cannot be stored in integer arrays).')
		Out = Stack
	else:
		Out = Stack.astype(np.float32, copy=True)
	Out[SpecularMasks] = np.nan
	return Out


def ReportSpecularLoss(SpecularMasks, CombinedMask, GoodFramesLabels, NWavelengths=16, Verbose=True):
	"""
	Quantify what the masking costs, per wavelength (frame index), using the frame LABELS.

	For each wavelength:
	  - 'any repeat masked' : valid pixels where at least one repeat is specular (averaged from fewer frames)
	  - 'all repeats masked': valid pixels where EVERY repeat is specular -> NaN after CombineFrames(NaN_aware=True)
	Finally, the fraction of valid pixels that lose at least one wavelength (these are skipped by the unmixing).

	Returns a dict with these numbers.
	"""
	Valid = ~np.asarray(CombinedMask, dtype=bool)
	NValid = max(int(Valid.sum()), 1)
	if len(GoodFramesLabels) != len(SpecularMasks):
		raise ValueError(f'{len(GoodFramesLabels)} labels for {len(SpecularMasks)} masks.')

	ByWav = {}
	for i, Label in enumerate(GoodFramesLabels):
		Match = re.search(r'Frame(\d+)', Label)
		if Match is None:
			print(f'WARNING: cannot parse wavelength index from label {Label!r}; ignoring it in the report.')
			continue
		ByWav.setdefault(int(Match.group(1)), []).append(i)

	Result = {'AnyRepeat': {}, 'AllRepeats': {}, 'NRepeats': {}}
	AnyLost = np.zeros(Valid.shape, dtype=bool)
	if Verbose:
		print(f'{"wav idx":>8s} {"repeats":>8s} {"any masked":>11s} {"all masked (-> NaN)":>20s}')
	for W in range(NWavelengths):
		Idx = ByWav.get(W, [])
		if not Idx:
			if Verbose:
				print(f'{W:>8d} {0:>8d}   (no frames kept for this wavelength!)')
			continue
		Sub = SpecularMasks[Idx]
		Any = Sub.any(axis=0) & Valid
		All = Sub.all(axis=0) & Valid
		AnyLost |= All
		Result['AnyRepeat'][W] = Any.sum() / NValid
		Result['AllRepeats'][W] = All.sum() / NValid
		Result['NRepeats'][W] = len(Idx)
		if Verbose:
			print(f'{W:>8d} {len(Idx):>8d} {100 * Result["AnyRepeat"][W]:>10.2f}% {100 * Result["AllRepeats"][W]:>19.2f}%')
	Result['PixelsLosingAWavelength'] = AnyLost.sum() / NValid
	if Verbose:
		print(f'Valid pixels losing at least one wavelength (NaN -> excluded from unmixing): '
			  f'{100 * Result["PixelsLosingAWavelength"]:.2f}%')
	return Result


def PolygonToMask(Vertices, Shape):
	"""
	Boolean mask (True inside) from polygon vertices given as (x, y) pixel coordinates,
	e.g. for the ExcludeMask argument, to keep the rigid endoscope rim (not tissue) from being flagged.
	Read the (x, y) coordinates off the matplotlib toolbar readout while hovering over the image.

		ExcludeMask = HySE.PolygonToMask([(0, 0), (260, 0), (0, 230)], CombinedMask.shape)
	"""
	Mask = np.zeros(Shape, dtype=np.uint8)
	cv2.fillPoly(Mask, [np.round(np.array(Vertices, dtype=float)).astype(np.int32)], 1)
	return Mask.astype(bool)


############################################################
# Saving / loading (label-checked)
############################################################

def SaveSpecularMasks(SavingPath, SpecularMasks, GoodFramesLabels, Params=None):
	"""Save the masks with their frame labels and the parameters used (compressed .npz)."""
	if len(GoodFramesLabels) != len(SpecularMasks):
		raise ValueError(f'{len(GoodFramesLabels)} labels for {len(SpecularMasks)} masks.')
	np.savez_compressed(SavingPath, SpecularMasks=SpecularMasks,
						Labels=np.array(GoodFramesLabels, dtype=str),
						Params=json.dumps(Params if Params is not None else {}))
	print(f'Saved {len(SpecularMasks)} specular masks to {SavingPath}')


def LoadSpecularMasks(Path, GoodFramesLabels):
	"""
	Load masks saved by SaveSpecularMasks and CHECK they belong to the current frames.
	If the saved labels differ from GoodFramesLabels (e.g. a frame was removed since), the
	mismatch is printed and a ValueError is raised rather than silently misaligning masks.

	Returns (SpecularMasks, Params dict).
	"""
	File = np.load(Path, allow_pickle=False)
	Saved = [str(s) for s in File['Labels']]
	Current = list(GoodFramesLabels)
	if Saved != Current:
		print(f'Label mismatch between {Path} and the current frames:')
		print(f'  only in saved masks : {[l for l in Saved if l not in Current]}')
		print(f'  only in current data: {[l for l in Current if l not in Saved]}')
		if sorted(Saved) == sorted(Current):
			print('  (same labels, different order)')
		raise ValueError('Saved specular masks are not aligned with the current frames. Re-run the detection.')
	return File['SpecularMasks'], json.loads(str(File['Params']))


############################################################
# Visual checks
############################################################

def PlotSpecularOverview(Hypercube, SpecularMasks, CombinedMask, GoodFramesLabels=None,
						 NCols=6, Downsample=3, SavingPath=None, Show=True):
	"""
	Contact sheet of every frame with the specular mask outlined in red.
	The area excluded by CombinedMask is dimmed. Good for a final check / for saving a QC figure.
	"""
	N, H, W = Hypercube.shape
	Labels = list(GoodFramesLabels) if GoodFramesLabels is not None else [f'Frame {i}' for i in range(N)]
	Valid = ~np.asarray(CombinedMask, dtype=bool)
	NRows = int(np.ceil(N / NCols))
	Fig, Ax = plt.subplots(NRows, NCols, figsize=(2 * NCols, 2 * NRows), squeeze=False)
	Size = (W // Downsample, H // Downsample)
	ValidS = cv2.resize(Valid.astype(np.uint8), Size, interpolation=cv2.INTER_NEAREST).astype(bool)
	K = np.ones((3, 3), np.uint8)
	for a in Ax.ravel():
		a.axis('off')
	for i in range(N):
		F = np.nan_to_num(Hypercube[i].astype(np.float32))
		FS = cv2.resize(F, Size, interpolation=cv2.INTER_AREA)
		Hi = np.percentile(FS[ValidS], 99.9) if ValidS.any() else FS.max()
		G = np.clip(FS / max(Hi, 1e-9), 0, 1) ** 0.8
		RGB = np.dstack([G, G, G])
		RGB[~ValidS] *= 0.4
		MS = cv2.resize(SpecularMasks[i].astype(np.uint8), Size, interpolation=cv2.INTER_NEAREST)
		Edge = (MS - cv2.erode(MS, K)).astype(bool)
		RGB[Edge] = [1, 0.1, 0.1]
		a = Ax[i // NCols, i % NCols]
		a.imshow(RGB)
		a.set_title(f'{Labels[i]}  {100 * SpecularMasks[i].sum() / max(Valid.sum(), 1):.1f}%', fontsize=9)
	plt.tight_layout()
	if SavingPath is not None:
		Fig.savefig(SavingPath, dpi=150)
		print(f'Saved overview to {SavingPath}')
	if Show:
		plt.show()
	return Fig


class SpecularMaskReviewer:
	"""
	Tune and check the specular mask frame by frame.

	Three linked panels (zoom/pan in one, all follow):
	    left   : the frame, untouched
	    middle : the same frame with the mask in red (+ outline). Pixels that would become NaN.
	    right  : the residual / Scale map the thresholds act on (jet). 0 = tissue, 1 = glint as bright as the whole tissue.
	The area excluded by CombinedMask is dimmed in the two left panels.

	Sliders:
	    Frame, SeedThreshold, GrowThreshold, Dilation, MinArea, OpeningSize, display min / max.
	Keys: left / right arrows change frame.
	Buttons:
	    'Apply to all frames' : run the current settings on every frame and print the per-frame report
	    'Finish / Save'       : same, then close the window; read the outcome with get_results()

	To run:
		%matplotlib qt
		Reviewer = HySE.SpecularMaskReviewer(CoregisteredHypercube, CombinedMask, GoodFramesLabels=GoodFramesLabels)

	Extract:
		Results = Reviewer.get_results()
		SpecularMasks, SpecularParams = Results['SpecularMasks'], Results['Params']

	With Claude Sonnet 5.5 from Anthropic.
	"""

	def __init__(self, Hypercube, CombinedMask, GoodFramesLabels=None, OpeningSize=25,
				 SeedThreshold=0.3, GrowThreshold=0.12, Dilation=2, MinArea=1,
				 ExcludeMask=None, colourmap='gray'):
		self.cube = Hypercube
		self.n_frames, self.H, self.W = Hypercube.shape
		self.combined_mask = np.asarray(CombinedMask, dtype=bool)
		if self.combined_mask.shape != (self.H, self.W):
			raise ValueError(f'CombinedMask {self.combined_mask.shape} does not match frames {(self.H, self.W)}.')
		self.valid = ~self.combined_mask
		self.exclude = None if ExcludeMask is None else np.asarray(ExcludeMask, dtype=bool)
		self.labels = list(GoodFramesLabels) if GoodFramesLabels is not None else [f'Frame {i}' for i in range(self.n_frames)]
		if len(self.labels) != self.n_frames:
			raise ValueError(f'{len(self.labels)} labels for {self.n_frames} frames.')
		self.cmap_name = colourmap
		self.curr_frame = 0
		self._cache = {}
		self._results = None
		self._results_params = None
		self._finished = False

		# ---------------- Layout ----------------
		self.fig = plt.figure(figsize=(18, 9.5))
		gs = self.fig.add_gridspec(1, 3, wspace=0.05)
		self.fig.subplots_adjust(bottom=0.33, right=0.93, left=0.03, top=0.92)
		self.ax_raw = self.fig.add_subplot(gs[0, 0])
		self.ax_over = self.fig.add_subplot(gs[0, 1], sharex=self.ax_raw, sharey=self.ax_raw)
		self.ax_res = self.fig.add_subplot(gs[0, 2], sharex=self.ax_raw, sharey=self.ax_raw)
		self.cbar_ax = self.fig.add_axes([0.945, 0.40, 0.012, 0.5])
		for a, t in ((self.ax_raw, 'Frame'), (self.ax_over, 'Specular mask'), (self.ax_res, 'Residual / Scale')):
			a.set_title(t)
			a.axis('off')

		blank = np.zeros((self.H, self.W))
		self.im_raw = self.ax_raw.imshow(blank, cmap=self.cmap_name, vmin=0, vmax=1)
		self.im_base = self.ax_over.imshow(blank, cmap=self.cmap_name, vmin=0, vmax=1)
		self.im_over = self.ax_over.imshow(np.zeros((self.H, self.W, 4)))
		self.im_res = self.ax_res.imshow(blank, cmap='jet', vmin=0, vmax=1)
		self.fig.colorbar(self.im_res, cax=self.cbar_ax)
		self.im_dim_raw = self.ax_raw.imshow(self._dim_rgba(), zorder=5)
		self.im_dim_over = self.ax_over.imshow(self._dim_rgba(), zorder=5)
		self.stats_text = self.fig.text(0.5, 0.955, '', ha='center', fontsize=14)

		# ---------------- Controls ----------------
		def _ax(rect):
			return self.fig.add_axes(rect)

		self.slider_frame = Slider(_ax([0.12, 0.26, 0.55, 0.022]), 'Frame', 0, self.n_frames - 1, valinit=0, valstep=1)
		self.slider_seed = Slider(_ax([0.12, 0.22, 0.55, 0.022]), 'Seed thr.', 0.02, 1.5, valinit=SeedThreshold)
		self.slider_grow = Slider(_ax([0.12, 0.18, 0.55, 0.022]), 'Grow thr.', 0.01, 1.5, valinit=GrowThreshold)
		self.slider_dil = Slider(_ax([0.12, 0.14, 0.55, 0.022]), 'Dilation', 0, 12, valinit=Dilation, valstep=1)
		self.slider_area = Slider(_ax([0.12, 0.10, 0.55, 0.022]), 'Min area', 1, 100, valinit=MinArea, valstep=1)
		self.slider_open = Slider(_ax([0.12, 0.06, 0.55, 0.022]), 'Opening size', 5, 81, valinit=OpeningSize, valstep=2)
		self.slider_min = Slider(_ax([0.75, 0.26, 0.2, 0.022]), 'Disp min', 0, 1, valinit=0.0)
		self.slider_max = Slider(_ax([0.75, 0.22, 0.2, 0.022]), 'Disp max', 0, 1, valinit=1.0)
		self.check = CheckButtons(_ax([0.75, 0.13, 0.12, 0.07]), ['Dim invalid area', 'Fill mask'], [True, True])
		self.btn_all = Button(_ax([0.75, 0.07, 0.2, 0.04]), 'Apply to all frames')
		self.btn_done = Button(_ax([0.75, 0.02, 0.2, 0.04]), 'Finish / Save')

		self.slider_frame.on_changed(self.update_frame)
		for s in (self.slider_seed, self.slider_grow, self.slider_dil, self.slider_area, self.slider_open):
			s.on_changed(self.update_mask)
		self.slider_min.on_changed(self.update_display)
		self.slider_max.on_changed(self.update_display)
		self.check.on_clicked(self.update_mask)
		self.btn_all.on_clicked(self.apply_to_all)
		self.btn_done.on_clicked(self.finish)
		self.fig.canvas.mpl_connect('key_press_event', self.on_key)

		self.update_frame(0)
		plt.show()

	# ---------------- helpers ----------------
	def _safe(method):
		def wrapped(self, *args, **kwargs):
			try:
				return method(self, *args, **kwargs)
			except Exception:
				print(f'[SpecularMaskReviewer ERROR in {method.__name__}]')
				traceback.print_exc()
		return wrapped

	def _dim_rgba(self):
		A = np.zeros((self.H, self.W, 4), dtype=np.float32)
		A[self.combined_mask, 3] = 0.6
		return A

	def _params(self):
		return dict(OpeningSize=int(self.slider_open.val), SeedThreshold=float(self.slider_seed.val),
					GrowThreshold=float(self.slider_grow.val), Dilation=int(self.slider_dil.val),
					MinArea=int(self.slider_area.val))

	def _residual(self, frame_idx, opening):
		key = (frame_idx, opening)
		if key not in self._cache:
			if len(self._cache) > 8:
				self._cache.pop(next(iter(self._cache)))
			self._cache[key] = ComputeSpecularResidual(self.cube[frame_idx], self.valid, OpeningSize=opening)
		return self._cache[key]

	def _normalised_frame(self):
		F = np.nan_to_num(self.cube[self.curr_frame].astype(np.float32))
		hi = np.percentile(F[self.valid], 99.9) if self.valid.any() else F.max()
		return np.clip(F / max(hi, 1e-9), 0, 1)

	# ---------------- callbacks ----------------
	@_safe
	def update_frame(self, val):
		self.curr_frame = int(self.slider_frame.val)
		self._frame_norm = self._normalised_frame()
		self.update_display(None)
		self.update_mask(None)

	@_safe
	def update_display(self, val):
		vmin, vmax = self.slider_min.val, self.slider_max.val
		if vmin >= vmax:
			vmax = vmin + 0.01
		for im in (self.im_raw, self.im_base):
			im.set_data(self._frame_norm)
			im.set_clim(vmin, vmax)
		self.fig.canvas.draw_idle()

	@_safe
	def update_mask(self, val):
		p = self._params()
		if p['GrowThreshold'] > p['SeedThreshold']:
			self.stats_text.set_text('Grow threshold must be <= seed threshold')
			self.fig.canvas.draw_idle()
			return
		residual, scale = self._residual(self.curr_frame, p['OpeningSize'])
		mask = ThresholdSpecularResidual(residual, scale, self.valid, SeedThreshold=p['SeedThreshold'],
										 GrowThreshold=p['GrowThreshold'], Dilation=p['Dilation'],
										 MinArea=p['MinArea'], ExcludeMask=self.exclude)
		dim, fill = self.check.get_status()
		overlay = np.zeros((self.H, self.W, 4), dtype=np.float32)
		if fill:
			overlay[mask] = [1.0, 0.1, 0.1, 0.55]
		edge = mask & ~cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
		overlay[edge] = [1.0, 1.0, 0.0, 1.0]
		self.im_over.set_data(overlay)
		self.im_res.set_data(np.clip(residual / scale, 0, 1) if scale > 0 else residual)
		for im in (self.im_dim_raw, self.im_dim_over):
			im.set_visible(bool(dim))
		self.stats_text.set_text(f'{self.labels[self.curr_frame]}  ({self.curr_frame + 1}/{self.n_frames})   |   '
								 f'{100 * mask.sum() / max(self.valid.sum(), 1):.2f}% of valid pixels masked, '
								 f'{ndi.label(mask)[1]} blobs')
		self.fig.canvas.draw_idle()

	@_safe
	def on_key(self, event):
		if event.key == 'right':
			self.slider_frame.set_val(min(self.curr_frame + 1, self.n_frames - 1))
		elif event.key == 'left':
			self.slider_frame.set_val(max(self.curr_frame - 1, 0))

	def _compute_all(self):
		p = self._params()
		print(f'\nApplying to all frames with: {p}')
		self._results = DetectSpecularReflections(self.cube, self.combined_mask, GoodFramesLabels=self.labels,
												  ExcludeMask=self.exclude, **p)
		self._results_params = p
		print('\nParameters to reuse without the GUI:')
		print(f'  SpecularParams = {p}')

	@_safe
	def apply_to_all(self, event):
		self._compute_all()

	@_safe
	def finish(self, event):
		self._compute_all()
		self._finished = True
		plt.close(self.fig)

	def get_results(self):
		"""
		Returns {'SpecularMasks': bool (Nframes, Y, X), 'Params': dict}.
		If the settings changed since the last 'Apply to all frames', the masks are recomputed
		so the returned masks always match the returned parameters.
		"""
		if self._results is None or self._results_params != self._params():
			self._compute_all()
		return {'SpecularMasks': self._results, 'Params': dict(self._results_params)}
