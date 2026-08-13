"""

Review of manually identified points

N.B. Functions and GUI written with Claude


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

## Points Reviewer

class PointsReviewer:
	"""
	Review and edit manually-placed landmark points, styled to match HySE's
	LandmarkPicker (GridSpec layout, per-panel 0-1 normalized contrast sliders,
	jet colormap point coding with colorbar, gray/magma/viridis radio buttons).

	Left panel: fixed/static frame with fixed_points (always visible for reference).
	Right panel: current moving frame, navigate with slider or left/right arrows.
	Double-click a point to select it (drawn with a black ring), then click
	elsewhere to move it. Right-click cancels a pending selection.
	Click 'Finish / Save' when done.

	To run:
		%matplotlib qt
		Reviewer = PointsReviewer(HypercubeForRegistration, AllLandmarkPoints, StaticIndex=index,GoodFramesLabels=GoodFramesLabels)

	Extract:
		UpdatedPoints = Reviewer.get_results()
		AllLandmarkPoints = UpdatedPoints 

	Then run the manual registration with AllLandmarkPoints already defined to re-compute the proper transforms

	With Claude Sonnet 5 from Anthropic.
	"""

	def __init__(self, HypercubeForRegistration, AllPoints, StaticIndex,
				 GoodFramesLabels=None, colourmap='magma', pick_radius=15):
		self.cube = HypercubeForRegistration
		self.n_frames = self.cube.shape[0]
		self.StaticIndex = StaticIndex
		self.labels = GoodFramesLabels if GoodFramesLabels is not None else \
			[f'Frame {i}' for i in range(self.n_frames)]

		self.fixed_points = np.array(AllPoints['fixed_points'], dtype=float)
		self.moving_points = [np.array(p, dtype=float) for p in AllPoints['moving_points']]
		self.n_points = len(self.fixed_points)

		for i, pts in enumerate(self.moving_points):
			if len(pts) != self.n_points:
				print(f'WARNING: {self.labels[i]} has {len(pts)} points, '
					  f'expected {self.n_points}. This WILL break navigation to that frame '
					  f'until fixed.')

		self.pick_radius = pick_radius
		self.curr_frame = 0
		self.selected_point = None
		self.cmap_name = colourmap
		self.plotted_artists = []

		# --- Layout: matches LandmarkPicker's GridSpec + colorbar ---
		self.fig = plt.figure(figsize=(16, 9))
		gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.1)
		self.fig.subplots_adjust(bottom=0.3, right=0.88)

		self.ax_fixed = self.fig.add_subplot(gs[0, 0])
		self.ax_moving = self.fig.add_subplot(gs[0, 1])
		self.cbar_ax = self.fig.add_axes([0.9, 0.3, 0.02, 0.55])

		# --- Normalize each panel's own reference frame 0-1, like LandmarkPicker ---
		self.fixed_img_norm = self._normalize(self.cube[self.StaticIndex])
		self.im_fixed_obj = self.ax_fixed.imshow(self.fixed_img_norm, cmap=self.cmap_name)
		self.ax_fixed.set_title(f'FIXED: {self.labels[self.StaticIndex]}')

		self.moving_img_norm = self._normalize(self.cube[self.curr_frame])
		self.im_moving_obj = self.ax_moving.imshow(self.moving_img_norm, cmap=self.cmap_name)

		ax_color = 'lightgoldenrodyellow'

		# --- Fixed-panel contrast sliders (0-1, same as LandmarkPicker) ---
		ax_fix_min = self.fig.add_axes([0.15, 0.15, 0.25, 0.03], facecolor=ax_color)
		ax_fix_max = self.fig.add_axes([0.15, 0.11, 0.25, 0.03], facecolor=ax_color)
		self.slider_fix_min = Slider(ax_fix_min, 'Fixed Min', 0.0, 1.0, valinit=0.0)
		self.slider_fix_max = Slider(ax_fix_max, 'Fixed Max', 0.0, 1.0, valinit=1.0)
		self.slider_fix_min.on_changed(self.update_fixed_clim)
		self.slider_fix_max.on_changed(self.update_fixed_clim)

		# --- Moving-panel contrast sliders ---
		ax_mov_min = self.fig.add_axes([0.55, 0.15, 0.25, 0.03], facecolor=ax_color)
		ax_mov_max = self.fig.add_axes([0.55, 0.11, 0.25, 0.03], facecolor=ax_color)
		self.slider_mov_min = Slider(ax_mov_min, 'Moving Min', 0.0, 1.0, valinit=0.0)
		self.slider_mov_max = Slider(ax_mov_max, 'Moving Max', 0.0, 1.0, valinit=1.0)
		self.slider_mov_min.on_changed(self.update_moving_clim)
		self.slider_mov_max.on_changed(self.update_moving_clim)

		# --- Colormap radio buttons ---
		ax_radio = self.fig.add_axes([0.92, 0.1, 0.07, 0.15], facecolor=ax_color)
		self.radio = RadioButtons(ax_radio, ('gray', 'magma', 'viridis'),
								   active={'gray': 0, 'magma': 1, 'viridis': 2}.get(colourmap, 1))
		self.radio.on_clicked(self.update_cmap)

		# --- Frame navigation + Finish (additions beyond LandmarkPicker) ---
		ax_slider = self.fig.add_axes([0.15, 0.04, 0.5, 0.03])
		self.slider_frame = Slider(ax_slider, 'Frame', 0, self.n_frames - 1, valinit=0, valstep=1)
		self.slider_frame.on_changed(self.on_frame_change)

		ax_finish = self.fig.add_axes([0.75, 0.03, 0.13, 0.05])
		self.btn_finish = Button(ax_finish, 'Finish / Save')
		self.btn_finish.on_clicked(self.finish)

		self.fig.canvas.mpl_connect('button_press_event', self.on_click)
		self.fig.canvas.mpl_connect('key_press_event', self.on_key)

		self._redraw()
		plt.show()

	def _normalize(self, img):
		"""Matches LandmarkPicker's absolute min/max normalization."""
		imin, imax = np.min(img), np.max(img)
		if imax == imin:
			return np.zeros_like(img, dtype=np.float32)
		return (img - imin) / (imax - imin)

	def _safe(method):
		def wrapped(self, *args, **kwargs):
			try:
				return method(self, *args, **kwargs)
			except Exception:
				print(f'[PointsReviewer ERROR in {method.__name__}]')
				traceback.print_exc()
		return wrapped

	@_safe
	def update_fixed_clim(self, val):
		vmin, vmax = self.slider_fix_min.val, self.slider_fix_max.val
		if vmin >= vmax:
			vmax = vmin + 0.01
		self.im_fixed_obj.set_clim(vmin, vmax)
		self.fig.canvas.draw_idle()

	@_safe
	def update_moving_clim(self, val):
		vmin, vmax = self.slider_mov_min.val, self.slider_mov_max.val
		if vmin >= vmax:
			vmax = vmin + 0.01
		self.im_moving_obj.set_clim(vmin, vmax)
		self.fig.canvas.draw_idle()

	@_safe
	def update_cmap(self, label):
		self.cmap_name = label
		self.im_fixed_obj.set_cmap(label)
		self.im_moving_obj.set_cmap(label)
		self.fig.canvas.draw_idle()

	@_safe
	def on_frame_change(self, val):
		self.curr_frame = int(val)
		self.selected_point = None
		self.moving_img_norm = self._normalize(self.cube[self.curr_frame])
		self.im_moving_obj.set_data(self.moving_img_norm)
		self._redraw()

	@_safe
	def on_key(self, event):
		if event.key == 'right':
			self.slider_frame.set_val(min(self.slider_frame.val + 1, self.slider_frame.valmax))
		elif event.key == 'left':
			self.slider_frame.set_val(max(self.slider_frame.val - 1, self.slider_frame.valmin))

	@_safe
	def on_click(self, event):
		if event.inaxes is None or event.xdata is None:
			return
		toolbar = self.fig.canvas.manager.toolbar
		if toolbar is not None and toolbar.mode != '':
			return

		which = 'fixed' if event.inaxes == self.ax_fixed else \
			'moving' if event.inaxes == self.ax_moving else None
		if which is None:
			return

		if event.button == 3:
			self.selected_point = None
			self._redraw()
			return

		pts = self.fixed_points if which == 'fixed' else self.moving_points[self.curr_frame]

		if event.dblclick:
			dists = np.hypot(pts[:, 0] - event.xdata, pts[:, 1] - event.ydata)
			nearest = np.argmin(dists)
			self.selected_point = (which, nearest) if dists[nearest] <= self.pick_radius else None
			if self.selected_point is not None:
				print(f'Point {nearest + 1} selected ({which}) - click the correct location to move it')
			self._redraw()
			return

		if self.selected_point is not None and self.selected_point[0] == which:
			_, pt_idx = self.selected_point
			if which == 'fixed':
				self.fixed_points[pt_idx] = [event.xdata, event.ydata]
				if self.StaticIndex == self.curr_frame:
					self.moving_points[self.StaticIndex][pt_idx] = [event.xdata, event.ydata]
			else:
				self.moving_points[self.curr_frame][pt_idx] = [event.xdata, event.ydata]
				if self.curr_frame == self.StaticIndex:
					self.fixed_points[pt_idx] = [event.xdata, event.ydata]
			print(f'Moved point {pt_idx + 1} ({which}) to ({event.xdata:.1f}, {event.ydata:.1f})')
			self.selected_point = None
			self._redraw()

	@_safe
	def _redraw(self):
		for artist in self.plotted_artists:
			artist.remove()
		self.plotted_artists.clear()
		self.cbar_ax.clear()

		moving_pts = self.moving_points[self.curr_frame]
		n_pts_here = len(moving_pts)

		cmap = plt.colormaps['jet']
		num_colors = self.n_points if self.n_points > 1 else 2
		colors = cmap(np.linspace(0, 1, num_colors))

		for i, (x, y) in enumerate(self.fixed_points):
			edge = 'black' if self.selected_point == ('fixed', i) else 'white'
			lw = 2.5 if self.selected_point == ('fixed', i) else 0.5
			p = self.ax_fixed.plot(x, y, 'o', markersize=8, mfc=colors[i], mec=edge, mew=lw, alpha=0.85)
			self.plotted_artists.extend(p)
			t = self.ax_fixed.text(x + 8, y, str(i + 1), color='white', fontsize=9, fontweight='bold')
			self.plotted_artists.append(t)

		for i, (x, y) in enumerate(moving_pts):
			edge = 'black' if self.selected_point == ('moving', i) else 'white'
			lw = 2.5 if self.selected_point == ('moving', i) else 0.5
			p = self.ax_moving.plot(x, y, 'o', markersize=8, mfc=colors[i], mec=edge, mew=lw, alpha=0.85)
			self.plotted_artists.extend(p)
			t = self.ax_moving.text(x + 8, y, str(i + 1), color='white', fontsize=9, fontweight='bold')
			self.plotted_artists.append(t)

		tag = ' [STATIC - same as left]' if self.curr_frame == self.StaticIndex else ''
		n_tag = '' if n_pts_here == self.n_points else f'  /!\\ {n_pts_here} pts (expected {self.n_points})'
		self.ax_moving.set_title(f'{self.labels[self.curr_frame]} ({self.curr_frame + 1}/{self.n_frames}){tag}{n_tag}')

		norm = mcolors.Normalize(vmin=1, vmax=num_colors)
		sm = cm.ScalarMappable(cmap=cmap, norm=norm)
		sm.set_array([])
		ticks = np.linspace(1, num_colors, min(num_colors, 10)).astype(int) if num_colors > 1 else [1]
		cbar = self.fig.colorbar(sm, cax=self.cbar_ax, ticks=ticks)
		cbar.set_label('Point Index')

		self.fig.canvas.draw()

	def finish(self, event):
		plt.close(self.fig)
		self.updated_points = {
			'fixed_points': self.fixed_points.tolist(),
			'moving_points': [pts.tolist() for pts in self.moving_points]
		}

	def get_results(self):
		if hasattr(self, 'updated_points'):
			return self.updated_points
		print('GUI not finished yet - click "Finish / Save" first.')
		return None

