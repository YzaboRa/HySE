# HySE
Functions to compute hypercube from endoscope videos 

## Installation
This is a python 3 code, it requires the following packages:

- numpy
- opencv-python
- DateTime
- scipy
- matplotlib
- ffmpeg
- natsort
- PIL
- imageio
- tqdm
- inspect
- math
- glob

The co-registration is done using SimpleITK.

Documentation available here:
https://simpleitk.org/

SimpleITK can be installed with
```python
pip install SimpleITK
```
N.B. A previous version required SimpleElastix, which cannot be installed properly - Only SimpleITK is now required

It also requires those additional libraries:
- time
- tqdm

If SimpleITK is not installed, or cannot be imported, the code will simply ignore it. This will not be a problem as long as the co-registration functions are not used.

------------------
## General information
Last updated: 16 June 2026

## Example notebooks
Three notebooks in the Example/ folder show examples of how to process the white calibration files, how to run the pre-processing pipeline and how to idenfity biopsied locations.

These notebooks include the full pipeline. This readme file gives details about the main functions, but cannot be used to run the full pipeline on its own.

## Import
The HySE library must be imported. 
There are different ways to do so, but this method incicating the path explicitely worked the most reliably:

```python
HySE.help('Unmixing')
```

## Help
Every function in the HySE library includes documentation, which can be accessed using the Help function.
A general help function allows to print all the modules and associated functions:

```python
PathToFunctions = 'GitHub/HySE/'
sys.path.append(PathToFunctions)
import HySE

Functions can then be accessed through "HySE.function()".
```

Adding a module in argument allows to print in more details the functinos it contains:
```python
HySE.help('Unmixing')
```

A detailed description of a given function with its inputs and outputs can be accessed by adding the function to the argument:
```python
HySE.help('Unmixing.MakeMixingMatrix')
```

Most functions also allow the input of 'Help=True' as an argument, which will print the same detailed function description. 
The output (0) will match the size of the expected output in order to allow debugging on the go.

```python
HySE.MakeMixingMatrix(_,_,Help=True)
```


<!---  % ------------------
## White Calibration
--->

------------------
## Data preparation

### Import 
The raw data needs to be imported and registered. 
Manual registration is done interactively with a QT GUI. To enable this feature in a jupyter notebook, ensure that the right matplotlib feature is set:
```python
%matplotlib qt
```

Define where the data is located. Note that a "Name" is defined in order to be added when saving outputs. This helps identify what each output is for.
```python
SavingPath = '/SavingPath/'

DataPath_General = SavingPath+'Videos/'
DarkPath = DataPath_General+'Dark.mp4'

DataPath = DataPath_General+'NegControl.mp4'
Name = 'NegControl'
RegistrationSavingPath = SavingPath+'Results/Registered/'

```

### Load Dark
This step only requires the dark video. If it doesn't exist, a dark estimate can be generated from the data itself. 
Save the computed dark to save computing time for other analyses using the same dataset.
```python
LongDark = HySE.GetDark_WholeVideo(DarkPath) #, CropImDimensions=CropImDimensions
HySE.PlotDark(LongDark)
np.savez(f'{ResultsPath}LongDark.npz', LongDark)

## Load:
# LongDark = np.load(f'{ResultsPath}LongDark.npz')['arr_0']
```

### Load Trace
The following parameters are set for the current implementation. The automatic edge detection is enabled: make sure that the edges have been properly identified. Extreme intensity variability inside the video can cause the automatic edge detection to fail and require tweaking.
Modify the matplotlib display parameters for this cell to allow the trace figure to display in the notebook.
```python
%matplotlib ipympl
EdgePos_Data = HySE.FindHypercube_RGB(DataPath, Automatic=True, PlateauSize=9, DarkMin=15, MaxPlateauSize=20, AutomaticThreshold=20, SaveFig=False)
np.savez(f'{ResultsPath}{Name}_EdgePos_Data.npz', EdgePos_Data)```
<p align="center">
  <img src="https://github.com/user-attachments/assets/7fd87cd3-ab97-457a-9f21-80580b51f00e" width="800"/>
</p>

### Extract Frames
This uses the edge positions found with the trace to extract usable frames. 
The buffer sets how many frames a thrown out at the start end the end of the plateau. Only the middle frames are used, where the illumination and Olympus post-processing is more stable.
For ease of readability, split the extract frames according to their RGB channels.
```python
Buffer = 3
Frames, RGB_Dark = HySE.ComputeHypercube_RGB(DataPath, EdgePos_Data, Buffer=Buffer, BlueShift=1, SaveArray=False)

Frames_Blue = Frames[:,:,:,:,:,0]
Frames_Green = Frames[:,:,:,:,:,1]
Frames_Red = Frames[:,:,:,:,:,2]
```


### Save Frames to Identify Good Sweeps

A GUI is available to faciliate the selection of which frames to use.

The frames must still be extracted, but all sweeps must be kept. The resulting array will have a shape (Nsweeps, Nwavelengths, Y, X).
```python
# Nframe = 1 ## Keep only the middle frame
Nframe = [0,1,2] ## Keep all usable frames (3, for a buffer = 3)


## # If taking all frames, all sweep
HypercubeForSelection = HySE.GetHypercubeForRegistration(0, Nframe, DataPath, EdgePos_Data, Wavelengths_list, 
                                                            Channel='R', Buffer=Buffer, AllSweeps=True, OnlySweep=False) #, OnlySweep=True

```
This array can be fed directly to the frame selector tool.
Make sure that the matplotlib display mode is back to popup QT windows, otherwise the code will run in an error.
```python
%matplotlib qt
GUI = HySE.FrameSelector(HypercubeForSelection)
```
Inside the GUI, the user can scan sweeps and frames independently. All frames are by default exluded; the user must manually tick the box to indicate good frames. This can be done by ticking the "keep" box manually, or clicking on the "k" key on the keyboard.

A menu keeps track of how many frames are kept for each frame index (corresponding to a specifici wavelength/wavelength combination). Make sure that there is at least one frame per index, otherwise the hypercube will be incomplete. 

To facilitate navigation, left/right arrows can be used to quickly switch between frames of a given sweep, and top/bottom arrows can be used to quickly switch between sweeps.

Once the main selection is done, a second menu allows to review all selected frames to ensure that they are all showing the same lesion and of a high enough quality.

<p align="center">
  <img src="https://github.com/user-attachments/assets/a58d9e78-0c77-4d49-9d21-6de429dfb30c" width="800"/>
</p>

Save this selection for future references
```python
good_frames_mask, good_indices = GUI.get_results()
np.savez(f'{RegistrationSavingPath}GoodFrames_Mask.npz', good_frames_mask)
np.savez(f'{RegistrationSavingPath}GoodFrames_Indices.npz', good_indices)
```

The selected frames must all be registered. This function creates an array of size (Nframes, Y,X) that can be fed to the manualr registration tool. The labels are generated in order to recombine the frames after registration appropriately.
```python
GoodFrames, GoodFramesLabels = HySE.MakeCombinedHypercubeForRegistration(HypercubeForSelection, GUI.get_results())
HypercubeForRegistration = GoodFrames
```

The selected frames can be loaded from a previous identification:
```python
good_frames_mask = np.load(RegistrationPath+f'{Name}_GoodFrames_Mask.npz')['arr_0']
good_indices = np.load(RegistrationPath+f'{Name}_GoodFrames_Indices.npz')['arr_0']
LoadedOutcome = (good_frames_mask, good_indices)

GoodFrames, GoodFramesLabels = HySE.MakeCombinedHypercubeForRegistration(HypercubeForSelection, LoadedOutcome)
print(GoodFramesLabels)
```


### Mask
Extract a mask representing the edges (corners) of the endoscopic image that can be used as a general mask,.
```python
_, Mask = HySE.NormaliseMixedHypercube(Frames_GreenD[0,:,0,:,:], Dark=LongDark, Wavelengths_list=Wavelengths_list, 
                                       SaveFigure=False, SavingPath=SavingPath+Name, vmax=160, Plot=False)

EdgeMask = HySE.GetBestEdgeMask(Mask)
```

## Registration
### Select Frames to Register
When registering frames previously identified from a GUI
```python
HypercubeForRegistration = GoodFrames
AllLandmarkPoints = None
SaveTransforms = True
```


### Manual Registration
Manual registration hinges on fixed points in each images that must be manually annotated. The code will create a surface (thin plate spline) that goes through all the points. This code was written with Gemini AI.
Previously defined points can be pre-loaded

```python
ManualRegistrationPointsPath_ = glob.glob(f'{RegistrationPath}*_UserDefinedPoints.npz')

if len(ManualRegistrationPointsPath_)==1:
    ManualRegistrationPointsPath = ManualRegistrationPointsPath_[0]
    print(f'Found Path at {ManualRegistrationPointsPath}')
    
    AllPoints_Loaded = np.load(ManualRegistrationPointsPath, allow_pickle=True)['arr_0'].item()

    HypercubeForRegistration = GoodFrames
    AllLandmarkPoints = AllPoints_Loaded
#     SaveTransforms = False
    SaveTransforms = True
else:
    print(f'There are {len(ManualRegistrationPointsPath_)} corresponding user identified points. Select exactly one.')
    print(f'{RegistrationPath}{Name}*.npz\n')
    for i in ManualRegistrationPointsPath_:
        print(i+'\n')

```

Otherwise points can be defined here. 
Running this function will prompt a window to appear, requesting the user to define points. The code will first prompt the user to define points on the fixed frame, before then requesting the user to define those same points in every other frame to be registered. 
There is no limit to the number of points defined, but every point must be present in all the frames.

A series of features are available to make this step slightly easier. Pressing "z" will undo the last point defined. Pressing "p" will toggle the labels on and off. Use this to make sure that the points are set in the same order each time, to avoid issues.
The colormap, min and max levels of the image can be adjusted. Normal matplotlib features remain available (zoom, saving, etc.)

Another function (CoRegisterHypercubeAndMask) allows the user to set specific ROIs, but uses the automatic registration method on that area (see below).

Due to the time commitment of this stage, every relevant information is saved following sucessful registration. If an error results in the code failling during registration, a recovery file called registration_backup.json will be automaticall generated saving the current progress. When launching the manual registration code, if a registration_backup.json file is located the code will automatically start the registration from where it left off.

```python
%matplotlib qt

Sigma = 1 ## Image smoothing
TransformSmoothing = 1000 
AnchorCorners = True


Manual = True
DeviationThreshold = 150


UserDefinedROI = False
# Cropping = 0
GridSpacing = 100
index = 0



if Manual:
    if AllLandmarkPoints is not None:
        print(f'Doing manual registration - Using provided fixed points')
        CoregisteredHypercube, AllTransforms, CombinedMask, AllPoints = HySE.ManualRegistration(HypercubeForRegistration, Wavelengths_list, 
                                                                                                EdgeMask=EdgeMask, #AllReflectionsMasks=AllReflectionsMasks,
                                                                                                Blurring=True, Sigma=Sigma, StaticIndex=index,
                                                                                                GoodFramesLabels=GoodFramesLabels,
                                                                                                DeviationThreshold=DeviationThreshold, 
                                                                                                AllLandmarkPoints=AllLandmarkPoints,
                                                                                                TransformSmoothing=TransformSmoothing, 
                                                                                                AnchorCorners=AnchorCorners)
        temp = AllPoints['fixed_points']
        Info = f'Blurring{Sigma}_ManualRegistration_{len(temp)}points_Provided_L{TransformSmoothing}'
    else:
        print(f'Doing manual registration - Prompting user to define fixed points_L{TransformSmoothing}')
        CoregisteredHypercube, AllTransforms, CombinedMask, AllPoints = HySE.ManualRegistration(HypercubeForRegistration, Wavelengths_list, 
                                                                                                EdgeMask=EdgeMask, #AllReflectionsMasks=AllReflectionsMasks, 
                                                                                                Blurring=True, Sigma=Sigma, StaticIndex=index,
                                                                                                GoodFramesLabels=GoodFramesLabels,
                                                                                                DeviationThreshold=DeviationThreshold, 
                                                                                                TransformSmoothing=TransformSmoothing, 
                                                                                                AnchorCorners=AnchorCorners)
        temp = AllPoints['fixed_points']
        Info = f'Blurring{Sigma}_ManualRegistration_{len(temp)}points_UserDefined_L{TransformSmoothing}'
    
else:
    if UserDefinedROI:
        print(f'Doing SimpleITK registration, prompting user to define square ROI')
        Info = f'Blurring{Sigma}_GridSpacing{GridSpacing}_i{index}_UserDefinedROI'
    else:
        print(f'Doing SimpleITK registration on the whole image')
        Info = f'Blurring{Sigma}_GridSpacing{GridSpacing}_i{index}'
    CoregisteredHypercube, AllTransforms, CombinedMask, AllROICoordinates = HySE.CoRegisterHypercubeAndMask(HypercubeForRegistration, Wavelengths_list, 
                                                                                                            Verbose=False, Transform='BSplineTransform', 
                                                                                                            Affine=False, Blurring=True, Sigma=Sigma, 
                                                                                                            Static_Index=index, 
                                                                                                            EdgeMask=EdgeMask, #AllReflectionsMasks=AllReflectionsMasks, 
                                                                                                            GradientMagnitude=False, HistogramMatch=False, 
                                                                                                            GoodFramesLabels=GoodFramesLabels,
                                                                                                            IntensityNorm=False, Cropping=60,
                                                                                                            TwoStage=True, Metric='AdvancedMattesMutualInformation', 
                                                                                                            GridSpacing=GridSpacing, MinVal=20, 
                                                                                                            InteractiveMasks=InteractiveMasks)



NMI_Sweep_before = HySE.GetNMI(HypercubeForRegistration)
NMI_Sweep_after = HySE.GetNMI(CoregisteredHypercube)
print(f'Average NMI before : {NMI_Sweep_before[0]:.4f}, Average NMI after : {NMI_Sweep_after[0]:.4f}')



if SaveTransforms:
    VideoSavingPath = f'{RegistrationPath}{Name}_Frame{Nframe}_{Info}_NMI{NMI_Sweep_after[0]:.2f}.mp4'


    ## Not CoRegistered
    OrigVideoSavingPath = f'{RegistrationPath}Frame{Nframe}_NoReg.mp4'
    HySE.MakeHypercubeVideo(HypercubeForRegistration, OrigVideoSavingPath)
    OrigArraySavingPath = OrigVideoSavingPath.replace('mp4', 'npz')
    np.savez(f'{OrigArraySavingPath}', HypercubeForRegistration)

    ## Whole image
    HySE.MakeHypercubeVideo(CoregisteredHypercube, VideoSavingPath, Normalise=True)
    ArraySavingPath = VideoSavingPath.replace('.mp4','.npz')
    np.savez(f'{ArraySavingPath}', CoregisteredHypercube)
    ## Masked
    VideoSavingPathMasked = VideoSavingPath.replace('.mp4','_Masked.mp4')
    HySE.MakeHypercubeVideo(CoregisteredHypercube, VideoSavingPathMasked, Mask=CombinedMask, Normalise=True)



    ## Plot the position of fixed points 
    SavingPathPointsFigure = f'{RegistrationPath}{Name}_{Info}_NMI{NMI_Sweep_after[0]:.2f}__FixedPoints.png'
    HySE.PlotFixedPoints(HypercubeForRegistration, index, AllPoints, SavingPath=SavingPathPointsFigure, Labels=False)

    # ## Masked
    # VideoSavingPathMasked = VideoSavingPath.replace('.mp4','_Masked.mp4')
    # HySE.MakeHypercubeVideo(CoregisteredHypercube, VideoSavingPathMasked, Mask=CombinedMask, Normalise=True)
    ## Mask
    MaskSavingPath = VideoSavingPath.replace('.mp4','_Mask.npz')
    np.savez(f'{MaskSavingPath}', CombinedMask)

    # ## Transforms
    # TransformsSavingPath = VideoSavingPath.replace('.mp4','_Transforms')
    # HySE.SaveTransforms(AllTransforms, TransformsSavingPath)

    PathToSaveTransforms = f'{RegistrationPath}{Info}_NMI{NMI_Sweep_after[0]:.2f}__Transforms.pkl'
    HySE.SaveAllTransforms(AllTransforms, GoodFramesLabels,filename=PathToSaveTransforms)

    # Wavelengths
    WavelengthsSavingPath_Sorted = VideoSavingPath.replace('.mp4','__SortedWavelengths.npz') # f'{SavingPath}{Name}_{NameSub}__SortedWavelengths.npz'
    WavelengthsSavingPath_Unsorted = VideoSavingPath.replace('.mp4','__UnsortedWavelengths.npz') #f'{SavingPath}{Name}_{NameSub}__UnsortedWavelengths.npz'
    np.savez(f'{WavelengthsSavingPath_Sorted}', Wavelengths_list_sorted)
    np.savez(f'{WavelengthsSavingPath_Unsorted}', Wavelengths_list)

    if Manual:
        PointsSavingPath = VideoSavingPath.replace('.mp4','_UserDefinedPoints.npz')
        np.savez(f'{PointsSavingPath}', AllPoints, allow_pickle=True)
    elif UserDefinedROI:
        ROISavingPath = VideoSavingPath.replace('.mp4','_UserDefinedROI')
        HySE.SaveTransforms(AllROICoordinates, ROISavingPath)
        
    # Wavelengths
    WavelengthsSavingPath_Sorted = VideoSavingPath.replace('.mp4','__SortedWavelengths.npz') # f'{SavingPath}{Name}_{NameSub}__SortedWavelengths.npz'
    WavelengthsSavingPath_Unsorted = VideoSavingPath.replace('.mp4','__UnsortedWavelengths.npz') #f'{SavingPath}{Name}_{NameSub}__UnsortedWavelengths.npz'
    np.savez(f'{WavelengthsSavingPath_Sorted}', Wavelengths_list_sorted)
    np.savez(f'{WavelengthsSavingPath_Unsorted}', Wavelengths_list)

    if Manual:
        PointsSavingPath = VideoSavingPath.replace('.mp4','_UserDefinedPoints.npz')
        np.savez(f'{PointsSavingPath}', AllPoints, allow_pickle=True)
    elif UserDefinedROI:
        ROISavingPath = VideoSavingPath.replace('.mp4','_UserDefinedROI')
        HySE.SaveTransforms(AllROICoordinates, ROISavingPath)

    print(f'\n\n Saved all data')


```


This is what the GUI looks like:
<p align="center">
  <img src="https://github.com/user-attachments/assets/0a0aa6c3-093c-4ac7-86e1-d0f5a88fcf48" width="800"/>
</p>

Automatic error detection flags cases when the corresponding point is further away than a pre-set maximal distance. This can help prevent cases when the user makes a mistake and for example forgets a point. It is however a very blunt tool, as it will not flag errors if the points confused are close enough to each other, and it will flag errors when the movement is larger than this set threhold (can be modified when launching the GUI). The errors do not affect the code and should be ignored when the user is confident that the point are where they should be, like in this example:
<p align="center">
  <img src="https://github.com/user-attachments/assets/8cf3d3bc-9d21-40b3-bb7c-03cd382df050" width="800"/>
</p>

## Manual Registration - Review points
To review and potentially modify the points' location after manual registration, run:

```python
%matplotlib qt
Reviewer = HySE.PointsReviewer(HypercubeForRegistration, AllLandmarkPoints, StaticIndex=index,GoodFramesLabels=GoodFramesLabels)
```
OR (if AllLandmarkPoints isn't defined but AllPoints - from the manual registration - is):
```python
%matplotlib qt
Reviewer = HySE.PointsReviewer(HypercubeForRegistration, AllPoints, StaticIndex=index,GoodFramesLabels=GoodFramesLabels)
```

In the Reviewer GUI, individual points can be moved by double-clicking on them before clicking to the new location. 
All frames can be reviewer in any order, and pressing the forward or backward arrows allows easy switching.

Once review is done, if any of the points was changed, the registration must be ran again to compute the updated transforms, this time by providing the updated fixed points:

```python
UpdatedPoints = Reviewer.get_results()
AllLandmarkPoints = UpdatedPoints

## Then go back to the manual registration cell and run it again - it should not prompt the user to define points and should run with the provided points
```
Note that running the manual registration code again will override the previously saved results and transforms. In doubt, make sure to save the previously identified fixed point somewhere else as backup.


## Manual Registration - Remove frames afterwards
Sometimes, frames that were previously selected reveal to be harder to register than expected. Provided that there are other frames properly registered for this wavelength combination, it can be best to remove it from the dataset to avoid contamination. This can be done the following way (with help from Claude Sonnet 5 from Anthropic)

First identify which frame to remove (this must be noted down when registering the data):
```python
LabelToRemove = 'S67_Frame2'

IndexToRemove = GoodFramesLabels.index(LabelToRemove)  # raises ValueError if not found
print(f'Removing frame at position {IndexToRemove}: {GoodFramesLabels[IndexToRemove]}')
```

The clear the data and results from that frame:
```python
StaticIndex = index   # whatever StaticIndex/`index` was used for this registration run

if IndexToRemove == StaticIndex:
    raise ValueError(
        "You're trying to remove the static/reference frame itself. "
        "This needs a new StaticIndex and a full re-registration - it can't be "
        "patched post-hoc, since every other frame's transform was computed "
        "relative to this one."
    )

NewStaticIndex = StaticIndex - 1 if IndexToRemove < StaticIndex else StaticIndex

NewGoodFramesLabels = GoodFramesLabels[:IndexToRemove] + GoodFramesLabels[IndexToRemove+1:]

NewCoregisteredHypercube = np.delete(CoregisteredHypercube, IndexToRemove, axis=0)
NewHypercubeForRegistration = np.delete(HypercubeForRegistration, IndexToRemove, axis=0)

# CombinedMask is a single shared 2D mask (1000, 1154) - not per-frame, so it's unaffected
NewCombinedMask = CombinedMask

# AllPoints: only moving_points is per-frame; fixed_points is just the static frame's points
NewAllPoints = {
    'fixed_points': AllPoints['fixed_points'],
    'moving_points': AllPoints['moving_points'][:IndexToRemove] + AllPoints['moving_points'][IndexToRemove+1:]
}

# AllTransforms: list aligned with GoodFramesLabels
NewAllTransforms = AllTransforms[:IndexToRemove] + AllTransforms[IndexToRemove+1:]

print(f'{len(NewGoodFramesLabels)} frames remain (was {len(GoodFramesLabels)}).')
print(f'StaticIndex: {StaticIndex} -> {NewStaticIndex}')

```
And lastly update everything and overwrite previous files, to make sure that the bad frame won't pop back up if the intermediary files are reloaded.
Note that this step overrides previous results (with the bad frame), so only run this code if you are certain the right frame has been removed. Saving previous results elsewhere as backup could help avoid bad surprises.

```python
# ---------------------------------------------------------------
# 3. Recompute NMI on the trimmed data
# ---------------------------------------------------------------
NMI_before = HySE.GetNMI(NewHypercubeForRegistration)
NMI_after = HySE.GetNMI(NewCoregisteredHypercube)
print(f'Average NMI before: {NMI_before[0]:.4f}, after: {NMI_after[0]:.4f}')

# ---------------------------------------------------------------
# 4. Re-save everything to the SAME paths, overwriting the originals
#    (adjust these to match your actual saved filenames)
# ---------------------------------------------------------------
np.savez(ArraySavingPath, NewCoregisteredHypercube)                 # registered array
np.savez(OrigArraySavingPath, NewHypercubeForRegistration)          # unregistered/NoReg array
np.savez(MaskSavingPath, NewCombinedMask)                           # mask
np.savez(PointsSavingPath, NewAllPoints, allow_pickle=True)         # landmark points

HySE.SaveAllTransforms(NewAllTransforms, NewGoodFramesLabels, filename=PathToSaveTransforms)

# Regenerate the videos so they stay in sync with the trimmed data (optional but recommended)
HySE.MakeHypercubeVideo(NewCoregisteredHypercube, VideoSavingPath, Normalise=True)
HySE.MakeHypercubeVideo(NewCoregisteredHypercube, VideoSavingPathMasked, Mask=NewCombinedMask, Normalise=True)
HySE.MakeHypercubeVideo(NewHypercubeForRegistration, OrigVideoSavingPath)

# ---------------------------------------------------------------
# 5. Update in-memory variables so the rest of the notebook uses the trimmed data
# ---------------------------------------------------------------
GoodFramesLabels = NewGoodFramesLabels
CoregisteredHypercube = NewCoregisteredHypercube
HypercubeForRegistration = NewHypercubeForRegistration
CombinedMask = NewCombinedMask
AllPoints = NewAllPoints
AllTransforms = NewAllTransforms
index = NewStaticIndex

# # ---------------------------------------------------------------
# # 6. Record the exclusion so a future GoodFrames reload
# #    (from good_frames_mask / good_indices) doesn't reintroduce this frame
# # ---------------------------------------------------------------
# ExcludedFramesPath = f'{RegistrationPath}{Name}_ExcludedFrameLabels.npz'
# try:
#     ExcludedLabels = list(np.load(ExcludedFramesPath, allow_pickle=True)['arr_0'])
# except FileNotFoundError:
#     ExcludedLabels = []
# if LabelToRemove not in ExcludedLabels:
#     ExcludedLabels.append(LabelToRemove)
# np.savez(ExcludedFramesPath, np.array(ExcludedLabels, dtype=object))
# print(f'Excluded labels for {Name}: {ExcludedLabels}')

import re

# ---------------------------------------------------------------
# Parse sweep/wav coordinates out of the label we already removed
# ---------------------------------------------------------------
m = re.match(r'S(\d+)_Frame(\d+)', LabelToRemove)
SweepToRemove, WavToRemove = int(m.group(1)), int(m.group(2))
print(f'Removing (sweep={SweepToRemove}, wav={WavToRemove}) from the frame selection files')

# ---------------------------------------------------------------
# Locate and cross-check against the row we already identified
# ---------------------------------------------------------------
match_rows = np.where((good_indices[:, 0] == SweepToRemove) &
                       (good_indices[:, 1] == WavToRemove))[0]
assert len(match_rows) == 1, f"Expected exactly one match, found {len(match_rows)}"
row_idx = match_rows[0]
assert row_idx == IndexToRemove, (
    f"good_indices row {row_idx} doesn't match GoodFramesLabels position {IndexToRemove} - "
    f"stop and double check ordering before overwriting anything."
)

# ---------------------------------------------------------------
# Update the mask and indices
# ---------------------------------------------------------------
NewGoodFramesMask = good_frames_mask.copy()
NewGoodFramesMask[SweepToRemove, WavToRemove] = False

NewGoodIndices = np.delete(good_indices, row_idx, axis=0)

print(f'good_frames_mask: {good_frames_mask.sum()} -> {NewGoodFramesMask.sum()} good frames')
print(f'good_indices: {good_indices.shape} -> {NewGoodIndices.shape}')

# ---------------------------------------------------------------
# Overwrite the three saved files
# ---------------------------------------------------------------
np.savez(f'{RegistrationPath}{Name}_GoodFrames_Mask.npz', NewGoodFramesMask)
np.savez(f'{RegistrationPath}{Name}_GoodFrames_Indices.npz', NewGoodIndices)
np.savez(f'{RegistrationPath}{Name}_GoodFrames_Labels.npz', GoodFramesLabels)  # already trimmed earlier

# ---------------------------------------------------------------
# Update in-memory variables too, for consistency in this session
# ---------------------------------------------------------------
good_frames_mask = NewGoodFramesMask
good_indices = NewGoodIndices
```

### Automatic Registration
<details markdown="1"><summary>Click to expand</summary>
Using SimpleITK, mutual information (metric). Does not always work very well. Can be applied after a quick manual registration.
Most implementations do first an affine transform, followed by a bspline transform. The GridSpacing paramter allows to set the unit size for the bspline transform. Too large grids do not allow to register finely enough (looks more like affine), while too small grids overfit the data/noise and introduces artefacts and distortions.

```python
# ## --- If going straight to the automatic registration: ---
# HypercubeForRegistration_Auto = HypercubeForRegistration

## OR
## --- If applying automatic registration after having done a manual pre-registration: ---

# HypercubeForRegistration_Auto = CoregisteredHypercube

## OR
## --- If Loading a previously pre-registered hypercube from path: ---

# ## Lesion2:
# TEMP = '/Users/iracicot/Library/CloudStorage/OneDrive-UniversityofCambridge/Data/HySE/Patient7_20251125/Results/'
# HypercubeForRegistration_Auto_Path = TEMP+'Lesion2_Registered/Manual_AllFeatures/Sweep3_Frame1_Blurring1_ManualRegistration_45points_Provided_NMI1.23.npz'
# HypercubeForRegistration_Auto = np.load(HypercubeForRegistration_Auto_Path)['arr_0']
# print(HypercubeForRegistration_Auto.shape)


# ## # If Averaging
# HypercubeForRegistration_Avg, _ = HySE.ComputeHypercube(DataPath, EdgePos_Data, Wavelengths_list, Buffer=Buffer, 
#                                                     Average=False, Order=False, Help=False, SaveFig=False, SaveArray=False, 
#                                                     Plot=False, ForCoRegistration=False)
# HypercubeForRegistration_Auto = HypercubeForRegistration_Avg[Nsweep,:,:,:]



Cropping = 100
Sigma = 1

index = 5

GridSpacing = 60

MinVal = 0 


#### Register:

CoregisteredHypercube_Auto, AllTransforms, CombinedMask, AllROICoordinates = HySE.CoRegisterHypercubeAndMask(HypercubeForRegistration_Auto, Wavelengths_list, 
                                                                                                        Verbose=False, Transform='BSplineTransform', 
                                                                                                        Cropping=Cropping,
                                                                                                        Affine=False, Blurring=True, 
                                                                                                        Sigma=Sigma, EdgeMask=EdgeMask,
                                                                                                        StaticIndex=index, GradientMagnitude=False, 
                                                                                                        HistogramMatch=False, IntensityNorm=False,
                                                                                                        TwoStage=True, Metric='AdvancedMattesMutualInformation', 
                                                                                                        GridSpacing=GridSpacing, MinVal=MinVal) 

Info = f'f{Nframe}_Blur{Sigma}_Grid{GridSpacing}_i{index}_RegisterAndMask'



# ####################################
# #### Compute NMI and save results:
# ####################################


NMI_Sweep_before = HySE.GetNMI(HypercubeForRegistration_Auto)
NMI_Sweep_after = HySE.GetNMI(CoregisteredHypercube_Auto)

print(f'Average NMI before : {NMI_Sweep_before[0]:.4f}, Average NMI after : {NMI_Sweep_after[0]:.4f}')



# print(f'\n\n')


VideoSavingPath = f'{RegistrationSavingPath}Sweep{Nsweep}_Frame{Nframe}_{Info}_NMI{NMI_Sweep_after[0]:.2f}.mp4'

## Not CoRegistered
OrigVideoSavingPath = f'{RegistrationSavingPath}Sweep{Nsweep}_Frame{Nframe}_NoReg.mp4'
HySE.MakeHypercubeVideo(HypercubeForRegistration_Auto, OrigVideoSavingPath)
OrigArraySavingPath = OrigVideoSavingPath.replace('mp4', 'npz')
np.savez(f'{OrigArraySavingPath}', HypercubeForRegistration_Auto)

## Whole image
HySE.MakeHypercubeVideo(CoregisteredHypercube_Auto, VideoSavingPath, Normalise=True)
ArraySavingPath = VideoSavingPath.replace('.mp4','.npz')
np.savez(f'{ArraySavingPath}', CoregisteredHypercube_Auto)
## Masked
VideoSavingPathMasked = VideoSavingPath.replace('.mp4','_Masked.mp4')
HySE.MakeHypercubeVideo(CoregisteredHypercube_Auto, VideoSavingPathMasked, Mask=CombinedMask, Normalise=True)



# ## Cropped image
# VideoSavingPathCropped = VideoSavingPath.replace('.mp4','_Cropped.mp4')
# HySE.MakeHypercubeVideo(CoregisteredHypercube_Auto[:,Cropping:-1*Cropping,Cropping:-1*Cropping], VideoSavingPathCropped, Normalise=True)


# Wavelengths
WavelengthsSavingPath_Sorted = VideoSavingPath.replace('.mp4','__SortedWavelengths.npz') # f'{SavingPath}{Name}_{NameSub}__SortedWavelengths.npz'
WavelengthsSavingPath_Unsorted = VideoSavingPath.replace('.mp4','__UnsortedWavelengths.npz') #f'{SavingPath}{Name}_{NameSub}__UnsortedWavelengths.npz'
np.savez(f'{WavelengthsSavingPath_Sorted}', Wavelengths_list_sorted)
np.savez(f'{WavelengthsSavingPath_Unsorted}', Wavelengths_list)


## Transforms
TransformsSavingPath = VideoSavingPath.replace('.mp4','_Transforms')
HySE.SaveTransforms(AllTransforms, TransformsSavingPath)

## Mask
MaskSavingPath = VideoSavingPath.replace('.mp4','_Mask.npz')
np.savez(f'{MaskSavingPath}', CombinedMask)
print(f'Saved all data')

```
</details>

------------------

## Normalisation
### Define Parameters

```python
indices_4x4 = [1,4,5,6]
indices_3x3 = [0,2,3]
indices_6x6 = [0,1,2,3,4,5,6]
index_418 = 7

Green_Exposure = 6.5
Blue_Exposure = 7
Red_Exposure = 1.6

Green_3Wav_1 = [[4, 5, 6],
                [1, 4, 6],
                [1, 5, 6],
                [1, 4, 5],
                [4, 5, 6, 7],
                [2, 3],
                [0, 3],
                [0, 2]]

MixingMatrix_Indices = Green_3Wav_1

print(f'Computing 6x6, 4x4 and 3x3 green matrices...')
GreenMatrix_6x6_Indices, GreenMatrix_4x4_Indices, GreenMatrix_3x3_Indices = HySE.GetShortGreenMatrices_3Wav(MixingMatrix_Indices, index_418=index_418, 
                                                                                indices_4x4=indices_4x4, indices_3x3=indices_3x3)
print(f'Computing blue matrix..')
BlueMatrix_Indices = HySE.GetBlueMatrix_3Wav(MixingMatrix_Indices, index_418=index_418)

print(f'computing short pannels...')
Wavs_6x6_P2, Wavs_6x6_P4, Wavs_4x4_P2, Wavs_4x4_P4, Wavs_3x3_P2, Wavs_3x3_P4 = HySE.GetShortPanelWavs_3Wav(Panel2_Wavelengths, Panel4_Wavelengths, 
                                                                                                      indices_4x4, indices_3x3, index_418)

Wavs_6x6 = np.sort(np.array(Wavs_6x6_P2+Wavs_6x6_P4))

Wavs_4x4 = np.sort(np.array(Wavs_4x4_P2+Wavs_4x4_P4))
Wavs_3x3 = np.sort(np.array(Wavs_3x3_P2+Wavs_3x3_P4))



#####################
Title = 'Green Matrix'
GreenMatrix_7x7 = HySE.MakeMixingMatrix_Flexible(Panel2_Wavelengths, MixingMatrix_Indices, 
                                         Panel4_Wavelengths, MixingMatrix_Indices, Plot=False,
                                         SaveFig=False, Title=Title, SavingPath='')

Title = 'Green Matrix 6x6'
GreenMatrix_6x6_sub_Indices = HySE.generate_local_index_matrix(GreenMatrix_6x6_Indices, indices_6x6)
GreenMatrix_6x6 = HySE.MakeMixingMatrix_Flexible(Wavs_6x6_P2, GreenMatrix_6x6_sub_Indices, 
                                         Wavs_6x6_P4, GreenMatrix_6x6_sub_Indices, Plot=False,
                                         SaveFig=False, Title=Title, SavingPath='')

Title = 'Blue Matrix'
BlueMatrix = HySE.MakeMixingMatrix_Flexible(Panel2_Wavelengths, BlueMatrix_Indices, 
                                         Panel4_Wavelengths, BlueMatrix_Indices, Plot=False,
                                         SaveFig=False, Title=Title, SavingPath='')

MixingMatrix_Sub1 = np.array([[1,1,1,0,0,0,0],
                              [1,0,1,0,0,1,0], 
                              [1,1,0,0,0,1,0], 
                              [0,1,1,0,0,1,0], 
                              [0,0,0,1,1,0,0], 
                              [0,0,0,1,0,0,1], 
                              [0,0,0,0,1,0,1],])

MixingMatrix_Sub2 = MixingMatrix_Sub1

Wavelengths_Sub1 = np.array([486,511,543,560,584,606,625])
Wavelengths_Sub2 = np.array([503,526,569,576,594,617,646])


Sub1_indices = [6,5,4,3,2,1,0]
Sub2_indices = [14,13,12,11,10,9,8]

Sub1_indices_frames = [0,1,2,3,5,6,7]
Sub2_indices_frames = [8,9,10,11,13,14,15]


MixingMatrix_SubSub1A = np.array([[1,1,1,0],
                                  [1,0,1,1],
                                  [1,1,0,1],
                                  [0,1,1,1]])

MixingMatrix_SubSub1B = np.array([[1,1,0],
                                  [1,0,1],
                                  [0,1,1]])

MixingMatrix_SubSub2A = MixingMatrix_SubSub1A
MixingMatrix_SubSub2B = MixingMatrix_SubSub1B

Wavelengths_SubSub1A = np.array([486,511,543,606])
Wavelengths_SubSub1B = np.array([560,584,625])
Wavelengths_SubSub2A = np.array([503,526,569,617])
Wavelengths_SubSub2B = np.array([576,594,646])

SubSub1A_indices_frames = [0,1,2,3]
SubSub1B_indices_frames = [5,6,7]
SubSub2A_indices_frames = [8,9,10,11]
SubSub2B_indices_frames = [13,14,15]

```

### White Calibration

```python
WhiteRefTraces = np.load(WhiteCalibrationTraces_Path)
for key, value in WhiteRefTraces.items() :
    print (key)
```

### Keep usable frames

```python
UsableFrames_Green = HySE.SelectUsableFrames(Frames_Green, good_indices, is_collapsed=True) #, target_nframe=Nframe
UsableFrames_Red = HySE.SelectUsableFrames(Frames_Red, good_indices, is_collapsed=True)
UsableFrames_Blue = HySE.SelectUsableFrames(Frames_Blue, good_indices, is_collapsed=True)

## Flatten array if needed:
UsableFrames_Green_flat, good_indices_flat_G = HySE.FlattenToPseudoSweeps(UsableFrames_Green, good_indices)
UsableFrames_Blue_flat, good_indices_flat_B = HySE.FlattenToPseudoSweeps(UsableFrames_Blue, good_indices)
UsableFrames_Red_flat, good_indices_flat_R = HySE.FlattenToPseudoSweeps(UsableFrames_Red, good_indices)
```

### Dark Subtraction
Subtract the dark from all the frames
```python
## Dark Subtract
UsableFrames_BlueD, MaskB = HySE.NormaliseMixedHypercube(UsableFrames_Blue_flat, Dark=LongDark, SaveFigure=False, Plot=False)
UsableFrames_GreenD, MaskG = HySE.NormaliseMixedHypercube(UsableFrames_Green_flat, Dark=LongDark, SaveFigure=False, Plot=False)
UsableFrames_RedD, MaskR = HySE.NormaliseMixedHypercube(UsableFrames_Red_flat, Dark=LongDark, SaveFigure=False, Plot=False)
```

### Spectral Normalisation
Use the white calibration to normalise the data (one number per wavelength combination representing the overall light intensity for this combination).
```python
UsableFrames_Red_DS = HySE.SpectralNormalise(UsableFrames_RedD, WhiteRefTraces['HySE_Red'],good_indices_flat_R) #good_indices
UsableFrames_Green_DS = HySE.SpectralNormalise(UsableFrames_GreenD, WhiteRefTraces['HySE_Green'],good_indices_flat_G) #good_indices
UsableFrames_Blue_DS = HySE.SpectralNormalise(UsableFrames_BlueC, WhiteRefTraces['HySE_Blue'],good_indices_flat_B) #good_indices
```

### Spatial Normalisation
Use the reference red channel to normalise the corresponding data frames.
```python
UsableFrames_Green_DSN = HySE.SpatialNormalisation(UsableFrames_Green_DS, UsableFrames_Red_DS)
UsableFrames_Blue_DSN = HySE.SpatialNormalisation(UsableFrames_Blue_DS, UsableFrames_Red_DS)
print(UsableFrames_Green_DSN.shape)
```

### Load Transforms
```python
TransformsPath = glob.glob(RegistrationPath+'*.pkl')
if len(TransformsPath)==1:
    TransformsPath = TransformsPath[0]
    NameSub = f'{Name}_Manual_L1000'
    print(NameSub)
    print(TransformsPath)
    
else:
    print(f'There needs to be exactly one corresponding pickled file on the path')
    print(f'There are currently {len(TransformsPath)}')
    for i in TransformsPath:
        print(i)
```

### Apply Transforms
The transforms are determined using the red frames, which are all obtained using the same illumination.
The data frames are in the green and blue channels. It is assumed that the movement between RGB frames is minimal, so that transforms from the red frames is applicable to equivalement GB frames.
```python
RegGreen, RegGreen_labels = HySE.ApplyAllTransforms(UsableFrames_Green_DCSN, good_indices_flat_G, TransformsPath) #good_indices HySE. good_indices_flat_G
RegBlue, RegBlue_labels = HySE.ApplyAllTransforms(UsableFrames_Blue_DCSN, good_indices_flat_B, TransformsPath) #good_indices
```


------------------
## Unmixing

### Combine corresponding Frames
If keeing more than one frame per wavelength combination, combine those (registered) frames before unmixing.
NB: if there are more than one frame for all wavelengths combination, it is also possible to combine them after unmixing.
```python
RegGreenCombined = HySE.CombineFrames(RegGreen, RegGreen_labels, 16)
RegBlueCombined = HySE.CombineFrames(RegBlue, RegBlue_labels, 16)
```

### Prep frames for unmixing
The green data channel can be unmixed using different sub-matrices. Prep the data according to the different unmixing strategies.
```python
FramesG_6x6 = HySE.omit_frames(RegGreenCombined, indices_6x6)

FramesG_Sub1 = RegGreenCombined[Sub1_indices_frames]
FramesG_Sub2 = RegGreenCombined[Sub2_indices_frames]

FramesG_SubSub1A = RegGreenCombined[SubSub1A_indices_frames]
FramesG_SubSub1B = RegGreenCombined[SubSub1B_indices_frames]
FramesG_SubSub2A = RegGreenCombined[SubSub2A_indices_frames]
FramesG_SubSub2B = RegGreenCombined[SubSub2B_indices_frames]
```

### Unmix
Note that these steps do all the possible ways the data can be unmixed. It can be shorten significantly by only doing one kind of unmixing (one for the blue channel, one for the green channel)
There are 3 functions in this package that can be used for umxing:
- UnmixData(): Basic matrix inversion using least squares
- UnmixDataNNLS(): Matrix inversition using non-negative least squares (NNLS), with noise filtering and rejection of pixels with negative or invalid intensity
- UnmixDataSmoothNNLS(): Matrix inversion using NNLS with with noise filtering and pixel rejection, imposing smoothness (regulation set weighted by lambda)
- UnmixDataSmoothNNLSPrior(): Matrix inversino using NNLS with two regulators, smoothnes (weighted by lambda_smooth) and a known spectral prior (weighted by lambda_prior)

The UnmixDataSmoothNNLS() function generally leads to the best results, with low lambda (~0.1)

Trying all combinations
```python
lambda_value = 0.1

## If applying all different unmixing strategies:

## Blue Matrix
t00 = time.time()
Unmixed_Blue = HySE.UnmixDataSmoothNNLS(RegBlueCombined, BlueMatrix, lambda_smooth=lambda_value)
t1 = time.time()
print(f'\nFull Blue - Smooth NNLS unmixing took {t1-t00:.0f}s')


## Green Matrix, all frames:
t0 = time.time()
Unmixed_Green_7x7 = HySE.UnmixDataSmoothNNLS(RegGreenCombined, GreenMatrix_7x7, lambda_smooth=lambda_value)
t1 = time.time()
print(f'Full Green - Smooth NNLS unmixing took {t1-t0:.0f}s')

## Green Matrix, 6x6:
t0 = time.time()
Unmixed_Green_6x6 = HySE.UnmixDataSmoothNNLS(FramesG_6x6, GreenMatrix_6x6, lambda_smooth=lambda_value)
t1 = time.time()
print(f'Green 6x6 - Smooth NNLS unmixing took {t1-t0:.0f}s')


## Green Matrix, Sub1:
t0 = time.time()
Unmixed_Green_Sub1 = HySE.UnmixDataSmoothNNLS(FramesG_Sub1, MixingMatrix_Sub1, lambda_smooth=lambda_value)
t1 = time.time()
print(f'Green Sub1 - Smooth NNLS unmixing took {t1-t0:.0f}s')

## Green Matrix, Sub2:
t0 = time.time()
Unmixed_Green_Sub2 = HySE.UnmixDataSmoothNNLS(FramesG_Sub2, MixingMatrix_Sub2, lambda_smooth=lambda_value)
t1 = time.time()
print(f'Green Sub2 - Smooth NNLS unmixing took {t1-t0:.0f}s')

## Green Matrix, SubSub1A:
t0 = time.time()
Unmixed_Green_SubSub1A = HySE.UnmixDataSmoothNNLS(FramesG_SubSub1A, MixingMatrix_SubSub1A, lambda_smooth=lambda_value)
t1 = time.time()
print(f'Green SubSub1A - Smooth NNLS unmixing took {t1-t0:.0f}s')

## Green Matrix, SubSub1B:
t0 = time.time()
Unmixed_Green_SubSub1B = HySE.UnmixDataSmoothNNLS(FramesG_SubSub1B, MixingMatrix_SubSub1B, lambda_smooth=lambda_value)
t1 = time.time()
print(f'Green SubSub1B - Smooth NNLS unmixing took {t1-t0:.0f}s')

## Green Matrix, SubSub2A:
t0 = time.time()
Unmixed_Green_SubSub2A = HySE.UnmixDataSmoothNNLS(FramesG_SubSub2A, MixingMatrix_SubSub2A, lambda_smooth=lambda_value)
t1 = time.time()
print(f'Green SubSub2A - Smooth NNLS unmixing took {t1-t0:.0f}s')

## Green Matrix, SubSub2B:
t0 = time.time()
Unmixed_Green_SubSub2B = HySE.UnmixDataSmoothNNLS(FramesG_SubSub2B, MixingMatrix_SubSub2B, lambda_smooth=lambda_value)
t1 = time.time()
print(f'Green SubSub2B - Smooth NNLS unmixing took {t1-t0:.0f}s')

print(f'\n\nAll the unmixing took {t1-t00:.0f}s')

```


### Recombine Frames from sub-Unmixing
```python
## in two parts:
Unmixed_GreenSub12_Combined, Wavs_Sub12_Combined = HySE.combine_hypercubes(Unmixed_Green_Sub1, Unmixed_Green_Sub2, 
                                                                           Wavelengths_Sub1, Wavelengths_Sub2)



## in four parts:
Unmixed_GreenSubSub1_Combined, Wavs_SubSub1_Combined = HySE.combine_hypercubes(Unmixed_Green_SubSub1A, Unmixed_Green_SubSub1B, 
                                                                               Wavelengths_SubSub1A, Wavelengths_SubSub1B)
Unmixed_GreenSubSub2_Combined, Wavs_SubSub2_Combined = HySE.combine_hypercubes(Unmixed_Green_SubSub2A, Unmixed_Green_SubSub2B, 
                                                                               Wavelengths_SubSub2A, Wavelengths_SubSub2B)

Unmixed_GreenSubSubAll_Combined, Wavs_SubSubAll_Combined = HySE.combine_hypercubes(Unmixed_GreenSubSub1_Combined, Unmixed_GreenSubSub2_Combined, 
                                                                                   Wavs_SubSub1_Combined, Wavs_SubSub2_Combined)

```
### Restore NaN values post-unmixing
```python
Unmixed_Blue_nan = HySE.RestoreNaNs(RegBlueCombined, Unmixed_Blue)
Unmixed_Green_7x7_nan = HySE.RestoreNaNs(RegGreenCombined, Unmixed_Green_7x7)
Unmixed_Green_6x6_nan = HySE.RestoreNaNs(RegGreenCombined[2:,:,:], Unmixed_Green_6x6)
Unmixed_Green_Split2_nan = HySE.RestoreNaNs(RegGreenCombined[2:,:,:], Unmixed_GreenSub12_Combined)
Unmixed_Green_Split4_nan = HySE.RestoreNaNs(RegGreenCombined[2:,:,:], Unmixed_GreenSubSubAll_Combined)
```


<!-- <p align="center">
  <img src="https://github.com/user-attachments/assets/49fd5867-b39b-40b0-8cfc-38f39294e873" width="500"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/650e1403-c03a-44b0-9424-341d801b709a" width="850"/>
</p> -->


<p align="center">
  <img src="https://github.com/user-attachments/assets/5a08ee78-a088-4881-aba4-c79da1dfaa03" width="550"/>
</p>


## BiopsyPicker
This GUI allows to draw ROIs where the user has determined the biopsies were taken.

As with the other PyQT GUI, the correct matplotlib settings must be set:
``` python
%matplotlib qt
```
### Load Results
``` python
## Where the data is
DataPath = ''
Name = 'HySE_ExamplePatientData'


## Enter diagnosis here for each ROI
## Refer to pathlogy report, and make sure to be 
## consistent about the labels
Pathology = ['Metaplasia', 'Intramucosal ADC']
Colours = ['tab:red', 'tab:blue']

Hypercube_Blue_Path = HySE.GetPath(DataPath, 'Umixed_Blue')
Hypercube_Green_Path = HySE.GetPath(DataPath, 'Green_Split2')
CoregistrationCube_Path = HySE.GetPath(DataPath, 'RegistrationHypercubeRed')

Wav_sub_Path = HySE.GetPath(DataPath, 'wavelengths_sub')
Wav_Path = HySE.GetPath(DataPath, 'wavelengths.npz', extension='')

ResultsPath = DataPath

Hypercube_Blue = np.load(Hypercube_Blue_Path)['arr_0']
Hypercube_Green = np.load(Hypercube_Green_Path)['arr_0']
Hypercube_Red = np.load(CoregistrationCube_Path)['arr_0']

Wav_sub = np.load(Wav_sub_Path)['arr_0']
Wav = np.load(Wav_Path)['arr_0']

```

### Draw the ROIs on the red reference channel images
This step involves having previously identified where the biopsies were taken, which can be determine with the biopsies clinical footage.

Using the red reference channel images allows to have a cleaner view of the features and facilitates identification. The GUI allows to switch between the different frames in the hypercube to visualise how the drown ROI looks on each of them.

Press "r" to start drawing a new ROI. Double-clock on an existing ROI to edit it. You can move individual points, add new ones by double-clicking next to existing points in the edit mode, or remove points by double-clicking on the point in edit mode. Press "e" to exit the edit mode, and "z" to undo the latest action.

``` python
%matplotlib qt
ROI_coordinates_Red, ROI_AvgSpectra_Red, ROI_AllSpectra_Red, preview_img_Red = HySE.GetBiopsyLocations(Hypercube_Red, display_frame_idx=1)

Coordinates = ROI_coordinates_Red
```

### Plot and save results
``` python
%matplotlib ipympl
plt.close()
fig, ax = plt.subplots(1, 1, figsize=(10,10))
ax.imshow(preview_img_Red)
ax.axis('off')
ax.set_title('Identified ROIs', fontsize=14)
plt.tight_layout()
plt.savefig(f'{ResultsPath}{Name}_ROIs_Red.png')
plt.savefig(f'{ResultsPath}{Name}_ROIs_Red.svg')
plt.show()

```

### Apply identified ROIs to data channels
``` python
%matplotlib qt
ROI_coordinates_Green, ROI_AvgSpectra_Green, ROI_AllSpectra_Green, preview_img_Green = HySE.GetBiopsyLocations(Hypercube_Green, initial_rois=Coordinates, display_frame_idx=5)
```


``` python
%matplotlib qt
ROI_coordinates_Blue, ROI_AvgSpectra_Blue, ROI_AllSpectra_Blue, preview_img_Blue = HySE.GetBiopsyLocations(Hypercube_Blue, initial_rois=Coordinates, display_frame_idx=5)
```

### Save resulting ROIs and average spectra
``` python
SavingPath = f'{DataPath}{Name}_ROIs_All.npz'
HySE.SaveBiopsiesData(SavingPath, Coordinates, 
                 ROI_AvgSpectra_Green, ROI_AllSpectra_Green, 
                 ROI_AvgSpectra_Blue, ROI_AllSpectra_Blue, 
                 Pathology)
```


### Plot
``` python
## Plot
fs_title = 14
fs_legend = 11



%matplotlib ipympl
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,4))

for i in range(0,len(ROI_AvgSpectra_Green)):
    AllSpectra = np.array(ROI_AllSpectra_Green[i])
    AllStd = np.std(AllSpectra, axis=0)
    ax.plot(Wav_sub, ROI_AvgSpectra_Green[i], label=f'Green ROI {i+1}: {Pathology[i]}', color=Colours[i])
    
for i in range(0,len(ROI_AvgSpectra_Blue)):
    AllSpectra = np.array(ROI_AllSpectra_Blue[i])
    AllStd = np.std(AllSpectra, axis=0)
    ax.plot(Wav, ROI_AvgSpectra_Blue[i], label=f'Blue ROI {i+1}: {Pathology[i]}', ls=':', color=Colours[i])
    
    
ax.legend(fontsize=fs_legend)
ax.set_xlabel(f'Wavelengths [nm]', fontsize=fs_title, fontweight='bold')
ax.set_ylabel(f'Average Spectra in ROI [a.u.]', fontsize=fs_title, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{ResultsPath}{Name}_ROIsSpectra_All.png')
plt.savefig(f'{ResultsPath}{Name}_ROIsSpectra_All.svg')
plt.show()

```

<p align="center">
  <img src="https://github.com/user-attachments/assets/978406ad-5385-4458-8e22-ab2f0f2e7db8" width="550"/>
</p>



### Registration Visualisation
Those functions allow to visualise the registration.

Load the unregistered and registered hypercubes:
```python
# Lesion
GeneralPath = '/GENERAL_PATH_LESION/'
SavingPath = GeneralPath

HypercubeForRegistration_Path = GeneralPath+'NoReg.npz'
CoregisteredHypercube_Path = GeneralPath+'ManualRegistration_45points_Provided_NMI1.23.npz'
Name = 'Sweep3_Frame1'

RegisteredMask_Path = GeneralPath+'ManualRegistration_45points_Provided_NMI1.23_Mask.npz'
Wavelengths_Path = GeneralPath+'wavelengths.npz'

HypercubeForRegistration = np.load(HypercubeForRegistration_Path)['arr_0']
CoregisteredHypercube = np.load(CoregisteredHypercube_Path)['arr_0']
RegisteredMask = np.load(RegisteredMask_Path)['arr_0']
Wavelengths_list = np.load(Wavelengths_Path)['arr_0']

```

Specular reflections dominate the sobel (edge detection). They must be filtered out
```python
Unregistered = HypercubeForRegistration
Registered = CoregisteredHypercube


Unregistered_ = np.clip(Unregistered, 0, None)

## Sweep 13:
##.        0.   1.   2.   3.   4.   5   6.   7.     8.   9.   10.  11.  12.  13.  14.  15. 
k_list = [4.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 2.0,   2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 2.0, 2.0]

HypercubeForRegistration_Masked, AllReflectionsMasks = HySE.RemoveSpecularReflections(Unregistered_, k=k_list, FillValue=0, Buffer=4, MaxSize=2800)
print(AllReflectionsMasks.shape)

fs=11
## Make sure reflections are masked appropriately
# Indices = [0,5,14]
Indices = [i for i in range(0,len(Wavelengths_list))]

plt.close()
fig, ax = plt.subplots(nrows=len(Indices), ncols=3, figsize=(10,2.5*len(Indices)))
for i in range(0,len(Indices)):
    u = Indices[i]
    m, M = np.nanmin(Unregistered_[u,:,:]), np.nanmax(Unregistered_[u,:,:])
    ax[i,0].imshow(Unregistered_[u,:,:], vmin=m, vmax=M, cmap='gray')
    ax[i,0].set_title(f'Hypercube u={u}', fontsize=fs)
    ax[i,1].imshow(HypercubeForRegistration_Masked[u,:,:], vmin=m, vmax=M, cmap='gray')
    ax[i,1].set_title(f'Masked hypercube u={u} - k={k_list[u]}', fontsize=fs)
    ax[i,2].imshow(AllReflectionsMasks[u,:,:])
    ax[i,2].set_title(f'Mask u={u} - Sum = {np.sum(AllReflectionsMasks[u,:,:])}', fontsize=fs)
    for j in range(0,3):
        ax[i,j].set_axis_off()
plt.tight_layout()
plt.show()
```

This function plots just the edges
```python

ReflectionsMask = AllReflectionsMasks
RegisteredMask = RegisteredMask

## Lesion
UnregCoords = [200,-45, 45,-45] ## ys,ye, xs,xe
RegCoords = [200,-45, 45,-45]


method = 'sobel'                   # sobel', 'canny', or 'raw_thresholded'.
sobel_ksize = 3
canny_sigma = 2.0                  # Sigma for Canny edge detection
cap_intensity = 0.2                # Clips high-intensity features (0.0 to 1.0)
raw_threshold = 0.4                # Threshold for 'raw_thresholded' -> Set everything below to 0
nframe = 1                         # Context frame index
mask_dilation_kernel = 10          # Size of dilation for reflection mask.
denoise_sigma = 0                  # Pre-filtering strength.
denoise_method = 'gaussian'          # 'gaussian' or 'median'.
mask_inpaint_method = 'inpaint'    # 'zero_fill' or 'inpaint'.
display_power_gamma = 2.0          # Power law (val^gamma) to darken background noise.
min_gradient_threshold = 0.20      # Hard cutoff (0.0 to 1.0). Signals below this are set to 0.

plt.close()
plt.close()

fig_unregistered, im_unregistered =  HySE.visualize_hypercube_movement(Unregistered[:,UnregCoords[0]:UnregCoords[1],UnregCoords[2]:UnregCoords[3]], 
                                    method=method,
                                    sobel_ksize=sobel_ksize, 
                                    canny_sigma=canny_sigma,
                                    cap_intensity=cap_intensity, 
                                    raw_threshold=raw_threshold, 
                                    reflection_mask=ReflectionsMask[:,UnregCoords[0]:UnregCoords[1],UnregCoords[2]:UnregCoords[3]], 
                                    nframe=nframe, 
                                    mask_dilation_kernel=mask_dilation_kernel, 
                                    denoise_sigma=denoise_sigma, 
                                    denoise_method=denoise_method,
                                    mask_inpaint_method=mask_inpaint_method, 
                                    display_power_gamma=display_power_gamma,
                                    min_gradient_threshold=min_gradient_threshold)



fig_registered, im_registered =  HySE.visualize_hypercube_movement(Registered[:,RegCoords[0]:RegCoords[1],RegCoords[2]:RegCoords[3]], 
                                    method=method,
                                    sobel_ksize=sobel_ksize,
                                    canny_sigma=canny_sigma, 
                                    cap_intensity=cap_intensity, 
                                    raw_threshold=raw_threshold, 
                                    reflection_mask=RegisteredMask[RegCoords[0]:RegCoords[1],RegCoords[2]:RegCoords[3]], 
                                    nframe=nframe, 
                                    mask_dilation_kernel=4, 
                                    denoise_sigma=denoise_sigma, 
                                    denoise_method=denoise_method,
                                    mask_inpaint_method=mask_inpaint_method, 
                                    display_power_gamma=display_power_gamma,
                                    min_gradient_threshold=min_gradient_threshold)

```
```python
plt.close()
fig, ax = plt.subplots(1, 2, figsize=(12,4))

ax[0].imshow(fig_unregistered)
ax[0].axis('off')
ax[0].set_title('Unregistered', fontsize=14)

ax[1].imshow(fig_registered)
ax[1].axis('off')
ax[1].set_title('Registered', fontsize=14)

plt.tight_layout()
plt.savefig(f'{SavingPath}{Name}_RegistrationCommparison.png', transparent=True)
plt.savefig(f'{SavingPath}{Name}_RegistrationCommparison.svg', transparent=True)
plt.show()
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/62a9f037-b6ed-4032-ad89-f6aff8c826b7" width="750"/>
</p>

While this function overlays the edges on top of a greyscale image of the lesion:
```python

method = 'sobel'                   # sobel', 'canny', or 'raw_thresholded'.
sobel_ksize = 15
canny_sigma = 2.0                  # Sigma for Canny edge detection
cap_intensity = 0.005                # Clips high-intensity features (0.0 to 1.0)
raw_threshold = 0.4                # Threshold for 'raw_thresholded' -> Set everything below to 0
nframe = 1                         # Context frame index
mask_dilation_kernel = 10          # Size of dilation for reflection mask.
denoise_sigma = 0                  # Pre-filtering strength.
denoise_method = 'gaussian'          # 'gaussian' or 'median'.
mask_inpaint_method = 'inpaint'    # 'zero_fill' or 'inpaint'.
min_gradient_threshold = 0.20      # Hard cutoff (0.0 to 1.0). Signals below this are set to 0.

opacity_gain = 1.2

plt.close()
plt.close()

overlay_unregistered =  HySE.visualize_edge_density_overlay(Unregistered[:,UnregCoords[0]:UnregCoords[1],UnregCoords[2]:UnregCoords[3]], 
                                                       sobel_ksize=sobel_ksize, 
                                                       cap_intensity=cap_intensity, 
                                                       raw_threshold=raw_threshold,
                                                       reflection_mask=ReflectionsMask[:,UnregCoords[0]:UnregCoords[1],UnregCoords[2]:UnregCoords[3]],
                                                       mask_dilation_kernel=mask_dilation_kernel,
                                                       denoise_sigma=denoise_sigma,
                                                       denoise_method=denoise_method,
                                                       mask_inpaint_method=mask_inpaint_method,
                                                       min_gradient_threshold=min_gradient_threshold,
                                                       opacity_gain=opacity_gain)

overlay_registered =  HySE.visualize_edge_density_overlay(Registered[:,RegCoords[0]:RegCoords[1],RegCoords[2]:RegCoords[3]], 
                                                       sobel_ksize=sobel_ksize, 
                                                       cap_intensity=cap_intensity, 
                                                       raw_threshold=raw_threshold,
                                                       reflection_mask=RegisteredMask[RegCoords[0]:RegCoords[1],RegCoords[2]:RegCoords[3]], 
                                                       mask_dilation_kernel=mask_dilation_kernel,
                                                       denoise_sigma=denoise_sigma,
                                                       denoise_method=denoise_method,
                                                       mask_inpaint_method=mask_inpaint_method,
                                                       min_gradient_threshold=min_gradient_threshold,
                                                       opacity_gain=opacity_gain)
```
```python
plt.close()
fig, ax = plt.subplots(1, 2, figsize=(12,4))

ax[0].imshow(overlay_unregistered)
ax[0].axis('off')
ax[0].set_title('Unregistered', fontsize=14)

ax[1].imshow(overlay_registered)
ax[1].axis('off')
ax[1].set_title('Registered', fontsize=14)

plt.tight_layout()
plt.savefig(f'{SavingPath}{Name}_RegistrationOverlay.png', transparent=True)
plt.savefig(f'{SavingPath}{Name}_RegistrationOverlay.svg', transparent=True)
plt.show()
```
<p align="center">
  <img src="https://github.com/user-attachments/assets/96662f5e-df99-4f84-97c8-0266aba6cb56" width="750"/>
</p>



















```

## Optional Functions

### Macbeth Colour Chart analysis

When imaging a Macbeth colour chart, it is helpful to look at the spectra from each patch.
Certain functions are designed to make this task easier.

```python

## Location of ground truth files

MacbethPath = 'MacBeth/Micro_Nano_CheckerTargetData.xls'

Macbeth_sRGBPath = 'MacBeth/Macbeth_sRGB.xlsx'
Macbeth_AdobePath = 'MacBeth/Macbeth_Adobe.xlsx'

MacBethSpectraData = np.array(pd.read_excel(MacbethPath, sheet_name='Spectra'))
MacBethSpectraColour = np.array(pd.read_excel(MacbethPath, sheet_name='Color_simple'))

MacBeth_RGB = np.array(pd.read_excel(Macbeth_AdobePath))
## Adobe RGB
MacBethSpectraRGB = np.array([MacBethSpectraColour[:,0], MacBethSpectraColour[:,7], MacBethSpectraColour[:,8], MacBethSpectraColour[:,9]])


## Crop around macbeth chart

xs, xe = 170,927
ys, ye = 310,944

macbeth = Unmixed_Hypercube_MacbethHySE_avg_ND[-1,ys:ye,xs:xe]

plt.close()
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,5))
ax.imshow(macbeth, cmap='gray')
plt.tight_layout()
plt.show()


## Identify where each patch is located

## This step requires referncing the plot previously generated, and usually requires a few iterations for each new dataset
## Make sure that the ROI squares cover the right patches and only those, by looking at the generated plot.
## The larger the ROI, the more pixels can be averaged to reduce noise
Patch1_pos = [558, 660]
## Size of a single macbeth patch, in pixel
Patch_size_x = 116
Patch_size_y = 115
Sample_size = 50
Image_angle = 0
 
Positions = HySE.GetPatchPos(Patch1_pos, Patch_size_x, Patch_size_y, Image_angle)
HySE.PlotPatchesDetection(macbeth, Positions, Sample_size)
```

<p align="center">
  <img src="https://github.com/user-attachments/assets/2fc5df12-4370-46ab-9119-1664b781fe36" width="400"/>
</p>

```python

## Compute the spectra for each patch

CropCoordinates = [xs, xe, ys, ye]

PatchesSpectra_Green7x7 = HySE.GetPatchesSpectrum(Unmixed_Green_7x7, Sample_size, Positions, CropCoordinates)
PatchesSpectra_Green6x6 = HySE.GetPatchesSpectrum(Unmixed_Green_6x6, Sample_size, Positions, CropCoordinates)
PatchesSpectra_Green4x4 = HySE.GetPatchesSpectrum(Unmixed_Green_4x4, Sample_size, Positions, CropCoordinates)
PatchesSpectra_Green3x3 = HySE.GetPatchesSpectrum(Unmixed_Green_3x3, Sample_size, Positions, CropCoordinates)

PatchesSpectra_Blue = HySE.GetPatchesSpectrum(Unmixed_Blue, Sample_size, Positions, CropCoordinates)
PatchesSpectra_GreenSub_Combined = HySE.GetPatchesSpectrum(Unmixed_GreenSub12_Combined, Sample_size, Positions, CropCoordinates)
PatchesSpectra_GreenSubSub_Combined = HySE.GetPatchesSpectrum(Unmixed_GreenSubSub_Combined, Sample_size, Positions, CropCoordinates)

## Plot

PatchesToPlot = [PatchesSpectra_Blue,
                 PatchesSpectra_Green7x7, 
                 PatchesSpectra_Green6x6,
                 PatchesSpectra_GreenSub_Combined, 
                 PatchesSpectra_GreenSubSub_Combined]

WavelengthsAll = [Wavelengths_list_sorted, 
                  Wavelengths_list_sorted,
                  Wavs_6x6,  
                  Wavs_6x6, 
                  Wavs_6x6]

Labels = ['Blue', 
          'Green full',
          'Green 6x6',  
          'Green sub combined', 
          'Green sub sub combined']

Colours = ['royalblue', 
           'forestgreen', 
           'limegreen', 
           'deeppink', 
           'cyan']


metric = 'SAM'

Name_ToSave = f'{SavingPath}{Name}'
HySE.PlotPatchesSpectra(PatchesToPlot, WavelengthsAll, MacBethSpectraData, MacBeth_RGB, Name, PlotLabels=Labels, SavingPath=Name_ToSave, XScale=[400,670],
                      Metric=metric, ChosenMethod=2, SaveFig=False, Colours=Colours)

```

<p align="center">
  <img src="https://github.com/user-attachments/assets/5fa369a3-d25c-49de-9eb7-1788d9319b4f" width="800"/>
</p>



<details>
<summary><b>Older Registration Documentation</b></summary>

## Co-Registration
The co-registration must be done before the unmixing. But since it requires SimpleITK and is only necessary for in vivo data (as opposed to testing data), I will for now put the relevant documentation here.
TO DO: Update description to incroporate registration at the right spot.

Co-registration is done with SimpleITK, which is an open-source interface trying to make the tooks of the the Insight Segmentation and Registration Toolkit (ITK) easier to use. It also uses functionalities from elastix, but all those are handled by SimpleITK directly.

The following documation is preliminary and indicates how to reproduce the best co-registration results obtained from HySE in vivo data, and gives a rough overview of all the options now available to play with.

Assuming that a video has been loaded, the trace computed and EdgePos computed with the HySE.FindHypercube() function, we may then start co-registering individual sweeps.

### Get raw hypercube for co-registration
This can be done either by averaging a given set of frames, or selecting a signle (good) frame. It is assumed that movement between individual RGB cycles is minimal (valid most of the time unless chromatic effects can be seen on the endoscopy monitor, which is very rare). Note that RGB frames have different exposure times, so there is typically one that has better SNR.

The sweep of interest must first be identified. Make sure that there are good frames (bubble-free, with good visibility) for all 16 wavelengths combinatino, and that movement is not too large that the same features can be identified in the whole sweep.

```python
Nsweep = 5

## ## If taking single frames
Nframe = 1
## (or Nframe = [1,2] if we want to co-register more frames - Note that this will result of a 'hypercube' or size (len(Nframe)*16,Y,X) )
HypercubeForRegistration = HySE.GetHypercubeForRegistration(Nsweep, Nframe, Lesion1_Path, EdgePos_Lesion1, Wavelengths_list, Buffer=12)


## ## If Averaging
HypercubeForRegistration_Avg, _ = HySE.ComputeHypercube(Lesion1_Path, EdgePos_Lesion1, Wavelengths_list, Buffer=Buffer, 
                                                    Average=False, Order=False, Help=False, SaveFig=False, SaveArray=False, 
                                                    Plot=False, ForCoRegistration=False)
HypercubeForRegistration = HypercubeForRegistration_Avg[Nsweep,:,:,:]
```

### Get Mask
#### Edges
Co-registration performs better when the sharp black edges from the endoscopy monitor display are masked

```python
## Use the data iself, or any normalisation, to estimate the sharp edges mask (anything works)
_, Mask = HySE.NormaliseMixedHypercube(Hypercube_Lesion1_all[2,:,:,:], Dark=LongDark, Wavelengths_list=Wavelengths_list, 
                                       SaveFigure=False, SavingPath=SavingPath+Name, vmax=160, Plot=False)

## Get the best mask to remove the edges of the image
EdgeMask = HySE.GetBestEdgeMask(Mask)
```

#### Specular Reflections
For some lesions, wet mucosa will lead to specular reflections, which can corrupt the registration. Those reflections have to be identified and masked for each frame.

```python
## Work on a single sweep that has been confirmed to have clear frames for the whole sweep, with minimal movement. Select a frame to work with (do not average images). If possible, use the (red) reference channel frames (which use all the same wavelengths and should be bright enough)
Nsweep = 13
Nframe = 1
## If necessary, clip the data to ensure no negative values (which might happen after dark subtraction)
HypercubeForRegistration = np.clip(Frames_GreenD[Nsweep, :, Nframe, :,:], 0, None)
```

Use the RemoveSpecularReflections() function to find masks to remove the specular reflections. If required (it probably will), you can upload a list of k values (lenght = # of frames) for to fine-tune the cutoff for each frame. The function accepts two methods: threshold (determined by k or cutoff, if specified. Lower k=most masking) and neighbourhood value (if NeighborhoodSize is specified). Use HySE.help('Masking.RemoveSpecularReflections') for details.

```python
## Find the right parameters to mask all specular reflections
## N
## Sweep 13:
##.        0.   1.   2.   3.   4.   5   6.   7.     8.   9.   10.  11.  12.  13.  14.  15. 
k_list = [3.1, 4.0, 3.1, 3.8, 3.8, 3.5, 4.0, 3.8,   3.7, 3.7, 3.6, 3.7, 3.7, 3.7, 4.0, 3.8]

HypercubeForRegistration_Masked, AllReflectionsMasks = HySE.RemoveSpecularReflections(HypercubeForRegistration, k=k_list, FillValue=0, Buffer=5) #FillValue=0, NeighborhoodSize=20, , Cutoff=200
print(AllReflectionsMasks.shape)
```
The fine-tuning of parameters will require some trial and error. Use the following code to visualise results:
```python
fs=11
## Make sure reflections are masked appropriately
Indices = [i for i in range(0,len(Wavelengths_list))]

fig, ax = plt.subplots(nrows=len(Indices), ncols=3, figsize=(10,2.5*len(Indices)))
# plt.close()
for i in range(0,len(Indices)):
    u = Indices[i]
    m, M = np.nanmin(HypercubeForRegistration[u,:,:]), np.nanmax(HypercubeForRegistration[u,:,:])
    ax[i,0].imshow(HypercubeForRegistration[u,:,:], vmin=m, vmax=M, cmap='gray')
    ax[i,0].set_title(f'Hypercube u={u}', fontsize=fs)
    ax[i,1].imshow(HypercubeForRegistration_Masked[u,:,:], vmin=m, vmax=M, cmap='gray')
    ax[i,1].set_title(f'Masked hypercube u={u} - k={k_list[u]}', fontsize=fs)
    ax[i,2].imshow(AllReflectionsMasks[u,:,:])
    ax[i,2].set_title(f'Mask u={u} - Sum = {np.sum(AllReflectionsMasks[u,:,:])}', fontsize=fs)
    for j in range(0,3):
        ax[i,j].set_axis_off()
plt.tight_layout()
plt.show()
```
An example of a subset of the figure generated can be seen here:
<p align="center">
  <img src="https://github.com/user-attachments/assets/03f16e61-9c8e-4af1-9c39-feaffbbb158d" width="800"/>
</p>

It's best to save those masks for furture registration. Registration codes will accept the EdgeMask and the AllReflectionsMasks.
```python
ArraySavingPath = f'{SavingPath}{Name}_{NameSub}_RawFrames/Sweep{Nsweep}_HypercubeForRegistration.npz'
np.savez(f'{ArraySavingPath}', HypercubeForRegistration)
ArraySavingPath = f'{SavingPath}{Name}_{NameSub}_RawFrames/Sweep{Nsweep}_AllReflectionsMasks.npz'
np.savez(f'{ArraySavingPath}', AllReflectionsMasks)
ArraySavingPath = f'{SavingPath}{Name}_{NameSub}_RawFrames/Sweep{Nsweep}_EdgeMask.npz'
np.savez(f'{ArraySavingPath}', EdgeMask)
```


### Perform Co-Registration

Then the co-registration. You will need to set:

- GridSpacing: This is the size of the grid used in the BSpline coregistration. Small grid spacings can co-registrate details better but tend to overfit and create distortions and artefacts. Large grid size preserve the image better but can lead to poorer registration. Some of the pattern artefact created by smaller grid spacings can be averaged out with appropriate blurring. GradSpacing of 20-50 has lead to the best results (100+ tends to not co-register everything)
- Sigma: This is the width of the Gaussian blurring function (in pixels) used to blur the image. Blurring is done twice, once on the images fed to get the coregistration transform, in order to smooth out noise that could mess with the registration, and once after registration to remove any artefact potentially introduced by the registration. Sigma of 1 has given good results.
- TwoStage: Set to true to first perform an affine transform, that roughly aligns the images, followed by a BSpline transform, that can correct distortions. Not doing the affine first leaves the BSpline transform to do all the coregistration works, which tends to create more artefacts
- Metric: Which metric to use when finding the best registration. 'AdvancedMattesMutualInformation' and 'NormalizedMutualInformation' give decent (and similar) results, other options can be found in the documentation (try HySE.help('CoRegistrationTools.CoRegisterImages') ), but have given worse results.

  
```python

GridSpacing=20
Sigma=1

CoregisteredHypercube, AllTransforms = HySE.CoRegisterHypercube(HypercubeForRegistration, ## Raw hypercube
                                                                Wavelengths_list, ## don't really need this
                                                                Verbose=False, ## Set to True to see all output from SimpleElastix
                                                                Transform='BSplineTransform', ## Transform to use. BSpline best for in vivo data, see documentation for options
                                                                Affine=False, ## Set to true to restrict to Affine transforms
                                                                Blurring=True, ## Set to true for blurring 
                                                                Sigma=Sigma, ## Blurring size
                                                                Mask=Mask_Invert, ## Inficate Mask
                                                                TwoStage=True, ## Set to true for a affine transform followed by BSpline
                                                                Metric='AdvancedMattesMutualInformation', # See documentations for options
                                                                GridSpacing=GridSpacing) ## Grid Spacing

```

### Save Results
And then you might want to calculate the normalised mutual information before and after to estimate the performance of the registration, and save the registered hypercube as a video to see the results.

```python
NMI_Sweep_before = HySE.GetNMI(HypercubeForRegistration)
NMI_Sweep_after = HySE.GetNMI(CoregisteredHypercube)
print(f'Average NMI before : {NMI_Sweep_before[0]:.4f}, Average NMI after : {NMI_Sweep_after[0]:.4f}')

Info = f'Frame{Nframe}_Blurring{Sigma}_AdvMI_TwoStage_GridSpacing{GridSpacing}'
VideoSavingPath = f'{SavingPath}{Name}_{NameSub}_RawFrames/Sweep{Nsweep}_SE_{Info}_NMI{NMI_Sweep_after[0]:.2f}.mp4'

## Whole image
HySE.MakeHypercubeVideo(CoregisteredHypercube, VideoSavingPath, Normalise=True)

## Cropped image
Cropping = 100
VideoSavingPathCropped = VideoSavingPath.replace('.mp4','_Cropped.mp4')
HySE.MakeHypercubeVideo(CoregisteredHypercube[:,Cropping:-1*Cropping,Cropping:-1*Cropping], VideoSavingPathCropped, Normalise=True)

## Not CoRegistered
OrigVideoSavingPath = f'{SavingPath}{Name}_{NameSub}_RawFrames/Sweep{Nsweep}_NoReg.mp4'
HySE.MakeHypercubeVideo(HypercubeForRegistration, OrigVideoSavingPath)
```

### Apply transforms to other data
The transforms optimised during the registration are stored in the AllTransforms list output by HySE.CoRegisterHypercube() function. Each element in the list is a specific transform applied to the associated frame. Each of those transforms can also be saved as .txt files.

To apply the transforms to new frames, use the ApplyTransforms() function:

```python
FramesTransformed = HySE.ApplyTransform(FramesForTransformApply, AllTransforms)
```
The new FramesTransformed array will now be transformed in the same way as the original data.


### Manual Registration
In some cases, SimpleITK is unable to provide decent registration. If the image has some visible features that we are trying to align (particularly lesions), a manual (brute force) registration can be used. For this registration, the user defines fixed points for all images, which are used to compute affine registration. Pop-up windows help to identify those fixed points manually.
The following example shows how to use manual registration.

```python
## Load unregistered data:
HypercubeForRegistration = np.load(HypercubeForRegistration_Path)['arr_0']
AllReflectionsMasks = np.load(AllReflectionsMasks_Path)['arr_0']
EdgeMask = np.load(EdgeMask_Path)['arr_0']

## Run the code to manually indentify fixed points and follow instructinos
Sigma = 2
DeviationThreshold=150
RegHypercube, AllTransforms, RegMask, AllPoints = HySE.CoRegisterHypercubeAndMask_Hybrid(HypercubeForRegistration, 
                                                                                         Wavelengths_list, 
                                                                                         Static_Index=9, 
                                                                                         AllReflectionsMasks=AllReflectionsMasks, 
                                                                                         EdgeMask=EdgeMask, 
                                                                                         InteractiveMasks=True,
                                                                                         Blurring=True, 
                                                                                         Sigma=Sigma, 
                                                                                         DeviationThreshold=DeviationThreshold)
```
At first a window prompting the user to identify fixed points on the static image will open. There is no limit to the number of fixed points, and the colourbar will expand as points are added. Press "z" to remove the latest point, and "p" to toggle the numbering of the points. 

<p align="center">
  <img src="https://github.com/user-attachments/assets/f4863536-f631-4cc1-bbd7-cbaa2edfde81" width="800"/>
</p>

Once all points on the fixed image have been identified, the window can be closed. A new window will then appear, with the static image and already idenfitied fixed points showed on the left frame. The right frame will show the first moving image, where the user must then identify the same fixed points. Like for the static image, pressing "z" will remove the latest point and "p" will toggle the numbering of the points. An automatic warning will apear if the distance between the fixed point in the moving image and the equivalent point in the static image is larger than DeviationThreshold (set to 150 pixels in this example). If this warning appears, it might be that the user has confused points (use "p" to show the points numbers), or that the moving image has enough movement that the same point has moved by more than DeviationThreshold. Once all fixed points have been identified, the window will indicate so and the user may then close the window. A new identical window will then appear, prompting the user to identify fixed points for the second moving image. The same thing will happen untill all moving images (typically 15) have been labelled.

<p align="center">
  <img src="https://github.com/user-attachments/assets/1e329078-526d-4049-94a9-9525ee440777" width="800"/>
</p>


Once all images have been labelled and if the user is confident in their labelling, it is best to save the coordinates for all points for reproducibility:
```python
Npoints = len(AllPoints['fixed_points'])
Info = f'ManualRegistration_{Npoints}points'
PointsSavingPath = f'{SavingPath}{Name}_{NameSub}_RawFrames/Sweep{Nsweep}_Crop{Cropping}_{Info}__Landmarkpoints.npz'
np.savez(f'{PointsSavingPath}', AllPoints, allow_pickle=True)
```

These saved coordinates can then be used to run the registration by adding the AllLandmarkPoints input in the CoRegisterHypercubeAndMask_Manual() function. When AllLandmarkPoints is indicated, the function does not prompt the user to label images but instead performs the registration based on those indicated points.
```python
AllLandmarkPoints_1 = np.load(PointsPath, allow_pickle=True)['arr_0'].item()
RegHypercube, AllTransforms, RegMask, AllPoints = HySE.CoRegisterHypercubeAndMask_Manual(HypercubeForRegistration, 
                                                                                         Wavelengths_list,
                                                                                         Static_Index=9, 
                                                                                         AllReflectionsMasks=AllReflectionsMasks, 
                                                                                         EdgeMask=EdgeMask, 
                                                                                         InteractiveMasks=True, 
                                                                                         Blurring=True, 
                                                                                         Sigma=Sigma, 
                                                                                         DeviationThreshold=DeviationThreshold,
                                                                                         AllLandmarkPoints=AllLandmarkPoints_1)

```

## Help 

A general help function allows to print all the modules and associated functions:
```python
HySE.help()
```

Adding a module in argument allows to print in more details the functinos it contains:
```python
HySE.help('Unmixing')
```

A detailed description of a given function with its inputs and outputs can be accessed by adding the function to the argument:
```python
HySE.help('Unmixing.MakeMixingMatrix')
```

Most functions also allow the input of 'Help=True' as an argument, which will print the same detailed function description. 
The output (0) will match the size of the expected output in order to allow debugging on the go.

```python
HySE.MakeMixingMatrix(_,_,Help=True)
```

</details>

## To do
- [ ] Channel Interplay

## Comments
- If the functions cannote be loaded, try adding an empty \__init__.py file in the same folder
