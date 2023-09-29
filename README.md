# PetReslice
Reslice a PET volume at a new z-axis slice spacing.  Default new slab thickness/spacing is 10 mm.

`PetReslice.py [-h] [-m {zoom,recombine,ics}] [-s SUV] [--version] D-IN D-OUT`

where:
D-IN  is a directory containing the PET series, or a "DICOMDIR"
D-OUT is a directory name to be created for the output series

if D-IN is "DICOMDIR" a QT selection box will launch and you can select your series from the DICOM CD archive.

Default resampling scheme is "ICS".  I can't remember the acronym but it's essentially resampling the integral along Z.  
Recombine will add discrete slices together and zoom is kind of a black box defined by the scipy image processing function.  Use ICS.

-s flag allows you to set a different SUV maximum for the output display windowing presets.  Default is 3.5.  This only af

Windowing presets are estblished so that they are consistent across the dataset and are comptatible with the ACR NILReader. I've tested this with Siemens and GE PET series.

I'm using python 3.9.  I am pretty sure it works with python 3.n where n<9 but how much less, I don't know.

Code dependencies other than standard are PyQt5, numpy, scipy, and pydicom.
