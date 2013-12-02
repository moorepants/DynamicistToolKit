The walk module provides tools to work with typical data collected during human
locomotion (gait). In general, the three dimensional coordinates throughout
time of a set of markers which are attached to anatomical features on the human
are tracked. Secondly, various analog signals are recorded. In particular,
voltages from force plate which are proportional to the appliead forces and
moments on one or two force plates, voltages from EMG measurements,
accelerometers, etc. All of these measurements are stored a discrete samples in
time.

D-Flow (and Cortex)
===================

The walk module includes a class that eases dealing with the data collected
from a typical D-Flow output.

Motek Medical sells packages which include treadmills with force plate
measurement capabilities and motion bases, motion capture systems, and other
sensors for various measurements. Their software D-Flow manages the data
streams from the various systems and is responsible for displaying interactive
visuals, sounds, and motions to the subject.

The Human Motion and Control Lab at Cleveland State University has such a
system. Our system includes:

- A ForceLink R-Mill (http://www.forcelink.nl/index.php/product/r-mill/) which
  has dual 6 DoF force plates, independent belts for each foot, and
  lateral and pitch motion capabilities.
- A 10 Camera Motion Analysis motion capture system which includes the Cortex
  software and hardware for collecting analog and camera data simultaneously.
- Motek Medical's D-Flow software and visual display system.
- Delsys wireless EMG + 3D Accelerometers.

Cortex alone is capable of delivering data from the cameras, force plates, and
analog sensors (EMG, Acclerometer), but D-Flow is required to collect data from
digital sensors and the treadmill's motion. In general, we use D-Flow to sample
all of the data and it outputs multiple files.

The treadmill's local coordinate system is such that the X coordinate points to
the right, the Y coordinate points upwards, and the Z coordinate follows from
the right-hand-rule, i.e. points backwards. The camera's coordinate system is
aligned to the treamdmill's frame during camera calibration.

Mocap Module
------------

D-Flow's Mocap Module has a file tab which allows you to export the time series
data collected from Cortex in two different file formats: tab seperated values
(tsv) and the C3D format (www.c3d.org).

The text file output from the mocap module in DFlow is a tab delimited file.
The first line is the header. The header contains the `TimeStamp` column which
is the system time on the DFlow computer when it receives the Cortex frame and
is thus not exactly at 100 hz, it has a light variable sample rate. The next
column is the `FrameNumber` column which is the Cortex frame number. Cortex
samples at 100hz and the frame numbers start at some positive integer value.
The remaining columns are samples of the computed marker positions at each
Cortex frame and the analog signals (force plate forces/moments, EMG,
accelerometers, etc). The analog signals are simply voltages that have been
scaled by some calibration function and they should have a reading at each
frame. The markers sometimes go missing (i.e. can't been seen by the cameras.
When a marker goes missing DFlow outputs the last non-missing value in all
three axes until the marker is visible again. The mocap file can also contain
variables computed by the real time implementation of the Human Body Model
(HBM). If the HBM computation fails at a D-Flow sample period, strings of
zeros, '0.000000', are inserted for missing values. Note that the order of the
"essential" measurements in the file must be retained if you expect to run the
file back into D-Flow for playback.

TimeStamp
   This column records the D-Flow system time when it receives a "frame" from
   Cortex. This is approximately at 100 hz, but has slight variability per
   sampel period (+/- 0.0012 s or so).
FrameNumber
   This column gives a positive integer to count the frame numbers delivered by
   Cortex. It seems as though, none of the frames are dropped but this should
   be verified.
Marker Coordinates
   The columns that correspond to marker coordinates have one of three
   suffixes: '.PosX', '.PosY', '.PosZ'. The prefix is the marker name which is
   set by providing a name to the marker in Cortex. There are specific names
   which are required for D-Flow's Human Body Model's computations.:wq

Treamdill reference frame
X: points to the right
Y: points upwards
Z: point backwards
The first 12 are raw voltage signals of the force plate measurements.
TODO : Need to figure out what these mean, would be nice to store the
calibration matrix from voltages to forces/momements also.
Channel1.Anlg
Channel2.Anlg
Channel3.Anlg
Channel4.Anlg
Channel5.Anlg
Channel6.Anlg
Channel7.Anlg
Channel8.Anlg
Channel9.Anlg
Channel10.Anlg
Channel11.Anlg
Channel12.Anlg
# the arrow on the accelerometers which aligns with the local X axis diretion is
# always pointing forward (i.e. aligned with the negative z direction)
# Front left
Channel13.Anlg : EMG
Channel14.Anlg : AccX
Channel15.Anlg : AccY
Channel16.Anlg : AccZ
# Back left
Channel17.Anlg : EMG
Channel18.Anlg : AccX
Channel19.Anlg : AccY
Channel20.Anlg : AccZ
# Front right
Channel21.Anlg : EMG
Channel22.Anlg : AccX
Channel23.Anlg : AccY
Channel24.Anlg : AccZ
# Back right
Channel25.Anlg : EMG
Channel26.Anlg : AccX
Channel27.Anlg : AccY
Channel28.Anlg : AccZ

Record Module
-------------

The record module in D-Flow allows one to sample any signal available in D-Flow
at the variable D-Flow sample rate which can vary from 0 to 300 Hz depending on
how fast D-Flow is completing it's computations. Any signal that you desire to
record that can't be recorded by

Gait Data
=========


Control Identification
======================


