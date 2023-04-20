# Rolling Shutter Test Post Processing Software

Post processing software for flicker method of rolling shutter testing. Process reads a video or still image
threosholds it to just above the black level. Then counts the number of lines in the exposed bands.

User can specify the light's pulse time and the software will calculate the rolling shutter time for the files using:

$$t_{line} = \frac{ exposure\\_time + flash\\_time }{exposed\\_rows}$$

$$t_{frame} = t_{line} \times num\\_lines$$


## Requirements
* numpy
* opencv-python
* matplotlib
* pyexiftool
