This folder contains the script used to generate the BirdSED dataset.

To get the audio files for the bird species that are used as the foreground, ...

Different white noise audio clips were downloaded from [freesound](freesound.org). For the background noises, multiple 
audio clips from the ESC50 dataset were taken. The list of audio files used can be found in esc50.txt

The file structure should look like this:

├── generate_dataset.py\
├── source\
│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── foreground\
│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── Accipiter_gentilis\
│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── ...\
│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│\
│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── background\
│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── crickets\
│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── footsteps\
│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── frog\
│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── insects\
│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── rain\
│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── white_noise\
│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;│   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── wind

With the audio clips in the respective folders

