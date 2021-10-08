![pase_icon](screenshots/pase_icon.png)
# Python Audio Spectrogram Explorer
### What you can do with this program:

- Visualize audio files as spectrograms

- Navigate through the spectrograms and listen in to selected areas in the spectrogram (adjustable playback speeds)

- Export selected areas in the spectrogram as .wav files

- Annotate areas in the spectrograms with custom labels and log each annotation's time-stamp and frequency 

- Export spectrograms as image files and automatically plot spectrograms for all selected files 

  ![screenshots/s1](screenshots/s1.JPG)

## How to install and start the program:
You can either download an executable or start the program by using the python source code. The windows executable is included in the repository (You can download it as .zip file on your computer).

A platform independent way to start the program is run the source code directly in python. To do so install python 3, ideally the latest Anaconda distribution (https://www.anaconda.com/products/individual), and download the file `python_audio_spectrogram_explorer.py`. Than open either the spyder IDE (or any other), open the downloaded python code and press the "run file" button (often a green play button). Now a graphical user interface (GUI) should open. 

Or from the command line, navigate to the folder that contains `python_audio_spectrogram_explorer.py` and start the program with this command: `python python_audio_spectrogram_explorer.py`. 

This program uses PyQT5 as GUI framework and numpy, scipy, pandas and matplotlib to manipulate and visualize the data. The module `simpleaudio` is used to playback sound. In case you are getting an error message due to a missing module, simply copy the module's name and install it using pip, for example `pip install simpleaudio` and `pip install soundfile`.

## How to use it:

### Open files with or without timestamps

The currently supported audio file types are: .wav .aif .aiff .aifc .ogg .flac

To get started, you first have to decide if you want to use real time-stamps (year-month-day hour:minute:seconds) or not. For simply looking at the spectrograms and exploring your audio-files, you do not need the real time-stamps. But as soon as you want to annotate your data, the program needs to know when each .wav file started recording. The default is using real time-stamps. 

**Without timestamps:**
- Delete the content of the field "filename key:"    
- Press the "Open .wav files" button

**With timestamps:**
- The start date and time of each recoding should be contained in the .wav file name

- Adjust the "filename key:" field so that the program recognizes the correct time-stamp. For example: `aural_%Y_%m_%d_%H_%M_%S.wav` or `%y%m%d_%H%M%S_AU_SO02.wav` Where %Y is year, %m is month, %d is day and so on.   Here is a list of the format strings:

  | **Directive** | **Meaning**                                                  | **Example**              |
  | ------------- | ------------------------------------------------------------ | ------------------------ |
  | `%a`          | Abbreviated weekday name.                                    | Sun, Mon, ...            |
  | `%A`          | Full weekday name.                                           | Sunday, Monday, ...      |
  | `%w`          | Weekday as a decimal number.                                 | 0, 1, ..., 6             |
  | `%d`          | Day of the month as a zero-padded decimal.                   | 01, 02, ..., 31          |
  | `%-d`         | Day of the month as a decimal number.                        | 1, 2, ..., 30            |
  | `%b`          | Abbreviated month name.                                      | Jan, Feb, ..., Dec       |
  | `%B`          | Full month name.                                             | January, February, ...   |
  | `%m`          | Month as a zero-padded decimal number.                       | 01, 02, ..., 12          |
  | `%-m`         | Month as a decimal number.                                   | 1, 2, ..., 12            |
  | `%y`          | Year without century as a zero-padded decimal number.        | 00, 01, ..., 99          |
  | `%-y`         | Year without century as a decimal number.                    | 0, 1, ..., 99            |
  | `%Y`          | Year with century as a decimal number.                       | 2013, 2019 etc.          |
  | `%H`          | Hour (24-hour clock) as a zero-padded decimal number.        | 00, 01, ..., 23          |
  | `%-H`         | Hour (24-hour clock) as a decimal number.                    | 0, 1, ..., 23            |
  | `%I`          | Hour (12-hour clock) as a zero-padded decimal number.        | 01, 02, ..., 12          |
  | `%-I`         | Hour (12-hour clock) as a decimal number.                    | 1, 2, ... 12             |
  | `%p`          | Locale’s AM or PM.                                           | AM, PM                   |
  | `%M`          | Minute as a zero-padded decimal number.                      | 00, 01, ..., 59          |
  | `%-M`         | Minute as a decimal number.                                  | 0, 1, ..., 59            |
  | `%S`          | Second as a zero-padded decimal number.                      | 00, 01, ..., 59          |
  | `%-S`         | Second as a decimal number.                                  | 0, 1, ..., 59            |
  | `%j`          | Day of the year as a zero-padded decimal number.             | 001, 002, ..., 366       |
  | `%-j`         | Day of the year as a decimal number.                         | 1, 2, ..., 366           |

- Press the "Open .wav files" button and select your .wav files with the dialogue.

### Plot and browse spectrograms 
- Select the spectrogram setting of your choice:
    - Minimum and maximum frequency (y-axis) as f_min and f_max
    - Linear or logarithmic (default) frequency scale 
    - The length (x-axis) of each spectrogram in seconds. If the field is left empty the spectrogram will be the length of the entire .wav file. 
    - The FFT size determines the spectral resolution. The higher it is, the more detail you will see in the lower part of the spectrogram, with less detail in the upper part 
    - The minimum and maximum dB values for the spectrogram color, will be determined automatically if left empty
    - The colormap from a dropdown menu
- Press next spectrogram (The Shortkey for this is the right arrow button)
- You can now navigate between the spectrograms using the "next/previous spectrogram" buttons or the left and right arrow keys. The time-stamp or filename of the current .wav file is displayed as title. 
- You can zoom and pan using the magnifying glass symbol in the matplotlib toolbar, where you can also save the spectrogram as image file. 
- Once you have reached the final spectrogram, the program will display a warning

Here is an example for the black and white colormap called "gist_yarg"
 ![screenshots/s3](screenshots/s3.JPG)

### Play audio and adjust playback speed, export the selected sound as .wav
- Press the "Play/Stop" button or the spacebar to play the .wav file.
- The program will only play what is visible in the current spectrogram (Sound above and below the frequency limits is filtered out)
- To listen to specific sounds, zoom in using the magnifying glass
- To listen to sound below or above the human hearing range, adjust the playback speed and press the Play button again.   
- To export the sound you selected as .wav file, press the "Export selected audio" button

### Automatically plot spectrograms of multiple .wav files 

- Select your .wav files with the "Open .wav files" button
- Select the spectrogram settings of your choice
- Press the "Plot all spectrograms" button and confirm the pop-up question
- The spectrograms will be saved as .jpg files with the same filename and location as your .wav files. 

### Annotate the spectrograms

- Make sure the "filename key" field contains the correct time-stamp information

- Now you can either choose to log you annotations in real time or save them later. I recommend using the "real-time logging" option. 

- Press the "real-time logging" check-box. Now the program will look if there are already log files existing for each .wav file. Log files are named by adding "_log.csv" to the .wav filename, for example "aural_2017_02_12_22_40_00_log.csv". You can choose to overwrite these log files. If you do not choose to overwrite them, the program ignores .wav files that already have and existing log file. This is useful if you want to work on a dataset over several sessions. 

- Now you can choose custom (or preset) labels for your annotations by changing the labels in the row "Annotation labels". If no label is selected (using the check-boxes) an empty string will be used as label. 

- To set an annotation, left-click at any location inside the spectrogram plot and draw a rectangle over the region of interest

- To remove the last annotation, click the right mouse button. 

- Once a new .wav file is opened, the annotations for the previous .wav file are saved as .csv file, for example  as "aural_2017_02_12_22_40_00_log.csv". If no annotations were set an empty table is saved. This indicates you have already screened this .wav file but found nothing to annotate. 

- The "...._log.csv" files are formated like this (t1,f1 and t2,f2 are the lower left and upper right of the annotation box):

|      | t1       |   t2   |       f1       | f2          | Label             |
| ---- | ------------------  | ---------------|------- | ----|--- |
| 0    | 2016-04-09 19:25:47.49 |2016-04-09 19:25:49.49  | 17.313 | 20.546   | FW_20_Hz  |
| 1    | 2016-05-10 17:36:13.94 | 2016-05-10 17:38:13.94 | 27.59109  | 34.57 | BW_Z_call |

- If you want to save your annotations separately, press the "Save annotation csv" button

### Remove the background from spectrogram

This feature can be useful to detect sounds hidden in background noise. It subtracts the average spectrum from the current spectrogram, so that the horizontal noise lines and slope in background noise disappear. To active this function toggle the checkbox called "Remove background".  For optimal use, test different dB minimum setting. Here is an example for the spectrogram shown above:
  ![screenshots/s2](screenshots/s2.JPG)

