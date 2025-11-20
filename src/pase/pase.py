# -*- coding: utf-8 -*-
"""
Python Audio Spectrogram Explorer (PASE)
Created on Mon Sep  6 17:28:37 2021

@author: Sebastian Menze, sebastian.menze@gmail.com
"""
import sys
from PyQt5 import QtCore, QtWidgets
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import scipy.io.wavfile as wav
import soundfile as sf
from scipy import signal
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import datetime as dt
import os

from matplotlib.widgets import RectangleSelector

# Replace simpleaudio with sounddevice
import sounddevice as sd

from skimage import filters, measure, morphology
from skimage.morphology import closing, disk
from matplotlib.path import Path
from skimage.transform import resize

from scipy.signal import find_peaks
from skimage.feature import match_template

# Optional moviepy import - only needed for video export
try:
    from moviepy.editor import VideoClip, AudioFileClip
    from moviepy.video.io.bindings import mplfig_to_npimage
    MOVIEPY_AVAILABLE = True
except (ImportError, RuntimeError) as e:
    MOVIEPY_AVAILABLE = False
    print("Warning: moviepy not available. Video export will be disabled.")
    print(f"Error: {e}")
    print("To enable video export, install ffmpeg: https://ffmpeg.org/download.html")


class MplCanvas(FigureCanvasQTAgg):
    def __init__(self, parent=None, dpi=150):
        self.fig = Figure(figsize=None, dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)


class gui(QtWidgets.QMainWindow):
    def __init__(self, *args, **kwargs):
        super(gui, self).__init__(*args, **kwargs)

        self.canvas = MplCanvas(self, dpi=150)
        
        # Audio playback control
        self.audio_stream = None
        self.audio_playing = False
        
        # Initialize UI elements
        self.f_min = QtWidgets.QLineEdit(self)
        self.f_min.setText('10')
        self.f_max = QtWidgets.QLineEdit(self)
        self.f_max.setText('16000')
        self.t_length = QtWidgets.QLineEdit(self)
        self.t_length.setText('120')
        self.db_saturation = QtWidgets.QLineEdit(self)
        self.db_saturation.setText('155')
        self.db_vmin = QtWidgets.QLineEdit(self)
        self.db_vmin.setText('30')
        self.db_vmax = QtWidgets.QLineEdit(self)
        self.db_vmax.setText('')
        
        self.fft_size = QtWidgets.QComboBox(self)
        self.fft_size.addItems(['1024', '2048', '4096', '8192', '16384', '32768', '65536', '131072'])
        self.fft_size.setCurrentIndex(4)
        
        self.colormap_plot = QtWidgets.QComboBox(self)
        self.colormap_plot.addItems(['plasma', 'viridis', 'inferno', 'gist_gray', 'gist_yarg'])
        self.colormap_plot.setCurrentIndex(2)
        
        self.checkbox_logscale = QtWidgets.QCheckBox('Log. scale')
        self.checkbox_logscale.setChecked(True)
        self.checkbox_background = QtWidgets.QCheckBox('Remove background')
        self.checkbox_background.setChecked(False)
        
        self.fft_overlap = QtWidgets.QComboBox(self)
        self.fft_overlap.addItems(['0.2', '0.5', '0.7', '0.9'])
        self.fft_overlap.setCurrentIndex(3)
        
        self.filename_timekey = QtWidgets.QLineEdit(self)
        
        self.playbackspeed = QtWidgets.QComboBox(self)
        self.playbackspeed.addItems(['0.5', '1', '2', '5', '10'])
        self.playbackspeed.setCurrentIndex(1)
        
        # Initialize data variables
        self.time = dt.datetime(2000, 1, 1, 0, 0, 0)
        self.f = None
        self.t = [-1, -1]
        self.Sxx = None
        self.draw_x = pd.Series(dtype='float')
        self.draw_y = pd.Series(dtype='float')
        self.cid1 = None
        self.cid2 = None
        
        self.plotwindow_startsecond = float(self.t_length.text())
        self.filecounter = -1
        self.filenames = np.array([])
        self.current_audiopath = None
        self.detectiondf = pd.DataFrame([])
        
        # Connect signals
        self.fft_size.currentIndexChanged.connect(self.new_fft_size_selected)
        self.colormap_plot.currentIndexChanged.connect(self.plot_spectrogram)
        self.checkbox_background.stateChanged.connect(self.plot_spectrogram)
        self.checkbox_logscale.stateChanged.connect(self.plot_spectrogram)
        
        self.checkbox_log = QtWidgets.QCheckBox('Real-time Logging')
        self.checkbox_log.toggled.connect(self.func_logging)
        
        # Create annotation labels
        self.checkbox_an_1 = QtWidgets.QCheckBox()
        self.an_1 = QtWidgets.QLineEdit(self)
        self.checkbox_an_2 = QtWidgets.QCheckBox()
        self.an_2 = QtWidgets.QLineEdit(self)
        self.checkbox_an_3 = QtWidgets.QCheckBox()
        self.an_3 = QtWidgets.QLineEdit(self)
        self.checkbox_an_4 = QtWidgets.QCheckBox()
        self.an_4 = QtWidgets.QLineEdit(self)
        self.checkbox_an_5 = QtWidgets.QCheckBox()
        self.an_5 = QtWidgets.QLineEdit(self)
        self.checkbox_an_6 = QtWidgets.QCheckBox()
        self.an_6 = QtWidgets.QLineEdit(self)
        
        self.bg = QtWidgets.QButtonGroup()
        self.bg.addButton(self.checkbox_an_1, 1)
        self.bg.addButton(self.checkbox_an_2, 2)
        self.bg.addButton(self.checkbox_an_3, 3)
        self.bg.addButton(self.checkbox_an_4, 4)
        self.bg.addButton(self.checkbox_an_5, 5)
        self.bg.addButton(self.checkbox_an_6, 6)
        
        # Setup menu bar
        self.setup_menubar()
        
        # Setup layouts
        self.setup_layouts()
        
        # Setup hotkeys
        self.msgSc1 = QtWidgets.QShortcut(QtCore.Qt.Key_Right, self)
        self.msgSc1.activated.connect(self.plot_next_spectro)
        self.msgSc2 = QtWidgets.QShortcut(QtCore.Qt.Key_Left, self)
        self.msgSc2.activated.connect(self.plot_previous_spectro)
        self.msgSc3 = QtWidgets.QShortcut(QtCore.Qt.Key_Space, self)
        self.msgSc3.activated.connect(self.func_playaudio)
        
        self.show()
    
    def setup_menubar(self):
        """Setup the menu bar"""
        menuBar = self.menuBar()
        
        openMenu = menuBar.addAction("Open files")
        openMenu.triggered.connect(self.openfilefunc)
        
        exportMenu = menuBar.addMenu("Export")
        exportMenu.addAction("Spectrogram as .wav file").triggered.connect(self.func_saveaudio)
        exportMenu.addAction("Spectrogram as animated video").triggered.connect(self.func_save_video)
        exportMenu.addAction("Spectrogram as .csv table").triggered.connect(self.export_zoomed_sgram_as_csv)
        exportMenu.addAction("All files as spectrogram images").triggered.connect(self.plot_all_spectrograms)
        exportMenu.addAction("Annotations as .csv table").triggered.connect(self.func_savecsv)
        exportMenu.addAction("Automatic detections as .csv table").triggered.connect(self.export_automatic_detector)
        
        drawMenu = menuBar.addAction("Draw")
        drawMenu.triggered.connect(self.func_draw_shape)
        
        autoMenu = menuBar.addMenu("Automatic detection")
        autoMenu.addAction("Shapematching on current file").triggered.connect(self.automatic_detector_shapematching)
        autoMenu.addAction("Shapematching on all files").triggered.connect(self.automatic_detector_shapematching_allfiles)
        autoMenu.addAction("Spectrogram correlation on current file").triggered.connect(self.automatic_detector_specgram_corr)
        autoMenu.addAction("Spectrogram correlation on all files").triggered.connect(self.automatic_detector_specgram_corr_allfiles)
        autoMenu.addAction("Show regions based on threshold").triggered.connect(self.plot_spectrogram_threshold)
        
        quitMenu = menuBar.addAction("Quit")
        quitMenu.triggered.connect(self.exitfunc)
    
    def setup_layouts(self):
        """Setup all UI layouts"""
        outer_layout = QtWidgets.QVBoxLayout()
        
        # Top layout with settings
        top2_layout = QtWidgets.QHBoxLayout()
        top2_layout.addWidget(self.checkbox_log)
        top2_layout.addWidget(self.checkbox_logscale)
        top2_layout.addWidget(self.checkbox_background)
        top2_layout.addWidget(QtWidgets.QLabel('Timestamp:'))
        top2_layout.addWidget(self.filename_timekey)
        top2_layout.addWidget(QtWidgets.QLabel('f_min[Hz]:'))
        top2_layout.addWidget(self.f_min)
        top2_layout.addWidget(QtWidgets.QLabel('f_max[Hz]:'))
        top2_layout.addWidget(self.f_max)
        top2_layout.addWidget(QtWidgets.QLabel('Spec. length [sec]:'))
        top2_layout.addWidget(self.t_length)
        top2_layout.addWidget(QtWidgets.QLabel('Saturation dB:'))
        top2_layout.addWidget(self.db_saturation)
        top2_layout.addWidget(QtWidgets.QLabel('dB min:'))
        top2_layout.addWidget(self.db_vmin)
        top2_layout.addWidget(QtWidgets.QLabel('dB max:'))
        top2_layout.addWidget(self.db_vmax)
        
        # Annotation labels layout
        top3_layout = QtWidgets.QHBoxLayout()
        top3_layout.addWidget(QtWidgets.QLabel('Annotation labels:'))
        for i in range(1, 7):
            checkbox = getattr(self, f'checkbox_an_{i}')
            lineedit = getattr(self, f'an_{i}')
            top3_layout.addWidget(checkbox)
            top3_layout.addWidget(lineedit)
        
        # Plot layout
        plot_layout = QtWidgets.QVBoxLayout()
        tnav = NavigationToolbar(self.canvas, self)
        
        toolbar = QtWidgets.QToolBar()
        
        button_plot_prevspectro = QtWidgets.QPushButton('<--Previous spectrogram')
        button_plot_prevspectro.clicked.connect(self.plot_previous_spectro)
        toolbar.addWidget(button_plot_prevspectro)
        
        toolbar.addWidget(QtWidgets.QLabel('  '))
        
        button_plot_spectro = QtWidgets.QPushButton('Next spectrogram-->')
        button_plot_spectro.clicked.connect(self.plot_next_spectro)
        toolbar.addWidget(button_plot_spectro)
        
        toolbar.addWidget(QtWidgets.QLabel('  '))
        
        button_play_audio = QtWidgets.QPushButton('Play/Stop [spacebar]')
        button_play_audio.clicked.connect(self.func_playaudio)
        toolbar.addWidget(button_play_audio)
        
        toolbar.addWidget(QtWidgets.QLabel('  '))
        toolbar.addWidget(QtWidgets.QLabel('Playback speed:'))
        toolbar.addWidget(QtWidgets.QLabel('  '))
        toolbar.addWidget(self.playbackspeed)
        toolbar.addWidget(QtWidgets.QLabel('  '))
        
        toolbar.addSeparator()
        toolbar.addWidget(QtWidgets.QLabel('  '))
        
        toolbar.addWidget(QtWidgets.QLabel('fft_size[bits]:'))
        toolbar.addWidget(QtWidgets.QLabel('  '))
        toolbar.addWidget(self.fft_size)
        toolbar.addWidget(QtWidgets.QLabel('  '))
        toolbar.addWidget(QtWidgets.QLabel('fft_overlap[0-1]:'))
        toolbar.addWidget(QtWidgets.QLabel('  '))
        toolbar.addWidget(self.fft_overlap)
        toolbar.addWidget(QtWidgets.QLabel('  '))
        toolbar.addWidget(QtWidgets.QLabel('Colormap:'))
        toolbar.addWidget(QtWidgets.QLabel('  '))
        toolbar.addWidget(self.colormap_plot)
        toolbar.addWidget(QtWidgets.QLabel('  '))
        
        toolbar.addSeparator()
        toolbar.addWidget(tnav)
        
        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(self.canvas)
        
        outer_layout.addLayout(top2_layout)
        outer_layout.addLayout(top3_layout)
        outer_layout.addLayout(plot_layout)
        
        widget = QtWidgets.QWidget()
        widget.setLayout(outer_layout)
        self.setCentralWidget(widget)
    
    def exitfunc(self):
        """Exit the application"""
        self.stop_audio()
        QtWidgets.QApplication.instance().quit()
        self.close()
    
    def stop_audio(self):
        """Stop any playing audio"""
        if self.audio_playing:
            sd.stop()
            self.audio_playing = False
    
    def find_regions(self, db_threshold):
        """Find regions in spectrogram above threshold"""
        y1 = int(self.f_min.text())
        y2 = int(self.f_max.text())
        if y2 > (self.fs / 2):
            y2 = (self.fs / 2)
        t1 = self.plotwindow_startsecond
        t2 = self.plotwindow_startsecond + self.plotwindow_length
        
        ix_time = np.where((self.t >= t1) & (self.t < t2))[0]
        ix_f = np.where((self.f >= y1) & (self.f < y2))[0]
        
        t = self.t[ix_time]
        minimum_patcharea = 5
        
        plotsxx = self.Sxx[int(ix_f[0]):int(ix_f[-1]), int(ix_time[0]):int(ix_time[-1])]
        spectrog = 10 * np.log10(plotsxx)
        
        # Filter out background
        spec_mean = np.median(spectrog, axis=1)
        sxx_background = np.transpose(np.broadcast_to(spec_mean, np.transpose(spectrog).shape))
        z = spectrog - sxx_background
        
        # Binary image, post-process the binary mask and compute labels
        mask = z > db_threshold
        mask = morphology.remove_small_objects(mask, 50, connectivity=30)
        mask = morphology.remove_small_holes(mask, 50, connectivity=30)
        mask = closing(mask, disk(3))
        
        labels = measure.label(mask)
        
        probs = measure.regionprops_table(labels, spectrog, properties=['label', 'area', 'mean_intensity',
                                                                         'orientation', 'major_axis_length',
                                                                         'minor_axis_length', 'weighted_centroid',
                                                                         'bbox'])
        df = pd.DataFrame(probs)
        
        # Get correct f and t
        ff = self.f[ix_f[0]:ix_f[-1]]
        ix = df['bbox-0'] > len(ff) - 1
        df.loc[ix, 'bbox-0'] = len(ff) - 1
        ix = df['bbox-2'] > len(ff) - 1
        df.loc[ix, 'bbox-2'] = len(ff) - 1
        
        df['f-1'] = ff[df['bbox-0']]
        df['f-2'] = ff[df['bbox-2']]
        df['f-width'] = df['f-2'] - df['f-1']
        
        ix = df['bbox-1'] > len(t) - 1
        df.loc[ix, 'bbox-1'] = len(t) - 1
        ix = df['bbox-3'] > len(t) - 1
        df.loc[ix, 'bbox-3'] = len(t) - 1
        
        df['t-1'] = t[df['bbox-1']]
        df['t-2'] = t[df['bbox-3']]
        df['duration'] = df['t-2'] - df['t-1']
        
        indices = np.where((df['area'] < minimum_patcharea) |
                          (df['bbox-3'] - df['bbox-1'] < 3) |
                          (df['bbox-2'] - df['bbox-0'] < 3))[0]
        df = df.drop(indices)
        df = df.reset_index()
        
        df['id'] = np.arange(len(df))
        
        # Get region dict
        patches = {}
        p_t_dict = {}
        p_f_dict = {}
        
        for ix in range(len(df)):
            m = labels == df.loc[ix, 'label']
            ix1 = df.loc[ix, 'bbox-1']
            ix2 = df.loc[ix, 'bbox-3']
            jx1 = df.loc[ix, 'bbox-0']
            jx2 = df.loc[ix, 'bbox-2']
            
            patch = m[jx1:jx2, ix1:ix2]
            pt = t[ix1:ix2]
            pt = pt - pt[0]
            pf = ff[jx1:jx2]
            
            patches[df['id'][ix]] = patch
            p_t_dict[df['id'][ix]] = pt
            p_f_dict[df['id'][ix]] = pf
        
        self.detectiondf = df
        self.patches = patches
        self.p_t_dict = p_t_dict
        self.p_f_dict = p_f_dict
        self.region_labels = labels
    
    def match_bbox_and_iou(self, template):
        """Match detected regions with template using IoU"""
        shape_f = template['Frequency_in_Hz'].values
        shape_t = template['Time_in_s'].values
        shape_t = shape_t - shape_t.min()
        
        df = self.detectiondf
        patches = self.patches
        p_t_dict = self.p_t_dict
        p_f_dict = self.p_f_dict
        
        score_ioubox = []
        smc_rs = []
        
        for ix in df.index:
            patch = patches[ix]
            pf = p_f_dict[ix]
            pt = p_t_dict[ix]
            pt = pt - pt[0]
            
            if df.loc[ix, 'f-1'] < shape_f.min():
                f1 = df.loc[ix, 'f-1']
            else:
                f1 = shape_f.min()
            if df.loc[ix, 'f-2'] > shape_f.max():
                f2 = df.loc[ix, 'f-2']
            else:
                f2 = shape_f.max()
            
            time_step = np.diff(pt)[0]
            f_step = np.diff(pf)[0]
            k_f = np.arange(f1, f2, f_step)
            
            if pt.max() > shape_t.max():
                k_t = pt
            else:
                k_t = np.arange(0, shape_t.max(), time_step)
            
            # IoU bounding box
            iou_kernel = np.zeros([k_f.shape[0], k_t.shape[0]])
            ixp2 = np.where((k_t >= shape_t.min()) & (k_t <= shape_t.max()))[0]
            ixp1 = np.where((k_f >= shape_f.min()) & (k_f <= shape_f.max()))[0]
            iou_kernel[ixp1[0]:ixp1[-1], ixp2[0]:ixp2[-1]] = 1
            
            iou_patch = np.zeros([k_f.shape[0], k_t.shape[0]])
            ixp2 = np.where((k_t >= pt[0]) & (k_t <= pt[-1]))[0]
            ixp1 = np.where((k_f >= pf[0]) & (k_f <= pf[-1]))[0]
            iou_patch[ixp1[0]:ixp1[-1], ixp2[0]:ixp2[-1]] = 1
            
            intersection = iou_kernel.astype('bool') & iou_patch.astype('bool')
            union = iou_kernel.astype('bool') | iou_patch.astype('bool')
            iou_bbox = np.sum(intersection) / np.sum(union)
            score_ioubox.append(iou_bbox)
            
            patch_rs = resize(patch, (50, 50))
            n_resize = 50
            k_t = np.linspace(0, shape_t.max(), n_resize)
            k_f = np.linspace(shape_f.min(), shape_f.max(), n_resize)
            kk_t, kk_f = np.meshgrid(k_t, k_f)
            x, y = kk_t.flatten(), kk_f.flatten()
            points = np.vstack((x, y)).T
            p = Path(list(zip(shape_t, shape_f)))
            grid = p.contains_points(points)
            kernel_rs = grid.reshape(kk_t.shape)
            smc_rs.append(np.sum(kernel_rs.astype('bool') == patch_rs.astype('bool')) / len(patch_rs.flatten()))
        
        smc_rs = np.array(smc_rs)
        score_ioubox = np.array(score_ioubox)
        
        score = score_ioubox * (smc_rs - 0.5) / 0.5
        return score
    
    def automatic_detector_specgram_corr(self):
        """Run spectrogram correlation detector on current file"""
        self.detectiondf = pd.DataFrame([])
        
        templatefiles, ok1 = QtWidgets.QFileDialog.getOpenFileNames(
            self, "QFileDialog.getOpenFileNames()", "", "CSV file (*.csv)")
        
        if not ok1:
            return
        
        templates = []
        for fnam in templatefiles:
            template = pd.read_csv(fnam, index_col=0)
            templates.append(template)
        
        corrscore_threshold, ok = QtWidgets.QInputDialog.getDouble(
            self, 'Input Dialog', 'Enter correlation threshold in (0-1):', decimals=2)
        
        if not ok:
            return
        
        corrscore_threshold = max(0, min(1, corrscore_threshold))
        
        if templates[0].columns[0] == 'Time_in_s':
            self._run_shape_correlation(templates, corrscore_threshold)
        else:
            self._run_image_correlation(templates, corrscore_threshold)
        
        print(self.detectiondf)
        print('done!!!')
        self.plot_spectrogram()
    
    def _run_shape_correlation(self, templates, corrscore_threshold):
        """Run shape-based correlation"""
        offset_f = 10
        offset_t = 0.5
        
        shape_f = np.array([])
        shape_t_raw = np.array([])
        for template in templates:
            shape_f = np.concatenate([shape_f, template['Frequency_in_Hz'].values])
            shape_t_raw = np.concatenate([shape_t_raw, template['Time_in_s'].values])
        shape_t = shape_t_raw - shape_t_raw.min()
        
        f_lim = [shape_f.min() - offset_f, shape_f.max() + offset_f]
        k_length_seconds = shape_t.max() + offset_t * 2
        
        time_step = np.diff(self.t)[0]
        k_t = np.linspace(0, k_length_seconds, int(k_length_seconds / time_step))
        ix_f = np.where((self.f >= f_lim[0]) & (self.f <= f_lim[1]))[0]
        k_f = self.f[ix_f[0]:ix_f[-1]]
        
        kk_t, kk_f = np.meshgrid(k_t, k_f)
        kernel = np.zeros([k_f.shape[0], k_t.shape[0]])
        
        x, y = kk_t.flatten(), kk_f.flatten()
        points = np.vstack((x, y)).T
        
        for template in templates:
            shf = template['Frequency_in_Hz'].values
            st = template['Time_in_s'].values
            st = st - shape_t_raw.min()
            p = Path(list(zip(st, shf)))
            grid = p.contains_points(points)
            kern = grid.reshape(kk_t.shape)
            kernel[kern > 0] = 1
        
        ix_f = np.where((self.f >= f_lim[0]) & (self.f <= f_lim[1]))[0]
        spectrog = 10 * np.log10(self.Sxx[ix_f[0]:ix_f[-1], :])
        
        result = match_template(spectrog, kernel)
        corr_score = result[0, :]
        t_score = np.linspace(self.t[int(kernel.shape[1] / 2)],
                             self.t[-int(kernel.shape[1] / 2)], corr_score.shape[0])
        
        peaks_indices = find_peaks(corr_score, height=corrscore_threshold)[0]
        
        if len(peaks_indices) > 0:
            t1, t2, f1, f2, score = [], [], [], [], []
            for ixpeak in peaks_indices:
                tstar = t_score[ixpeak] - k_length_seconds / 2 - offset_t
                tend = t_score[ixpeak] + k_length_seconds / 2 - offset_t
                t1.append(tstar)
                t2.append(tend)
                f1.append(f_lim[0] + offset_f)
                f2.append(f_lim[1] - offset_f)
                score.append(corr_score[ixpeak])
            
            df = pd.DataFrame({'t-1': t1, 't-2': t2, 'f-1': f1, 'f-2': f2, 'score': score})
            self.detectiondf = df.copy()
            self.detectiondf['audiofilename'] = self.current_audiopath
            self.detectiondf['threshold'] = corrscore_threshold
    
    def _run_image_correlation(self, templates, corrscore_threshold):
        """Run image-based correlation"""
        template = templates[0]
        
        k_length_seconds = float(template.columns[-1]) - float(template.columns[0])
        f_lim = [int(template.index[0]), int(template.index[-1])]
        
        ix_f = np.where((self.f >= f_lim[0]) & (self.f <= f_lim[1]))[0]
        spectrog = 10 * np.log10(self.Sxx[ix_f[0]:ix_f[-1], :])
        specgram_t_step = self.t[1] - self.t[0]
        n_f = spectrog.shape[0]
        n_t = int(k_length_seconds / specgram_t_step)
        
        kernel = resize(template.values, [n_f, n_t])
        
        result = match_template(spectrog, kernel)
        corr_score = result[0, :]
        t_score = np.linspace(self.t[int(kernel.shape[1] / 2)],
                             self.t[-int(kernel.shape[1] / 2)], corr_score.shape[0])
        
        peaks_indices = find_peaks(corr_score, height=corrscore_threshold)[0]
        
        if len(peaks_indices) > 0:
            t1, t2, f1, f2, score = [], [], [], [], []
            for ixpeak in peaks_indices:
                tstar = t_score[ixpeak] - k_length_seconds / 2
                tend = t_score[ixpeak] + k_length_seconds / 2
                t1.append(tstar)
                t2.append(tend)
                f1.append(f_lim[0])
                f2.append(f_lim[1])
                score.append(corr_score[ixpeak])
            
            df = pd.DataFrame({'t-1': t1, 't-2': t2, 'f-1': f1, 'f-2': f2, 'score': score})
            self.detectiondf = df.copy()
            self.detectiondf['audiofilename'] = self.current_audiopath
            self.detectiondf['threshold'] = corrscore_threshold
    
    def automatic_detector_specgram_corr_allfiles(self):
        """Run spectrogram correlation detector on all files"""
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(f"Are you sure you want to run the detector over {self.file_blocks.shape[0]} files?")
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        returnValue = msg.exec()
        
        if returnValue != QtWidgets.QMessageBox.Yes:
            return
        
        templatefiles, ok1 = QtWidgets.QFileDialog.getOpenFileNames(
            self, "QFileDialog.getOpenFileNames()", "", "CSV file (*.csv)")
        
        if not ok1:
            return
        
        templates = []
        for fnam in templatefiles:
            template = pd.read_csv(fnam, index_col=0)
            templates.append(template)
        
        corrscore_threshold, ok = QtWidgets.QInputDialog.getDouble(
            self, 'Input Dialog', 'Enter correlation threshold in (0-1):', decimals=2)
        
        if not ok:
            return
        
        corrscore_threshold = max(0, min(1, corrscore_threshold))
        self.detectiondf_all = pd.DataFrame([])
        
        for i_block in range(len(self.file_blocks)):
            audiopath = self.file_blocks.loc[i_block, 'fname']
            
            if self.filename_timekey.text() == '':
                self.time = dt.datetime(2000, 1, 1, 0, 0, 0)
            else:
                try:
                    self.time = dt.datetime.strptime(audiopath.split('/')[-1], self.filename_timekey.text())
                except:
                    self.time = dt.datetime(2000, 1, 1, 0, 0, 0)
            
            if self.file_blocks.loc[i_block, 'start'] > 0:
                secoffset = self.file_blocks.loc[i_block, 'start'] / self.fs
                self.time = self.time + pd.Timedelta(seconds=secoffset)
            
            self.x, self.fs = sf.read(audiopath, dtype='int16',
                                      start=self.file_blocks.loc[i_block, 'start'],
                                      stop=self.file_blocks.loc[i_block, 'end'])
            print(f'Processing file: {audiopath}')
            
            if len(self.x.shape) > 1:
                if np.shape(self.x)[1] > 1:
                    self.x = self.x[:, 0]
            
            db_saturation = float(self.db_saturation.text())
            x = self.x / 32767
            p = np.power(10, (db_saturation / 20)) * x
            
            fft_size = int(self.fft_size.currentText())
            fft_overlap = float(self.fft_overlap.currentText())
            
            self.f, self.t, self.Sxx = signal.spectrogram(
                p, self.fs, window='hamming', nperseg=fft_size, noverlap=int(fft_size * fft_overlap))
            
            if self.file_blocks.loc[i_block, 'start'] > 0:
                secoffset = self.file_blocks.loc[i_block, 'start'] / self.fs
                self.t = self.t + secoffset
            
            if templates[0].columns[0] == 'Time_in_s':
                self._run_shape_correlation(templates, corrscore_threshold)
            else:
                self._run_image_correlation(templates, corrscore_threshold)
            
            if hasattr(self, 'detectiondf') and not self.detectiondf.empty:
                self.detectiondf_all = pd.concat([self.detectiondf_all, self.detectiondf])
                self.detectiondf_all = self.detectiondf_all.reset_index(drop=True)
        
        self.detectiondf = self.detectiondf_all
        self.read_wav()
        self.plot_spectrogram()
        print('Done!!!')
    
    def automatic_detector_shapematching(self):
        """Run shape matching detector on current file"""
        self.detectiondf = pd.DataFrame([])
        
        templatefiles, ok1 = QtWidgets.QFileDialog.getOpenFileNames(
            self, "QFileDialog.getOpenFileNames()", "", "CSV file (*.csv)")
        
        if not ok1:
            return
        
        db_threshold, ok = QtWidgets.QInputDialog.getInt(
            self, 'Input Dialog', 'Enter signal-to-noise threshold in dB:')
        
        if not ok:
            return
        
        print(f'Threshold: {db_threshold} dB')
        self.detectiondf = pd.DataFrame([])
        
        self.find_regions(db_threshold)
        self.detectiondf['score'] = np.zeros(len(self.detectiondf))
        
        for fnam in templatefiles:
            template = pd.read_csv(fnam, index_col=0)
            score_new = self.match_bbox_and_iou(template)
            ix_better = score_new > self.detectiondf['score'].values
            self.detectiondf.loc[ix_better, 'score'] = score_new[ix_better]
        
        ixdel = np.where(self.detectiondf['score'] < 0.01)[0]
        self.detectiondf = self.detectiondf.drop(ixdel)
        self.detectiondf = self.detectiondf.reset_index(drop=True)
        self.detectiondf['audiofilename'] = self.current_audiopath
        self.detectiondf['threshold'] = db_threshold
        
        print(self.detectiondf)
        self.plot_spectrogram()
    
    def automatic_detector_shapematching_allfiles(self):
        """Run shape matching detector on all files"""
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(f"Are you sure you want to run the detector over {self.filenames.shape[0]} files?")
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        returnValue = msg.exec()
        
        if returnValue != QtWidgets.QMessageBox.Yes:
            return
        
        templatefiles, ok1 = QtWidgets.QFileDialog.getOpenFileNames(
            self, "QFileDialog.getOpenFileNames()", "", "CSV file (*.csv)")
        
        if not ok1:
            return
        
        db_threshold, ok = QtWidgets.QInputDialog.getInt(
            self, 'Input Dialog', 'Enter signal-to-noise threshold in dB:')
        
        if not ok:
            return
        
        self.detectiondf_all = pd.DataFrame([])
        
        for i_block in range(len(self.file_blocks)):
            audiopath = self.file_blocks.loc[i_block, 'fname']
            
            if self.filename_timekey.text() == '':
                self.time = dt.datetime(2000, 1, 1, 0, 0, 0)
            else:
                try:
                    self.time = dt.datetime.strptime(audiopath.split('/')[-1], self.filename_timekey.text())
                except:
                    self.time = dt.datetime(2000, 1, 1, 0, 0, 0)
            
            self.x, self.fs = sf.read(audiopath, dtype='int16',
                                      start=self.file_blocks.loc[i_block, 'start'],
                                      stop=self.file_blocks.loc[i_block, 'end'])
            print(f'Processing file: {audiopath}')
            
            if len(self.x.shape) > 1:
                if np.shape(self.x)[1] > 1:
                    self.x = self.x[:, 0]
            
            db_saturation = float(self.db_saturation.text())
            x = self.x / 32767
            p = np.power(10, (db_saturation / 20)) * x
            
            fft_size = int(self.fft_size.currentText())
            fft_overlap = float(self.fft_overlap.currentText())
            
            self.f, self.t, self.Sxx = signal.spectrogram(
                p, self.fs, window='hamming', nperseg=fft_size, noverlap=int(fft_size * fft_overlap))
            
            if self.file_blocks.loc[i_block, 'start'] > 0:
                secoffset = self.file_blocks.loc[i_block, 'start'] / self.fs
                self.t = self.t + secoffset
            
            self.plotwindow_startsecond = 0
            self.plotwindow_length = self.t.max()
            
            self.detectiondf = pd.DataFrame([])
            self.find_regions(db_threshold)
            self.detectiondf['score'] = np.zeros(len(self.detectiondf))
            
            for fnam in templatefiles:
                template = pd.read_csv(fnam, index_col=0)
                score_new = self.match_bbox_and_iou(template)
                ix_better = score_new > self.detectiondf['score'].values
                self.detectiondf.loc[ix_better, 'score'] = score_new[ix_better]
            
            ixdel = np.where(self.detectiondf['score'] < 0.01)[0]
            self.detectiondf = self.detectiondf.drop(ixdel)
            self.detectiondf = self.detectiondf.reset_index(drop=True)
            self.detectiondf['audiofilename'] = audiopath
            self.detectiondf['threshold'] = db_threshold
            
            self.detectiondf_all = pd.concat([self.detectiondf_all, self.detectiondf])
            self.detectiondf_all = self.detectiondf_all.reset_index(drop=True)
            
            print(self.detectiondf_all)
        
        self.detectiondf = self.detectiondf_all
        print('Done!!!')
    
    def export_automatic_detector(self):
        """Export automatic detection results to CSV"""
        if self.detectiondf.shape[0] > 0:
            savename = QtWidgets.QFileDialog.getSaveFileName(
                self, "QFileDialog.getSaveFileName()", "", "csv files (*.csv)")
            if len(savename[0]) > 0:
                self.detectiondf.to_csv(savename[0])
    
    def openfilefunc(self):
        """Open audio files dialog"""
        fname_candidates, ok = QtWidgets.QFileDialog.getOpenFileNames(
            self, "QFileDialog.getOpenFileNames()", '',
            "Audio Files (*.wav *.aif *.aiff *.aifc *.ogg *.flac)")
        
        if len(fname_candidates) == 0:
            return
        
        self.filenames = np.array(fname_candidates)
        self.filecounter = -1
        self.plotwindow_startsecond = float(self.t_length.text())
        
        self.annotation = pd.DataFrame({
            't1': pd.Series(dtype='datetime64[ns]'),
            't2': pd.Series(dtype='datetime64[ns]'),
            'f1': pd.Series(dtype='float'),
            'f2': pd.Series(dtype='float'),
            'label': pd.Series(dtype='object'),
            'audiofilename': pd.Series(dtype='object')
        })
        self.detectiondf = pd.DataFrame([])
        
        # Create file blocks for large files
        fid_names = []
        fid_start = []
        fid_end = []
        max_element_length_sec = 60 * 10
        
        for fname in self.filenames:
            a = sf.info(fname)
            
            if a.duration < max_element_length_sec:
                fid_names.append(fname)
                fid_start.append(0)
                fid_end.append(a.frames)
            else:
                s = 0
                while s < a.frames:
                    fid_names.append(fname)
                    fid_start.append(s)
                    e = s + max_element_length_sec * a.samplerate
                    fid_end.append(e)
                    s = s + max_element_length_sec * a.samplerate
        
        self.file_blocks = pd.DataFrame({
            'fname': fid_names,
            'start': fid_start,
            'end': fid_end
        })
        
        print(self.file_blocks)
        self.plotwindow_startsecond = 0
        self.plot_next_spectro()
    
    def read_wav(self):
        """Read WAV file and compute spectrogram"""
        if self.filecounter < 0:
            return
        
        self.current_audiopath = self.file_blocks.loc[self.filecounter, 'fname']
        
        self.x, self.fs = sf.read(
            self.current_audiopath,
            start=self.file_blocks.loc[self.filecounter, 'start'],
            stop=self.file_blocks.loc[self.filecounter, 'end'],
            dtype='int16'
        )
        
        if self.filename_timekey.text() == '':
            self.time = dt.datetime(2000, 1, 1, 0, 0, 0)
        else:
            try:
                self.time = dt.datetime.strptime(
                    self.current_audiopath.split('/')[-1],
                    self.filename_timekey.text()
                )
            except:
                print('Wrong filename format')
                self.time = dt.datetime(2000, 1, 1, 0, 0, 0)
        
        if self.file_blocks.loc[self.filecounter, 'start'] > 0:
            secoffset = self.file_blocks.loc[self.filecounter, 'start'] / self.fs
            self.time = self.time + pd.Timedelta(seconds=secoffset)
        
        print(f'Open new file: {self.current_audiopath}')
        print(f'FS: {self.fs}, x: {np.shape(self.x)}')
        
        if len(self.x.shape) > 1:
            if np.shape(self.x)[1] > 1:
                self.x = self.x[:, 0]
        
        db_saturation = float(self.db_saturation.text())
        x = self.x / 32767
        p = np.power(10, (db_saturation / 20)) * x
        
        fft_size = int(self.fft_size.currentText())
        fft_overlap = float(self.fft_overlap.currentText())
        
        self.f, self.t, self.Sxx = signal.spectrogram(
            p, self.fs, window='hamming', nperseg=fft_size, noverlap=int(fft_size * fft_overlap))
        
        if self.file_blocks.loc[self.filecounter, 'start'] > 0:
            secoffset = self.file_blocks.loc[self.filecounter, 'start'] / self.fs
            self.t = self.t + secoffset
    
    def plot_annotation_box(self, annotation_row):
        """Plot annotation box on spectrogram"""
        x1 = annotation_row.iloc[0, 0]
        x2 = annotation_row.iloc[0, 1]
        
        xt = pd.Series([x1, x2])
        tt = xt - np.array(self.time).astype('datetime64[ns]')
        xt = tt.dt.seconds + tt.dt.microseconds / 10**6
        x1 = xt.iloc[0]
        x2 = xt.iloc[1]
        
        y1 = annotation_row.iloc[0, 2]
        y2 = annotation_row.iloc[0, 3]
        c_label = annotation_row.iloc[0, 4]
        
        line_x = [x2, x1, x1, x2, x2]
        line_y = [y1, y1, y2, y2, y1]
        
        xmin = np.min([x1, x2])
        ymax = np.max([y1, y2])
        
        self.canvas.axes.plot(line_x, line_y, '-b', linewidth=0.75)
        self.canvas.axes.text(xmin, ymax, c_label, size=8)
    
    def plot_spectrogram(self):
        """Plot spectrogram"""
        if self.filecounter < 0:
            return
        
        self.canvas.fig.clf()
        self.canvas.axes = self.canvas.fig.add_subplot(111)
        
        if self.t_length.text() == '':
            self.plotwindow_length = self.t[-1]
            self.plotwindow_startsecond = self.t[0]
        else:
            self.plotwindow_length = float(self.t_length.text())
            if self.t[-1] < self.plotwindow_length:
                self.plotwindow_startsecond = self.t[0]
                self.plotwindow_length = self.t[-1]
        
        y1 = int(self.f_min.text())
        y2 = int(self.f_max.text())
        if y2 > (self.fs / 2):
            y2 = (self.fs / 2)
        t1 = self.plotwindow_startsecond
        t2 = self.plotwindow_startsecond + self.plotwindow_length
        
        ix_time = np.where((self.t >= t1) & (self.t < t2))[0]
        ix_f = np.where((self.f >= y1) & (self.f < y2))[0]
        
        plotsxx = self.Sxx[int(ix_f[0]):int(ix_f[-1]), int(ix_time[0]):int(ix_time[-1])]
        plotsxx_db = 10 * np.log10(plotsxx)
        
        if self.checkbox_background.isChecked():
            spec_mean = np.median(plotsxx_db, axis=1)
            sxx_background = np.transpose(np.broadcast_to(spec_mean, np.transpose(plotsxx_db).shape))
            plotsxx_db = plotsxx_db - sxx_background
            plotsxx_db = plotsxx_db - np.min(plotsxx_db.flatten())
        
        colormap_plot = self.colormap_plot.currentText()
        img = self.canvas.axes.imshow(
            plotsxx_db, aspect='auto', cmap=colormap_plot, origin='lower', extent=[t1, t2, y1, y2])
        
        self.canvas.axes.set_ylabel('Frequency [Hz]')
        self.canvas.axes.set_xlabel('Time [sec]')
        
        if self.checkbox_logscale.isChecked():
            self.canvas.axes.set_yscale('log')
        else:
            self.canvas.axes.set_yscale('linear')
        
        if self.filename_timekey.text() == '':
            self.canvas.axes.set_title(self.current_audiopath.split('/')[-1])
        else:
            self.canvas.axes.set_title(self.time)
        
        clims = img.get_clim()
        if (self.db_vmin.text() == '') & (self.db_vmax.text() != ''):
            img.set_clim([clims[0], float(self.db_vmax.text())])
        if (self.db_vmin.text() != '') & (self.db_vmax.text() == ''):
            img.set_clim([float(self.db_vmin.text()), clims[1]])
        if (self.db_vmin.text() != '') & (self.db_vmax.text() != ''):
            img.set_clim([float(self.db_vmin.text()), float(self.db_vmax.text())])
        
        self.canvas.fig.colorbar(img, label=r'PSD [dB re $1 \mu Pa Hz^{-1}$]')
        
        # Plot annotations
        if self.annotation.shape[0] > 0:
            ix = (self.annotation['t1'] > (np.array(self.time).astype('datetime64[ns]') +
                                          pd.Timedelta(self.plotwindow_startsecond, unit="s"))) & \
                 (self.annotation['t1'] < (np.array(self.time).astype('datetime64[ns]') +
                                          pd.Timedelta(self.plotwindow_startsecond + self.plotwindow_length, unit="s"))) & \
                 (self.annotation['audiofilename'] == self.current_audiopath)
            
            if np.sum(ix) > 0:
                ix = np.where(ix)[0]
                for ix_x in ix:
                    a = pd.DataFrame([self.annotation.iloc[ix_x, :]])
                    self.plot_annotation_box(a)
        
        # Plot detections
        cmap = plt.get_cmap('cool')
        if self.detectiondf.shape[0] > 0:
            for i in range(self.detectiondf.shape[0]):
                insidewindow = (self.detectiondf.loc[i, 't-1'] > self.plotwindow_startsecond) & \
                              (self.detectiondf.loc[i, 't-2'] < (self.plotwindow_startsecond + self.plotwindow_length)) & \
                              (self.detectiondf.loc[i, 'audiofilename'] == self.current_audiopath)
                
                scoremin = self.detectiondf['score'].min()
                scoremax = self.detectiondf['score'].max()
                
                if (self.detectiondf.loc[i, 'score'] >= 0.01) & insidewindow:
                    xx1 = self.detectiondf.loc[i, 't-1']
                    xx2 = self.detectiondf.loc[i, 't-2']
                    yy1 = self.detectiondf.loc[i, 'f-1']
                    yy2 = self.detectiondf.loc[i, 'f-2']
                    scorelabel = str(np.round(self.detectiondf.loc[i, 'score'], 2))
                    snorm = (self.detectiondf.loc[i, 'score'] - scoremin) / (scoremax - scoremin)
                    scorecolor = cmap(snorm)
                    
                    line_x = [xx2, xx1, xx1, xx2, xx2]
                    line_y = [yy1, yy1, yy2, yy2, yy1]
                    
                    xmin = np.min([xx1, xx2])
                    ymax = np.max([yy1, yy2])
                    self.canvas.axes.plot(line_x, line_y, '-', color=scorecolor, linewidth=0.75)
                    self.canvas.axes.text(xmin, ymax, scorelabel, size=8, color=scorecolor)
        
        self.canvas.axes.set_ylim([y1, y2])
        self.canvas.axes.set_xlim([t1, t2])
        
        self.canvas.fig.tight_layout()
        self.toggle_selector = RectangleSelector(
            self.canvas.axes, self.box_select_callback,
            useblit=False, button=[1],
            interactive=False, 
            props=dict(facecolor="blue", edgecolor="black", alpha=0.1, fill=True))
        
        self.canvas.draw()
        self.cid1 = self.canvas.fig.canvas.mpl_connect('button_press_event', self.onclick)
    
    def plot_spectrogram_threshold(self):
        """Plot spectrogram with threshold regions"""
        if self.filecounter < 0:
            return
        
        db_threshold, ok = QtWidgets.QInputDialog.getInt(
            self, 'Input Dialog', 'Enter signal-to-noise threshold in dB:')
        
        if not ok:
            return
        
        self.find_regions(db_threshold)
        self.detectiondf = pd.DataFrame([])
        
        self.canvas.fig.clf()
        self.canvas.axes = self.canvas.fig.add_subplot(111)
        
        self.canvas.axes.set_ylabel('Frequency [Hz]')
        
        if self.checkbox_logscale.isChecked():
            self.canvas.axes.set_yscale('log')
        else:
            self.canvas.axes.set_yscale('linear')
        
        img = self.canvas.axes.imshow(
            self.region_labels > 0, aspect='auto', cmap='gist_yarg', origin='lower')
        
        self.canvas.fig.colorbar(img)
        self.canvas.fig.tight_layout()
        self.canvas.draw()
    
    def export_zoomed_sgram_as_csv(self):
        """Export zoomed spectrogram as CSV"""
        if self.filecounter < 0:
            return
        
        spectrog = 10 * np.log10(self.Sxx)
        
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText("Remove background?")
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        returnValue = msg.exec()
        
        if returnValue == QtWidgets.QMessageBox.Yes:
            rectime = pd.to_timedelta(self.t, 's')
            spg = pd.DataFrame(np.transpose(spectrog), index=rectime)
            bg = spg.resample('3min').mean().copy()
            bg = bg.resample('1s').interpolate(method='time')
            bg = bg.reindex(rectime, method='nearest')
            background = np.transpose(bg.values)
            z = spectrog - background
        else:
            z = spectrog
        
        self.f_limits = self.canvas.axes.get_ylim()
        self.t_limits = self.canvas.axes.get_xlim()
        y1 = int(self.f_limits[0])
        y2 = int(self.f_limits[1])
        t1 = self.t_limits[0]
        t2 = self.t_limits[1]
        
        ix_time = np.where((self.t >= t1) & (self.t < t2))[0]
        ix_f = np.where((self.f >= y1) & (self.f < y2))[0]
        
        plotsxx_db = z[int(ix_f[0]):int(ix_f[-1]), int(ix_time[0]):int(ix_time[-1])]
        
        sgram = pd.DataFrame(data=plotsxx_db, index=self.f[ix_f[:-1]], columns=self.t[ix_time[:-1]])
        print(sgram)
        
        savename = QtWidgets.QFileDialog.getSaveFileName(self, "", "csv files (*.csv)")
        if len(savename[0]) > 0:
            if savename[0][-4:] != '.csv':
                savename = savename[0] + '.csv'
            else:
                savename = savename[0]
            sgram.to_csv(savename)
    
    def box_select_callback(self, eclick, erelease):
        """Callback for box selection"""
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata
        
        x1 = self.time + pd.to_timedelta(x1, unit='s')
        x2 = self.time + pd.to_timedelta(x2, unit='s')
        
        t1 = np.min([x1, x2])
        t2 = np.max([x1, x2])
        f1 = np.min([y1, y2])
        f2 = np.max([y1, y2])
        
        if self.bg.checkedId() == -1:
            c_label = ''
        else:
            c_label = eval(f'self.an_{self.bg.checkedId()}.text()')
        
        a = pd.DataFrame({
            't1': pd.Series(t1, dtype='datetime64[ns]'),
            't2': pd.Series(t2, dtype='datetime64[ns]'),
            'f1': pd.Series(f1, dtype='float'),
            'f2': pd.Series(f2, dtype='float'),
            'label': pd.Series(c_label, dtype='object'),
            'audiofilename': self.current_audiopath
        })
        
        self.annotation = pd.concat([self.annotation, a], ignore_index=True)
        self.plot_annotation_box(a)
    
    def onclick(self, event):
        """Handle mouse clicks"""
        if event.button == 3:
            self.annotation = self.annotation.head(-1)
            self.plot_spectrogram()
    
    def end_of_filelist_warning(self):
        """Show end of file list warning"""
        msg_listend = QtWidgets.QMessageBox()
        msg_listend.setIcon(QtWidgets.QMessageBox.Information)
        msg_listend.setText("End of file list reached!")
        msg_listend.exec_()
    
    def plot_next_spectro(self):
        """Plot next spectrogram"""
        if len(self.filenames) == 0:
            return
        
        print(f'Old filecounter: {self.filecounter}')
        
        if self.t_length.text() == '' or ((self.filecounter >= 0) and (self.t[-1] < float(self.t_length.text()))):
            self.filecounter = self.filecounter + 1
            if self.filecounter > self.file_blocks.shape[0] - 1:
                self.filecounter = self.file_blocks.shape[0] - 1
                print('That was it')
                self.end_of_filelist_warning()
            self.plotwindow_length = self.t[-1]
            self.plotwindow_startsecond = self.t[0]
            self.read_wav()
            self.plot_spectrogram()
        else:
            self.plotwindow_length = float(self.t_length.text())
            self.plotwindow_startsecond = self.plotwindow_startsecond + self.plotwindow_length
        
        print([self.plotwindow_startsecond, self.t[0], self.t[-1]])
        
        if self.plotwindow_startsecond > self.t[-1]:
            # Save log
            if self.checkbox_log.isChecked():
                tt = self.annotation['t1'] - self.time
                t_in_seconds = np.array(tt.values * 1e-9, dtype='float16')
                reclength = np.array(self.t[-1], dtype='float16')
                
                ix = (t_in_seconds > 0) & (t_in_seconds < reclength)
                
                calldata = self.annotation.iloc[ix, :]
                print(calldata)
                savename = self.current_audiopath
                nn = savename[:-4] + f'_log_sec{int(self.t[0])}_to_sec{int(self.t[-1])}.csv'
                calldata.to_csv(nn)
                print(f'Writing log: {nn}')
            
            # New file
            self.filecounter = self.filecounter + 1
            if self.filecounter >= self.file_blocks.shape[0] - 1:
                self.filecounter = self.file_blocks.shape[0] - 1
                print('That was it')
                self.end_of_filelist_warning()
            self.read_wav()
            self.plotwindow_startsecond = self.t[0]
            self.plot_spectrogram()
        else:
            self.plot_spectrogram()
    
    def plot_previous_spectro(self):
        """Plot previous spectrogram"""
        if len(self.filenames) == 0:
            return
        
        print(f'Old filecounter: {self.filecounter}')
        
        if self.t_length.text() == '' or ((self.filecounter >= 0) and (self.t[-1] < float(self.t_length.text()))):
            self.filecounter = self.filecounter - 1
            if self.filecounter < 0:
                self.filecounter = 0
                print('That was it')
                self.end_of_filelist_warning()
            self.plotwindow_length = self.t[-1]
            self.plotwindow_startsecond = self.t[0]
            self.read_wav()
            self.plot_spectrogram()
        else:
            self.plotwindow_startsecond = self.plotwindow_startsecond - self.plotwindow_length
            print([self.plotwindow_startsecond, self.t[0], self.t[-1]])
            
            if self.plotwindow_startsecond < self.t[0]:
                self.filecounter = self.filecounter - 1
                
                if self.filecounter < 0:
                    self.filecounter = 0
                    print('That was it')
                    self.end_of_filelist_warning()
                
                self.read_wav()
                self.plot_spectrogram()
            else:
                self.plot_spectrogram()
    
    def new_fft_size_selected(self):
        """Handle FFT size change"""
        self.read_wav()
        self.plot_spectrogram()
    
    def func_savecsv(self):
        """Save annotations to CSV"""
        savename = QtWidgets.QFileDialog.getSaveFileName(
            self, "QFileDialog.getSaveFileName()", "", "csv files (*.csv)")
        print(f'Location: {savename[0]}')
        if len(savename[0]) > 0:
            self.annotation.to_csv(savename[0])
    
    def func_logging(self):
        """Handle logging checkbox"""
        if self.checkbox_log.isChecked():
            print('Logging enabled')
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Information)
            msg.setText("Overwrite existing log files?")
            msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
            returnValue = msg.exec()
            
            if returnValue == QtWidgets.QMessageBox.No:
                ix_delete = []
                for i, fn in enumerate(self.filenames):
                    logpath = fn[:-4] + '_log.csv'
                    if os.path.isfile(logpath):
                        ix_delete.append(i)
                
                self.filenames = np.delete(self.filenames, ix_delete)
                print('Updated filelist:')
                print(self.filenames)
    
    def plot_all_spectrograms(self):
        """Plot all spectrograms and save as images"""
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText(f"Are you sure you want to plot {self.filenames.shape[0]} spectrograms?")
        msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
        returnValue = msg.exec()
        
        if returnValue != QtWidgets.QMessageBox.Yes:
            return
        
        for audiopath in self.filenames:
            if self.filename_timekey.text() == '':
                self.time = dt.datetime(2000, 1, 1, 0, 0, 0)
            else:
                try:
                    self.time = dt.datetime.strptime(audiopath.split('/')[-1], self.filename_timekey.text())
                except:
                    self.time = dt.datetime(2000, 1, 1, 0, 0, 0)
            
            self.x, self.fs = sf.read(audiopath, dtype='int16')
            print(f'Processing file: {audiopath}')
            
            db_saturation = float(self.db_saturation.text())
            x = self.x / 32767
            p = np.power(10, (db_saturation / 20)) * x
            
            fft_size = int(self.fft_size.currentText())
            fft_overlap = float(self.fft_overlap.currentText())
            
            self.f, self.t, self.Sxx = signal.spectrogram(
                p, self.fs, window='hamming', nperseg=fft_size, noverlap=int(fft_size * fft_overlap))
            
            self.plotwindow_startsecond = 0
            self.plot_spectrogram()
            self.canvas.axes.set_title(audiopath.split('/')[-1])
            self.canvas.fig.savefig(audiopath[:-4] + '.jpg', dpi=150)
    
    def func_draw_shape_plot(self):
        """Plot spectrogram for shape drawing"""
        if self.filecounter < 0:
            return
        
        self.canvas.fig.clf()
        self.canvas.axes = self.canvas.fig.add_subplot(111)
        
        if self.t_length.text() == '':
            self.plotwindow_length = self.t[-1]
            self.plotwindow_startsecond = 0
        else:
            self.plotwindow_length = float(self.t_length.text())
            if self.t[-1] < self.plotwindow_length:
                self.plotwindow_startsecond = 0
                self.plotwindow_length = self.t[-1]
        
        y1 = int(self.f_min.text())
        y2 = int(self.f_max.text())
        if y2 > (self.fs / 2):
            y2 = (self.fs / 2)
        t1 = self.plotwindow_startsecond
        t2 = self.plotwindow_startsecond + self.plotwindow_length
        
        ix_time = np.where((self.t >= t1) & (self.t < t2))[0]
        ix_f = np.where((self.f >= y1) & (self.f < y2))[0]
        
        plotsxx = self.Sxx[int(ix_f[0]):int(ix_f[-1]), int(ix_time[0]):int(ix_time[-1])]
        plotsxx_db = 10 * np.log10(plotsxx)
        
        if self.checkbox_background.isChecked():
            spec_mean = np.median(plotsxx_db, axis=1)
            sxx_background = np.transpose(np.broadcast_to(spec_mean, np.transpose(plotsxx_db).shape))
            plotsxx_db = plotsxx_db - sxx_background
            plotsxx_db = plotsxx_db - np.min(plotsxx_db.flatten())
        
        colormap_plot = self.colormap_plot.currentText()
        img = self.canvas.axes.imshow(
            plotsxx_db, aspect='auto', cmap=colormap_plot, origin='lower', extent=[t1, t2, y1, y2])
        
        self.canvas.axes.set_ylabel('Frequency [Hz]')
        self.canvas.axes.set_xlabel('Time [sec]')
        
        if self.checkbox_logscale.isChecked():
            self.canvas.axes.set_yscale('log')
        else:
            self.canvas.axes.set_yscale('linear')
        
        clims = img.get_clim()
        if (self.db_vmin.text() == '') & (self.db_vmax.text() != ''):
            img.set_clim([clims[0], float(self.db_vmax.text())])
        if (self.db_vmin.text() != '') & (self.db_vmax.text() == ''):
            img.set_clim([float(self.db_vmin.text()), clims[1]])
        if (self.db_vmin.text() != '') & (self.db_vmax.text() != ''):
            img.set_clim([float(self.db_vmin.text()), float(self.db_vmax.text())])
        
        self.canvas.fig.colorbar(img, label=r'PSD [dB re $1 \mu Pa Hz^{-1}$]')
        
        # Plot annotations
        if self.annotation.shape[0] > 0:
            ix = (self.annotation['t1'] > (np.array(self.time).astype('datetime64[ns]') +
                                          pd.Timedelta(self.plotwindow_startsecond, unit="s"))) & \
                 (self.annotation['t1'] < (np.array(self.time).astype('datetime64[ns]') +
                                          pd.Timedelta(self.plotwindow_startsecond + self.plotwindow_length, unit="s")))
            
            if np.sum(ix) > 0:
                ix = np.where(ix)[0]
                for ix_x in ix:
                    a = pd.DataFrame([self.annotation.iloc[ix_x, :]])
                    self.plot_annotation_box(a)
        
        if hasattr(self, 't_limits') and self.t_limits is not None:
            self.canvas.axes.set_ylim(self.f_limits)
            self.canvas.axes.set_xlim(self.t_limits)
        else:
            self.canvas.axes.set_ylim([y1, y2])
            self.canvas.axes.set_xlim([t1, t2])
        
        self.canvas.fig.tight_layout()
        self.canvas.axes.plot(self.draw_x, self.draw_y, '.-g')
        self.canvas.draw()
        self.cid2 = self.canvas.fig.canvas.mpl_connect('button_press_event', self.onclick_draw)
    
    def onclick_draw(self, event):
        """Handle clicks in draw mode"""
        if event.button == 1 and event.dblclick:
            self.draw_x = pd.concat([self.draw_x, pd.Series([event.xdata])], ignore_index=True)
            self.draw_y = pd.concat([self.draw_y, pd.Series([event.ydata])], ignore_index=True)
            self.f_limits = self.canvas.axes.get_ylim()
            self.t_limits = self.canvas.axes.get_xlim()
            
            line = self.line_2.pop(0)
            line.remove()
            self.line_2 = self.canvas.axes.plot(self.draw_x, self.draw_y, '.-g')
            self.canvas.draw()
        
        if event.button == 3:
            self.draw_x = self.draw_x.head(-1)
            self.draw_y = self.draw_y.head(-1)
            self.f_limits = self.canvas.axes.get_ylim()
            self.t_limits = self.canvas.axes.get_xlim()
            
            line = self.line_2.pop(0)
            line.remove()
            self.line_2 = self.canvas.axes.plot(self.draw_x, self.draw_y, '.-g')
            self.canvas.draw()
    
    def func_draw_shape_exit(self):
        """Exit draw mode and save shape"""
        print(f'Save shape: {self.draw_x.shape}')
        self.canvas.fig.canvas.mpl_disconnect(self.cid2)
        self.plot_spectrogram()
        print('Back to boxes')
        self.drawexitm.setEnabled(False)
        
        if self.draw_x.shape[0] > 0:
            savename = QtWidgets.QFileDialog.getSaveFileName(
                self, "QFileDialog.getSaveFileName()", "csv files (*.csv)")
            
            if len(savename[0]) > 0:
                if savename[0][-4:] != '.csv':
                    savename = savename[0] + '.csv'
                else:
                    savename = savename[0]
                
                drawcsv = pd.DataFrame({'Time_in_s': self.draw_x, 'Frequency_in_Hz': self.draw_y})
                drawcsv.to_csv(savename)
    
    def func_draw_shape(self):
        """Enter draw mode"""
        msg = QtWidgets.QMessageBox()
        msg.setIcon(QtWidgets.QMessageBox.Information)
        msg.setText("Add points with double left click.\n"
                   "Remove latest point with single right click.\n"
                   "Exit draw mode and save CSV by pushing enter")
        msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
        returnValue = msg.exec()
        
        if returnValue != QtWidgets.QMessageBox.Ok:
            return
        
        print('Drawing mode enabled')
        self.draw_x = pd.Series(dtype='float')
        self.draw_y = pd.Series(dtype='float')
        self.f_limits = self.canvas.axes.get_ylim()
        self.t_limits = self.canvas.axes.get_xlim()
        self.canvas.fig.canvas.mpl_disconnect(self.cid1)
        self.cid2 = self.canvas.fig.canvas.mpl_connect('button_press_event', self.onclick_draw)
        self.line_2 = self.canvas.axes.plot(self.draw_x, self.draw_y, '.-g')
        self.func_draw_shape_plot()
        self.drawexitm = QtWidgets.QShortcut(QtCore.Qt.Key_Return, self)
        self.drawexitm.activated.connect(self.func_draw_shape_exit)
    
    def func_playaudio(self):
        """Play/stop audio"""
        if self.filecounter < 0:
            return
        
        # Stop if already playing
        if self.audio_playing:
            self.stop_audio()
            return
        
        # Get audio parameters
        new_rate = 32000
        t_limits = list(self.canvas.axes.get_xlim())
        t_limits = np.array(t_limits) - self.file_blocks.loc[self.filecounter, 'start'] / self.fs
        f_limits = list(self.canvas.axes.get_ylim())
        if f_limits[1] >= (self.fs / 2):
            f_limits[1] = self.fs / 2 - 10
        
        print(f'Time limits: {t_limits}')
        print(f'Frequency limits: {f_limits}')
        
        # Extract and filter audio
        x_select = self.x[int(t_limits[0] * self.fs):int(t_limits[1] * self.fs)]
        
        sos = signal.butter(8, f_limits, 'bandpass', fs=self.fs, output='sos')
        x_select = signal.sosfilt(sos, x_select)
        
        # Resample
        playback_speed = float(self.playbackspeed.currentText())
        number_of_samples = round(len(x_select) * (new_rate / playback_speed) / self.fs)
        x_resampled = np.array(signal.resample(x_select, number_of_samples)).astype('int')
        
        # Normalize
        maximum_x = 32767 * 0.8
        old_max = np.max(np.abs([x_resampled.min(), x_resampled.max()]))
        x_resampled = x_resampled * (maximum_x / old_max)
        x_resampled = x_resampled.astype(np.int16)
        
        print(f'Audio range: [{x_resampled.min()}, {x_resampled.max()}]')
        
        # Play using sounddevice
        self.audio_playing = True
        sd.play(x_resampled, new_rate)
    
    def func_saveaudio(self):
        """Export selected audio to WAV file"""
        if self.filecounter < 0:
            return
        
        savename = QtWidgets.QFileDialog.getSaveFileName(
            self, "QFileDialog.getSaveFileName()", "wav files (*.wav)")
        
        if len(savename[0]) == 0:
            return
        
        savename = savename[0]
        new_rate = 32000
        
        t_limits = self.canvas.axes.get_xlim()
        f_limits = list(self.canvas.axes.get_ylim())
        if f_limits[1] >= (self.fs / 2):
            f_limits[1] = self.fs / 2 - 10
        
        print(f'Time limits: {t_limits}')
        print(f'Frequency limits: {f_limits}')
        
        x_select = self.x[int(t_limits[0] * self.fs):int(t_limits[1] * self.fs)]
        
        sos = signal.butter(8, f_limits, 'bandpass', fs=self.fs, output='sos')
        x_select = signal.sosfilt(sos, x_select)
        
        playback_speed = float(self.playbackspeed.currentText())
        number_of_samples = round(len(x_select) * (new_rate / playback_speed) / self.fs)
        x_resampled = np.array(signal.resample(x_select, number_of_samples)).astype('int')
        
        # Normalize
        maximum_x = 32767 * 0.8
        old_max = np.max(np.abs([x_resampled.min(), x_resampled.max()]))
        x_resampled = x_resampled * (maximum_x / old_max)
        x_resampled = x_resampled.astype(np.int16)
        
        if savename[-4:] != '.wav':
            savename = savename + '.wav'
        
        wav.write(savename, new_rate, x_resampled)
        print(f'Saved audio to: {savename}')
    
    def func_save_video(self):
        """Export spectrogram as animated video"""
        if self.filecounter < 0:
            return
        
        # Check if moviepy is available
        if not MOVIEPY_AVAILABLE:
            msg = QtWidgets.QMessageBox()
            msg.setIcon(QtWidgets.QMessageBox.Warning)
            msg.setWindowTitle("Video Export Not Available")
            msg.setText("Video export requires ffmpeg to be installed.")
            msg.setInformativeText(
                "To enable video export:\n\n"
                "1. Install ffmpeg from https://ffmpeg.org/download.html\n"
                "2. Or install via conda: conda install -c conda-forge ffmpeg\n"
                "3. Restart this application\n\n"
                "Alternatively, use 'Export > Spectrogram as .wav file' to export audio only."
            )
            msg.exec_()
            return
        
        savename = QtWidgets.QFileDialog.getSaveFileName(
            self, "QFileDialog.getSaveFileName()", "video files (*.mp4)")
        
        if len(savename[0]) == 0:
            return
        
        savename = savename[0]
        new_rate = 32000
        
        t_limits = self.canvas.axes.get_xlim()
        f_limits = list(self.canvas.axes.get_ylim())
        if f_limits[1] >= (self.fs / 2):
            f_limits[1] = self.fs / 2 - 10
        
        print(f'Time limits: {t_limits}')
        print(f'Frequency limits: {f_limits}')
        
        x_select = self.x[int(t_limits[0] * self.fs):int(t_limits[1] * self.fs)]
        
        sos = signal.butter(8, f_limits, 'bandpass', fs=self.fs, output='sos')
        x_select = signal.sosfilt(sos, x_select)
        
        playback_speed = float(self.playbackspeed.currentText())
        number_of_samples = round(len(x_select) * (new_rate / playback_speed) / self.fs)
        x_resampled = np.array(signal.resample(x_select, number_of_samples)).astype('int')
        
        # Normalize
        maximum_x = 32767 * 0.8
        old_max = np.max(np.abs([x_resampled.min(), x_resampled.max()]))
        x_resampled = x_resampled * (maximum_x / old_max)
        x_resampled = x_resampled.astype(np.int16)
        
        if savename[-4:] == '.wav':
            savename = savename[:-4]
        if savename[-4:] == '.mp4':
            savename = savename[:-4]
        
        wav.write(savename + '.wav', new_rate, x_resampled)
        
        audioclip = AudioFileClip(savename + '.wav')
        duration = audioclip.duration
        
        self.canvas.axes.set_title(None)
        self.line_2 = self.canvas.axes.plot([t_limits[0], t_limits[0]], f_limits, '-k')
        
        def make_frame(x):
            s = t_limits[1] - t_limits[0]
            xx = x / duration * s + t_limits[0]
            line = self.line_2.pop(0)
            line.remove()
            self.line_2 = self.canvas.axes.plot([xx, xx], f_limits, '-k')
            return mplfig_to_npimage(self.canvas.fig)
        
        animation = VideoClip(make_frame, duration=duration)
        animation = animation.set_audio(audioclip)
        animation.write_videofile(savename + ".mp4", fps=24, preset='fast')
        
        self.plot_spectrogram()


def start():
    """Start the application"""
    app = QtWidgets.QApplication(sys.argv)
    app.setApplicationName("Python Audio Spectrogram Explorer")
    w = gui()
    sys.exit(app.exec_())


if __name__ == "__main__":
    start()
