# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 17:28:37 2021

@author: Sebastian Menze, sebastian.menze@gmail.com
"""


import sys
import matplotlib
# matplotlib.use('Qt5Agg')

from PyQt5 import QtCore, QtGui, QtWidgets

# from PyQt5.QtWidgets import QShortcut
# from PyQt5.QtGui import QKeySequence

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

import scipy.io.wavfile as wav
from scipy import signal
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import datetime as dt
import time
import os

# from pydub import AudioSegment
# from pydub.playback import play
# import threading

import simpleaudio as sa


class MplCanvas(FigureCanvasQTAgg ):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        super(MplCanvas, self).__init__(self.fig)



class MainWindow(QtWidgets.QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)

        self.canvas =  MplCanvas(self, width=5, height=4, dpi=200)
                
        # self.call_time=pd.Series()
        # self.call_frec=pd.Series()
        
        self.f_min = QtWidgets.QLineEdit(self)
        self.f_min.setText('10')
        self.f_max = QtWidgets.QLineEdit(self)
        self.f_max.setText('16000')
        self.t_length = QtWidgets.QLineEdit(self)
        self.t_length.setText('120')
        self.db_saturation=QtWidgets.QLineEdit(self)
        self.db_saturation.setText('155')
        
        # self.fft_size = QtWidgets.QLineEdit(self)
        # self.fft_size.setText('32768')
        self.fft_size = QtWidgets.QComboBox(self)
        self.fft_size.addItem('1024')        
        self.fft_size.addItem('2048')        
        self.fft_size.addItem('4096')        
        self.fft_size.addItem('8192')        
        self.fft_size.addItem('16384')        
        self.fft_size.addItem('32768')        
        self.fft_size.addItem('65536')    
        self.fft_size.addItem('131072')    
        self.fft_size.setCurrentIndex(4)
        
        self.fft_overlap = QtWidgets.QLineEdit(self)
        self.fft_overlap.setText('0.8')
 
        self.filename_timekey = QtWidgets.QLineEdit(self)
        self.filename_timekey.setText('aural_%Y_%m_%d_%H_%M_%S.wav')       
 
        self.playbackspeed = QtWidgets.QComboBox(self)
        self.playbackspeed.addItem('0.5')        
        self.playbackspeed.addItem('1')        
        self.playbackspeed.addItem('2')        
        self.playbackspeed.addItem('5')        
        self.playbackspeed.addItem('10')        
        self.playbackspeed.setCurrentIndex(1)
        
        
        self.time=None
        self.f=None
        self.t=[-1,-1]
        self.Sxx=None
             
        self.plotwindow_startsecond=0
        # self.plotwindow_length=120
        
        openfilebutton=QtWidgets.QPushButton('Open .wav files')
        def openfilefunc():
            self.filecounter=-1
            self.call_time=pd.Series(dtype='datetime64[ns]')
            self.call_frec=pd.Series()            
            self.call_label=pd.Series()            

            options = QtWidgets.QFileDialog.Options()
            # options |= QtWidgets.QFileDialog.DontUseNativeDialog
            self.filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", r"C:\Users\a5278\Documents\passive_acoustics\detector_delevopment\detector_validation_subset","Wav Files (*.wav)", options=options)
            self.filenames = np.array( self.filenames )
            print(self.filenames)
        openfilebutton.clicked.connect(openfilefunc)
        
        
        def read_wav():
                 
            audiopath=self.filenames[self.filecounter]
            
            # if self.filename_timekey.text()=='':
            #     self.time= dt.datetime(1,1,1,0,0,0)
            # else:     
            #     self.time= dt.datetime.strptime( audiopath.split('/')[-1], self.filename_timekey.text() )
           
            if self.filename_timekey.text()=='':
                # self.time= dt.datetime(1,1,1,0,0,0)
                self.time= dt.datetime.now()
            else:     
                self.time= dt.datetime.strptime( audiopath.split('/')[-1], self.filename_timekey.text() )

            self.fs, self.x = wav.read(audiopath)
            print('open new file: '+audiopath)
            
            # factor=60
            # x=signal.decimate(x,factor,ftype='fir')
            
            db_saturation=float( self.db_saturation.text() )
            x=self.x/32767 
            p =np.power(10,(db_saturation/20))*x #convert data.signal to uPa    
            
            fft_size=int( self.fft_size.currentText() )
            fft_overlap=float(  self.fft_overlap.text() )
            self.f, self.t, self.Sxx = signal.spectrogram(p, self.fs, window='hamming',nperseg=fft_size,noverlap=fft_size*fft_overlap)
            # self.t=self.time +  pd.to_timedelta( t  , unit='s')           
            
        def plot_spectrogram():
        
            # self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
            # self.setCentralWidget(self.canvas)
            self.canvas.fig.clf() 
            self.canvas.axes = self.canvas.fig.add_subplot(111)
            # self.canvas.axes.cla()
            
            if self.t_length.text()=='':
                self.plotwindow_length= self.t[-1] 
                self.plotwindow_startsecond=0
            else:    
                self.plotwindow_length=float( self.t_length.text() )
                
           
            y1=int(self.f_min.text())    
            y2=int(self.f_max.text())    
            t1=self.plotwindow_startsecond
            t2=self.plotwindow_startsecond+self.plotwindow_length
            # tt,ff=np.meshgrid(self.t,self.f)
            # ix_time=(tt>=self.plotwindow_startsecond) & (tt<(self.plotwindow_startsecond+self.plotwindow_length))
            # ix_f=(ff>=y1) & (ff<y2)
            # plotsxx=self.Sxx[ ix_f & ix_time]
            ix_time=np.where( (self.t>=self.plotwindow_startsecond) & (self.t<(self.plotwindow_startsecond+self.plotwindow_length)) )[0]
            ix_f=np.where((self.f>=y1) & (self.f<y2))[0]
            # print(ix_time.shape)
            # print(ix_f.shape)
            plotsxx= self.Sxx[ int(ix_f[0]):int(ix_f[-1]),int(ix_time[0]):int(ix_time[-1]) ] 
            

            # print(plotsxx.shape)
  
            # img=self.canvas.axes.pcolormesh(self.t, self.f, 10*np.log10(self.Sxx) ,cmap='plasma')
            img=self.canvas.axes.imshow( 10*np.log10(plotsxx) , aspect='auto',cmap='plasma',origin = 'lower',extent = [t1, t2, y1, y2])
          
            # img=self.canvas.axes.pcolormesh(self.t[ int(ix_time[0]):int(ix_time[-1])], self.f[int(ix_f[0]):int(ix_f[-1])], 10*np.log10(plotsxx) , shading='flat',cmap='plasma')

            
            self.canvas.axes.set_ylabel('Frequency [Hz]')
            self.canvas.axes.set_xlabel('Time [sec]')
            if self.checkbox_logscale.isChecked():
                self.canvas.axes.set_yscale('log')
            else:
                self.canvas.axes.set_yscale('linear')        
                
 

            
            if self.filename_timekey.text()=='':
                audiopath=self.filenames[self.filecounter]
                self.canvas.axes.set_title(audiopath.split('/')[-1])
            else:     
                self.canvas.axes.set_title(self.time)

            # img.set_clim([ 40 ,10*np.log10( np.max(np.array(plotsxx).ravel() )) ] )
            clims=img.get_clim()
            img.set_clim([ 30 ,clims[1]] )
                        
            self.canvas.fig.colorbar(img,label='PSD [dB re $1 \ \mu Pa \ Hz^{-1}$]')
            
            # print(self.time)        
            # print(self.call_time)

       # plot annotations
            if self.call_time.shape[0]>0:
                # ix=self.call_time > (np.array(self.time).astype('datetime64[ns]')+pd.Timedelta(self.plotwindow_startsecond, unit="s") )  
                ix=(self.call_time > (np.array(self.time).astype('datetime64[ns]')+pd.Timedelta(self.plotwindow_startsecond, unit="s") )  ) & (self.call_time < (np.array(self.time).astype('datetime64[ns]')+pd.Timedelta(self.plotwindow_startsecond+self.plotwindow_length, unit="s") )  )
                
                if np.sum(ix)>0:
                    tt= self.call_time - np.array(self.time).astype('datetime64[ns]')  
                    # print(tt)                           
                    # self.canvas.axes.plot(tt[ix]*1e-9, self.call_frec[ix],'xb')
                    self.canvas.axes.plot(tt[ix].dt.seconds, self.call_frec[ix],'xb')
                    
                    x=tt[ix].dt.seconds.values
                    y=self.call_frec[ix].values
                    txt=self.call_label[ix].values
                    print(x)
                    print(txt)

    
                    for i in range(x.shape[0]):
                        self.canvas.axes.text(x[i],y[i],txt[i],size=5)
              
    # plt.annotate(txt, (fpr_mat[i], tpr_mat[i]))
   
                # self.canvas.axes.text(tt[ix]*1e-9, self.call_frec[ix],self.call_label[ix])
              
            self.canvas.axes.set_ylim([y1,y2])
            self.canvas.axes.set_xlim([self.plotwindow_startsecond,self.plotwindow_startsecond+self.plotwindow_length])
                      
                
            self.canvas.fig.tight_layout()
            self.canvas.draw()

        def onclick(event):
            # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
            #       ('double' if event.dblclick else 'single', event.button,
            #        event.x, event.y, event.xdata, event.ydata))
            if event.dblclick:
                # print(self.bg.checkedId())

                clicktime=self.time +  pd.to_timedelta( event.xdata  , unit='s')                
                self.call_time=self.call_time.append( pd.Series(clicktime) ,ignore_index=True )
                self.call_frec=self.call_frec.append( pd.Series(event.ydata),ignore_index=True )    
                if self.bg.checkedId()==-1:
                    c_label=''
                else:
                    c_label=eval( 'self.an_'+str(self.bg.checkedId())+'.text()' )
                # print(c_label)
                self.call_label=self.call_label.append( pd.Series(c_label),ignore_index=True )    
                # print(self.call_label)
                # print( self.call_time) 
                # self.call_time=self.call_time.astype('datetime64[ns]')
                plot_spectrogram()              
              
            if event.button==3:
                self.call_time=self.call_time.head(-1)
                self.call_frec=self.call_frec.head(-1)
                self.call_label=self.call_label.head(-1)

                plot_spectrogram()              

        def plot_next_spectro():
            
            if self.t_length.text()=='':
                self.plotwindow_length= self.t[-1] 
                self.plotwindow_startsecond=0
                # new file    
                self.filecounter=self.filecounter+1
                read_wav()
                plot_spectrogram()
                    
            else:    
                self.plotwindow_length=float( self.t_length.text() )       
                self.plotwindow_startsecond=self.plotwindow_startsecond + self.plotwindow_length
                
            print( [self.plotwindow_startsecond,  self.t[-1] ] )
            
            if self.plotwindow_startsecond > self.t[-1]:              
                #save log
                if checkbox_log.isChecked():
                    
                    tt= self.call_time - self.time             
                    t_in_seconds=np.array( tt.values*1e-9 ,dtype='float16')
                    reclength=np.array( self.t[-1] ,dtype='float16')
    
                    ix=(t_in_seconds>0) & (t_in_seconds<reclength)
    
                    calldata=pd.concat([ self.call_time[ix] , self.call_frec[ix],self.call_label[ix]], axis=1)
                    calldata.columns=['Timestamp','Frequency','Label']
                    print(calldata)
                    savename=self.filenames[self.filecounter]
                    calldata.to_csv(savename[:-4]+'_log.csv')                  
                    print('writing log: '+savename[:-4]+'_log.csv')
                # new file    
                self.filecounter=self.filecounter+1
                
                if self.filecounter>self.filenames.shape[0]:
                    print('That was it')
                    self.canvas.fig.clf() 
                    self.canvas.axes = self.canvas.fig.add_subplot(111)
                    self.canvas.axes.set_title('That was it')
                    self.canvas.draw()
                else:      
                    read_wav()
                    self.plotwindow_startsecond=0
                    plot_spectrogram()
            else:
                plot_spectrogram()
         

                    
        def plot_previous_spectro():
            
            if self.t_length.text()=='':
                self.plotwindow_startsecond = -100
            else:                
                self.plotwindow_startsecond=self.plotwindow_startsecond -self.plotwindow_length
                print( [self.plotwindow_startsecond,  self.t[-1] ] )
            
            if self.plotwindow_startsecond < 0: 
                self.plotwindow_startsecond = self.t[-1] + self.plotwindow_startsecond                
                # old file    
                self.filecounter=self.filecounter-1
                
                if self.filecounter<0:
                    print('That was it')
                    self.canvas.fig.clf() 
                    self.canvas.axes = self.canvas.fig.add_subplot(111)
                    self.canvas.axes.set_title('That was it')
                    self.canvas.draw()
                else:      
                    read_wav()
                    # self.plotwindow_startsecond=0
                    plot_spectrogram()
            else:
                plot_spectrogram()
         
                        
        self.canvas.fig.canvas.mpl_connect('button_press_event', onclick)
        
        # QtGui.QShortcut(QtCore.Qt.Key_Right, MainWindow, plot_next_spectro())
        # self.msgSc = QShortcut(QKeySequence(u"\u2192"), self)
        # self.msgSc.activated.connect(plot_next_spectro)
        button_plot_spectro=QtWidgets.QPushButton('Next spectrogram-->')
        button_plot_spectro.clicked.connect(plot_next_spectro)
        
        button_plot_prevspectro=QtWidgets.QPushButton('<--Previous spectrogram')
        button_plot_prevspectro.clicked.connect(plot_previous_spectro)
    
        button_save=QtWidgets.QPushButton('Save annotation csv')
        def func_savecsv():         
            options = QtWidgets.QFileDialog.Options()
            savename = QtWidgets.QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()", r"C:\Users\a5278\Documents\passive_acoustics\detector_delevopment\detector_validation_subset", "csv files (*.csv)",options=options)
            calldata=pd.concat([ self.call_time , self.call_frec,self.call_label], axis=1)
            calldata.columns=['Timestamp','Frequency','Label']
            print(calldata)
            calldata.to_csv(savename[0])         
        button_save.clicked.connect(func_savecsv)
        
        button_quit=QtWidgets.QPushButton('Quit')
        button_quit.clicked.connect(QtWidgets.QApplication.instance().quit)
        
        
        
        def func_logging():    
            if checkbox_log.isChecked():
                print('logging')
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Information)   
                msg.setText("Overwrite existing log files?")
                msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                returnValue = msg.exec()
                if returnValue == QtWidgets.QMessageBox.No:
                                 
                    ix_delete=[]
                    i=0
                    for fn in self.filenames:
                        logpath=fn[:-4]+'_log.csv'
                        # print(logpath)
                        if os.path.isfile( logpath):
                            ix_delete.append(i)    
                        i=i+1
                    # print(ix_delete)
        
                    self.filenames=np.delete(self.filenames,ix_delete)
                    print('Updated filelist:')
                    print(self.filenames)
                


        checkbox_log=QtWidgets.QCheckBox('Real-time Logging')
        checkbox_log.toggled.connect(func_logging)
        
        button_plot_all_spectrograms=QtWidgets.QPushButton('Plot all spectrograms')
        def plot_all_spectrograms():         
            
           msg = QtWidgets.QMessageBox()
           msg.setIcon(QtWidgets.QMessageBox.Information)   
           msg.setText("Are you sure you want to plot "+ str(self.filenames.shape[0]) +" spectrograms?")
           msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
           returnValue = msg.exec()

           if returnValue == QtWidgets.QMessageBox.Yes:
                for audiopath in self.filenames:
                                    
                    if self.filename_timekey.text()=='':
                        self.time= dt.datetime(1,1,1,0,0,0)
                    else:     
                        self.time= dt.datetime.strptime( audiopath.split('/')[-1], self.filename_timekey.text() )
        
                    self.fs, self.x = wav.read(audiopath)
                    print('open new file: '+audiopath)
                    
                    db_saturation=float( self.db_saturation.text() )
                    x=self.x/32767 
                    p =np.power(10,(db_saturation/20))*x #convert data.signal to uPa    
                    
                    fft_size=int( self.fft_size.currentText() )
                    fft_overlap=float(  self.fft_overlap.text() )
                    
                    
                    self.f, self.t, self.Sxx = signal.spectrogram(p, self.fs, window='hamming',nperseg=fft_size,noverlap=fft_size*fft_overlap)
                    
                    self.plotwindow_startsecond=0
                    
                    plot_spectrogram()
                    self.canvas.axes.set_title(audiopath.split('/')[-1])
                    self.canvas.fig.savefig( audiopath[:-4]+'.jpg',dpi=150 )    
            
            
        button_plot_all_spectrograms.clicked.connect(plot_all_spectrograms)        
       
        ####### play audio
        button_play_audio=QtWidgets.QPushButton('Play/Stop [spacebar]')
        def func_playaudio():
            if not hasattr(self, "play_obj"):
                new_rate = 32000          

                t_limits=self.canvas.axes.get_xlim()
                print(t_limits)
                x_select=self.x[int(t_limits[0]*self.fs) : int(t_limits[1]*self.fs) ]          
                number_of_samples = round(len(x_select) * (float(new_rate)/ float(self.playbackspeed.currentText())) / self.fs)
                x_resampled = np.array(signal.resample(x_select, number_of_samples)).astype('int')            
                wave_obj = sa.WaveObject(x_resampled, 1, 2, new_rate)
                self.play_obj = wave_obj.play()   
            else:    
                if self.play_obj.is_playing():
                    sa.stop_all()
                else:    
                    new_rate = 32000          
                    t_limits=self.canvas.axes.get_xlim()
                    print(t_limits)
                    x_select=self.x[int(t_limits[0]*self.fs) : int(t_limits[1]*self.fs) ]          
                    # number_of_samples = round(len(x_select) * float(new_rate) / self.fs)
                    number_of_samples = round(len(x_select) * (float(new_rate)/ float(self.playbackspeed.currentText())) / self.fs)

                    x_resampled = np.array(signal.resample(x_select, number_of_samples)).astype('int')            
                    wave_obj = sa.WaveObject(x_resampled, 1, 2, new_rate)
                    self.play_obj = wave_obj.play()
            
        button_play_audio.clicked.connect(func_playaudio)        
                 
       
        ######## layout
        outer_layout = QtWidgets.QVBoxLayout()
        
        top_layout = QtWidgets.QHBoxLayout()       
        
        top_layout.addWidget(openfilebutton)
        top_layout.addWidget(checkbox_log)
        top_layout.addWidget(button_plot_prevspectro)
        top_layout.addWidget(button_plot_spectro)
        top_layout.addWidget(button_plot_all_spectrograms)
        top_layout.addWidget(button_play_audio)
        top_layout.addWidget(QtWidgets.QLabel('Playback speed:'))        
        top_layout.addWidget(self.playbackspeed)


        top_layout.addWidget(button_save)            
        top_layout.addWidget(button_quit)

        top2_layout = QtWidgets.QHBoxLayout()   
        
        # self.f_min = QtWidgets.QLineEdit(self)
        # self.f_max = QtWidgets.QLineEdit(self)
        # self.t_start = QtWidgets.QLineEdit(self)
        # self.t_end = QtWidgets.QLineEdit(self)
        # self.fft_size = QtWidgets.QLineEdit(self)
        top2_layout.addWidget(QtWidgets.QLabel('filename key:'))        
        top2_layout.addWidget(self.filename_timekey)
        top2_layout.addWidget(QtWidgets.QLabel('f_min[Hz]:'))
        top2_layout.addWidget(self.f_min)     
        top2_layout.addWidget(QtWidgets.QLabel('f_max[Hz]:'))
        top2_layout.addWidget(self.f_max)
        top2_layout.addWidget(QtWidgets.QLabel('Spectrogram length [sec]:'))
        top2_layout.addWidget(self.t_length)
 
        top2_layout.addWidget(QtWidgets.QLabel('fft_size[bits]:'))
        top2_layout.addWidget(self.fft_size) 
        top2_layout.addWidget(QtWidgets.QLabel('fft_overlap[0-1]:'))
        top2_layout.addWidget(self.fft_overlap) 
        
        self.checkbox_logscale=QtWidgets.QCheckBox('log scale')
        self.checkbox_logscale.setChecked(True)
        top2_layout.addWidget(self.checkbox_logscale)
        top2_layout.addWidget(QtWidgets.QLabel('Saturation [dB re 1 muPa]:'))
        top2_layout.addWidget(self.db_saturation)
        
        # annotation label area
        top3_layout = QtWidgets.QHBoxLayout()   
        top3_layout.addWidget(QtWidgets.QLabel('Annotation labels:'))


        self.checkbox_an_1=QtWidgets.QCheckBox()
        top3_layout.addWidget(self.checkbox_an_1)
        self.an_1 = QtWidgets.QLineEdit(self)
        top3_layout.addWidget(self.an_1) 
        self.an_1.setText('FW_20_Hz')
        
        self.checkbox_an_2=QtWidgets.QCheckBox()
        top3_layout.addWidget(self.checkbox_an_2)
        self.an_2 = QtWidgets.QLineEdit(self)
        top3_layout.addWidget(self.an_2) 
        self.an_2.setText('FW_40_Hz')
        
        self.checkbox_an_3=QtWidgets.QCheckBox()
        top3_layout.addWidget(self.checkbox_an_3)
        self.an_3 = QtWidgets.QLineEdit(self)
        top3_layout.addWidget(self.an_3) 
        self.an_3.setText('BW_D_call')
        
        self.checkbox_an_4=QtWidgets.QCheckBox()
        top3_layout.addWidget(self.checkbox_an_4)
        self.an_4 = QtWidgets.QLineEdit(self)
        top3_layout.addWidget(self.an_4) 
        self.an_4.setText('BW_Z_call')
        
        self.checkbox_an_5=QtWidgets.QCheckBox()
        top3_layout.addWidget(self.checkbox_an_5)
        self.an_5 = QtWidgets.QLineEdit(self)
        top3_layout.addWidget(self.an_5) 
        self.an_5.setText('US_100-200')

        self.checkbox_an_6=QtWidgets.QCheckBox()
        top3_layout.addWidget(self.checkbox_an_6)
        self.an_6 = QtWidgets.QLineEdit(self)
        top3_layout.addWidget(self.an_6) 
        self.an_6.setText('')
        
        self.checkbox_an_7=QtWidgets.QCheckBox()
        top3_layout.addWidget(self.checkbox_an_7)
        self.an_7 = QtWidgets.QLineEdit(self)
        top3_layout.addWidget(self.an_7) 
        self.an_7.setText('')

        self.checkbox_an_8=QtWidgets.QCheckBox()
        top3_layout.addWidget(self.checkbox_an_8)
        self.an_8 = QtWidgets.QLineEdit(self)
        top3_layout.addWidget(self.an_8) 
        self.an_8.setText('')
           
        self.bg = QtWidgets.QButtonGroup()
        self.bg.addButton(self.checkbox_an_1,1)
        self.bg.addButton(self.checkbox_an_2,2)
        self.bg.addButton(self.checkbox_an_3,3)
        self.bg.addButton(self.checkbox_an_4,4)
        self.bg.addButton(self.checkbox_an_5,5)
        self.bg.addButton(self.checkbox_an_6,6)
        self.bg.addButton(self.checkbox_an_7,7)
        self.bg.addButton(self.checkbox_an_8,8)
        


        
        # combine layouts together
        
        plot_layout = QtWidgets.QVBoxLayout()
        toolbar = NavigationToolbar( self.canvas, self)
        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(self.canvas)
        
        outer_layout.addLayout(top_layout)
        outer_layout.addLayout(top2_layout)
        outer_layout.addLayout(top3_layout)

        outer_layout.addLayout(plot_layout)
        
        # self.setLayout(outer_layout)
        
        # Create a placeholder widget to hold our toolbar and canvas.
        widget = QtWidgets.QWidget()
        widget.setLayout(outer_layout)
        self.setCentralWidget(widget)
        
        #### hotkeys
        self.msgSc1 = QtWidgets.QShortcut(QtCore.Qt.Key_Right, self)
        self.msgSc1.activated.connect(plot_next_spectro)
        self.msgSc2 = QtWidgets.QShortcut(QtCore.Qt.Key_Left, self)
        self.msgSc2.activated.connect(plot_previous_spectro)        
        self.msgSc3 = QtWidgets.QShortcut(QtCore.Qt.Key_Space, self)
        self.msgSc3.activated.connect(func_playaudio)                
        ####
        # layout = QtWidgets.QVBoxLayout()
        # layout.addWidget(openfilebutton)
        # layout.addWidget(button_plot_spectro)
        # layout.addWidget(button_save)            
        # layout.addWidget(button_quit)


        # layout.addWidget(toolbar)
        # layout.addWidget(self.canvas)

        # # Create a placeholder widget to hold our toolbar and canvas.
        # widget = QtWidgets.QWidget()
        # widget.setLayout(layout)
        # self.setCentralWidget(widget)

        self.show()


app = QtWidgets.QApplication(sys.argv)
app.setApplicationName("Python Audio Spectrogram Explorer")

w = MainWindow()
sys.exit(app.exec_())

#%%
