# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 17:28:37 2021

@author: Sebastian Menze, sebastian.menze@gmail.com
"""

def start():
    
    import sys
    import matplotlib
    # matplotlib.use('Qt5Agg')

    from PyQt5 import QtCore, QtGui, QtWidgets

    # from PyQt5.QtWidgets import QShortcut
    # from PyQt5.QtGui import QKeySequence

    from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT as NavigationToolbar
    from matplotlib.figure import Figure

    import scipy.io.wavfile as wav

    import soundfile as sf


    from scipy import signal
    import numpy as np
    from matplotlib import pyplot as plt
    import pandas as pd
    import datetime as dt
    import time
    import os

    from matplotlib.widgets import RectangleSelector


    # from pydub import AudioSegment
    # from pydub.playback import play
    # import threading

    import simpleaudio as sa

    from skimage import data, filters, measure, morphology
    from skimage.morphology import (erosion, dilation, opening, closing,  # noqa
                                    white_tophat)
    from skimage.morphology import disk  # noqa
    from matplotlib.path import Path
    from skimage.transform import rescale, resize, downscale_local_mean

    from scipy.signal import find_peaks
    from skimage.feature import match_template
     
    from moviepy.editor import VideoClip, AudioFileClip
    from moviepy.video.io.bindings import mplfig_to_npimage
    



    class MplCanvas(FigureCanvasQTAgg ):

        def __init__(self, parent=None, width=5, height=4, dpi=100):
            self.fig = Figure(figsize=(width, height), dpi=dpi)
            self.axes = self.fig.add_subplot(111)
            super(MplCanvas, self).__init__(self.fig)



    class MainWindow(QtWidgets.QMainWindow):
        
            
        def __init__(self, *args, **kwargs):
            super(MainWindow, self).__init__(*args, **kwargs)

            self.canvas =  MplCanvas(self, width=5, height=4, dpi=150)
                    
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
            self.db_vmin=QtWidgets.QLineEdit(self)
            self.db_vmin.setText('30')
            self.db_vmax=QtWidgets.QLineEdit(self)
            self.db_vmax.setText('')       
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
            
            
            self.colormap_plot = QtWidgets.QComboBox(self)
            self.colormap_plot.addItem('plasma')        
            self.colormap_plot.addItem('viridis')        
            self.colormap_plot.addItem('inferno')        
            self.colormap_plot.addItem('gist_gray')           
            self.colormap_plot.addItem('gist_yarg')           
            self.colormap_plot.setCurrentIndex(2)
            
            self.checkbox_logscale=QtWidgets.QCheckBox('Log. scale')
            self.checkbox_logscale.setChecked(True)
            self.checkbox_background=QtWidgets.QCheckBox('Remove background')
            self.checkbox_background.setChecked(False)
            
            # self.fft_overlap = QtWidgets.QLineEdit(self)
            # self.fft_overlap.setText('0.9')
            
            self.fft_overlap = QtWidgets.QComboBox(self)
            self.fft_overlap.addItem('0.2')        
            self.fft_overlap.addItem('0.5')        
            self.fft_overlap.addItem('0.7')        
            self.fft_overlap.addItem('0.9')  
            self.fft_overlap.setCurrentIndex(3)
         
            
            
            self.filename_timekey = QtWidgets.QLineEdit(self)
            # self.filename_timekey.setText('aural_%Y_%m_%d_%H_%M_%S.wav')       
     
            self.playbackspeed = QtWidgets.QComboBox(self)
            self.playbackspeed.addItem('0.5')        
            self.playbackspeed.addItem('1')        
            self.playbackspeed.addItem('2')        
            self.playbackspeed.addItem('5')        
            self.playbackspeed.addItem('10')        
            self.playbackspeed.setCurrentIndex(1)
            
            
            self.time= dt.datetime(2000,1,1,0,0,0)
            self.f=None
            self.t=[-1,-1]
            self.Sxx=None
            self.draw_x=pd.Series(dtype='float')
            self.draw_y=pd.Series(dtype='float')
            self.cid1=None
            self.cid2=None
            
            self.plotwindow_startsecond= float( self.t_length.text() )
            # self.plotwindow_length=120
            self.filecounter=-1
            self.filenames=np.array( [] )
            self.current_audiopath=None

            self.detectiondf=pd.DataFrame([])

            
            def find_regions(db_threshold):

                y1=int(self.f_min.text())    
                y2=int(self.f_max.text())    
                if y2>(self.fs/2):
                    y2=(self.fs/2)
                t1=self.plotwindow_startsecond
                t2=self.plotwindow_startsecond+self.plotwindow_length
             
                ix_time=np.where( (self.t>=t1) & (self.t<t2 ))[0]
                ix_f=np.where((self.f>=y1) & (self.f<y2))[0]
                
                # f_lim=[self.f_min,self.f_max]
                t=self.t[ix_time]
                
                # db_threshold=10
                minimum_patcharea=2*5
                
                plotsxx= self.Sxx[ int(ix_f[0]):int(ix_f[-1]),int(ix_time[0]):int(ix_time[-1]) ] 
                spectrog=10*np.log10(plotsxx)
                
                # filter out background
                spec_mean=np.median(spectrog,axis=1) 
                sxx_background=np.transpose(np.broadcast_to(spec_mean,np.transpose(spectrog).shape))
                z = spectrog - sxx_background
                
                # z=spectrog - np.min(spectrog.flatten())
                
                # rectime= pd.to_timedelta( t ,'s')
                # spg=pd.DataFrame(np.transpose(spectrog),index=rectime)
                # bg=spg.resample('3min').mean().copy()
                # bg=bg.resample('1s').interpolate(method='time')
                # bg=    bg.reindex(rectime,method='nearest')
                
                # background=np.transpose(bg.values)   
                # z=spectrog-background
                              
                # Binary image, post-process the binary mask and compute labels
                mask = z > db_threshold
                mask = morphology.remove_small_objects(mask, 50,connectivity=30)
                mask = morphology.remove_small_holes(mask, 50,connectivity=30)
                
                mask = closing(mask,  disk(3) )
                # op_and_clo = opening(closed,  disk(1) )
                
                labels = measure.label(mask)
                  
                probs=measure.regionprops_table(labels,spectrog,properties=['label','area','mean_intensity','orientation','major_axis_length','minor_axis_length','weighted_centroid','bbox'])
                df=pd.DataFrame(probs)
                
                # get corect f anf t
                ff=self.f[ ix_f[0]:ix_f[-1] ]
                ix=df['bbox-0']>len(ff)-1
                df.loc[ix,'bbox-0']=len(ff)-1
                ix=df['bbox-2']>len(ff)-1
                df.loc[ix,'bbox-2']=len(ff)-1
                
                df['f-1']=ff[df['bbox-0']] 
                df['f-2']=ff[df['bbox-2']] 
                df['f-width']=df['f-2']-df['f-1']
                
                ix=df['bbox-1']>len(t)-1
                df.loc[ix,'bbox-1']=len(t)-1
                ix=df['bbox-3']>len(t)-1
                df.loc[ix,'bbox-3']=len(t)-1
                
                df['t-1']=t[df['bbox-1']] 
                df['t-2']=t[df['bbox-3']] 
                df['duration']=df['t-2']-df['t-1']               
                
                indices=np.where( (df['area']<minimum_patcharea) | (df['bbox-3']-df['bbox-1']<3) | (df['bbox-2']-df['bbox-0']<3)  )[0]
                df=df.drop(indices)
                df=df.reset_index()    
                
                df['id']=  np.arange(len(df))
                
                # df['filename']=audiopath 
                
                # get region dict
                # sgram={}
                patches={}
                p_t_dict={}
                p_f_dict={}
                
                for ix in range(len(df)):
                    m=labels== df.loc[ix,'label']
                    ix1=df.loc[ix,'bbox-1']
                    ix2=df.loc[ix,'bbox-3']
                    jx1=df.loc[ix,'bbox-0']
                    jx2=df.loc[ix,'bbox-2'] 
                
                    patch=m[jx1:jx2,ix1:ix2]
                    pt=t[ix1:ix2]
                    pt=pt-pt[0]     
                    pf=ff[jx1:jx2]
                    
                    # contour = measure.find_contours(m, 0.5)[0]
                    # y, x = contour.T
                               
                    patches[ df['id'][ix]  ] = patch
                    p_t_dict[ df['id'][ix]  ] = pt
                    p_f_dict[ df['id'][ix]  ] = pf
                       
                    # ix1=ix1-10
                    # if ix1<=0: ix1=0
                    # ix2=ix2+10
                    # if ix2>=spectrog.shape[1]: ix2=spectrog.shape[1]-1       
                    # sgram[ df['id'][ix]  ] = spectrog[:,ix1:ix2]
                self.detectiondf = df    
                self.patches = patches    
                self.p_t_dict = p_t_dict    
                self.p_f_dict = p_f_dict    
                
                self.region_labels=labels

                # return df, patches,p_t_dict,p_f_dict         

            def match_bbox_and_iou(template):
            
                shape_f=template['Frequency_in_Hz'].values
                shape_t=template['Time_in_s'].values
                shape_t=shape_t-shape_t.min()
                
                df=self.detectiondf
                patches=self.patches
                p_t_dict=self.p_t_dict
                p_f_dict=self.p_f_dict
                
                # f_lim=[ shape_f.min()-10 ,shape_f.max()+10  ]
            
                # score_smc=[]
                score_ioubox=[] 
                smc_rs=[]
            
                for ix in df.index:   
                    
                    # breakpoint()
                    patch=patches[ix]
                    pf=p_f_dict[ix]
                    pt=p_t_dict[ix]
                    pt=pt-pt[0]
                    
                    
                    if df.loc[ix,'f-1'] < shape_f.min():
                        f1= df.loc[ix,'f-1'] 
                    else:
                        f1= shape_f.min()
                    if df.loc[ix,'f-2'] > shape_f.max():
                        f2= df.loc[ix,'f-2'] 
                    else:
                        f2= shape_f.max()      
                        
                    # f_lim=[ f1,f2  ]
                            
                    time_step=np.diff(pt)[0]
                    f_step=np.diff(pf)[0]
                    k_f=np.arange(f1,f2,f_step )
                    
                    if pt.max()>shape_t.max():
                        k_t=pt
            
                    else:
                        k_t=np.arange(0,shape_t.max(),time_step)
            
                 
                    ### iou bounding box
                    
                    iou_kernel=np.zeros( [ k_f.shape[0] ,k_t.shape[0] ] ) 
                    ixp2=np.where((k_t>=shape_t.min())  & (k_t<=shape_t.max()))[0]     
                    ixp1=np.where((k_f>=shape_f.min())  & (k_f<=shape_f.max()))[0]     
                    iou_kernel[ ixp1[0]:ixp1[-1] , ixp2[0]:ixp2[-1] ]=1
            
                    iou_patch=np.zeros( [ k_f.shape[0] ,k_t.shape[0] ] ) 
                    ixp2=np.where((k_t>=pt[0])  & (k_t<=pt[-1]))[0]     
                    ixp1=np.where((k_f>=pf[0])  & (k_f<=pf[-1]))[0]  
                    iou_patch[ ixp1[0]:ixp1[-1] , ixp2[0]:ixp2[-1] ]=1
                   
                    intersection=  iou_kernel.astype('bool') & iou_patch.astype('bool')
                    union=  iou_kernel.astype('bool') | iou_patch.astype('bool')
                    iou_bbox =  np.sum( intersection ) /  np.sum( union )
                    score_ioubox.append(iou_bbox)
                        
                    patch_rs = resize(patch, (50,50))
                    n_resize=50       
                    k_t=np.linspace(0,shape_t.max(),n_resize )
                    k_f=np.linspace(shape_f.min(), shape_f.max(),n_resize )   
                    kk_t,kk_f=np.meshgrid(k_t,k_f)   
                    # kernel=np.zeros( [ k_f.shape[0] ,k_t.shape[0] ] )
                    x, y = kk_t.flatten(), kk_f.flatten()
                    points = np.vstack((x,y)).T 
                    p = Path(list(zip(shape_t, shape_f))) # make a polygon
                    grid = p.contains_points(points)
                    kernel_rs = grid.reshape(kk_t.shape) # now you have a mask with points inside a polygon  
                    smc_rs.append(  np.sum( kernel_rs.astype('bool') == patch_rs.astype('bool') ) /  len( patch_rs.flatten() ) )
            
                smc_rs=np.array(smc_rs)
                score_ioubox=np.array(score_ioubox)
            
                df['score'] =score_ioubox * (smc_rs-.5)/.5
                
                self.detectiondf = df.copy()    
                
            def automatic_detector_specgram_corr():
                # open template
                self.detectiondf=pd.DataFrame([])
                
                templatefiles, ok1 = QtWidgets.QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", r"C:\Users","CSV file (*.csv)")
                if ok1:

                    templates=[]
                    for fnam in templatefiles:
                       template=pd.read_csv(fnam,index_col=0)
                       templates.append( template )
                    
                    corrscore_threshold, ok = QtWidgets.QInputDialog.getDouble(self, 'Input Dialog',
                                                    'Enter correlation threshold in (0-1):',decimals=2)
                    if corrscore_threshold>1:
                        corrscore_threshold=1
                    if corrscore_threshold<0:
                        corrscore_threshold=0
                        
                    if templates[0].columns[0]=='Time_in_s': 
                         
                            # print(template)
                        offset_f=10
                        offset_t=0.5    
                        
                        # shape_f=template['Frequency_in_Hz'].values
                        # shape_t=template['Time_in_s'].values
                        # shape_t=shape_t-shape_t.min()
                        shape_f=np.array([])
                        shape_t_raw=np.array([])
                        for template in templates:            
                            shape_f=np.concatenate( [shape_f, template['Frequency_in_Hz'].values ] )
                            shape_t_raw=np.concatenate( [shape_t_raw, template['Time_in_s'].values ])
                        shape_t=shape_t_raw-shape_t_raw.min()
                        
                        
                        f_lim=[ shape_f.min() - offset_f ,  shape_f.max() + offset_f ]
                        k_length_seconds=shape_t.max()+offset_t*2
    
                        # generate kernel  
                        time_step=np.diff(self.t)[0]
                        
                        k_t=np.linspace(0,k_length_seconds,int(k_length_seconds/time_step) )
                        ix_f=np.where((self.f>=f_lim[0]) & (self.f<=f_lim[1]))[0]
                        k_f=self.f[ix_f[0]:ix_f[-1]]
                        # k_f=np.linspace(f_lim[0],f_lim[1], int( (f_lim[1]-f_lim[0]) /f_step)  )
                        
                        kk_t,kk_f=np.meshgrid(k_t,k_f)   
                        kernel_background_db=0
                        kernel_signal_db=1
                        kernel=np.ones( [ k_f.shape[0] ,k_t.shape[0] ] ) * kernel_background_db
                        # find wich grid points are inside the shape
                        x, y = kk_t.flatten(), kk_f.flatten()
                        points = np.vstack((x,y)).T 
                        # p = Path(list(zip(shape_t, shape_f))) # make a polygon
                        # grid = p.contains_points(points)
                        # mask = grid.reshape(kk_t.shape) # now you have a mask with points inside a polygon  
                        # kernel[mask]=kernel_signal_db
                        for template in templates:       
                            shf= template['Frequency_in_Hz'].values 
                            st= template['Time_in_s'].values 
                            st=st-shape_t_raw.min()
                            p = Path(list(zip(st, shf))) # make a polygon
                            grid = p.contains_points(points)
                            kern = grid.reshape(kk_t.shape) # now you have a mask with points inside a polygon  
                            kernel[kern>0]=kernel_signal_db   
                    # print(kernel_rs)
                    
                        
                        ix_f=np.where((self.f>=f_lim[0]) & (self.f<=f_lim[1]))[0]
                        spectrog =10*np.log10( self.Sxx[ ix_f[0]:ix_f[-1],: ] )
                    
                        result = match_template(spectrog, kernel)
                        corr_score=result[0,:]
                        t_score=np.linspace( self.t[int(kernel.shape[1]/2)] , self.t[-int(kernel.shape[1]/2)], corr_score.shape[0] )
    
                        peaks_indices = find_peaks(corr_score, height=corrscore_threshold)[0]
                     
                        
                        t1=[]
                        t2=[]
                        f1=[]
                        f2=[]
                        score=[]
                        
                        if len(peaks_indices)>0: 
                            t2_old=0
                            for ixpeak in peaks_indices:     
                                tstar=t_score[ixpeak] - k_length_seconds/2 - offset_t
                                tend=t_score[ixpeak] + k_length_seconds/2 - offset_t
                                # if tstar>t2_old:
                                t1.append(tstar)
                                t2.append(tend)
                                f1.append(f_lim[0]+offset_f)
                                f2.append(f_lim[1]-offset_f)
                                score.append(corr_score[ixpeak])
                                # t2_old=tend
                            df=pd.DataFrame()
                            df['t-1']=t1
                            df['t-2']=t2
                            df['f-1']=f1
                            df['f-2']=f2
                            df['score']=score
                            
                            self.detectiondf = df.copy()  
                            self.detectiondf['audiofilename']= self.current_audiopath 
                            self.detectiondf['threshold']= corrscore_threshold 

                            plot_spectrogram()
                    else:  # image kernel
   
                        template=templates[0]
                        
                        k_length_seconds= float(template.columns[-1]) -float(template.columns[0])
                        
                        f_lim=[ int(template.index[0]) , int( template.index[-1] )]
                        ix_f=np.where((self.f>=f_lim[0]) & (self.f<=f_lim[1]))[0]
                        spectrog =10*np.log10( self.Sxx[ ix_f[0]:ix_f[-1],: ] )
                        specgram_t_step= self.t[1] - self.t[0]
                        n_f=spectrog.shape[0]
                        n_t= int(k_length_seconds/ specgram_t_step)
                        
                        kernel= resize( template.values , [ n_f,n_t] )
                        

                        result = match_template(spectrog, kernel)
                        corr_score=result[0,:]
                        t_score=np.linspace( self.t[int(kernel.shape[1]/2)] , self.t[-int(kernel.shape[1]/2)], corr_score.shape[0] )
    
                        peaks_indices = find_peaks(corr_score, height=corrscore_threshold)[0]
                        
                        # print(corr_score)
                        
                        t1=[]
                        t2=[]
                        f1=[]
                        f2=[]
                        score=[]
                        
                        if len(peaks_indices)>0: 
                            t2_old=0
                            for ixpeak in peaks_indices:     
                                tstar=t_score[ixpeak] - k_length_seconds/2 
                                tend=t_score[ixpeak] + k_length_seconds/2 
                                # if tstar>t2_old:
                                t1.append(tstar)
                                t2.append(tend)
                                f1.append(f_lim[0])
                                f2.append(f_lim[1])
                                score.append(corr_score[ixpeak])
                                t2_old=tend
                            df=pd.DataFrame()
                            df['t-1']=t1
                            df['t-2']=t2
                            df['f-1']=f1
                            df['f-2']=f2
                            df['score']=score
                            
                            self.detectiondf = df.copy()   
                            self.detectiondf['audiofilename']= self.current_audiopath 
                            self.detectiondf['threshold']= corrscore_threshold 

                            plot_spectrogram()
                    
            def automatic_detector_specgram_corr_allfiles():
               msg = QtWidgets.QMessageBox()
               msg.setIcon(QtWidgets.QMessageBox.Information)   
               msg.setText("Are you sure you want to run the detector over "+ str(self.file_blocks.shape[0]) +" ?")
               msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
               returnValue = msg.exec()

               if returnValue == QtWidgets.QMessageBox.Yes:
                   
                templatefiles, ok1 = QtWidgets.QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", r"C:\Users","CSV file (*.csv)")
                if ok1:
                    templates=[]
                    for fnam in templatefiles:
                       template=pd.read_csv(fnam,index_col=0)
                       templates.append( template )
                                        
                    corrscore_threshold, ok = QtWidgets.QInputDialog.getDouble(self, 'Input Dialog',
                                                    'Enter correlation threshold in (0-1):',decimals=2)
                    if corrscore_threshold>1:
                        corrscore_threshold=1
                    if corrscore_threshold<0:
                        corrscore_threshold=0
                        
                    self.detectiondf_all=pd.DataFrame([])
                
                    for i_block in range(len(self.file_blocks)):
                        
                        audiopath=self.file_blocks.loc[i_block,'fname']
                        
                        
                        if self.filename_timekey.text()=='':
                            self.time= dt.datetime(2000,1,1,0,0,0)
                        else:  
                            try:
                                self.time= dt.datetime.strptime( audiopath.split('/')[-1], self.filename_timekey.text() )
                            except:
                                self.time= dt.datetime(2000,1,1,0,0,0)
                                
                        if self.file_blocks.loc[i_block,'start']>0:           
                             secoffset = self.file_blocks.loc[self.filecounter,'start'] / self.fs
                             self.time=self.time + pd.Timedelta(seconds=secoffset)
                                 
                                
                        self.x,self.fs  =  sf.read(audiopath,dtype='int16', start=self.file_blocks.loc[i_block,'start'] , stop=self.file_blocks.loc[i_block,'end'])
                        print('open new file: '+audiopath)
                        
                        db_saturation=float( self.db_saturation.text() )
                        x=self.x/32767 
                        p =np.power(10,(db_saturation/20))*x #convert data.signal to uPa    
                        
                        fft_size=int( self.fft_size.currentText() )
                        fft_overlap=float(  self.fft_overlap.currentText() )
                        
                        self.f, self.t, self.Sxx = signal.spectrogram(p, self.fs, window='hamming',nperseg=fft_size,noverlap=fft_size*fft_overlap)
                        if self.file_blocks.loc[i_block,'start']>0:     
                            secoffset = self.file_blocks.loc[i_block,'start'] / self.fs
                            self.t=  self.t + secoffset     
                                                   
                        # self.plotwindow_startsecond=0
                        # self.plotwindow_length = self.t.max()
                
                        if templates[0].columns[0]=='Time_in_s': 
                            
                            # print(template)
                            offset_f=10
                            offset_t=0.5                  
                             # shape_f=template['Frequency_in_Hz'].values
                            # shape_t=template['Time_in_s'].values
                            # shape_t=shape_t-shape_t.min()
                            shape_f=np.array([])
                            shape_t_raw=np.array([])
                            for template in templates:            
                                shape_f=np.concatenate( [shape_f, template['Frequency_in_Hz'].values ] )
                                shape_t_raw=np.concatenate( [shape_t_raw, template['Time_in_s'].values ])
                            shape_t=shape_t_raw-shape_t_raw.min()
                            
                            f_lim=[ shape_f.min() - offset_f ,  shape_f.max() + offset_f ]
                            k_length_seconds=shape_t.max()+offset_t*2
        
                            # generate kernel  
                            time_step=np.diff(self.t)[0]
                            
                            k_t=np.linspace(0,k_length_seconds,int(k_length_seconds/time_step) )
                            ix_f=np.where((self.f>=f_lim[0]) & (self.f<=f_lim[1]))[0]
                            k_f=self.f[ix_f[0]:ix_f[-1]]
                            # k_f=np.linspace(f_lim[0],f_lim[1], int( (f_lim[1]-f_lim[0]) /f_step)  )
                            
                            kk_t,kk_f=np.meshgrid(k_t,k_f)   
                            kernel_background_db=0
                            kernel_signal_db=1
                            kernel=np.ones( [ k_f.shape[0] ,k_t.shape[0] ] ) * kernel_background_db
                            # find wich grid points are inside the shape
                            x, y = kk_t.flatten(), kk_f.flatten()
                            points = np.vstack((x,y)).T 
                            # p = Path(list(zip(shape_t, shape_f))) # make a polygon
                            # grid = p.contains_points(points)
                            # mask = grid.reshape(kk_t.shape) # now you have a mask with points inside a polygon  
                            # kernel[mask]=kernel_signal_db
                            for template in templates:       
                                shf= template['Frequency_in_Hz'].values 
                                st= template['Time_in_s'].values 
                                st=st-shape_t_raw.min()
                                p = Path(list(zip(st, shf))) # make a polygon
                                grid = p.contains_points(points)
                                kern = grid.reshape(kk_t.shape) # now you have a mask with points inside a polygon  
                                kernel[kern>0]=kernel_signal_db   
                            
                            ix_f=np.where((self.f>=f_lim[0]) & (self.f<=f_lim[1]))[0]
                            spectrog =10*np.log10( self.Sxx[ ix_f[0]:ix_f[-1],: ] )
                        
                            result = match_template(spectrog, kernel)
                            corr_score=result[0,:]
                            t_score=np.linspace( self.t[int(kernel.shape[1]/2)] , self.t[-int(kernel.shape[1]/2)], corr_score.shape[0] )
        
                            peaks_indices = find_peaks(corr_score, height=corrscore_threshold)[0]
                         
                            
                            t1=[]
                            t2=[]
                            f1=[]
                            f2=[]
                            score=[]
                            
                            if len(peaks_indices)>0: 
                                t2_old=0
                                for ixpeak in peaks_indices:     
                                    tstar=t_score[ixpeak] - k_length_seconds/2 - offset_t
                                    tend=t_score[ixpeak] + k_length_seconds/2 - offset_t
                                    # if tstar>t2_old:
                                    t1.append(tstar)
                                    t2.append(tend)
                                    f1.append(f_lim[0]+offset_f)
                                    f2.append(f_lim[1]-offset_f)
                                    score.append(corr_score[ixpeak])
                                    t2_old=tend
                                df=pd.DataFrame()
                                df['t-1']=t1
                                df['t-2']=t2
                                df['f-1']=f1
                                df['f-2']=f2
                                df['score']=score
                                
                                self.detectiondf = df.copy()  
                                self.detectiondf['audiofilename']= audiopath 
                                self.detectiondf['threshold']= corrscore_threshold 
                        else:  # image kernel                        
                            template=templates[0]
                            
                            k_length_seconds= float(template.columns[-1]) -float(template.columns[0])
                            
                            f_lim=[ int(template.index[0]) , int( template.index[-1] )]
                            ix_f=np.where((self.f>=f_lim[0]) & (self.f<=f_lim[1]))[0]
                            spectrog =10*np.log10( self.Sxx[ ix_f[0]:ix_f[-1],: ] )
                            specgram_t_step= self.t[1] - self.t[0]
                            n_f=spectrog.shape[0]
                            n_t= int(k_length_seconds/ specgram_t_step)
                            
                            kernel= resize( template.values , [ n_f,n_t] )
                            
    
                            result = match_template(spectrog, kernel)
                            corr_score=result[0,:]
                            t_score=np.linspace( self.t[int(kernel.shape[1]/2)] , self.t[-int(kernel.shape[1]/2)], corr_score.shape[0] )
        
                            peaks_indices = find_peaks(corr_score, height=corrscore_threshold)[0]
                            
                            # print(corr_score)
                            
                            t1=[]
                            t2=[]
                            f1=[]
                            f2=[]
                            score=[]
                            
                            if len(peaks_indices)>0: 
                                t2_old=0
                                for ixpeak in peaks_indices:     
                                    tstar=t_score[ixpeak] - k_length_seconds/2 
                                    tend=t_score[ixpeak] + k_length_seconds/2 
                                    # if tstar>t2_old:
                                    t1.append(tstar)
                                    t2.append(tend)
                                    f1.append(f_lim[0])
                                    f2.append(f_lim[1])
                                    score.append(corr_score[ixpeak])
                                    t2_old=tend
                                df=pd.DataFrame()
                                df['t-1']=t1
                                df['t-2']=t2
                                df['f-1']=f1
                                df['f-2']=f2
                                df['score']=score
                                
                                self.detectiondf = df.copy()   
                                self.detectiondf['audiofilename']= self.current_audiopath 
                                self.detectiondf['threshold']= corrscore_threshold 
                    
                        self.detectiondf_all=pd.concat([ self.detectiondf_all,self.detectiondf ])
                        self.detectiondf_all=self.detectiondf_all.reset_index(drop=True)

                        print(self.detectiondf_all)
                        
                        
                    self.detectiondf= self.detectiondf_all   
                    # self.detectiondf=self.detectiondf.reset_index(drop=True)
                    read_wav()
                    plot_spectrogram()
                    print('done!!!')

                    
            def automatic_detector_shapematching():
                # open template
                self.detectiondf=pd.DataFrame([])
                
                templatefile, ok1 = QtWidgets.QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileNames()", r"C:\Users","CSV file (*.csv)")
                if ok1:               
                    template=pd.read_csv(templatefile,index_col=0)

                    # print(template)
                    # set db threshold   
                    db_threshold, ok = QtWidgets.QInputDialog.getInt(self, 'Input Dialog',
                                                    'Enter signal-to-noise threshold in dB:')
                    if ok:
                        print(db_threshold)
                        self.detectiondf=pd.DataFrame([])

                        find_regions(db_threshold)            
                        match_bbox_and_iou(template)    
                        ixdel=np.where(self.detectiondf['score']<0.01)[0]
                        self.detectiondf=self.detectiondf.drop(ixdel)
                        self.detectiondf=self.detectiondf.reset_index(drop=True)
                        self.detectiondf['audiofilename']= self.current_audiopath 
                        self.detectiondf['threshold']= db_threshold 

                        print(self.detectiondf)
            
                        # plot results
                        plot_spectrogram()
                        
            def automatic_detector_shapematching_allfiles():
               msg = QtWidgets.QMessageBox()
               msg.setIcon(QtWidgets.QMessageBox.Information)   
               msg.setText("Are you sure you want to run the detector over "+ str(self.filenames.shape[0]) +" ?")
               msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
               returnValue = msg.exec()

               if returnValue == QtWidgets.QMessageBox.Yes:
                    templatefile, ok1 = QtWidgets.QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileNames()", r"C:\Users","CSV file (*.csv)")
                    template=pd.read_csv(templatefile)
                    db_threshold, ok = QtWidgets.QInputDialog.getInt(self, 'Input Dialog',
                                                    'Enter signal-to-noise threshold in dB:')                   
                   
                    self.detectiondf_all=pd.DataFrame([])
                   
                    for i_block in range(len(self.file_blocks)):
                        
                        audiopath=self.file_blocks.loc[i_block,'fname']
                        
                                                                
                        if self.filename_timekey.text()=='':
                            self.time= dt.datetime(2000,1,1,0,0,0)
                        else:  
                            try:
                                self.time= dt.datetime.strptime( audiopath.split('/')[-1], self.filename_timekey.text() )
                            except:
                                self.time= dt.datetime(2000,1,1,0,0,0)
                                
                        self.x,self.fs  =  sf.read(audiopath,dtype='int16', start=self.file_blocks.loc[i_block,'start'] , stop=self.file_blocks.loc[i_block,'end'])
                        print('open new file: '+audiopath)
                        
                        db_saturation=float( self.db_saturation.text() )
                        x=self.x/32767 
                        p =np.power(10,(db_saturation/20))*x #convert data.signal to uPa    
                        
                        fft_size=int( self.fft_size.currentText() )
                        fft_overlap=float(  self.fft_overlap.currentText() )
                        
                        self.f, self.t, self.Sxx = signal.spectrogram(p, self.fs, window='hamming',nperseg=fft_size,noverlap=fft_size*fft_overlap)
                        if self.file_blocks.loc[i_block,'start']>0:     
                            secoffset = self.file_blocks.loc[i_block,'start'] / self.fs
                            self.t=  self.t + secoffset     
                            
                        self.plotwindow_startsecond=0
                        self.plotwindow_length = self.t.max()
                
                        self.detectiondf=pd.DataFrame([])

                        find_regions(db_threshold)            
                        match_bbox_and_iou(template)    
                        ixdel=np.where(self.detectiondf['score']<0.01)[0]
                        self.detectiondf=self.detectiondf.drop(ixdel)
                        self.detectiondf=self.detectiondf.reset_index(drop=True)
                        self.detectiondf['audiofilename']= audiopath
                        self.detectiondf['threshold']= db_threshold 

                        self.detectiondf_all=pd.concat([ self.detectiondf_all,self.detectiondf ])
                        self.detectiondf_all=self.detectiondf_all.reset_index(drop=True)

                        print(self.detectiondf_all)
                        
                        
                    self.detectiondf= self.detectiondf_all   
                    # self.detectiondf=self.detectiondf.reset_index(drop=True)
                    print('done!!!')
                 
            def export_automatic_detector_shapematching():
                if self.detectiondf.shape[0]>0:
                    savename = QtWidgets.QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()", r"C:\Users", "csv files (*.csv)")
                    print('location is:' + savename[0])
                    if len(savename[0])>0:
                        self.detectiondf.to_csv(savename[0])                

            openfilebutton=QtWidgets.QPushButton('Open files')
            def openfilefunc():
                self.filecounter=-1   
                self.plotwindow_startsecond= float( self.t_length.text() )

                self.annotation= pd.DataFrame({'t1': pd.Series(dtype='datetime64[ns]'),
                       't2': pd.Series(dtype='datetime64[ns]'),
                       'f1': pd.Series(dtype='float'),
                       'f2': pd.Series(dtype='float'),
                       'label': pd.Series(dtype='object'),
                       'audiofilename': pd.Series(dtype='object')})
                self.detectiondf=pd.DataFrame([])
        
                # self.annotation=pd.DataFrame(columns=['t1','t2','f1','f2','label'],dtype=[("t1", "datetime64[ns]"), ("t2", "datetime64[ns]"), ("f1", "float"), ("f2", "float"), ("label", "object")] )
                
                # annotation=pd.DataFrame(dtype=[("t1", "datetime64[ns]"), ("t2", "datetime64[ns]"), ("f1", "float"), ("f2", "float"), ("label", "object")] )

                # ,dtype=('datetime64[ns]','datetime64[ns]','float','float','object'))
                # self.call_t_1=pd.Series(dtype='datetime64[ns]')
                # self.call_f_1=pd.Series(dtype='float')            
                # self.call_t_2=pd.Series(dtype='datetime64[ns]')
                # self.call_f_2=pd.Series(dtype='float') 
                # self.call_label=pd.Series(dtype='object')            

                options = QtWidgets.QFileDialog.Options()
                # options |= QtWidgets.QFileDialog.DontUseNativeDialog
                self.filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(self,"QFileDialog.getOpenFileNames()", r"C:\Users\a5278\Documents\passive_acoustics\detector_delevopment\detector_validation_subset","Audio Files (*.wav *.aif *.aiff *.aifc *.ogg *.flac)", options=options)
                self.filenames = np.array( self.filenames )      
                
                if len(self.filenames)>0:                  
                    fid_names=[]
                    fid_start=[]
                    fid_end=[]
            
                    max_element_length_sec=60*10
    
                    for fname in self.filenames:
                        a = sf.info( fname )
                        
                        if a.duration < max_element_length_sec:
                            fid_names.append( fname )
                            fid_start.append( 0 )
                            fid_end.append(a.frames )
                        else: 
    
                            s=0
                            while s < a.frames:
                                fid_names.append( fname )
                                fid_start.append( s )
                                e =  s + max_element_length_sec * a.samplerate
                                fid_end.append( e )
                                s=s+max_element_length_sec * a.samplerate
                                
                    self.file_blocks = pd.DataFrame([])               
                    self.file_blocks['fname']=fid_names
                    self.file_blocks['start']=fid_start
                    self.file_blocks['end']=fid_end
            
                    print( self.file_blocks)
                    self.plotwindow_startsecond=0
    
                    plot_next_spectro()
                    
            openfilebutton.clicked.connect(openfilefunc)
            
            
            def read_wav():
              if self.filecounter>=0:        
                self.current_audiopath=self.file_blocks.loc[self.filecounter,'fname']
                
                # if self.filename_timekey.text()=='':
                #     self.time= dt.datetime(1,1,1,0,0,0)
                # else:     
                #     self.time= dt.datetime.strptime( audiopath.split('/')[-1], self.filename_timekey.text() )
               
             
                
                # if audiopath[-4:]=='.wav':
                    
                    
                self.x,self.fs  =  sf.read(self.current_audiopath, start=self.file_blocks.loc[self.filecounter,'start'] , stop=self.file_blocks.loc[self.filecounter,'end'], dtype='int16')

                if self.filename_timekey.text()=='':
                    self.time= dt.datetime(2000,1,1,0,0,0)
                    #self.time= dt.datetime.now()
                else:     
                    try:
                        self.time= dt.datetime.strptime( self.current_audiopath.split('/')[-1], self.filename_timekey.text() )
                    except: 
                        print('wrongfilename')
                 
                if self.file_blocks.loc[self.filecounter,'start']>0:           
                    secoffset = self.file_blocks.loc[self.filecounter,'start'] / self.fs
                    self.time=self.time + pd.Timedelta(seconds=secoffset)
                        
                # if audiopath[-4:]=='.aif' | audiopath[-4:]=='.aiff' | audiopath[-4:]=='.aifc':
                #     obj = aifc.open(audiopath,'r')
                #     self.fs, self.x = wav.read(audiopath)     
                print('open new file: '+self.current_audiopath)
                print('FS: '+str(self.fs) +' x: '+str(np.shape(self.x)))
                if len(self.x.shape)>1:
                    if np.shape(self.x)[1]>1:
                        self.x=self.x[:,0]

                # factor=60
                # x=signal.decimate(x,factor,ftype='fir')
                
                db_saturation=float( self.db_saturation.text() )
                x=self.x/32767 
                p =np.power(10,(db_saturation/20))*x #convert data.signal to uPa    
                
                fft_size=int( self.fft_size.currentText() )
                fft_overlap=float(  self.fft_overlap.currentText() )
                self.f, self.t, self.Sxx = signal.spectrogram(p, self.fs, window='hamming',nperseg=fft_size,noverlap=int(fft_size*fft_overlap))
                # self.t=self.time +  pd.to_timedelta( t  , unit='s')           
                if self.file_blocks.loc[self.filecounter,'start']>0:     
                    secoffset = self.file_blocks.loc[self.filecounter,'start'] / self.fs
                    self.t=  self.t + secoffset     
                # print(self.t)
                
            def plot_annotation_box(annotation_row):
                print('row:')
                # print(annotation_row.dtypes)
                x1=annotation_row.iloc[0,0]
                x2=annotation_row.iloc[0,1]
                
                xt=pd.Series([x1,x2])
                print(xt)
                print(xt.dtype)

                # print(np.dtype(np.array(self.time).astype('datetime64[ns]') ))
                tt=xt - np.array(self.time).astype('datetime64[ns]')  
                xt=tt.dt.seconds + tt.dt.microseconds/10**6
                x1=xt[0]
                x2=xt[1]
                
                # tt=x1 - np.array(self.time).astype('datetime64[ns]')  
                # x1=tt.dt.seconds + tt.dt.microseconds/10**6
                # tt=x2 - np.array(self.time).astype('datetime64[ns]')  
                # x2=tt.dt.seconds + tt.dt.microseconds/10**6
       
                y1=annotation_row.iloc[0,2]
                y2=annotation_row.iloc[0,3]
                c_label=annotation_row.iloc[0,4]
                
                line_x=[x2,x1,x1,x2,x2]
                line_y=[y1,y1,y2,y2,y1]
                
                xmin=np.min([x1,x2])
                ymax=np.max([y1,y2])
                
                self.canvas.axes.plot(line_x,line_y,'-b',linewidth=.75)
                self.canvas.axes.text(xmin,ymax,c_label,size=5)
                  
                
                
            def plot_spectrogram():
             if self.filecounter>=0:
                # self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
                # self.setCentralWidget(self.canvas)
                self.canvas.fig.clf() 
                self.canvas.axes = self.canvas.fig.add_subplot(111)
                # self.canvas.axes.cla()
                
                if self.t_length.text()=='':
                    self.plotwindow_length= self.t[-1] 
                    self.plotwindow_startsecond=self.t[0]
                else:    
                    self.plotwindow_length=float( self.t_length.text() )
                    if self.t[-1]<self.plotwindow_length:
                        self.plotwindow_startsecond=self.t[0]
                        self.plotwindow_length=self.t[-1]
                        
                y1=int(self.f_min.text())    
                y2=int(self.f_max.text())    
                if y2>(self.fs/2):
                    y2=(self.fs/2)
                t1=self.plotwindow_startsecond
                t2=self.plotwindow_startsecond+self.plotwindow_length
                
                # if self.t_length.text=='':
                #     t2=self.t[-1]
                # else:    
                #     if self.t[-1]<float(self.t_length.text()):
                #         t2=self.t[-1]
                #     else:    
                #         t2=self.plotwindow_startsecond+self.plotwindow_length
                        
                # tt,ff=np.meshgrid(self.t,self.f)
                # ix_time=(tt>=self.plotwindow_startsecond) & (tt<(self.plotwindow_startsecond+self.plotwindow_length))
                # ix_f=(ff>=y1) & (ff<y2)
                # plotsxx=self.Sxx[ ix_f & ix_time]
                ix_time=np.where( (self.t>=t1) & (self.t<t2 ))[0]
                ix_f=np.where((self.f>=y1) & (self.f<y2))[0]
                # print(ix_time.shape)
                print([self.t,t1,t2])
                plotsxx= self.Sxx[ int(ix_f[0]):int(ix_f[-1]),int(ix_time[0]):int(ix_time[-1]) ] 
                plotsxx_db=10*np.log10(plotsxx)
                
                if self.checkbox_background.isChecked():
                   spec_mean=np.median(plotsxx_db,axis=1) 
                   sxx_background=np.transpose(np.broadcast_to(spec_mean,np.transpose(plotsxx_db).shape))
                   plotsxx_db = plotsxx_db - sxx_background
                   plotsxx_db=plotsxx_db - np.min(plotsxx_db.flatten())
                # print(plotsxx.shape)
      
                # img=self.canvas.axes.pcolormesh(self.t, self.f, 10*np.log10(self.Sxx) ,cmap='plasma')
                colormap_plot=self.colormap_plot.currentText()
                img=self.canvas.axes.imshow( plotsxx_db , aspect='auto',cmap=colormap_plot,origin = 'lower',extent = [t1, t2, y1, y2])
              
                # img=self.canvas.axes.pcolormesh(self.t[ int(ix_time[0]):int(ix_time[-1])], self.f[int(ix_f[0]):int(ix_f[-1])], 10*np.log10(plotsxx) , shading='flat',cmap='plasma')

                
                self.canvas.axes.set_ylabel('Frequency [Hz]')
                self.canvas.axes.set_xlabel('Time [sec]')
                if self.checkbox_logscale.isChecked():
                    self.canvas.axes.set_yscale('log')
                else:
                    self.canvas.axes.set_yscale('linear')        
                    
     

                
                if self.filename_timekey.text()=='':
                    # audiopath=self.filenames[self.filecounter]
                    self.canvas.axes.set_title(self.current_audiopath.split('/')[-1])
                else:     
                    self.canvas.axes.set_title(self.time)

                # img.set_clim([ 40 ,10*np.log10( np.max(np.array(plotsxx).ravel() )) ] )
                clims=img.get_clim()
                if (self.db_vmin.text()=='') & (self.db_vmax.text()!=''):
                    img.set_clim([ clims[0] , float(self.db_vmax.text())] )
                if (self.db_vmin.text()!='') & (self.db_vmax.text()==''):
                    img.set_clim([ float(self.db_vmin.text()) ,clims[1]] )
                if (self.db_vmin.text()!='') & (self.db_vmax.text()!=''):
                    img.set_clim([ float(self.db_vmin.text()) ,float(self.db_vmax.text()) ] )        
                    
                self.canvas.fig.colorbar(img,label='PSD [dB re $1 \ \mu Pa \ Hz^{-1}$]')
                
                # print(self.time)        
                # print(self.call_time)

            # plot annotations
                if self.annotation.shape[0]>0:
                     ix=(self.annotation['t1'] > (np.array(self.time).astype('datetime64[ns]')+pd.Timedelta(self.plotwindow_startsecond, unit="s") )  ) & \
                         (self.annotation['t1'] < (np.array(self.time).astype('datetime64[ns]')+pd.Timedelta(self.plotwindow_startsecond+self.plotwindow_length, unit="s") )  )   &\
                         (self.annotation['audiofilename'] ==  self.current_audiopath  )
                     if np.sum(ix)>0:
                         ix=np.where(ix)[0]
                         print('ix is')
                         print(ix)
                         for ix_x in ix:
                           a= pd.DataFrame([self.annotation.iloc[ix_x,:] ])
                           print(a)
                           plot_annotation_box(a)
      
              # plot detections
                cmap = plt.cm.get_cmap('cool')  
                if self.detectiondf.shape[0]>0:                
                    for i in range(self.detectiondf.shape[0]):     
                        
                        insidewindow=(self.detectiondf.loc[i,'t-1'] > self.plotwindow_startsecond  ) & (self.detectiondf.loc[i,'t-2'] < (self.plotwindow_startsecond+self.plotwindow_length)   )   &\
                                                    (self.detectiondf.loc[i,'audiofilename'] ==  self.current_audiopath  )
 
                        scoremin=self.detectiondf['score'].min()
                        scoremax=self.detectiondf['score'].max()
                       
                        if (self.detectiondf.loc[i,'score']>=0.01) & insidewindow:
                   
                            xx1=self.detectiondf.loc[i,'t-1']
                            xx2=self.detectiondf.loc[i,'t-2']
                            yy1=self.detectiondf.loc[i,'f-1']
                            yy2=self.detectiondf.loc[i,'f-2']
                            scorelabel=str(np.round(self.detectiondf.loc[i,'score'],2))
                            snorm=(self.detectiondf.loc[i,'score']-scoremin) / (scoremax-scoremin)
                            scorecolor = cmap(snorm)
            
                            line_x=[xx2,xx1,xx1,xx2,xx2]
                            line_y=[yy1,yy1,yy2,yy2,yy1]
                            
                            xmin=np.min([xx1,xx2])
                            ymax=np.max([yy1,yy2])                
                            self.canvas.axes.plot(line_x,line_y,'-',color=scorecolor,linewidth=.75)
                            self.canvas.axes.text(xmin,ymax,scorelabel,size=5,color=scorecolor)               
      
                            
                  
                self.canvas.axes.set_ylim([y1,y2])
                self.canvas.axes.set_xlim([t1,t2])
                          
                    
                self.canvas.fig.tight_layout()
                toggle_selector.RS=RectangleSelector(self.canvas.axes, box_select_callback,
                                           drawtype='box', useblit=False,
                                           button=[1],  # disable middle button                                  
                                           interactive=False,rectprops=dict(facecolor="blue", edgecolor="black", alpha=0.1, fill=True))


                self.canvas.draw()
                self.cid1=self.canvas.fig.canvas.mpl_connect('button_press_event', onclick)

                
            def plot_spectrogram_threshold():
             if self.filecounter>=0:
                 
                db_threshold, ok = QtWidgets.QInputDialog.getInt(self, 'Input Dialog',
                                                    'Enter signal-to-noise threshold in dB:')
                find_regions(db_threshold)   
                self.detectiondf=pd.DataFrame([])
                
                # plot_spectrogram()
                # self.canvas.axes2 = self.canvas.axes.twiny()

                # self.canvas.axes2.contour( self.region_labels>0 , [0.5] , color='g')
   
                self.canvas.fig.clf() 
                self.canvas.axes = self.canvas.fig.add_subplot(111)
                
                
                self.canvas.axes.set_ylabel('Frequency [Hz]')
                # self.canvas.axes.set_xlabel('Time [sec]')
                if self.checkbox_logscale.isChecked():
                    self.canvas.axes.set_yscale('log')
                else:
                    self.canvas.axes.set_yscale('linear')        
                    
     
                
                img=self.canvas.axes.imshow( self.region_labels>0 , aspect='auto',cmap='gist_yarg',origin = 'lower')
                
                self.canvas.fig.colorbar(img)
                
                self.canvas.fig.tight_layout()
                self.canvas.draw()


            def export_zoomed_sgram_as_csv():
             if self.filecounter>=0:
                 
                 
                    # filter out background
                spectrog = 10*np.log10(self.Sxx )  
                
                msg = QtWidgets.QMessageBox()
                msg.setIcon(QtWidgets.QMessageBox.Information)   
                msg.setText("Remove background?")
                msg.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No)
                returnValue = msg.exec()
                
                if returnValue == QtWidgets.QMessageBox.Yes:
                    rectime= pd.to_timedelta( self.t ,'s')
                    spg=pd.DataFrame(np.transpose(spectrog),index=rectime)
                    bg=spg.resample('3min').mean().copy()
                    bg=bg.resample('1s').interpolate(method='time')
                    bg=    bg.reindex(rectime,method='nearest')
                    background=np.transpose(bg.values)   
                    z=spectrog-background
                else:    
                    z=spectrog
                
                self.f_limits=self.canvas.axes.get_ylim()
                self.t_limits=self.canvas.axes.get_xlim()
                y1=int(self.f_limits[0])    
                y2=int(self.f_limits[1])    
                t1=self.t_limits[0]
                t2=self.t_limits[1]

                        
                ix_time=np.where( (self.t>=t1) & (self.t<t2 ))[0]
                ix_f=np.where((self.f>=y1) & (self.f<y2))[0]
                # print(ix_time.shape)
                # print(ix_f.shape)
                plotsxx_db= z[ int(ix_f[0]):int(ix_f[-1]),int(ix_time[0]):int(ix_time[-1]) ] 
                
                sgram = pd.DataFrame(data=plotsxx_db, index=self.f[ix_f[:-1]] , columns=self.t[ix_time[:-1]] )
                print(sgram)
                
                savename = QtWidgets.QFileDialog.getSaveFileName(self,"", "csv files (*.csv)")
                if len(savename[0])>0:     
                    if savename[-4:]!='.csv':
                        savename=savename[0]+'.csv'
                    sgram.to_csv(savename)                 
           
           
      
                            
                
            def box_select_callback(eclick, erelease):

                x1, y1 = eclick.xdata, eclick.ydata
                x2, y2 = erelease.xdata, erelease.ydata

                x1 =self.time +  pd.to_timedelta( x1 , unit='s') 
                x2 =self.time +  pd.to_timedelta( x2 , unit='s') 
                
                # sort to increasing values
                t1=np.min([x1,x2])
                t2=np.max([x1,x2])
                f1=np.min([y1,y2])
                f2=np.max([y1,y2])
                
                if self.bg.checkedId()==-1:
                    c_label=''
                else:
                    c_label=eval( 'self.an_'+str(self.bg.checkedId())+'.text()' )
                
                # a=pd.DataFrame(columns=['t1','t2','f1','f2','label'])
                # a.iloc[0,:]=np.array([x1,x2,y1,y2,c_label ])
                a=pd.DataFrame({'t1': pd.Series(t1,dtype='datetime64[ns]'),
                       't2': pd.Series(t2,dtype='datetime64[ns]'),
                       'f1': pd.Series(f1,dtype='float'),
                       'f2': pd.Series(f2,dtype='float'),
                       'label': pd.Series(c_label,dtype='object') , 
                        'audiofilename':  self.current_audiopath })
                
                # a=pd.DataFrame(data=[ [x1,x2,y1,y2,c_label ] ],columns=['t1','t2','f1','f2','label'])
                # print('a:')
                # print(a.dtypes)
                # self.annotation.append(a, ignore_index = True)
                self.annotation=pd.concat([ self.annotation ,a ] , ignore_index = True)

                # print(self.annotation.dtypes)  
                plot_annotation_box(a)

            def toggle_selector(event):
                # toggle_selector.RS.set_active(True)
                print('select')
                # if event.key == 't':
                #     if toggle_selector.RS.active:
                #         print(' RectangleSelector deactivated.')
                #         toggle_selector.RS.set_active(False)
                #     else:
                #         print(' RectangleSelector activated.')
                #         toggle_selector.RS.set_active(True)
            


            def onclick(event):
                if event.button==3:
                    self.annotation=self.annotation.head(-1)
                    # print(self.annotation)            
                    plot_spectrogram()              
            
            def end_of_filelist_warning(): 
                msg_listend = QtWidgets.QMessageBox()
                msg_listend.setIcon(QtWidgets.QMessageBox.Information)
                msg_listend.setText("End of file list reached!")
                msg_listend.exec_()

            def plot_next_spectro():
             if len(self.filenames)>0:
                print('old filecounter is: '+str(self.filecounter))
               
                if self.t_length.text()=='' or ((self.filecounter>=0) & (self.t[-1]<float(self.t_length.text())) ):
                    self.filecounter=self.filecounter+1
                    if self.filecounter>self.file_blocks.shape[0]-1:
                        self.filecounter=self.file_blocks.shape[0]-1
                        print('That was it')
                        end_of_filelist_warning()  
                    self.plotwindow_length= self.t[-1] 
                    self.plotwindow_startsecond=self.t[0]
                    # new file    
                    # self.filecounter=self.filecounter+1
                    read_wav()
                    plot_spectrogram()
                    
                    # print('hello!!!!!')
                        
                else:    
                    self.plotwindow_length=float( self.t_length.text() )       
                    self.plotwindow_startsecond=self.plotwindow_startsecond + self.plotwindow_length
                    
                print( [self.plotwindow_startsecond, self.t[0], self.t[-1] ] )      
                
                if self.plotwindow_startsecond > self.t[-1]:              
                    #save log
                    if checkbox_log.isChecked():
                        
                        tt= self.annotation['t1'] - self.time             
                        t_in_seconds=np.array( tt.values*1e-9 ,dtype='float16')
                        reclength=np.array( self.t[-1] ,dtype='float16')
        
                        ix=(t_in_seconds>0) & (t_in_seconds<reclength)
                        
                        calldata=self.annotation.iloc[ix,:]
                        print(calldata)
                        savename=self.current_audiopath
                        nn = savename[:-4]+'_log_sec' + str(int(self.t[0])) +'_to_sec'+str(int(self.t[-1])) +'.csv'
                        calldata.to_csv(nn)                  
                        print('writing log: '+ nn)
                    # new file    
                    self.filecounter=self.filecounter+1
                    if self.filecounter>=self.file_blocks.shape[0]-1:
                        self.filecounter=self.file_blocks.shape[0]-1
                        print('That was it')
                        end_of_filelist_warning()  
                    read_wav()
                    self.plotwindow_startsecond=self.t[0]
                    plot_spectrogram()
                else:
                    plot_spectrogram()
             

                        
            def plot_previous_spectro():
             if len(self.filenames)>0:   
                print('old filecounter is: '+str(self.filecounter))
             
                if self.t_length.text()=='' or ((self.filecounter>=0) & (self.t[-1]<float(self.t_length.text())) ) :
                    self.filecounter=self.filecounter-1
                    if self.filecounter<0:
                        self.filecounter=0
                        print('That was it')
                        end_of_filelist_warning()  
                    self.plotwindow_length= self.t[-1] 
                    self.plotwindow_startsecond=self.t[0]
                    # new file    
                    # self.filecounter=self.filecounter+1
                    read_wav()
                    plot_spectrogram()
                    
                else:                
                    self.plotwindow_startsecond=self.plotwindow_startsecond -self.plotwindow_length
                    print( [self.plotwindow_startsecond, self.t[0], self.t[-1] ] )      
                    if self.plotwindow_startsecond < self.t[0] : 
                        # self.plotwindow_startsecond=self.t[0]
                        # if self.file_blocks.loc[self.filecounter,'start']==0:
                        #     self.plotwindow_startsecond = self.t[-1] + self.plotwindow_startsecond                
                        # old file    
                        self.filecounter=self.filecounter-1
                        
                        if self.filecounter<0:
                            self.filecounter=0
                            print('That was it')
                            end_of_filelist_warning()  
                        
                        # if self.filecounter<0:
                        #     print('That was it')
                        #     self.canvas.fig.clf() 
                        #     self.canvas.axes = self.canvas.fig.add_subplot(111)
                        #     self.canvas.axes.set_title('That was it')
                        #     self.canvas.draw()                 
                        # elif self.filecounter>=self.filenames.shape[0]-1:
                        #     print('That was it')
                        #     self.canvas.fig.clf() 
                        #     self.canvas.axes = self.canvas.fig.add_subplot(111)
                        #     self.canvas.axes.set_title('That was it')
                        #     self.canvas.draw()
       
                        read_wav()
                        # self.plotwindow_startsecond=0
                        plot_spectrogram()
                    else:
                        plot_spectrogram()
             
                    
             
            def new_fft_size_selected():
                 read_wav()
                 plot_spectrogram()
            self.fft_size.currentIndexChanged.connect(new_fft_size_selected)

            self.colormap_plot.currentIndexChanged.connect( plot_spectrogram)
            self.checkbox_background.stateChanged.connect(plot_spectrogram  )      
            self.checkbox_logscale.stateChanged.connect(plot_spectrogram   )         
             
                            
            # self.canvas.fig.canvas.mpl_connect('button_press_event', onclick)
            # self.canvas.fig.canvas.mpl_connect('key_press_event', toggle_selector)


            
            # QtGui.QShortcut(QtCore.Qt.Key_Right, MainWindow, plot_next_spectro())
            # self.msgSc = QShortcut(QKeySequence(u"\u2192"), self)
            # self.msgSc.activated.connect(plot_next_spectro)
            # button_plot_spectro=QtWidgets.QPushButton('Next spectrogram-->')
            # button_plot_spectro.clicked.connect(plot_next_spectro)
            
            # button_plot_prevspectro=QtWidgets.QPushButton('<--Previous spectrogram')
            # button_plot_prevspectro.clicked.connect(plot_previous_spectro)
        
            button_save=QtWidgets.QPushButton('Save annotation csv')
            def func_savecsv():         
                options = QtWidgets.QFileDialog.Options()
                savename = QtWidgets.QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()", r"C:\Users\a5278\Documents\passive_acoustics\detector_delevopment\detector_validation_subset", "csv files (*.csv)",options=options)
                print('location is:' + savename[0])
                if len(savename[0])>0:
                    self.annotation.to_csv(savename[0])         
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
                            self.time= dt.datetime(2000,1,1,0,0,0)
                        else:  
                            try:
                                self.time= dt.datetime.strptime( audiopath.split('/')[-1], self.filename_timekey.text() )
                            except:
                                self.time= dt.datetime(2000,1,1,0,0,0)
                                
                        self.x,self.fs  =  sf.read(audiopath,dtype='int16')
                        print('open new file: '+audiopath)
                        
                        db_saturation=float( self.db_saturation.text() )
                        x=self.x/32767 
                        p =np.power(10,(db_saturation/20))*x #convert data.signal to uPa    
                        
                        fft_size=int( self.fft_size.currentText() )
                        fft_overlap=float(  self.fft_overlap.currentText() )
                        
                        
                        self.f, self.t, self.Sxx = signal.spectrogram(p, self.fs, window='hamming',nperseg=fft_size,noverlap=fft_size*fft_overlap)
                        
                        self.plotwindow_startsecond=0
                        
                        plot_spectrogram()
                        self.canvas.axes.set_title(audiopath.split('/')[-1])
                        self.canvas.fig.savefig( audiopath[:-4]+'.jpg',dpi=150 )    
                
                
            button_plot_all_spectrograms.clicked.connect(plot_all_spectrograms)        

            button_draw_shape=QtWidgets.QPushButton('Draw shape')
            def func_draw_shape_plot(): 
                   if self.filecounter>=0:
                        self.canvas.fig.clf() 
                        self.canvas.axes = self.canvas.fig.add_subplot(111)                    
                        if self.t_length.text()=='':
                            self.plotwindow_length= self.t[-1] 
                            self.plotwindow_startsecond=0
                        else:    
                            self.plotwindow_length=float( self.t_length.text() )
                            if self.t[-1]<self.plotwindow_length:
                                self.plotwindow_startsecond=0
                                self.plotwindow_length=self.t[-1]
                                
                        y1=int(self.f_min.text())    
                        y2=int(self.f_max.text())    
                        if y2>(self.fs/2):
                            y2=(self.fs/2)
                        t1=self.plotwindow_startsecond
                        t2=self.plotwindow_startsecond+self.plotwindow_length
                    
                        ix_time=np.where( (self.t>=t1) & (self.t<t2 ))[0]
                        ix_f=np.where((self.f>=y1) & (self.f<y2))[0]

                        plotsxx= self.Sxx[ int(ix_f[0]):int(ix_f[-1]),int(ix_time[0]):int(ix_time[-1]) ] 
                        plotsxx_db=10*np.log10(plotsxx)
                        
                        if self.checkbox_background.isChecked():
                           spec_mean=np.median(plotsxx_db,axis=1) 
                           sxx_background=np.transpose(np.broadcast_to(spec_mean,np.transpose(plotsxx_db).shape))
                           plotsxx_db = plotsxx_db - sxx_background
                           plotsxx_db=plotsxx_db - np.min(plotsxx_db.flatten())
              
                        colormap_plot=self.colormap_plot.currentText()
                        img=self.canvas.axes.imshow( plotsxx_db , aspect='auto',cmap=colormap_plot,origin = 'lower',extent = [t1, t2, y1, y2])
                                            
                        self.canvas.axes.set_ylabel('Frequency [Hz]')
                        self.canvas.axes.set_xlabel('Time [sec]')
                        if self.checkbox_logscale.isChecked():
                            self.canvas.axes.set_yscale('log')
                        else:
                            self.canvas.axes.set_yscale('linear')        
                            
                        # if self.filename_timekey.text()=='':
                        #     self.canvas.axes.set_title(self.current_audiopath.split('/')[-1])
                        # else:     
                        #     self.canvas.axes.set_title(self.time)
            
                        clims=img.get_clim()
                        if (self.db_vmin.text()=='') & (self.db_vmax.text()!=''):
                            img.set_clim([ clims[0] , float(self.db_vmax.text())] )
                        if (self.db_vmin.text()!='') & (self.db_vmax.text()==''):
                            img.set_clim([ float(self.db_vmin.text()) ,clims[1]] )
                        if (self.db_vmin.text()!='') & (self.db_vmax.text()!=''):
                            img.set_clim([ float(self.db_vmin.text()) ,float(self.db_vmax.text()) ] )        
                            
                        self.canvas.fig.colorbar(img,label='PSD [dB re $1 \ \mu Pa \ Hz^{-1}$]')
                        
                    # plot annotations
                        if self.annotation.shape[0]>0:
                             ix=(self.annotation['t1'] > (np.array(self.time).astype('datetime64[ns]')+pd.Timedelta(self.plotwindow_startsecond, unit="s") )  ) & (self.annotation['t1'] < (np.array(self.time).astype('datetime64[ns]')+pd.Timedelta(self.plotwindow_startsecond+self.plotwindow_length, unit="s") )  )          
                             if np.sum(ix)>0:
                                 ix=np.where(ix)[0]
                                 print('ix is')
                                 print(ix)
                                 for ix_x in ix:
                                   a= pd.DataFrame([self.annotation.iloc[ix_x,:] ])
                                   print(a)
                                   plot_annotation_box(a)
                                   
                        if self.t_limits==None:           
                            self.canvas.axes.set_ylim([y1,y2])
                            self.canvas.axes.set_xlim([t1,t2])
                        else:
                          self.canvas.axes.set_ylim(self.f_limits)
                          self.canvas.axes.set_xlim(self.t_limits)
                     
     
                                                      
                        self.canvas.fig.tight_layout()
                    
                        self.canvas.axes.plot(self.draw_x,self.draw_y,'.-g')
      
                        self.canvas.draw()    
                        self.cid2=self.canvas.fig.canvas.mpl_connect('button_press_event', onclick_draw)
                        
            def onclick_draw(event):
                    # print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
                    #       ('double' if event.dblclick else 'single', event.button,
                    #        event.x, event.y, event.xdata, event.ydata))
                    if event.button==1 & event.dblclick:
                        self.draw_x=self.draw_x.append( pd.Series(event.xdata) ,ignore_index=True )
                        self.draw_y=self.draw_y.append( pd.Series(event.ydata) ,ignore_index=True )
                        self.f_limits=self.canvas.axes.get_ylim()
                        self.t_limits=self.canvas.axes.get_xlim()   
                        
                        line = self.line_2.pop(0)
                        line.remove()        
                        self.line_2 =self.canvas.axes.plot(self.draw_x,self.draw_y,'.-g')      
                        self.canvas.draw()    
                                 
                        # func_draw_shape_plot()   
                      
                    if event.button==3:
                        self.draw_x=self.draw_x.head(-1)
                        self.draw_y=self.draw_y.head(-1)
                        self.f_limits=self.canvas.axes.get_ylim()
                        self.t_limits=self.canvas.axes.get_xlim()
                        # func_draw_shape_plot()              
                        line = self.line_2.pop(0)
                        line.remove()        
                        self.line_2 =self.canvas.axes.plot(self.draw_x,self.draw_y,'.-g')     
                        self.canvas.draw()    
                                  
                        # func_draw_shape_plot()   
                        
            def func_draw_shape_exit():
                print('save shape' + str(self.draw_x.shape))
                self.canvas.fig.canvas.mpl_disconnect(self.cid2)
                plot_spectrogram()
                print('back to boxes')
                ## deactive shortcut
                self.drawexitm.setEnabled(False)  

                if self.draw_x.shape[0]>0:
                    options = QtWidgets.QFileDialog.Options()
                    savename = QtWidgets.QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()", "csv files (*.csv)",options=options)
                    if len(savename[0])>0:     
                        if savename[-4:]!='.csv':
                            savename=savename[0]+'.csv'
                        # drawcsv=pd.concat([self.draw_x,self.draw_y],axis=1)
                        drawcsv=pd.DataFrame(columns=['Time_in_s','Frequency_in_Hz'])
                        drawcsv['Time_in_s']=self.draw_x
                        drawcsv['Frequency_in_Hz']=self.draw_y
                        drawcsv.to_csv(savename)   
                        
          
            def func_draw_shape():                  
               msg = QtWidgets.QMessageBox()
               msg.setIcon(QtWidgets.QMessageBox.Information)   
               msg.setText("Add points with double left click.\nRemove latest point with single right click. \nExit draw mode and save CSV by pushing enter")
               msg.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)
               returnValue = msg.exec()    
               if returnValue == QtWidgets.QMessageBox.Ok:
                   print('drawing')  
                   self.draw_x=pd.Series(dtype='float')
                   self.draw_y=pd.Series(dtype='float')
                   self.f_limits=self.canvas.axes.get_ylim()
                   self.t_limits=self.canvas.axes.get_xlim()
                   self.canvas.fig.canvas.mpl_disconnect(self.cid1)    
                   self.cid2=self.canvas.fig.canvas.mpl_connect('button_press_event', onclick_draw)
                   self.line_2 =self.canvas.axes.plot(self.draw_x,self.draw_y,'.-g')        
                   func_draw_shape_plot()   
                   self.drawexitm = QtWidgets.QShortcut(QtCore.Qt.Key_Return, self)
                   self.drawexitm.activated.connect(func_draw_shape_exit)  
                                                                        
            button_draw_shape.clicked.connect(func_draw_shape)
                   
            ####### play audio
            button_play_audio=QtWidgets.QPushButton('Play/Stop [spacebar]')
            def func_playaudio():
                if self.filecounter>=0:
                    if not hasattr(self, "play_obj"):
                        new_rate = 32000          
        
                        t_limits=list(self.canvas.axes.get_xlim())
                        t_limits=t_limits - self.file_blocks.loc[self.filecounter,'start']/self.fs
                        f_limits=list(self.canvas.axes.get_ylim())
                        if f_limits[1]>=(self.fs/2):
                            f_limits[1]= self.fs/2-10
                        print(t_limits)
                        print(f_limits)

                        x_select=self.x[int(t_limits[0]*self.fs) : int(t_limits[1]*self.fs) ]     
                        
                        sos=signal.butter(8, f_limits, 'bandpass', fs=self.fs, output='sos')
                        x_select = signal.sosfilt(sos, x_select)
                        
                        number_of_samples = round(len(x_select) * (float(new_rate)/ float(self.playbackspeed.currentText())) / self.fs)
                        x_resampled = np.array(signal.resample(x_select, number_of_samples)).astype('int')            
                       
                        #normalize sound level
                        maximum_x=32767*0.8
                        old_max=np.max(np.abs([x_resampled.min(),x_resampled.max()]))
                        x_resampled=x_resampled * (maximum_x/old_max)
                        x_resampled = x_resampled.astype(np.int16)

                        print( [x_resampled.min(),x_resampled.max()]  )                   
                        wave_obj = sa.WaveObject(x_resampled, 1, 2, new_rate)
                        self.play_obj = wave_obj.play()   
                    else:    
                        if self.play_obj.is_playing():
                            sa.stop_all()
                        else:    
                            new_rate = 32000          
                            t_limits=list(self.canvas.axes.get_xlim())
                            t_limits=t_limits - self.file_blocks.loc[self.filecounter,'start']/self.fs
                            f_limits=list(self.canvas.axes.get_ylim())
                            if f_limits[1]>=(self.fs/2):
                                f_limits[1]= self.fs/2-10
                            print(t_limits)
                            print(f_limits)
                        
                            x_select=self.x[int(t_limits[0]*self.fs) : int(t_limits[1]*self.fs) ]    
                            sos=signal.butter(8, f_limits, 'bandpass', fs=self.fs, output='sos')
                            x_select = signal.sosfilt(sos, x_select)
                        
                            # number_of_samples = round(len(x_select) * float(new_rate) / self.fs)
                            number_of_samples = round(len(x_select) * (float(new_rate)/ float(self.playbackspeed.currentText())) / self.fs)

                            x_resampled = np.array(signal.resample(x_select, number_of_samples)).astype('int')            
                             #normalize sound level
                            maximum_x=32767*0.8
                            old_max=np.max(np.abs([x_resampled.min(),x_resampled.max()]))
                            x_resampled=x_resampled * (maximum_x/old_max)
                            x_resampled = x_resampled.astype(np.int16)
                            print( [x_resampled.min(),x_resampled.max()]  )
                            wave_obj = sa.WaveObject(x_resampled, 1, 2, new_rate)
                            self.play_obj = wave_obj.play()
                
            button_play_audio.clicked.connect(func_playaudio)        
            
            button_save_audio=QtWidgets.QPushButton('Export selected audio')         
            def func_saveaudio():
                if self.filecounter>=0:
                    options = QtWidgets.QFileDialog.Options()
                    savename = QtWidgets.QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()", "wav files (*.wav)",options=options)
                    if len(savename[0])>0:     
                        savename=savename[0]
                        new_rate = 32000          
        
                        t_limits=self.canvas.axes.get_xlim()
                        f_limits=list(self.canvas.axes.get_ylim())
                        if f_limits[1]>=(self.fs/2):
                            f_limits[1]= self.fs/2-10
                        print(t_limits)
                        print(f_limits)
                        x_select=self.x[int(t_limits[0]*self.fs) : int(t_limits[1]*self.fs) ]     
                        
                        sos=signal.butter(8, f_limits, 'bandpass', fs=self.fs, output='sos')
                        x_select = signal.sosfilt(sos, x_select)
                                           
                        number_of_samples = round(len(x_select) * (float(new_rate)/ float(self.playbackspeed.currentText())) / self.fs)
                        x_resampled = np.array(signal.resample(x_select, number_of_samples)).astype('int') 
                                            #normalize sound level
                        maximum_x=32767*0.8
                        old_max=np.max(np.abs([x_resampled.min(),x_resampled.max()]))
                        x_resampled=x_resampled * (maximum_x/old_max)
                        x_resampled = x_resampled.astype(np.int16)
                        
                        if savename[-4:]!='.wav':
                            savename=savename+'.wav'
                        wav.write(savename, new_rate, x_resampled)
            button_save_audio.clicked.connect(func_saveaudio)        
 
            button_save_video=QtWidgets.QPushButton('Export video')         
            def func_save_video():
                if self.filecounter>=0:
                    savename = QtWidgets.QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()", "video files (*.mp4)")
                    if len(savename[0])>0:     
                        savename=savename[0]
                        new_rate = 32000          
        
                        t_limits=self.canvas.axes.get_xlim()
                        f_limits=list(self.canvas.axes.get_ylim())
                        if f_limits[1]>=(self.fs/2):
                            f_limits[1]= self.fs/2-10
                        print(t_limits)
                        print(f_limits)
                        x_select=self.x[int(t_limits[0]*self.fs) : int(t_limits[1]*self.fs) ]     
                        
                        sos=signal.butter(8, f_limits, 'bandpass', fs=self.fs, output='sos')
                        x_select = signal.sosfilt(sos, x_select)
                                           
                        number_of_samples = round(len(x_select) * (float(new_rate)/ float(self.playbackspeed.currentText())) / self.fs)
                        x_resampled = np.array(signal.resample(x_select, number_of_samples)).astype('int') 
                                            #normalize sound level
                        maximum_x=32767*0.8
                        old_max=np.max(np.abs([x_resampled.min(),x_resampled.max()]))
                        x_resampled=x_resampled * (maximum_x/old_max)
                        x_resampled = x_resampled.astype(np.int16)
                        
                        if savename[:-4]=='.wav':
                            savename=savename[:-4]
                        if savename[:-4]=='.mp4':
                            savename=savename[:-4]
                        wav.write(savename+'.wav', new_rate, x_resampled)
                        
                        # self.f_limits=self.canvas.axes.get_ylim()
                        # self.t_limits=self.canvas.axes.get_xlim()
   
                        audioclip = AudioFileClip(savename+'.wav')
                        duration=audioclip.duration
                        # func_draw_shape_plot()     

                        self.canvas.axes.set_title(None)                                              
                        # self.canvas.axes.set_ylim(f_limits)
                        # self.canvas.axes.set_xlim(t_limits)
                        self.line_2=self.canvas.axes.plot([t_limits[0]  ,t_limits[0]  ],f_limits,'-k')  
                        def make_frame(x):
                           s=t_limits[1] - t_limits[0]
                           xx= x/duration * s + t_limits[0]            
                           line = self.line_2.pop(0)
                           line.remove()                         
                           self.line_2=self.canvas.axes.plot([xx,xx],f_limits,'-k')    
                           
                    
                           return mplfig_to_npimage(self.canvas.fig)
                       
                        animation = VideoClip(make_frame, duration = duration )                        
                        animation = animation.set_audio(audioclip)
                        animation.write_videofile(savename+".mp4",fps=24,preset='fast')

                        plot_spectrogram()  
                        # self.canvas.fig.canvas.mpl_disconnect(self.cid2)
                                          
            button_save_video.clicked.connect(func_save_video)        
              
############# menue
            menuBar = self.menuBar()

            # Creating menus using a title
            openMenu = menuBar.addAction("Open files")
            openMenu.triggered.connect(openfilefunc)

            
            exportMenu = menuBar.addMenu("Export")
            e1 =exportMenu.addAction("Spectrogram as .wav file")
            e1.triggered.connect(func_saveaudio)
            e2 =exportMenu.addAction("Spectrogram as animated video")
            e2.triggered.connect(func_save_video)
            e3 =exportMenu.addAction("Spectrogram as .csv table")
            e3.triggered.connect(export_zoomed_sgram_as_csv)            
            e4 =exportMenu.addAction("All files as spectrogram images")
            e4.triggered.connect(plot_all_spectrograms)
            e5 =exportMenu.addAction("Annotations as .csv table")
            e5.triggered.connect(func_savecsv)                  
            e6 =exportMenu.addAction("Automatic detections as .csv table")
            e6.triggered.connect(export_automatic_detector_shapematching)            
            
            drawMenu = menuBar.addAction("Draw")
            drawMenu.triggered.connect(func_draw_shape)            
            
            
            autoMenu = menuBar.addMenu("Automatic detection")
            a1 =autoMenu.addAction("Shapematching on current file")
            a1.triggered.connect(automatic_detector_shapematching)
            a3 =autoMenu.addAction("Shapematching on all files")
            a3.triggered.connect(automatic_detector_shapematching_allfiles)
            
            a2 =autoMenu.addAction("Spectrogram correlation on current file")
            a2.triggered.connect(automatic_detector_specgram_corr)
   
            a4 =autoMenu.addAction("Spectrogram correlation on all files")
            a4.triggered.connect(automatic_detector_specgram_corr_allfiles)
            
            a5 =autoMenu.addAction("Show regions based on threshold")
            a5.triggered.connect(plot_spectrogram_threshold)
            
            
            
            quitMenu = menuBar.addAction("Quit")
            quitMenu.triggered.connect(QtWidgets.QApplication.instance().quit)

#################
             
            ######## layout
            outer_layout = QtWidgets.QVBoxLayout()
            
            # top_layout = QtWidgets.QHBoxLayout()       
            
            # top_layout.addWidget(openfilebutton)
            
            # top_layout.addWidget(button_plot_prevspectro)
            # top_layout.addWidget(button_plot_spectro)
            # # top_layout.addWidget(button_plot_all_spectrograms)
            # top_layout.addWidget(button_play_audio)
            # top_layout.addWidget(QtWidgets.QLabel('Playback speed:'))        
            # top_layout.addWidget(self.playbackspeed)
            # # top_layout.addWidget(button_save_audio)    
            # # top_layout.addWidget(button_save_video)    
            # # top_layout.addWidget(button_draw_shape)            

            # top_layout.addWidget(button_save)            
            # top_layout.addWidget(button_quit)

            top2_layout = QtWidgets.QHBoxLayout()   
            
            # self.f_min = QtWidgets.QLineEdit(self)
            # self.f_max = QtWidgets.QLineEdit(self)
            # self.t_start = QtWidgets.QLineEdit(self)
            # self.t_end = QtWidgets.QLineEdit(self)
            # self.fft_size = QtWidgets.QLineEdit(self)
            top2_layout.addWidget(checkbox_log)
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
     
            # top2_layout.addWidget(QtWidgets.QLabel('fft_size[bits]:'))
            # top2_layout.addWidget(self.fft_size) 
            # top2_layout.addWidget(QtWidgets.QLabel('fft_overlap[0-1]:'))
            # top2_layout.addWidget(self.fft_overlap) 
            

            
            # top2_layout.addWidget(QtWidgets.QLabel('Colormap:'))
            # top2_layout.addWidget( self.colormap_plot)        
            
            
            top2_layout.addWidget(QtWidgets.QLabel('Saturation dB:'))
            top2_layout.addWidget(self.db_saturation)
            
            top2_layout.addWidget(QtWidgets.QLabel('dB min:'))
            top2_layout.addWidget(self.db_vmin)
            top2_layout.addWidget(QtWidgets.QLabel('dB max:'))
            top2_layout.addWidget(self.db_vmax)
            

            
            # annotation label area
            top3_layout = QtWidgets.QHBoxLayout()   
            top3_layout.addWidget(QtWidgets.QLabel('Annotation labels:'))


            self.checkbox_an_1=QtWidgets.QCheckBox()
            top3_layout.addWidget(self.checkbox_an_1)
            self.an_1 = QtWidgets.QLineEdit(self)
            top3_layout.addWidget(self.an_1) 
            self.an_1.setText('')
            
            self.checkbox_an_2=QtWidgets.QCheckBox()
            top3_layout.addWidget(self.checkbox_an_2)
            self.an_2 = QtWidgets.QLineEdit(self)
            top3_layout.addWidget(self.an_2) 
            self.an_2.setText('')
            
            self.checkbox_an_3=QtWidgets.QCheckBox()
            top3_layout.addWidget(self.checkbox_an_3)
            self.an_3 = QtWidgets.QLineEdit(self)
            top3_layout.addWidget(self.an_3) 
            self.an_3.setText('')
            
            self.checkbox_an_4=QtWidgets.QCheckBox()
            top3_layout.addWidget(self.checkbox_an_4)
            self.an_4 = QtWidgets.QLineEdit(self)
            top3_layout.addWidget(self.an_4) 
            self.an_4.setText('')
            
            self.checkbox_an_5=QtWidgets.QCheckBox()
            top3_layout.addWidget(self.checkbox_an_5)
            self.an_5 = QtWidgets.QLineEdit(self)
            top3_layout.addWidget(self.an_5) 
            self.an_5.setText('')

            self.checkbox_an_6=QtWidgets.QCheckBox()
            top3_layout.addWidget(self.checkbox_an_6)
            self.an_6 = QtWidgets.QLineEdit(self)
            top3_layout.addWidget(self.an_6) 
            self.an_6.setText('')
            
            # self.checkbox_an_7=QtWidgets.QCheckBox()
            # top3_layout.addWidget(self.checkbox_an_7)
            # self.an_7 = QtWidgets.QLineEdit(self)
            # top3_layout.addWidget(self.an_7) 
            # self.an_7.setText('')
            

            # button_export_sgramcsv=QtWidgets.QPushButton('Export spectrog. csv')         
            # button_export_sgramcsv.clicked.connect(export_zoomed_sgram_as_csv)        
            # top3_layout.addWidget(button_export_sgramcsv)    
            
            # button_autodetect_shape=QtWidgets.QPushButton('Shapematching')         
            # button_autodetect_shape.clicked.connect(automatic_detector_shapematching)        
            # top3_layout.addWidget(button_autodetect_shape)            

            # button_autodetect_corr=QtWidgets.QPushButton('Spectrog. correlation')         
            # button_autodetect_corr.clicked.connect(automatic_detector_specgram_corr)        
            # top3_layout.addWidget(button_autodetect_corr)            
                        

            # button_saveautodetect=QtWidgets.QPushButton('Export auto-detec.')         
            # button_saveautodetect.clicked.connect(export_automatic_detector_shapematching)        
            # top3_layout.addWidget(button_saveautodetect)    
            
            

            # self.checkbox_an_8=QtWidgets.QCheckBox()
            # top3_layout.addWidget(self.checkbox_an_8)
            # self.an_8 = QtWidgets.QLineEdit(self)
            # top3_layout.addWidget(self.an_8) 
            # self.an_8.setText('')
               
            self.bg = QtWidgets.QButtonGroup()
            self.bg.addButton(self.checkbox_an_1,1)
            self.bg.addButton(self.checkbox_an_2,2)
            self.bg.addButton(self.checkbox_an_3,3)
            self.bg.addButton(self.checkbox_an_4,4)
            self.bg.addButton(self.checkbox_an_5,5)
            self.bg.addButton(self.checkbox_an_6,6)
            # self.bg.addButton(self.checkbox_an_7,7)
            # self.bg.addButton(self.checkbox_an_8,8)
            


            
            # combine layouts together
            
            plot_layout = QtWidgets.QVBoxLayout()
            tnav = NavigationToolbar( self.canvas, self)
            
            toolbar = QtWidgets.QToolBar()
        

            # toolbar.addAction('test')
            # toolbar.addWidget(button_plot_prevspectro)
            # toolbar.addWidget(button_plot_spectro)
            
            # b1=QtWidgets.QToolButton()
            # b1.setText('<--Previous spectrogram')
            # # b1.setStyleSheet("background-color: yellow; font-size: 18pt")
            # b1.clicked.connect(plot_previous_spectro)      
            # toolbar.addWidget(b1)

            # b2=QtWidgets.QToolButton()
            # b2.setText('Next spectrogram-->')
            # b2.clicked.connect(plot_next_spectro)      
            # toolbar.addWidget(b2)

            # b3=QtWidgets.QToolButton()
            # b3.setText('Play/Stop [spacebar]')
            # b3.clicked.connect(func_playaudio)      
            # toolbar.addWidget(b3)
            
            button_plot_prevspectro=QtWidgets.QPushButton('<--Previous spectrogram')
            button_plot_prevspectro.clicked.connect(plot_previous_spectro)            
            toolbar.addWidget(button_plot_prevspectro)
            
            ss='  '
            toolbar.addWidget(QtWidgets.QLabel(ss))     
            
            button_plot_spectro=QtWidgets.QPushButton('Next spectrogram-->')
            button_plot_spectro.clicked.connect(plot_next_spectro)
            toolbar.addWidget(button_plot_spectro)

            toolbar.addWidget(QtWidgets.QLabel(ss))     

           
            toolbar.addWidget(button_play_audio)
            toolbar.addWidget(QtWidgets.QLabel(ss))     
            
            toolbar.addWidget(QtWidgets.QLabel('Playback speed:'))        
            toolbar.addWidget(QtWidgets.QLabel(ss))     
            toolbar.addWidget(self.playbackspeed)
            toolbar.addWidget(QtWidgets.QLabel(ss))     

            toolbar.addSeparator()
            toolbar.addWidget(QtWidgets.QLabel(ss))     
         

            toolbar.addWidget(QtWidgets.QLabel('fft_size[bits]:'))
            toolbar.addWidget(QtWidgets.QLabel(ss))     
            toolbar.addWidget(self.fft_size) 
            toolbar.addWidget(QtWidgets.QLabel(ss))     
            toolbar.addWidget(QtWidgets.QLabel('fft_overlap[0-1]:'))
            toolbar.addWidget(QtWidgets.QLabel(ss))     
            toolbar.addWidget(self.fft_overlap) 
            
            toolbar.addWidget(QtWidgets.QLabel(ss))     

            
            toolbar.addWidget(QtWidgets.QLabel('Colormap:'))
            toolbar.addWidget(QtWidgets.QLabel(ss))     
            toolbar.addWidget( self.colormap_plot)  
            toolbar.addWidget(QtWidgets.QLabel(ss))     
            
            toolbar.addSeparator()
         
            toolbar.addWidget(tnav)

            
            plot_layout.addWidget(toolbar)
            plot_layout.addWidget(self.canvas)
            
            # outer_layout.addLayout(top_layout)
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