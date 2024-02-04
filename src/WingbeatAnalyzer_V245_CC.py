# -*- coding: utf-8 -*-
"""
==========================
Biogents Wingbeat Analyzer
==========================
Author:     Michael Weber (michael.weber@biogents.com)
Revision:   06/12/22

License:  Biogents Wingbeat Analyzer
© 2014 by Michael Weber, onVector Technology
© 2019 by Biogents AG
is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International.
To view a copy of this license, open license.txt or visit http://creativecommons.org/licenses/by-nc-sa/4.0/,

Python 3.9 (Ananconda Distribution)
"""

version = 'V2.45CC'
print(__doc__)

import matplotlib
matplotlib.use("Qt5Agg")
matplotlib.rcParams['font.size'] = 8
from matplotlib.backends.backend_qt5agg import (
	FigureCanvasQTAgg as FigureCanvas,
	NavigationToolbar2QT as NavigationToolbar)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates

# The PyQt GUI layout is developed on MacOS.
# It will work on other platforms but may look different.
from PyQt5.Qt import Qt
from PyQt5.QtCore import QDate, QTime
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QApplication, QWidget, QFileDialog, \
	QDialog, QTableWidget, QTableWidgetItem, QHeaderView, \
    QVBoxLayout, QHBoxLayout, QLabel, QStyleFactory, \
	QSizePolicy, QDesktopWidget, QMainWindow, QAction, QProgressBar, QStatusBar

import sys
import os
import pickle
from platform import system as platform
import datetime
import time
from functools import partial
import operator
import csv
import shutil
import multiprocessing

import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import scipy
from scipy import signal, interpolate
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.stats import entropy
import statistics
import itertools
import pywt
import cv2
from collections import Counter

from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sklearn.mixture import BayesianGaussianMixture
from sklearn import metrics
from sklearn.preprocessing import StandardScaler, Normalizer, RobustScaler
from sklearn import manifold
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPClassifier, BernoulliRBM

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from xgboost import XGBClassifier
from xgboost import plot_importance

# import soundfile as sf
import peakutils
import librosa
import soundfile
from playsound import playsound
from operator import itemgetter


class App(QMainWindow):
	'''class for the main experiment window'''

	def __init__(self):
		super().__init__()

		# set initial working directory
		self.cwd = os.getcwd()

		# create the various objects
		# they are referenced as experiment.object
		self.collection = Collection()
		# self.features = Features()
		self.tsne = TSNE()
		self.model = Model()

		# set title and initial size
		self.title = 'Biogents Wingbeat Analyzer ' + version
		self.setWindowTitle(self.title)
		screenShape = QDesktopWidget().screenGeometry()
		self.resize(int(9 / 10 * screenShape.width()), int(9 / 10 * screenShape.height()))

		self.layoutGUI()

	# lay out the GUI
	def layoutGUI(self):

		# working directory label
		self.cwd_label = QLabel(self)

		# the chart and table widgets
		self.myChartWidget = MyChartWidget(self)
		self.myTableWidget = MyTableWidget(self)

		# create a vertical layout, and add path and charts
		self.vlayout1 = QVBoxLayout()
		self.vlayout1.addWidget(self.cwd_label)
		self.vlayout1.addWidget(self.myChartWidget)

		# create another vertical layout, and add tables
		self.vlayout2 = QVBoxLayout()
		self.vlayout2.addWidget(self.myTableWidget)

		# create a horizontal layout, and add the vertical layouts
		self.hlayout = QHBoxLayout()
		self.hlayout.addLayout(self.vlayout1)
		self.hlayout.addLayout(self.vlayout2)

		# create a widget to contain the layout, and set as central widget
		self.main_window = QWidget()
		self.main_window.setLayout(self.hlayout)
		self.setCentralWidget(self.main_window)

		# set up the menus
		self.menuSystem()

		# activate the status bar
		self.statusBar = QStatusBar()
		self.setStatusBar(self.statusBar)
		self.statusBar.showMessage('Ready')

		# progress bar
		self.progress = QProgressBar()
		self.statusBar.insertPermanentWidget(0, self.progress)
		self.progress.setFixedWidth(250)
		self.progress.setVisible(False)

	# define menus
	def menuSystem(self):
		self.mainMenu = self.menuBar()

		if platform() == 'Darwin':  # How Mac OS X is identified by Python
			self.mainMenu.setNativeMenuBar(False)

		self.fileMenu = self.mainMenu.addMenu('File')
		self.editMenu = self.mainMenu.addMenu('Clean')
		self.featuresMenu = self.mainMenu.addMenu('Features')
		self.classifierMenu = self.mainMenu.addMenu('Classifier')
		self.clusterMenu = self.mainMenu.addMenu('Cluster')
		self.viewMenu = self.mainMenu.addMenu('View')

		# certain menus are initially disabled
		self.editMenu.setEnabled(False)
		self.featuresMenu.setEnabled(False)
		self.classifierMenu.setEnabled(False)
		self.viewMenu.setEnabled(False)

		# File menu
		# =========
		self.actionLoadDirectory = QAction('Load WAV Directory...', self)
		self.actionLoadDirectory.triggered.connect(self.loadDirectory)
		self.fileMenu.addAction(self.actionLoadDirectory)

		wavScaleMenu = self.fileMenu.addMenu('Set WAV Scale Factor')
		wav_scale_factors = ['Invert', 'Normalize', '1', '2', '5', '10', '20', '50', '100']
		wav_scale_action = []
		for i, value in enumerate(wav_scale_factors):
			wav_scale_action.append(wavScaleMenu.addAction(value))
			wav_scale_action[i].triggered.connect(partial(self.setWavScaleFactor, value))

		self.fileMenu.addSeparator()

		self.copy_clusterButton = QAction('Copy WAV Files to New Directory...', self)
		self.copy_clusterButton.triggered.connect(self.copy_cluster)
		self.fileMenu.addAction(self.copy_clusterButton)
		self.copy_clusterButton.setEnabled(False)

		self.write_csvButton = QAction('Write Data to CSV', self)
		self.write_csvButton.triggered.connect(self.write_csv)
		self.fileMenu.addAction(self.write_csvButton)
		self.write_csvButton.setEnabled(False)

		self.save_featuresButton = QAction('Save Feature Data', self)
		self.save_featuresButton.triggered.connect(self.save_features)
		self.fileMenu.addAction(self.save_featuresButton)
		self.save_featuresButton.setEnabled(False)

		self.save_CWTButton = QAction('Save CWT Images...', self)
		self.save_CWTButton.triggered.connect(self.save_cwt)
		self.fileMenu.addAction(self.save_CWTButton)
		self.save_CWTButton.setEnabled(False)

		self.fileMenu.addSeparator()

		closeButton = QAction('Close', self)
		closeButton.triggered.connect(self.close)
		self.fileMenu.addAction(closeButton)

		# Clean menu
		# ==========
		delLowS2NMenu = self.editMenu.addMenu('Delete Low s/n')
		low_S2N_range = [5, 10, 15, 20, 50, 100]
		low_S2N_action = []
		for i, s2n in enumerate(low_S2N_range):
			low_S2N_action.append(delLowS2NMenu.addAction('<' + str(s2n)))
			low_S2N_action[i].triggered.connect(partial(self.delLowS2N, s2n))

		delLowTriggerMenu = self.editMenu.addMenu('Delete Low Power')
		low_trigger_range = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10]
		low_trigger_action = []
		for i, trigger in enumerate(low_trigger_range):
			low_trigger_action.append(delLowTriggerMenu.addAction('<' + str(trigger) + ' (x1000)'))
			low_trigger_action[i].triggered.connect(partial(self.delLowTriggers, trigger))

		delLFTriggerMenu = self.editMenu.addMenu('Delete Low-Frequency Power')
		lf_trigger_ratios = [1, 1.5, 2, 2.5, 3]
		lf_trigger_action = []
		for i, ratio in enumerate(lf_trigger_ratios):
			lf_trigger_action.append(delLFTriggerMenu.addAction('Ratio <' + str(ratio)))
			lf_trigger_action[i].triggered.connect(partial(self.delLFTriggers, ratio))

		delLowHScoreMenu = self.editMenu.addMenu('Delete Low Harmonic Count')
		low_hscore_range = np.arange(1, 6)
		low_hscore_action = []
		for i, hscore in enumerate(low_hscore_range):
			low_hscore_action.append(delLowHScoreMenu.addAction('<' + '{:}'.format(hscore)))
			low_hscore_action[i].triggered.connect(partial(self.delLowHCount, hscore))

		# Features menu
		# =============
		self.usePSDButton = QAction('Use PSD', self)
		self.usePSDButton.setCheckable(True)
		self.usePSDButton.setChecked(False)
		self.usePSDButton.setEnabled(True)
		self.usePSDButton.triggered.connect(self.usePSD)
		self.featuresMenu.addAction(self.usePSDButton)

		self.useMelButton = QAction('Use MFCC', self)
		self.useMelButton.setCheckable(True)
		self.useMelButton.setChecked(True)
		self.useMelButton.setEnabled(False)
		self.useMelButton.triggered.connect(self.useMel)
		self.featuresMenu.addAction(self.useMelButton)

		self.useDWTButton = QAction('Use DWT', self)
		self.useDWTButton.setCheckable(True)
		self.useDWTButton.setChecked(False)
		self.useDWTButton.setEnabled(True)
		self.useDWTButton.triggered.connect(self.useDWT)
		self.featuresMenu.addAction(self.useDWTButton)

		self.DWTMenu = self.featuresMenu.addMenu('Apply Wavelet')
		self.wavelets = pywt.wavelist(kind='discrete')
		self.wavelet_actions = []
		self.wavelet = None
		for i, wavelet in enumerate(self.wavelets):
			self.wavelet_actions.append(self.DWTMenu.addAction(wavelet))
			self.wavelet_actions[i].triggered.connect(partial(self.applyDWT, wavelet))
		self.DWTMenu.setEnabled(False)

		self.useF1Button = QAction('Use Fundamental Frequency', self)
		self.useF1Button.setCheckable(True)
		self.useF1Button.setChecked(False)
		self.useF1Button.setEnabled(True)
		self.useF1Button.triggered.connect(self.useF1)
		self.featuresMenu.addAction(self.useF1Button)

		self.useRawButton = QAction('Use Raw Data', self)
		self.useRawButton.setCheckable(True)
		self.useRawButton.setChecked(False)
		self.useRawButton.setEnabled(True)
		self.useRawButton.triggered.connect(self.useRaw)
		self.featuresMenu.addAction(self.useRawButton)

		self.featuresMenu.addSeparator()

		self.applyRBMButton = QAction('Apply Bernoulli RBM', self)
		self.applyRBMButton.setCheckable(True)
		self.applyRBMButton.setChecked(False)
		self.applyRBMButton.triggered.connect(self.applyRBM)
		self.featuresMenu.addAction(self.applyRBMButton)

		self.applyPCAButton = QAction('Apply PCA', self)
		self.applyPCAButton.setCheckable(True)
		self.applyPCAButton.setChecked(False)
		self.applyPCAButton.triggered.connect(self.applyPCA)
		self.featuresMenu.addAction(self.applyPCAButton)

		self.featuresMenu.addSeparator()

		self.includeTButton = QAction('Include Temperature', self)
		self.includeTButton.setCheckable(True)
		self.includeTButton.setChecked(False)
		self.includeTButton.triggered.connect(self.includeTemp)
		self.featuresMenu.addAction(self.includeTButton)

		self.includeSPButton = QAction('Include Signal Power', self)
		self.includeSPButton.setCheckable(True)
		self.includeSPButton.setChecked(False)
		self.includeSPButton.triggered.connect(self.includeSP)
		self.featuresMenu.addAction(self.includeSPButton)

		self.featuresMenu.addSeparator()

		self.scaler = global_scalers[0]
		self.scalerMenu = self.featuresMenu.addMenu('Apply Scaler: ' + self.scaler)
		self.scaler_actions = []
		for i, scaler in enumerate(global_scalers):
			self.scaler_actions.append(self.scalerMenu.addAction(scaler))
			if scaler == self.scaler:
				self.scaler_actions[i].setChecked(True)
			self.scaler_actions[i].triggered.connect(partial(self.applyScaler, scaler))


		# Classifier menu
		# ===============
		self.train_xgboost_button = QAction('Train: XGBoost', self)
		self.train_xgboost_button.triggered.connect(self.train_xgboost)
		self.classifierMenu.addAction(self.train_xgboost_button)

		self.train_mlp_button = QAction('Train: MLP (Neural Net)', self)
		self.train_mlp_button.triggered.connect(self.train_mlp)
		self.classifierMenu.addAction(self.train_mlp_button)

		self.classifierMenu.addSeparator()

		self.modelSaveButton = QAction('Save Model', self)
		self.modelSaveButton.triggered.connect(self.saveModel)
		self.modelSaveButton.setEnabled(False)
		self.classifierMenu.addAction(self.modelSaveButton)

		self.modelLoadButton = QAction('Load Model...', self)
		self.modelLoadButton.triggered.connect(self.loadModel)
		self.classifierMenu.addAction(self.modelLoadButton)

		# Cluster menu
		# ============
		self.clusterPredictMenu = self.clusterMenu.addMenu('Predict: No Model')
		clusterPredict_mp = ['Normal', 'p≥0.6', 'p≥0.7', 'p≥0.8', 'p≥0.9']
		clusterPredict_action = []
		for i, mp in enumerate(clusterPredict_mp):
			clusterPredict_action.append(self.clusterPredictMenu.addAction(mp))
			clusterPredict_action[i].triggered.connect(partial(self.clusterPredict, mp))
		self.clusterPredictMenu.setEnabled(False)

		self.clusterMenu.addSeparator()

		clusterGMMenu = self.clusterMenu.addMenu('Bayesian Gaussian Mixture')
		clusterGM_n = [2, 3, 4, 5]
		clusterGM_action = []
		for i, n in enumerate(clusterGM_n):
			clusterGM_action.append(clusterGMMenu.addAction('N=' + str(n)))
			clusterGM_action[i].triggered.connect(partial(self.clusterGM, n))

		# clusterCKMeansMenu = self.transformMenu.addMenu('Clustering: CKMeans (f1 only)')
		# clusterCKMeans_n = [2, 3, 4, 5]
		# clusterCKMeans_action = []
		# for i, n in enumerate(clusterCKMeans_n):
		# 	clusterCKMeans_action.append(clusterCKMeansMenu.addAction('N=' + str(n)))
		# 	clusterCKMeans_action[i].triggered.connect(partial(self.clusterCKMeans, n))

		clusterSpectralMenu = self.clusterMenu.addMenu('Spectral')
		clusterSpectral_n = [2, 3, 4, 5]
		clusterSpectral_action = []
		for i, n in enumerate(clusterSpectral_n):
			clusterSpectral_action.append(clusterSpectralMenu.addAction('N=' + str(n)))
			clusterSpectral_action[i].triggered.connect(partial(self.clusterSpectral, n))

		clusterKMeansMenu = self.clusterMenu.addMenu('KMeans')
		clusterKMeans_n = [2, 3, 4, 5]
		clusterKMeans_action = []
		for i, n in enumerate(clusterKMeans_n):
			clusterKMeans_action.append(clusterKMeansMenu.addAction('N=' + str(n)))
			clusterKMeans_action[i].triggered.connect(partial(self.clusterKMeans, n))

		self.clusterMenu.addSeparator()

		DBScanMenu = self.clusterMenu.addMenu('DBScan')
		eps_range = [0.01, 0.1, 1, 10]
		eps_multiplier = [1, 2, 3, 4, 5, 6, 7, 8, 9]
		epsAction = []
		epsRangeMenu = []
		for i, range in enumerate(eps_range):
			epsRangeMenu.append(DBScanMenu.addMenu('Range: ' + str(range)))
			for j, multiplier in enumerate(eps_multiplier):
				epsAction.append(epsRangeMenu[i].addAction('ɛ=' + '{:0.2f}'.format(range * multiplier)))
				epsAction[i * 9 + j].triggered.connect(partial(self.clusterDBScan, range, multiplier))

		epsAdjUpButton = QAction('Increase ɛ by 10% of range', self)
		epsAdjUpButton.setShortcut('Ctrl++')
		epsAdjUpButton.triggered.connect(partial(self.clusterDBScanEpsAdj, 1))
		self.clusterMenu.addAction(epsAdjUpButton)

		epsAdjUpButton2 = QAction('Increase ɛ by 1% of range', self)
		epsAdjUpButton2.setShortcut('alt++')
		epsAdjUpButton2.triggered.connect(partial(self.clusterDBScanEpsAdj, 0.1))
		self.clusterMenu.addAction(epsAdjUpButton2)

		epsAdjDownButton2 = QAction('Decrease ɛ by 1% of range', self)
		epsAdjDownButton2.setShortcut('alt+-')
		epsAdjDownButton2.triggered.connect(partial(self.clusterDBScanEpsAdj, -0.1))
		self.clusterMenu.addAction(epsAdjDownButton2)

		epsAdjDownButton = QAction('Decrease ɛ by 10% of range', self)
		epsAdjDownButton.setShortcut('Ctrl+-')
		epsAdjDownButton.triggered.connect(partial(self.clusterDBScanEpsAdj, -1))
		self.clusterMenu.addAction(epsAdjDownButton)

		self.clusterMenu.addSeparator()

		clusterOriginal = QAction('Reset (File Group)', self)
		clusterOriginal.triggered.connect(self.clusterDefault)
		self.clusterMenu.addAction(clusterOriginal)

		# View menu
		# =========
		self.s2n_dist_Button = QAction('s/n Distribution', self)
		self.s2n_dist_Button.setCheckable(True)
		self.s2n_dist_Button.setChecked(True)
		self.s2n_dist_Button.setEnabled(False)
		self.s2n_dist_Button.triggered.connect(self.view_s2n_dist)
		self.viewMenu.addAction(self.s2n_dist_Button)

		self.power_dist_Button = QAction('Power Distribution', self)
		self.power_dist_Button.setCheckable(True)
		self.power_dist_Button.setChecked(False)
		self.power_dist_Button.setEnabled(True)
		self.power_dist_Button.triggered.connect(self.view_power_dist)
		self.viewMenu.addAction(self.power_dist_Button)

		self.f1_vs_T_Button = QAction('Temperature Effect', self)
		self.f1_vs_T_Button.setCheckable(True)
		self.f1_vs_T_Button.setChecked(False)
		self.f1_vs_T_Button.setEnabled(True)
		self.f1_vs_T_Button.triggered.connect(self.view_f1_vs_T)
		self.viewMenu.addAction(self.f1_vs_T_Button)

		self.f1_vs_RH_Button = QAction('Humidity Effect', self)
		self.f1_vs_RH_Button.setCheckable(True)
		self.f1_vs_RH_Button.setChecked(False)
		self.f1_vs_RH_Button.setEnabled(True)
		self.f1_vs_RH_Button.triggered.connect(self.view_f1_vs_RH)
		self.viewMenu.addAction(self.f1_vs_RH_Button)

		self.TSNEButton = QAction('Projection (t-SNE)', self)
		self.TSNEButton.setCheckable(True)
		self.TSNEButton.setChecked(False)
		self.TSNEButton.setEnabled(True)
		self.TSNEButton.triggered.connect(self.view_TSNE)
		self.viewMenu.addAction(self.TSNEButton)

		self.confusionMatrixButton = QAction('Confusion Matrix', self)
		self.confusionMatrixButton.triggered.connect(self.view_confusionMatrix)
		self.confusionMatrixButton.setCheckable(True)
		self.confusionMatrixButton.setChecked(False)
		self.confusionMatrixButton.setEnabled(False)
		self.viewMenu.addAction(self.confusionMatrixButton)

		self.viewMenu.addSeparator()

		self.originalSort = QAction('PSD Map: Original File Order', self)
		self.originalSort.setShortcut('F1')
		self.originalSort.setStatusTip('Sort by Original File Order')
		self.originalSort.setCheckable(True)
		self.originalSort.setChecked(True)
		self.originalSort.setEnabled(False)
		self.originalSort.triggered.connect(self.setOrderOriginal)
		self.viewMenu.addAction(self.originalSort)

		self.by_S2NSort = QAction('PSD Map: s/n', self)
		self.by_S2NSort.setShortcut('F2')
		self.by_S2NSort.setStatusTip('Sort by Signal-to-Noise Ratio')
		self.by_S2NSort.setCheckable(True)
		self.by_S2NSort.setChecked(False)
		self.by_S2NSort.setEnabled(True)
		self.by_S2NSort.triggered.connect(self.setOrderByS2N)
		self.viewMenu.addAction(self.by_S2NSort)

		self.by_FreqSort = QAction('PSD Map: Frequency', self)
		self.by_FreqSort.setShortcut('F3')
		self.by_FreqSort.setStatusTip('Sort by Frequency')
		self.by_FreqSort.setCheckable(True)
		self.by_FreqSort.setChecked(False)
		self.by_FreqSort.setEnabled(True)
		self.by_FreqSort.triggered.connect(self.setOrderByFreq)
		self.viewMenu.addAction(self.by_FreqSort)

		self.by_ClusterSort = QAction('PSD Map: Cluster', self)
		self.by_ClusterSort.setShortcut('F4')
		self.by_ClusterSort.setStatusTip('Sort by Cluster')
		self.by_ClusterSort.setCheckable(True)
		self.by_ClusterSort.setChecked(False)
		self.by_ClusterSort.setEnabled(True)
		self.by_ClusterSort.triggered.connect(self.setOrderByCluster)
		self.viewMenu.addAction(self.by_ClusterSort)

		self.tempButton = QAction('Temperature Overlay', self)
		self.tempButton.setShortcut('F3')
		self.tempButton.setStatusTip('Show Temperature Profile')
		self.tempButton.setCheckable(True)
		self.tempButton.setChecked(False)
		self.tempButton.setEnabled(True)
		self.tempButton.triggered.connect(self.showTemp)
		self.viewMenu.addAction(self.tempButton)

		self.viewMenu.addSeparator()

		self.CWTMenu = self.viewMenu.addMenu('Continuous Wavelet Transform')
		self.cwavelet_actions = []
		for i, wavelet in enumerate(cwt_wavelets):
			self.cwavelet_actions.append(self.CWTMenu.addAction(wavelet))
			self.cwavelet_actions[i].triggered.connect(partial(self.viewCWT, wavelet))
		self.cwavelet = cwt_default_cwavelet
		self.CWTMenu.setEnabled(False)

	# ask for and load working directory
	# (returns empty string on cancel)
	def loadDirectory(self):

		cwd = QFileDialog.getExistingDirectory(
			None,
			"Select Folder",
			self.cwd
		)
		if cwd == '':  # return if no directory was selected
			return

		self.cwd = cwd  # set the new working directory
		self.loadData()  # load the data, and update
		self.activateWindow()

	def loadData(self):
		self.statusBar.showMessage('Loading: ' + self.cwd)
		QApplication.processEvents()  # this will update the status bar

		if not self.collection.getData(self.cwd):  # returns false if there are no wav files
			ex.statusBar.showMessage('No wav data found!')
			QApplication.processEvents()  # this will update the status bar
			return

		# we have data!
		self.features = Features()          # initialize Features object
		self.cluster = Cluster()            # create cluster object
		self.save_featuresButton.setEnabled(True)
		self.save_CWTButton.setEnabled(True)
		self.myChartWidget.clear_overview() # clear any previous data
		self.myChartWidget.clear_view()
		self.view_resetButtons()            # view s/n distribution
		self.s2n_dist_Button.setChecked(True)
		self.s2n_dist_Button.setEnabled(False)
		self.enableTrain()                  # enable training (if file cluster)
		self.resetOrderButtons()            # show original file order
		self.write_csvButton.setEnabled(True)
		self.CWTMenu.setEnabled(True)
		self.updateAll()                    # update the data display
		self.statusBar.showMessage('Ready')
		QApplication.processEvents()  # this will update the status bar

	# set scale factor for wav files
	def setWavScaleFactor(self, value):
		global wavScaleFactor
		if value[0] == 'I':
			value = -1
		elif value[0] == 'N':
			value = int(0)
		else:
			value = int(value)
		wavScaleFactor = value



	# export results to csv file
	def write_csv(self):
		ex.collection.write_csv()

	# save features to file
	def save_features(self):
		ex.features.save()

	# save cwt images
	def save_cwt(self):
		# ask for directory
		# (returns empty string on cancel)
		cwd = QFileDialog.getExistingDirectory(
			None,
			"Select Folder for CWT Image Files",
			self.cwd
		)
		if cwd == '':  # return if no directory was selected
			return

		self.statusBar.showMessage('Performing CWT and saving images...')
		QApplication.processEvents()  # this will update the status bar

		self.progress.setMaximum(self.collection.count)
		self.progress.setValue(0)
		self.progress.setVisible(True)
		progress = 0

		# create the subdirectories
		dst_dir_list = []
		for label in self.cluster.group_labels:
			path = os.path.join(cwd, label)
			if os.path.isdir(path):
				shutil.rmtree(path)
			os.mkdir(path)
			dst_dir_list.append(path)

		for i, recording in enumerate(ex.collection.reclist):
			index = self.cluster.class_labels[i]
			dst = os.path.join(dst_dir_list[index], recording.displayname + '.jpg')
			[power, frequencies] = getCWT(recording)
			slope = 255 /(cwt_log_max - cwt_log_min)
			offset = cwt_log_min
			image = (power - offset) * slope
			image[image>255] = 255
			image[image<0] = 0
			image = (image + 0.5).astype(int)
			cv2.imwrite(dst, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
			# update progress bar
			progress += 1
			if progress % 10 == 0:
				self.progress.setValue(progress)
				QApplication.processEvents()

		# hide the progress bar and update the status bar
		ex.progress.setVisible(False)
		self.statusBar.showMessage('Ready')
		QApplication.processEvents()  # this will update the status bar

	# copy files based on clustering results
	def copy_cluster(self):
		# ask for directory
		# (returns empty string on cancel)
		cwd = QFileDialog.getExistingDirectory(
			None,
			"Select Folder for Clustered WAV Files",
			self.cwd
		)
		if cwd == '':  # return if no directory was selected
			return

		self.statusBar.showMessage('Copying WAV files...')
		QApplication.processEvents()  # this will update the status bar

		# create the subdirectories
		dst_dir_list = []
		for label in self.cluster.group_labels:
			path = os.path.join(cwd, label)
			if os.path.isdir(path):
				shutil.rmtree(path)
			os.mkdir(path)
			dst_dir_list.append(path)

		# copy the files
		for i, recording in enumerate(ex.collection.reclist):
			src = recording.filespec
			index = self.cluster.class_labels[i]
			dst = os.path.join(dst_dir_list[index], recording.filename)
			shutil.copyfile(src, dst)

		self.statusBar.showMessage('Ready')
		QApplication.processEvents()  # this will update the status bar

	# delete low s/n
	def delLowS2N(self, s2n):
		self.statusBar.showMessage('Deleting low S/N recordings...')
		QApplication.processEvents()  # this will update the status bar
		self.collection.setOrder(self.collection.by_cluster)
		self.collection.delLowS2N(s2n)
		self.myChartWidget.updateAll()

	# delete false triggers
	def delLowTriggers(self, min_power):
		self.statusBar.showMessage('Deleting low signal recordings...')
		QApplication.processEvents()  # this will update the status bar
		self.collection.setOrder(self.collection.by_cluster)
		self.collection.delFalseTriggers(min_power / 1000)
		self.myChartWidget.updateAll()

	# delete LF triggers
	def delLFTriggers(self, lf_ratio):
		self.statusBar.showMessage('Deleting low frequency recordings...')
		QApplication.processEvents()  # this will update the status bar
		self.collection.setOrder(self.collection.by_cluster)
		self.collection.delLFTriggers(lf_ratio)
		self.myChartWidget.updateAll()  # update the GUI and data display

	def delLowHCount(self, hscore):
		self.statusBar.showMessage('Deleting low harmonic count recordings...')
		QApplication.processEvents()  # this will update the status bar
		self.collection.setOrder(self.collection.by_cluster)
		self.collection.delPoorHarmonics(hscore)
		self.myChartWidget.updateAll()  # update the GUI and data display

	# use PSD
	def usePSD(self):
		self.features_resetButtons()
		self.usePSDButton.setChecked(True)
		self.usePSDButton.setEnabled(False)
		self.setFeatureOptions()
		self.update()
		self.myChartWidget.updateClusterDisplay()

	# use Mel
	def useMel(self):
		self.features_resetButtons()
		self.useMelButton.setChecked(True)
		self.useMelButton.setEnabled(False)
		self.setFeatureOptions()
		self.update()
		self.myChartWidget.updateClusterDisplay()

	# use DWT
	def useDWT(self):
		self.features_resetButtons()
		self.useDWTButton.setChecked(True)
		self.useDWTButton.setEnabled(False)
		self.DWTMenu.setEnabled(True)

	# select wavelet
	def applyDWT(self, wavelet):
		self.wavelet = wavelet
		self.DWTMenu.setTitle('Apply DWT: ' + wavelet)
		self.setFeatureOptions()
		self.update()
		self.myChartWidget.updateClusterDisplay()

	# use f1
	def useF1(self):
		self.features_resetButtons()
		self.useF1Button.setChecked(True)
		self.useF1Button.setEnabled(False)
		self.setFeatureOptions()
		self.update()
		self.myChartWidget.updateClusterDisplay()

	# use raw
	def useRaw(self):
		self.features_resetButtons()
		self.useRawButton.setChecked(True)
		self.useRawButton.setEnabled(False)
		self.setFeatureOptions()
		self.update()
		self.myChartWidget.updateClusterDisplay()

	def features_resetButtons(self):
		self.usePSDButton.setChecked(False)
		self.usePSDButton.setEnabled(True)
		self.useMelButton.setChecked(False)
		self.useMelButton.setEnabled(True)
		self.useDWTButton.setChecked(False)
		self.useDWTButton.setEnabled(True)
		self.DWTMenu.setEnabled(False)
		self.useF1Button.setChecked(False)
		self.useF1Button.setEnabled(True)
		self.useRawButton.setChecked(False)
		self.useRawButton.setEnabled(True)

	# include temperature
	def includeTemp(self):
		self.setFeatureOptions()
		self.update()
		self.myChartWidget.updateClusterDisplay()

	# include signal power
	def includeSP(self):
		self.setFeatureOptions()
		self.update()
		self.myChartWidget.updateClusterDisplay()

	# transformation: RBM
	def applyRBM(self):
		self.applyPCAButton.setChecked(False)
		self.setFeatureOptions()
		self.update()
		self.myChartWidget.updateClusterDisplay()

	# transformation: PCA
	def applyPCA(self):
		self.applyRBMButton.setChecked(False)
		self.setFeatureOptions()
		self.update()
		self.myChartWidget.updateClusterDisplay()

	# apply scaler
	def applyScaler(self, scaler):
		self.scaler = scaler
		self.scalerMenu.setTitle('Apply Scaler: ' + scaler)
		self.setFeatureOptions()
		self.update()
		self.myChartWidget.updateClusterDisplay()

	# features options
	def setFeatureOptions(self):
		ex.features.setOptions(
			self.usePSDButton.isChecked(),
			self.useMelButton.isChecked(),
			self.useDWTButton.isChecked(),
			self.wavelet,
			self.useF1Button.isChecked(),
			self.useRawButton.isChecked(),
			self.includeTButton.isChecked(),
			self.includeSPButton.isChecked(),
			self.applyRBMButton.isChecked(),
			self.applyPCAButton.isChecked(),
			self.scaler
		)
		ex.features.set()
		self.checkModel()

	# enable train function
	def enableTrain(self):
		self.train_xgboost_button.setEnabled(False)
		self.train_mlp_button.setEnabled(False)
		self.copy_clusterButton.setEnabled(False)
		if self.cluster.n_clusters > 1:
			self.train_xgboost_button.setEnabled(True)
			self.train_mlp_button.setEnabled(True)
			self.copy_clusterButton.setEnabled(True)

	# train the respective model, and show the confusion matrix
	def train_xgboost(self):
		self.model = XGBoost()
		self.model.train()
		self.activateModel()

	def train_mlp(self):
		self.model = MLP()
		self.model.train()
		self.activateModel()

	# enable appropriate buttons and view confusion matrix
	def activateModel(self):
		self.modelSaveButton.setEnabled(True)
		self.clusterPredictMenu.setTitle('Predict: ' + self.model.name + ' (' + self.model.descriptor + ')')
		self.checkModel()
		self.view_confusionMatrix()
		self.confusionMatrixButton.setEnabled(True)

	# check the model and enable menu choice,
	# if there is a trained model and the model matches the features
	def checkModel(self):
		self.clusterPredictMenu.setEnabled(False)
		try:
			if self.model.descriptor == self.features.descriptor and self.model.trained:
				self.clusterPredictMenu.setEnabled(True)
		except:
			pass

	# save the model
	def saveModel(self):
		self.model.save(self.cwd)

	# load the model
	def loadModel(self):
		# ask for the model file
		# returns false on cancel
		file_path, _ = QFileDialog.getOpenFileName(
			None,
			"Select File",
			self.cwd,
			"Wingbeat Analyzer Model (*.wam)"
		)
		if file_path == '': return  # cancel was pressed

		self.model.load(file_path)
		self.activateModel()


	# clustering: none
	def clusterDefault(self):
		self.cluster = Cluster()
		self.cluster.update()
		self.setOrderByCluster()
		self.myChartWidget.updateClusterDisplay()
		self.enableTrain()

	# Clustering: trained model
	def clusterPredict(self, mp):
		if mp == 'Normal':
			mp = 0
		else:
			mp = float(mp[2:])
		self.cluster = Predict(self.model, mp)
		self.cluster.update()
		self.setOrderByCluster()
		self.myChartWidget.updateClusterDisplay()
		self.enableTrain()

	# clustering: Bayesian Gaussian Mixture
	def clusterGM(self, n_clusters):
		self.cluster = Gaussian(n_clusters)
		self.cluster.update()
		self.setOrderByCluster()
		self.myChartWidget.updateClusterDisplay()
		self.enableTrain()

	# clustering: CKMeans (f1 only)
	def clusterCKMeans(self, n_clusters):
		self.cluster = CKMeans(n_clusters)
		self.useF1Button.setChecked(True)
		self.cluster.update()
		self.setOrderByCluster()
		self.myChartWidget.updateClusterDisplay()
		self.enableTrain()

	# clustering: KMeans
	def clusterKMeans(self, n_clusters):
		self.cluster = myKMeans(n_clusters)
		self.cluster.update()
		self.setOrderByCluster()
		self.myChartWidget.updateClusterDisplay()
		self.enableTrain()

	# clustering: Spectral
	def clusterSpectral(self, n_clusters):
		self.cluster = Spectral(n_clusters)
		self.cluster.update()
		self.setOrderByCluster()
		self.myChartWidget.updateClusterDisplay()
		self.enableTrain()

	# clustering: DBScan
	def clusterDBScan(self, range, multiplier):
		self.range = range
		self.eps = range * multiplier
		self.adj = 0
		self.cluster = DBScan(self.eps)
		self.cluster.update()
		self.setOrderByCluster()
		self.myChartWidget.updateClusterDisplay()
		self.enableTrain()

	def clusterDBScanEpsAdj(self, adj):
		if self.eps == 0: return
		self.adj = self.range / 10 * adj
		self.eps += self.adj
		if self.eps <= 0: self.eps = abs(self.adj)
		self.cluster = DBScan(self.eps)
		self.cluster.update()
		self.setOrderByCluster()
		self.myChartWidget.updateClusterDisplay()
		self.enableTrain()

	# view transformation: t-SNE
	def view_TSNE(self):
		self.view_resetButtons()
		self.TSNEButton.setChecked(True)
		self.TSNEButton.setEnabled(False)
		self.myChartWidget.clear_view()
		self.tsne.update()
		self.myChartWidget.update_TSNE()
		self.myChartWidget.commit_changes()

	# view confusion matrix
	def view_confusionMatrix(self):
		self.view_resetButtons()
		self.confusionMatrixButton.setChecked(True)
		self.confusionMatrixButton.setEnabled(False)
		self.myChartWidget.clear_view()
		self.myChartWidget.update_ConfusionMatrix()
		self.myChartWidget.commit_changes()

	# view s/n histogram
	def view_s2n_dist(self):
		self.view_resetButtons()
		self.s2n_dist_Button.setChecked(True)
		self.s2n_dist_Button.setEnabled(False)
		self.myChartWidget.clear_view()
		self.myChartWidget.update_s2n_dist()
		self.myChartWidget.commit_changes()

	# view power histogram
	def view_power_dist(self):
		self.view_resetButtons()
		self.power_dist_Button.setChecked(True)
		self.power_dist_Button.setEnabled(False)
		self.myChartWidget.clear_view()
		self.myChartWidget.update_power_dist()
		self.myChartWidget.commit_changes()

	# view f1 versus temperature
	def view_f1_vs_T(self):
		self.view_resetButtons()
		self.f1_vs_T_Button.setChecked(True)
		self.f1_vs_T_Button.setEnabled(False)
		self.myChartWidget.clear_view()
		self.myChartWidget.update_f1_vs_T()
		self.myChartWidget.commit_changes()

	# view f1 versus humidity
	def view_f1_vs_RH(self):
		self.view_resetButtons()
		self.f1_vs_RH_Button.setChecked(True)
		self.f1_vs_RH_Button.setEnabled(False)
		self.myChartWidget.clear_view()
		self.myChartWidget.update_f1_vs_RH()
		self.myChartWidget.commit_changes()

	def viewCWT(self, wavelet):
		self.cwavelet = wavelet
		self.myChartWidget.updateRecording()

	def view_resetButtons(self):
		self.s2n_dist_Button.setChecked(False)
		self.s2n_dist_Button.setEnabled(True)
		self.power_dist_Button.setChecked(False)
		self.power_dist_Button.setEnabled(True)
		self.f1_vs_T_Button.setChecked(False)
		self.f1_vs_T_Button.setEnabled(True)
		self.f1_vs_RH_Button.setChecked(False)
		self.f1_vs_RH_Button.setEnabled(True)
		self.TSNEButton.setChecked(False)
		self.TSNEButton.setEnabled(True)
		self.confusionMatrixButton.setChecked(False)
		if self.clusterPredictMenu.isEnabled():
			self.confusionMatrixButton.setEnabled(True)

	def resetOrderButtons(self):
		self.originalSort.setChecked(True)
		self.originalSort.setEnabled(False)
		self.by_S2NSort.setChecked(False)
		self.by_S2NSort.setEnabled(True)
		self.by_FreqSort.setChecked(False)
		self.by_FreqSort.setEnabled(True)
		self.by_ClusterSort.setChecked(False)
		self.by_ClusterSort.setEnabled(True)
		self.tempButton.setChecked(False)
		self.tempButton.setEnabled(True)
		self.collection.cursor_reset()

	# sort by cluster
	def setOrderByCluster(self):
		self.originalSort.setChecked(False)
		self.originalSort.setEnabled(True)
		self.by_S2NSort.setChecked(False)
		self.by_S2NSort.setEnabled(True)
		self.by_FreqSort.setChecked(False)
		self.by_FreqSort.setEnabled(True)
		self.by_ClusterSort.setChecked(True)
		self.by_ClusterSort.setEnabled(False)
		self.tempButton.setChecked(False)
		self.tempButton.setEnabled(False)
		self.myChartWidget.setOrderByCluster()

	# sort by signal-to-noise
	def setOrderByS2N(self):
		self.originalSort.setChecked(False)
		self.originalSort.setEnabled(True)
		self.by_S2NSort.setChecked(True)
		self.by_S2NSort.setEnabled(False)
		self.by_FreqSort.setChecked(False)
		self.by_FreqSort.setEnabled(True)
		self.by_ClusterSort.setChecked(False)
		self.by_ClusterSort.setEnabled(True)
		self.tempButton.setChecked(False)
		self.tempButton.setEnabled(False)
		self.myChartWidget.setOrderByS2N()

	# sort by frequency
	def setOrderByFreq(self):
		self.originalSort.setChecked(False)
		self.originalSort.setEnabled(True)
		self.by_S2NSort.setChecked(False)
		self.by_S2NSort.setEnabled(True)
		self.by_FreqSort.setChecked(True)
		self.by_FreqSort.setEnabled(False)
		self.by_ClusterSort.setChecked(False)
		self.by_ClusterSort.setEnabled(True)
		self.tempButton.setChecked(False)
		self.tempButton.setEnabled(False)
		self.myChartWidget.setOrderByFreq()

	# set original sort order
	def setOrderOriginal(self):
		self.originalSort.setChecked(True)
		self.originalSort.setEnabled(False)
		self.by_S2NSort.setChecked(False)
		self.by_S2NSort.setEnabled(True)
		self.by_FreqSort.setChecked(False)
		self.by_FreqSort.setEnabled(True)
		self.by_ClusterSort.setChecked(False)
		self.by_ClusterSort.setEnabled(True)
		self.tempButton.setChecked(False)
		self.tempButton.setEnabled(True)
		self.myChartWidget.setOrderOriginal()

	# enable temperature plot
	def showTemp(self):
		self.myChartWidget.showTemp()

	def update(self):
		if not isinstance(self.cluster, Predict):
			self.cluster.update()
		elif self.model.descriptor == self.features.descriptor:
			self.cluster.update()
		if self.TSNEButton.isChecked():
			self.tsne.update()

	def updateAll(self):
		self.cwd_label.setText('Directory: ' + self.cwd)
		self.setFeatureOptions()
		self.update()
		self.myChartWidget.updateAll()
		self.myTableWidget.updateAll()

		# enable previously disabled GUI items
		self.editMenu.setEnabled(True)
		self.featuresMenu.setEnabled(True)
		self.classifierMenu.setEnabled(True)
		self.viewMenu.setEnabled(True)
		self.myChartWidget.setVisible(True)
		self.myTableWidget.setVisible(True)

		# dismiss temporary status bar messages
		self.statusBar.showMessage('Ready')

	def close(self):
		sys.exit()  # this is the widget for the charts

# secondary window to display a figure
class FigureWindow(QMainWindow):
	def __init__(self, figure, title):
		super().__init__()

		# keep the window on top
		self.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)

		# set title and size
		self.setWindowTitle(title)
		# screenShape = QDesktopWidget().screenGeometry()
		# self.resize(1 / 3 * screenShape.width(), 1 / 4 * screenShape.height())

		# create the central widget
		self.cw = QWidget()
		# create the canvas
		self.canvas = FigureCanvas(figure)
		self.canvas.setSizePolicy(QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding)
		self.canvas.setParent(self.cw)
		# create the toolbar
		self.toolbar = NavigationToolbar(self.canvas, self.cw)
		# create a layout for this tab, and add the figure canvas and toolbar
		self.layout = QVBoxLayout()
		self.layout.addWidget(self.canvas)
		self.layout.addWidget(self.toolbar)

		# assign the layout
		self.cw.setLayout(self.layout)

		# now set the central widget
		self.setCentralWidget(self.cw)

		self.show()


class MyChartWidget(QWidget):
	def __init__(self, parent):
		super(QWidget, self).__init__(parent)

		# widget is initially not visible
		self.setVisible(False)

		# layout will be vertical box
		self.layout = QVBoxLayout(self)

		# add the charts
		# set up figure for plotting the PSD map, histogram and a recording
		self.fig = Figure()  # create the figure
		self.gs = GridSpec(5, 8)
		self.ax_sct = self.fig.add_subplot(self.gs[0:2, 0:2])  # the amplitude-frequency scatter plot
		self.ax_xf = self.fig.add_subplot(self.gs[0:2, 2:4])  # the transform scatter plot
		self.ax_hist = self.fig.add_subplot(self.gs[0:2, 4:6])  # the frequency histogram
		self.ax_ts = self.fig.add_subplot(self.gs[0:2, 6:])  # the time series
		self.ax2_ts = self.ax_ts.twinx()    # the frequency overlay
		self.ax_map = self.fig.add_subplot(self.gs[2, 0:])  # the psd map
		self.ax_temp = self.ax_map.twinx()  # the temperature overlay
		self.ax_temp.axis('off')
		self.ax_rec = self.fig.add_subplot(self.gs[3:, 0:3])  # the recording
		self.ax_cwt = self.fig.add_subplot(self.gs[3:, 3:5])  # the recording
		self.ax_psd = self.fig.add_subplot(self.gs[3:, 5:])  # the psd [dB]
		self.ax2_psd = self.ax_psd.twinx()  # second scale for the original psd

		# embed figure and toolbar in widget
		self.canvas = FigureCanvas(self.fig)
		self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
		self.canvas.setFocusPolicy(Qt.StrongFocus)
		self.toolbar = NavigationToolbar(self.canvas, self)


		# Add the widgets to the layout
		self.layout.addWidget(self.canvas)
		self.layout.addWidget(self.toolbar)
		self.setLayout(self.layout)

		# set up events in matplotlib figure
		cid1 = self.canvas.mpl_connect('resize_event', self.mpl_resize)
		cid2 = self.canvas.mpl_connect('pick_event', self.mpl_pick)
		cid3 = self.canvas.mpl_connect('key_press_event', self.mpl_key_press)
		cid3 = self.canvas.mpl_connect('key_release_event', self.mpl_key_release)
		cid4 = self.canvas.mpl_connect('button_press_event', self.mpl_button_press)
		cid5 = self.canvas.mpl_connect('button_release_event', self.mpl_button_release)

		# create the scatter plots
		self.f1_vs_T = ScatterPlot(
			'Temperature Effect',
			'Temperature [°C]',
			'Fundamental Frequency [Hz]',
			True
		)

		self.f1_vs_RH = ScatterPlot(
			'Humidity Effect',
			'Humidity [%]',
			'Fundamental Frequency [Hz]',
			True
		)

		self.tSNE = ScatterPlot(
			't-SNE',
			'x [AU]',
			'y [AU]',
			False
		)

		self.power_vs_f1 = ScatterPlot(
			'Overview',
			'Fundamental Frequency [Hz]',
			'Signal Power [V**2, dB]',
			True
		)

		self.f1_vs_time = ScatterPlot(
			'',
			'',
			'Frequency [Hz]',
			True
		)

	# set sort by cluster
	def setOrderByCluster(self):
		rec = ex.collection.cursor_get_rec()
		ex.collection.setOrder(ex.collection.by_cluster)
		ex.collection.hide_temp_overlay(self.ax_temp)  # hide temperature overlay
		ex.collection.psd_map(self.ax_map)  # plot the map
		ex.collection.cursor_set_pos(ex.collection.cursor_get_pos(rec))
		self.commit_changes()
		# self.updateRecording(rec)

	# set sort by signal-to-noise
	def setOrderByS2N(self):
		rec = ex.collection.cursor_get_rec()
		ex.collection.setOrder(ex.collection.by_s2n)
		ex.collection.hide_temp_overlay(self.ax_temp)  # hide temperature overlay
		ex.collection.psd_map(self.ax_map)  # plot the map
		ex.collection.cursor_set_pos(ex.collection.cursor_get_pos(rec))
		self.commit_changes()
		# self.updateRecording(rec)

	# set sort by frequency
	def setOrderByFreq(self):
		rec = ex.collection.cursor_get_rec()
		ex.collection.setOrder(ex.collection.by_freq)
		ex.collection.hide_temp_overlay(self.ax_temp)  # hide temperature overlay
		ex.collection.psd_map(self.ax_map)  # plot the map
		ex.collection.cursor_set_pos(ex.collection.cursor_get_pos(rec))
		self.commit_changes()
		# self.updateRecording(rec)

	# set original sort order
	def setOrderOriginal(self):
		rec = ex.collection.cursor_get_rec()
		ex.collection.setOrder(ex.collection.by_cluster)
		ex.collection.psd_map(self.ax_map)  # plot the original map
		ex.collection.cursor_set_pos(ex.collection.cursor_get_pos(rec))
		self.commit_changes()
		# self.updateRecording(rec)

	# show temperature overlay
	def showTemp(self):
		if ex.tempButton.isChecked():
			ex.collection.plot_temp_overlay(self.ax_temp)
		else:
			ex.collection.hide_temp_overlay(self.ax_temp)
		self.commit_changes()

	# clear the overview, get ready for new view
	def clear_overview(self):
		self.ax_sct.clear()

	# clear the last view, get ready for new view
	def clear_view(self):
		self.ax_xf.clear()

	# update (draw) confusion matrix
	def update_ConfusionMatrix(self):
		self.view = 'CM'
		ex.model.drawConfusionMatrix(self.ax_xf)
		self.commit_changes()

	# update (draw) s/n histogram
	def update_s2n_dist(self):
		self.view = 'S/N'
		ex.cluster.draw_s2n_dist(self.ax_xf)

	# update (draw) power histogram
	def update_power_dist(self):
		self.view = 'Signal Power'
		ex.cluster.draw_power_dist(self.ax_xf)

	# update (draw) f1 vs T
	def update_f1_vs_T(self):
		self.view = 'f1_vs_T'
		ex.collection.plot_f1_vs_T(self.f1_vs_T, self.ax_xf)
		self.f1_vs_T.show_highlight(self.ax_xf)

	# update (draw) f1 vs RH
	def update_f1_vs_RH(self):
		self.view = 'f1_vs_RH'
		ex.collection.plot_f1_vs_RH(self.f1_vs_RH, self.ax_xf)
		self.f1_vs_RH.show_highlight(self.ax_xf)

	# update (draw) tsne
	def update_TSNE(self):
		self.view = 'TSNE'
		ex.tsne.plot(self.tSNE, self.ax_xf)
		self.tSNE.show_highlight(self.ax_xf)

	# update all plots (after loading data or deleting recordings)
	def updateAll(self):
		# recalculate everything
		ex.update()

		# update all cluster plots
		self.updateClusterDisplay()

		# draw the first recording
		self.updateRecording()

	# update cluster display only (without recalculation)
	def updateClusterDisplay(self):
		# draw cluster items
		ex.collection.plot_power_vs_f1(self.power_vs_f1, self.ax_sct)
		ex.cluster.drawHistogram(self.ax_hist)
		ex.collection.plot_f1_vs_time(self.f1_vs_time, self.ax2_ts)
		ex.cluster.drawTimeseries(self.ax_ts)


		# show the psd plot
		ex.collection.psd_map(self.ax_map)

		# draw view elements
		if ex.s2n_dist_Button.isChecked(): self.update_s2n_dist()
		if ex.power_dist_Button.isChecked(): self.update_power_dist()
		if ex.TSNEButton.isChecked(): self.update_TSNE()
		if ex.f1_vs_T_Button.isChecked(): self.update_f1_vs_T()
		if ex.f1_vs_RH_Button.isChecked(): self.update_f1_vs_RH()

		# update the collection table
		ex.myTableWidget.collection_table.updateAll()

		self.commit_changes()

	def updateRecording(self):
		# plot the recording at the cursor position
		n = ex.collection.cursor_get_rec()
		recording = ex.collection.getRecording(n)
		recording.plot_rec(self.ax_rec)
		recording.plot_cwt(self.ax_cwt)
		recording.plot_psd(self.ax_psd, self.ax2_psd)

		# highlight the recording in the scatter plots
		self.power_vs_f1.show_highlight(self.ax_sct)
		self.f1_vs_time.show_highlight(self.ax2_ts)
		if ex.TSNEButton.isChecked(): self.tSNE.show_highlight(self.ax_xf)
		if ex.f1_vs_T_Button.isChecked(): self.f1_vs_T.show_highlight(self.ax_xf)
		if ex.f1_vs_RH_Button.isChecked(): self.f1_vs_RH.show_highlight(self.ax_xf)

		self.commit_changes()

		# update the recording table
		ex.myTableWidget.recording_table.updateAll(n)

	def commit_changes(self):
		self.fig.tight_layout
		self.canvas.draw()

		# catch up with events
		# and dismiss temporary status bar messages
		ex.statusBar.clearMessage()
		QApplication.processEvents()

		self.canvas.setFocus()

	def mpl_resize(self, event):
		"""
		This function responds to a resize event
		"""
		event.canvas.figure.tight_layout()

	def mpl_pick(self, event):
		"""
		This function responds to pick event
		"""
		n = ex.collection.cursor_get_pos(event.ind[0])  # cursor position for recording
		ex.collection.cursor_set_pos(n)
		self.updateRecording()

	def mpl_key_press(self, event):
		"""
		This function responds to key press event
		"""

		# check if control key was pressed
		if event.key == 'control':
			self.ctrl_is_pressed = True
			return

		rec = ex.collection.cursor_x    # get the current cursor location (recording number)

		if event.key == 'right':        # the right button was clicked!
			rec = rec + 1
			ex.collection.cursor_set_pos(rec)
		elif event.key == 'left':       # the left button was clicked!
			rec = rec - 1
			ex.collection.cursor_set_pos(rec)
		elif event.key == 'backspace':  # the d(elete) button was clicked
			ex.collection.delRecording()# delete the recording
			ex.features.set(True)       # reset the features
			ex.collection.psd_map(self.ax_map, True)
			ex.collection.cursor_set_pos(rec)
			self.updateAll()
			return
		elif event.key == 'r':          # r(eset) was pressed
			if event.inaxes == self.ax_sct:  # reset the scatter plot
				self.power_vs_f1.reset(self.ax_sct)
				self.canvas.draw()
			return
		elif event.key == 'p':          # p(lay) was pressed
			ex.collection.playRecording()
			return
		elif event.key == 'alt':        # the option key was pressed
			ex.cluster.change_label(ex.collection.latest)
			self.updateClusterDisplay()
			ex.collection.cursor_set_pos(rec)
		else:  # no valid key
			print('Invalid key: ', event.key)
			return

		self.updateRecording()

	def mpl_key_release(self, event):
		"""
		This function responds to key release event
		"""
		if event.key == 'control':
			self.ctrl_is_pressed = False
			return

	def mpl_button_press(self, event):
		"""
		This function responds to a button press event
		"""
		try:
			if self.ctrl_is_pressed:
				self.mpl_zoom(event)
				return
		except AttributeError:
			pass

		# check if button was pressed in map
		if event.inaxes != self.ax_map and event.inaxes != self.ax_temp:
			return
		if event.button != 1:  # make sure left mouse button pressed
			return
		rec = int(event.xdata + 0.5)  # mouse was clicked, get new cursor position
		ex.collection.cursor_set_pos(rec)
		self.updateRecording()

	def mpl_button_release(self, event):
		"""
		This function responds to a button release event in the PSD plot
		"""

	def mpl_zoom(self, event):
		"""
		This function zooms an axis in a new figure
		"""

		# access the figure; if it doesn't exist, create it
		try:
			plt.figure(self.zoom_fig.number)
		except (AttributeError):
			self.zoom_fig = plt.figure('Zoom')
			self.zoom_fig.canvas.manager.window.setWindowFlags(self.windowFlags() | Qt.WindowStaysOnTopHint)
			self.ax_zoom = self.zoom_fig.add_subplot(111)
			cid1 = self.zoom_fig.canvas.mpl_connect('resize_event', self.mpl_resize)
		self.ax_zoom.clear()
		try:
			self.ax2_zoom.remove()
		except:
			pass

		if event.inaxes == self.ax_sct:
			self.power_vs_f1.draw(self.ax_zoom)
			self.ax_zoom.grid('True', axis='x', which='minor', color='b', linestyle='--')
			self.ax_zoom.minorticks_on()
		elif event.inaxes == self.ax_xf:
			if self.view == 'CM': ex.model.drawConfusionMatrix(self.ax_zoom)
			if self.view == 'f1_vs_T': self.f1_vs_T.draw(self.ax_zoom)
			if self.view == 'f1_vs_RH': self.f1_vs_RH.draw(self.ax_zoom)
			if self.view == 'TSNE': ex.tsne.plot(self.tSNE, self.ax_zoom)
			if self.view == 'S/N': ex.cluster.draw_s2n_dist(self.ax_zoom)
			if self.view == 'Signal Power': ex.cluster.draw_power_dist(self.ax_zoom)
		elif event.inaxes == self.ax_hist:
			ex.cluster.drawHistogram(self.ax_zoom)
		elif event.inaxes == self.ax2_ts:
			self.ax2_zoom = self.ax_zoom.twinx()
			self.f1_vs_time.draw(self.ax2_zoom)
			self.ax2_zoom.grid('True', axis='y', which='both')
			self.ax2_zoom.grid('True', axis='y', which='minor', color='b', linestyle='--')
			self.ax2_zoom.minorticks_on()
			ex.cluster.drawTimeseries(self.ax_zoom)
		elif event.inaxes == self.ax_rec:
			ex.collection.getLatestRecording().plot_rec(self.ax_zoom)
		elif event.inaxes == self.ax_cwt:
			ex.collection.getLatestRecording().plot_cwt(self.ax_zoom)
		elif event.inaxes == self.ax2_psd:
			self.ax2_zoom = self.ax_zoom.twinx()
			ex.collection.getLatestRecording().plot_psd(self.ax_zoom, self.ax2_zoom)
		else:
			return

		self.zoom_fig.tight_layout()
		self.zoom_fig.canvas.draw()
		self.zoom_fig.show()

		# clear flag
		self.ctrl_is_pressed = False


# this is the widget for the result tables
class MyTableWidget(QWidget):
	def __init__(self, parent):
		super(QWidget, self).__init__(parent)

		# widget is initially not visible
		self.setVisible(False)

		# layout will be vertical box
		self.layout = QVBoxLayout(self)

		self.collection_table = CollectionTable()
		self.recording_table = RecordingTable()

		self.layout.addWidget(self.collection_table)
		self.layout.addWidget(self.recording_table)
		self.setLayout(self.layout)

	def updateAll(self):
		# update the collection table
		self.collection_table.updateAll()

		# update the recording table
		self.recording_table.updateAll()


class CollectionTable(QTableWidget):
	"""Summary results for the collection"""

	def __init__(self):
		super().__init__()
		self.setColumnCount(1)
		self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)
		self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
		self.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

	def updateAll(self):
		# prepare the table
		self.setHorizontalHeaderLabels(['Collection'])
		self.setRowCount(0)

		# number of samples, median, mean, std dev
		row = self.rowCount()
		self.insertRow(row)
		self.setVerticalHeaderItem(row, QTableWidgetItem('All' + '\n' +
														 '(n = {:.0f}'.format(ex.collection.count) + ')'))
		# self.verticalHeaderItem(row).setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
		self.setItem(row, 0, QTableWidgetItem('{:.0f}'.format(np.mean(ex.collection.f1)) + ' (' +
											  '{:.0f}'.format(np.median(ex.collection.f1)) + ') ± ' +
											  '{:.0f}'.format(np.std(ex.collection.f1)) + ' Hz'))

		# average duration of recordings
		row = self.rowCount()
		self.insertRow(row)
		self.setVerticalHeaderItem(row, QTableWidgetItem('Duration'))
		# self.verticalHeaderItem(row).setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
		self.setItem(row, 0, QTableWidgetItem("%.0f" % np.mean(ex.collection.dur_ms) + " ± " +
											  "%.0f" % np.std(ex.collection.dur_ms) + 'ms'))

		# temperature dependence
		row = self.rowCount()
		self.insertRow(row)
		coef, score = ex.cluster.f_slope()
		self.setVerticalHeaderItem(row, QTableWidgetItem('Temperature' + '\n' + 'Coefficient'))
		self.setItem(row, 0, QTableWidgetItem('{:.1f}'.format(coef) + ' Hz/℃' + '\n' +
											  '(R^2 = {:.3f}'.format(score) + ')'))

		# clustering results (show if more than one cluster)
		if ex.cluster.n_clusters < 2:
			return

		row = self.rowCount()  # clustering method
		self.insertRow(row)
		self.setVerticalHeaderItem(row, QTableWidgetItem(ex.cluster.spec[0]))
		self.setItem(row, 0, QTableWidgetItem(ex.cluster.spec[1]))

		row = self.rowCount()  # clustering score
		self.insertRow(row)
		self.setVerticalHeaderItem(row, QTableWidgetItem('Silhouette Score \n(-1 to 1 = best)'))
		self.setItem(row, 0, QTableWidgetItem('{:.3f}'.format(ex.cluster.silhouette_score)))

		# set of unique labels.  These are not necessarily contiguous and/or starting at 0.
		unique_labels = set(ex.cluster.class_labels)
		for k in unique_labels:  # show cluster counts and frequencies
			row = self.rowCount()
			self.insertRow(row)
			coef, score = ex.cluster.f_slope(k)
			if k == -1:
				r = 255
				g = 255
				b = 0
			else:
				r = int(255 * ex.cluster.colors[k][0])
				g = int(255 * ex.cluster.colors[k][1])
				b = int(255 * ex.cluster.colors[k][2])
			v_header = QTableWidgetItem('Cluster ' + str(k + 1) + '\n' +
										'(n = {:.0f}'.format(ex.cluster.class_labels.tolist().count(k)) + ')\n' +
										ex.cluster.group_labels[k])
			v_header.setBackground(QColor(r, g, b))
			self.setVerticalHeaderItem(row, v_header)
			cell = QTableWidgetItem('{:.0f}'.format(ex.cluster.f_mean(k)) + ' (' +
									'{:.0f}'.format(ex.cluster.f_median(k)) + ') ± ' +
									'{:.0f}'.format(ex.cluster.f_std(k)) + ' Hz' + '\n' +
									'{:.1f}'.format(coef) + ' Hz/℃' + '\n' +
									'(R^2 = {:0.3f}'.format(score) + ')')
			self.setItem(row, 0, cell)


class RecordingTable(QTableWidget):
	'''Summary results for a recording'''

	def __init__(self):
		super().__init__()
		# set up table for individal results
		self.setColumnCount(1)
		self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.MinimumExpanding)
		self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
		self.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)

	# show the parameters in the provided table
	# n is the recording number
	def updateAll(self, n=0):
		# prepare the table
		self.setRowCount(0)

		# get the recording and cluster
		rec = ex.collection.getRecording(n)
		cluster = ex.cluster.class_labels[ex.collection.latest]
		self.setHorizontalHeaderLabels(['# ' + str(n + 1) + ' (' + str(cluster + 1) + ')'])

		# fundamental frequency (best guess)
		row = self.rowCount()
		self.insertRow(row)
		self.setVerticalHeaderItem(row, QTableWidgetItem('f1 [Hz]'))
		self.setItem(row, 0, QTableWidgetItem("%3.0f" % rec.f1))

		# fundamental frequency (autocorrelation)
		row = self.rowCount()
		self.insertRow(row)
		self.setVerticalHeaderItem(row, QTableWidgetItem('Autocorrelation f1 [Hz]'))
		self.setItem(row, 0, QTableWidgetItem("%3.0f" % rec.ac_f1))

		# autocorrelation value
		row = self.rowCount()
		self.insertRow(row)
		self.setVerticalHeaderItem(row, QTableWidgetItem('Correlation Coefficient'))
		self.setItem(row, 0, QTableWidgetItem("%4.2f" % rec.ac_coeff))

		# fundamental frequency (harmonic analysis)
		row = self.rowCount()
		self.insertRow(row)
		self.setVerticalHeaderItem(row, QTableWidgetItem('Harmonic Analysis f1 [Hz]'))
		self.setItem(row, 0, QTableWidgetItem("%3.0f" % rec.ha_f1))

		# number of PSD peaks matching harmonic series
		row = self.rowCount()
		self.insertRow(row)
		self.setVerticalHeaderItem(row, QTableWidgetItem('Harmonics Count'))
		self.setItem(row, 0, QTableWidgetItem("%1.0f" % rec.ha_count
											  + "/" + "%1.0f" % ha_n_peaks))

		# harmonic match
		row = self.rowCount()
		self.insertRow(row)
		self.setVerticalHeaderItem(row, QTableWidgetItem('Harmonic Match'))
		self.setItem(row, 0, QTableWidgetItem(rec.ha_verbose))

		# first peak in PSD
		row = self.rowCount()
		self.insertRow(row)
		self.setVerticalHeaderItem(row, QTableWidgetItem('First PSD Peak [Hz]'))
		self.setItem(row, 0, QTableWidgetItem("%3.0f" % rec.peak_freq[0]))

		# harmonic median frequency spacing
		row = self.rowCount()
		self.insertRow(row)
		self.setVerticalHeaderItem(row, QTableWidgetItem('PSD Peak Spacing [Hz]'))
		self.setItem(row, 0, QTableWidgetItem("%2.0f" % rec.psd_delta))

		# duration
		row = self.rowCount()
		self.insertRow(row)
		self.setVerticalHeaderItem(row, QTableWidgetItem('Duration [ms]'))
		self.setItem(row, 0, QTableWidgetItem("%2.0f" % rec.dur_ms))

		# rms
		row = self.rowCount()
		self.insertRow(row)
		self.setVerticalHeaderItem(row, QTableWidgetItem('RMS (x1000)'))
		self.setItem(row, 0, QTableWidgetItem("%5.1f" % (1000 * rec.rms)))

		# s/n
		row = self.rowCount()
		self.insertRow(row)
		self.setVerticalHeaderItem(row, QTableWidgetItem('s/n @ f[Hz]'))
		self.setItem(row, 0, QTableWidgetItem('{:3.1f} @ {:0.0f}'.format(rec.fd_s2n[0],rec.fd_s2n[1])))

		# power (frequency domain)
		row = self.rowCount()
		self.insertRow(row)
		self.setVerticalHeaderItem(row, QTableWidgetItem('Signal Power (x1000)'))
		self.setItem(row, 0, QTableWidgetItem("%5.3f" % (1000 * rec.fd_power)))

		# baseline or noise (frequency domain)
		row = self.rowCount()
		self.insertRow(row)
		self.setVerticalHeaderItem(row, QTableWidgetItem('Baseline Power (x1000)'))
		self.setItem(row, 0, QTableWidgetItem("%5.3f" % (1000 * rec.fd_power_baseline)))

		# power in low-frequency background
		row = self.rowCount()
		self.insertRow(row)
		self.setVerticalHeaderItem(row, QTableWidgetItem('Low Frequency Power (x1000)'))
		self.setItem(row, 0, QTableWidgetItem("%5.3f" % (1000 * rec.fd_power_lf)))

		# temperature
		row = self.rowCount()
		self.insertRow(row)
		self.setVerticalHeaderItem(row, QTableWidgetItem('Temperature [°C]'))
		self.setItem(row, 0, QTableWidgetItem("%4.1f" % (rec.temp)))

		# humidity
		row = self.rowCount()
		self.insertRow(row)
		self.setVerticalHeaderItem(row, QTableWidgetItem('Rel Humidity [%]'))
		self.setItem(row, 0, QTableWidgetItem("%4.1f" % (rec.rh)))


class Collection():
	"""A collection of Recordings"""

	def __init__(self):
		self.clear()
		self.cursor_x = 1

	# clear all fields in the collection
	def clear(self):
		self.reclist = []
		self.group_labels = []
		self.count = 0
		self.by_s2n = 1     # sorted by s2n
		self.by_freq = 2    # sorted by frequency
		self.by_cluster = 3 # sorted by cluster
		self.f1 = []        # all the fundamental frequencies
		self.s2n = []       # all the s/n ratios
		self.fd_power_db = []       # all the signal power levels
		self.fd_baseline_db = []    # all the baselines in dB
		self.draw_as_y = [] # y values for scatter plot
		self.features = []  # features for clustering
		self.f_max = f_max
		self.psd_min = 0
		self.order = self.by_cluster
		self.dt = []
		self.temp = []
		self.rh = []
		self.dur_ms = []

	# get the data
	# (returns False if no files were found)
	def getData(self, cwd):
		# proceed with loading the data
		# if no data loaded, do nothing
		if not self.load(cwd):
			return False
		# we have data
		self.cwd = cwd
		return True

	# add a recording
	def addRecording(self, rec, group):
		self.reclist.append(rec)
		self.group_labels.append(group)
		self.f1.append(rec.f1)
		self.s2n.append(rec.fd_s2n)
		self.fd_power_db.append(10 * np.log10(rec.fd_power))
		self.fd_baseline_db.append(10 * np.log10(rec.fd_power_baseline))
		self.f_max = min(rec.f_max, self.f_max)
		self.psd_min = min(self.psd_min, rec.psd_min)
		self.draw_as_y.append(10 * np.log10(rec.fd_power))
		self.dt.append(rec.dt)  # timestamp for current recording
		rec.td = np.inf         # calculate timedelta to previous
		n = len(self.dt)
		if n>1:
			rec.td = (rec.dt - self.dt[n-2]).total_seconds()
		self.temp.append(rec.temp)
		self.rh.append(rec.rh)
		self.dur_ms.append(rec.dur_ms)
		self.features.append(rec.features)
		self.count += 1
		# self.updateRecListParams()

	# delete the latest recording that was retrieved
	def delRecording(self):
		rec_to_delete = self.reclist.pop(self.latest)
		del self.group_labels[self.latest]
		del self.f1[self.latest]
		del self.s2n[self.latest]
		del self.fd_power_db[self.latest]
		del self.fd_baseline_db[self.latest]
		del self.draw_as_y[self.latest]
		del self.features[self.latest]
		del self.dt[self.latest]
		del self.temp[self.latest]
		del self.rh[self.latest]
		del self.dur_ms[self.latest]
		os.remove(rec_to_delete.filespec)  # remove the wav file
		self.count -= 1
		# self.updateRecListParams()
		if self.latest >= self.count:
			self.latest = self.count - 1

	# play the latest recording
	def playRecording(self):
		recording = self.reclist[self.latest]
		recording.play()

	# update collection parameters
	def updateRecListParams(self):
		# self.count = len(self.reclist)
		# update the key (in case recordings were deleted)
		map_param = []
		for i in range(self.count):
			map_param.append([i, self.s2n[i], self.f1[i], ex.cluster.class_labels[i]])
		self.index_by_parameter = sorted(map_param, key=itemgetter(self.order))

	# delete low S/N recordings
	def delLowS2N(self, threshold):
		ex.statusBar.showMessage('Deleting...')
		QApplication.processEvents()  # this will update the status bar
		# make a list of recordings to delete, starting at the end
		for n in reversed(range(self.count)):
			rec = self.getRecording(n)
			if rec.fd_s2n[0] < threshold:
				self.delRecording()
		ex.features.set(True)  # reset the features

	# delete false (noise) triggers
	def delFalseTriggers(self, threshold):
		ex.statusBar.showMessage('Deleting...')
		QApplication.processEvents()  # this will update the status bar
		# make a list of recordings to delete, starting at the end
		for n in reversed(range(self.count)):
			rec = self.getRecording(n)
			if rec.fd_power < threshold:
				self.delRecording()
		ex.features.set(True)  # reset the features

	# delete low frequency triggers
	def delLFTriggers(self, lf_ratio):
		ex.statusBar.showMessage('Deleting...')
		QApplication.processEvents()  # this will update the status bar
		# make a list of recordings to delete, starting at the end
		for n in reversed(range(self.count)):
			rec = self.getRecording(n)
			if rec.fd_power_lf * lf_ratio > rec.fd_power:
				self.delRecording()
		ex.features.set(True)  # reset the features

	# delete non-harmonic recordings
	def delPoorHarmonics(self, ha_count):
		ex.statusBar.showMessage('Deleting...')
		QApplication.processEvents()  # this will update the status bar
		# make a list of recordings to delete, starting at the end
		for n in reversed(range(self.count)):
			rec = self.getRecording(n)
			if rec.ha_count < ha_count: self.delRecording()
		ex.features.set(True)  # reset the features

	def setOrder(self, order):
		self.order = order

	def getRecording(self, n=0):
		self.latest = n
		return self.reclist[self.latest]

	def getLatestRecording(self):
		return self.reclist[self.latest]

	def getPSDTable(self):
		self.updateRecListParams()
		PSDTable = []
		for i in range(0, self.count):
			n = self.index_by_parameter[i][0]
			recording = self.getRecording(n)
			PSDTable.append(recording.psd_db -
							recording.psd_baseline_db
							)
		return PSDTable

	# plot psd map in the provided axes as colormap image
	def psd_map(self, ax, reset = False):
		color_map = plt.cm.YlOrRd
		the_scale = [0.5, self.count + 0.5, self.f_max, 0]
		ax.clear()
		ax.set_xlim(0.5, self.count + 0.5)
		ax.set_ylim(self.f_max, 0)
		ax.set_xlabel('Recording#')
		ax.set_ylabel('Freq [Hz]')
		ax.imshow(np.transpose(self.getPSDTable()),
				  aspect='auto', extent=the_scale, cmap=color_map
				  )
		# reset the cursor object
		self.cursor_init(ax)
		if reset:
			self.cursor_x = 1
		self.cursor_set_pos(self.cursor_x)

		if self.order == self.by_s2n or self.order == self.by_freq:
			# draw a grid only
			ax.grid(True, color='k', linestyle='--')
		else:
			# draw lines delineating the classes
			x = 1
			for k in sorted(set(ex.cluster.class_labels)):
				group = ex.cluster.group_labels[k]
				count = ex.cluster.group_counts()[k]
			# for group, count in zip(ex.cluster.group_labels, ex.cluster.group_counts()):
				if count > 0:
					ax.text(x + self.count * .005 + 0.5, self.f_max * 0.99,
							group + '\n'
							+ str(count)
							)
					x = x + count
					ax.axvline(x, color='k', linestyle='--')

	# plot temperature overlay
	def plot_temp_overlay(self, ax):
		ax.clear()
		ax.set_xlim(0.5, self.count + 0.5)
		ax.plot(range(1, self.count + 1), self.temp)
		# ax.set_ylim(temp_min, temp_max)
		ax.set_ylabel('Temperature [℃]')
		ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:2.0f}'))

	# hide temperature overlay
	def hide_temp_overlay(self, ax):
		ax.clear()
		ax.axis('off')

	# initialize cursor
	def cursor_init(self, ax):
		self.line = ax.axvline(color='b')
		self.text = ax.text(1, 0, 1, ha='center')

	# reset cursor
	def cursor_reset(self):
		self.cursor_x = 1

	# set cursor at a position
	def cursor_set_pos(self, pos):
		self.cursor_x = pos                 # check the new cursor and store it
		if pos < 1:
			self.cursor_x = 1
		if pos > self.count:
			self.cursor_x = self.count
		self.line.set_xdata(self.cursor_x)  # draw the cursor line at recording
		self.text.set_x(self.cursor_x)      # show the position
		self.text.set_text(self.cursor_x)   # label the cursor
		return self.cursor_x - 1            # index for the recording

	# get cursor position for a recording number
	def cursor_get_pos(self, n):
		for i in range(self.count):
			if self.index_by_parameter[i][0] == n:
				return i + 1

	# get index of a recording for cursor position
	def cursor_get_rec(self):
		return self.index_by_parameter[self.cursor_x - 1][0]

	# draw f1 vs T in a scatter plot
	def plot_f1_vs_T(self, the_scatter_plot, ax):
		the_scatter_plot.update(ax, self.temp, self.f1)

	# draw f1 vs RH in a scatter plot
	def plot_f1_vs_RH(self, the_scatter_plot, ax):
		the_scatter_plot.update(ax, self.rh, self.f1)

	# draw power vs f1 in a scatter plot
	def plot_power_vs_f1(self, the_scatter_plot, ax):
		the_scatter_plot.update(ax, self.f1, self.draw_as_y)

	# draw f1 vs time in a scatter plot
	def plot_f1_vs_time(self, the_scatter_plot, ax):
		offset = datetime.datetime(2017, 1, 1)  # arbitrary date
		# for each event, get seconds since midnight
		t_event = [(t.hour * 3600 + t.minute * 60 + t.second) for t in ex.collection.dt]
		t_event = [offset + datetime.timedelta(seconds = t) for t in t_event]
		the_scatter_plot.update(ax, t_event, self.f1, labelbottom=False, show_title=False)

	# read the data from file in the selected working directory (cwd)
	# returns true if data were read
	def load(self, cwd):
		# get the count of files and set up the progress bar
		file_count = progress = 0
		for [root, dirs, files] in os.walk(cwd):
			file_count += len(files)
		ex.progress.setMaximum(file_count)
		ex.progress.setValue(0)
		ex.progress.setVisible(True)

		# Set up the target names
		for root, dirs, files in os.walk(cwd):
			break
		# dirs is now a list of all the subdirectories under cwd.
		# save this list for grouping the files
		self.groups = sorted(dirs)  # sort and save the groups
		self.cwd = cwd  # save the working directory

		# if empty (no subdirectory), select the current working directory
		if self.groups == []:
			self.cwd, current_subdir = os.path.split(self.cwd)
			self.groups.append(current_subdir)
		# initialize the number of files in each subdirectory
		self.group_filecounts = []

		# clear any previous data
		self.clear()
		print('\nLoading files...')

		# pool = multiprocessing.Pool()       # set up pool

		with multiprocessing.get_context("spawn").Pool() as pool:
			for i, group in enumerate(self.groups):
				self.group_filecounts.append(0)  # initialize target count
				path = os.path.join(self.cwd, group)
				for [root, dirs, files] in os.walk(path, topdown=False):
					filespecs = []
					for filename in sorted(files):      # get the full path to file
						filespecs.append(os.path.join(root, filename))
					args = zip(filespecs, itertools.repeat(wavScaleFactor, len(filespecs)))
					for recording in pool.imap(load_recording, args, chunksize=10):
						if recording != None:
							self.addRecording(recording, i)
							self.group_filecounts[i] += 1
							# update progress bar
							progress += 1
							if progress % 10 == 0:
								ex.progress.setValue(progress)
								QApplication.processEvents()
				print(self.groups[i], ' ({:0.0f})'.format(self.group_filecounts[i]))

		# pool.close()
		# pool.terminate()
		# pool.join()

		# hide the progress bar
		ex.progress.setVisible(False)

		# check if some files were read
		if not sum(self.group_filecounts) > 0: return False

		# assume all recordings have the same length and sampling frequency
		print('Total recordings =', self.count)
		print('Sampling frequency =', self.getRecording().fs, 'Hz')
		print('Recording length =', self.getRecording().samples, 'samples')

		return True

	# export summary results to csv file
	def write_csv(self):
		# do processing for timestamps
		timestamp = []
		elapsed = []
		for dt in self.dt:
			timestamp.append(datetime.datetime.strftime(dt, '%Y-%b-%d %H:%M:%S'))
			td = dt - self.dt[0]
			elapsed.append(td.days * 86400 + td.seconds)
		# now write the results
		filespec = os.path.join(self.cwd, 'results.csv')
		with open(filespec, 'w', newline='', encoding='UTF-8') as csvfile:
			writer = csv.writer(csvfile)
			writer.writerow(['DateTime',
			                 'Elapsed[s]',
			                 'Cluster',
			                 'Temperature[C]',
			                 'RH[%]',
			                 'AC Frequency[Hz]',
			                 'AC Coefficient',
			                 'HA Frequency[Hz]',
			                 'HA Peaks',
			                 'Duration[ms]',
			                 'S/N',
			                 '@Freq[Hz]',
			                 'Power',
			                 'Fs[Hz]',
			                 'Waveform'])
			for i, rec in enumerate(self.reclist):
				data = [timestamp[i],
				        elapsed[i],
				        ex.cluster.class_labels[i],
				        rec.temp,
				        rec.rh,
				        rec.ac_f1,
				        rec.ac_coeff,
				        rec.ha_f1,
				        rec.ha_count,
				        rec.dur_ms,
				        rec.fd_s2n[0],
				        rec.fd_s2n[1],
				        rec.fd_power,
				        rec.fs]
				data.extend(list(rec.waveform))
				writer.writerow(data)

	# save cwt images to directory
	def save_cwt(self):
		pass


# load a single recording using multiprocessing
def load_recording(args):
	# unpack arguments
	filespec = args[0]
	wavScaleFactor = args[1]
	root, filename = os.path.split(filespec)
	name, ext = os.path.splitext(filename)
	if ext == '.wav':
		filespec = os.path.join(root, filename)
		try:
			rawdata, fs = librosa.core.load(filespec, mono=True, sr=sr_librosa)
		# data, fs = sf.read(filespec)
		except (RuntimeError, EOFError):
			print('Read Error: ', filename)
			return None
		# pad with zeros as required to get minimum number of samples
		data = rawdata
		if (len(rawdata)) < min_rec_length:
			width = round((min_rec_length - len(rawdata)) / 2)
			data = np.pad(rawdata, (width, width), 'constant')
		return Recording(data, fs, filename, filespec, wavScaleFactor, rawdata)

class Recording():
	"""Single recording object"""

	def __init__(self, waveform, fs, filename, filespec, wavScaleFactor, rawdata):
		if wavScaleFactor == 0:
			self.waveform = waveform / np.max(abs(waveform))
		else:
			self.waveform = waveform * wavScaleFactor
		self.rawdata = rawdata
		self.samples = len(self.waveform)
		self.fs = fs
		self.t_max = self.samples / self.fs * 1000
		self.t = np.arange(0, self.t_max, 1 / self.fs * 1000)
		self.filename = filename
		self.filespec = filespec
		self.displayname, _ = os.path.splitext(self.filename)

		# print (fs, rawdata.shape)

		# get datetime, temperature and humidity (as available) from the filename
		self.dt, self.temp, self.rh = self.decodeFilename(filename)

		# calculate normalized temperature
		self.temp_norm = (self.temp - temp_min) / (temp_max - temp_min)

		# pad the waveform with leading zeros,
		# to avoid window function suppressing beginning of short signals

		# padded = np.concatenate((np.zeros(welch_noverlap), waveform))
		# Calculate Welch's estimate of Power Spectral Density
		self.f, self.psd = signal.welch(self.waveform,
										fs=fs,
										window=welch_window,
										nperseg=welch_nperseg,
										noverlap=welch_noverlap,
										nfft=welch_nfft,
										detrend='constant'
										)
		# truncate spectrum to maximum analysis frequency
		idx = self.f <= f_max
		self.f = self.f[idx]
		self.psd = self.psd[idx] + welch_epsilon

		# get maximum frequency.  should be same for all recordings
		self.f_max = max(self.f)

		# calculate the mel spectrogram (need to reshape the psd first)
		psd = self.psd.reshape((-1,1))
		self.melspec = librosa.feature.melspectrogram(S=psd, sr=fs, n_mels=n_mels)
		self.melspec_db = 10 * np.log10(self.melspec)
		self.mel_f = np.arange(0, self.f_max, self.f_max / n_mels)
		self.mfcc = librosa.feature.mfcc(S=self.melspec_db, n_mfcc=n_mfcc, sr=fs)
		# self.mfcc = librosa.feature.mfcc(y=self.waveform, n_mfcc=n_mfcc, sr=fs)
		# print (self.mfcc, self.mfcc_1)

		# calculate dB, and get minimum (for plot scaling)
		self.psd_db = 10 * np.log10(self.psd)
		self.psd_min = min(self.psd_db)

		# find PSD peaks
		self.findPSDPeaks()

		# calculate total power in signal in frequency domain
		self.fd_power = self.f[1] * np.sum(self.psd[self.f >= f_min])

		# calculate total power in baseline in frequency domain
		self.fd_power_baseline = self.f[1] * np.sum(self.psd_baseline[self.f >= f_min])

		# calculate total power in low-frequency background
		self.fd_power_lf = self.f[1] * np.sum(self.psd[self.f < f_min])

		# calculate s/n
		self.fd_s2n = self.calcS2N()

		# calculate total power in signal in time domain
		self.td_power = np.mean(self.waveform ** 2)

		# calculate rms
		self.rms = np.sqrt(np.mean(self.waveform ** 2))

		# interpolate the psd
		self.freq_interp, \
		self.psd_interp \
			= self.interpolatePSD(self.f, self.psd)

		# get harmonic product spectrum (HPS), f1, and number of frequencies in analysis
		# self.hps_f,     \
		# self.hps_y,     \
		# self.hps_f1,    \
		# self.hps_n_hf   \
		# 	= self.getHPS(self.freq_interp, self.psd_interp, self.f1_est)

		# harmonic analysis yields  estimate for fundamental frequency
		self.ha_f1, self.psd_delta, self.ha_score, self.ha_verbose, self.ha_count \
			= self.harmonicAnalysis(self.peak_freq, self.peak_psd_db)

		# autocorrelation yields estimate for fundamental frequency
		self.baseline, \
		self.ac_data, \
		self.ac_f1, \
		self.ac_coeff, \
		self.ac_f, \
		self.ac_y = self.autocorrelation_result(self.waveform, fs)

		# select fundamental frequency for display
		# use autocorrelation
		# self.f1 = self.ac_f1
		# use harmonic analysis
		self.f1 = self.ha_f1

		# get duration
		self.dur_ms = self.getDuration(self.waveform, fs)

		self.addFeatures()

	# add features for machine learning
	def addFeatures(self):
		# normalize the temperature in range from 0 to 1
		temp_norm = (self.temp - temp_min) / (temp_max - temp_min)

		# one-hot encoding of temperature
		temp_one_hot = np.zeros(1 + round((temp_max - temp_min) / temp_step))
		temp_one_hot[round((self.temp - temp_min) / temp_step)] = 1

		# feature list (list of numpy arrays)
		self.features = [self.psd_db,
						 self.mfcc,
						 np.asarray(self.f1),
						 self.temp_norm,
						 np.asarray(10 * np.log10(self.fd_power))]

	def getDuration(self, waveform, fs):
		# returns duration of waveform in ms
		# approach:
		# - calculate power for each sample, +/- 1 ms interval
		# - count up all samples which are >3x the lowest interval
		# - this is the signal length

		# samples per ms
		n_per_ms = round(fs / 1000)

		# baseline power per ms
		bp_per_ms = dur_noisefloor / (len(waveform) / (fs / 1000))

		# power for each sample based on window
		power = []
		for i in range(dur_hw * n_per_ms, len(waveform) - dur_hw * n_per_ms + 1):
			power.append(1 / (2 * dur_hw) * np.mean(waveform[i - n_per_ms:i + n_per_ms] ** 2))

		# the duration is all samples > threshold times the minimum power
		condition = (np.asarray(power) > (bp_per_ms * dur_factor))
		return condition.sum() / (fs / 1000)


	# peak finding with baseline subtraction;
	# with the peakutils library, we must make sure the argument functions are positive.
	def findPSDPeaks(self):
		# note: must do peakfinding on positive values (problem in peakutils)
		db_offset = min(self.psd_db)  # this should be a negative number
		baseline = peakutils.baseline(self.psd_db - db_offset, deg=9)
		self.pf_data = (self.psd_db - db_offset - baseline)
		self.peak_idx = peakutils.indexes(self.pf_data,
										  thres=psd_pf_threshold,
										  min_dist=round(f1_min / max(self.f) * len(self.f) - 1)
										  )
		self.psd_baseline_db = baseline + db_offset

		# convert baseline from dB
		self.psd_baseline = 10 ** (self.psd_baseline_db / 10)

		# check if peaks were found
		if len(self.peak_idx) > 0:  # if peaks are found, use second order curve for exact pos
			self.peak_freq = []
			self.peak_psd_db = []
			self.peak_psd_db_baseline = []
			for n in self.peak_idx:
				alpha = self.psd_db[n - 1]
				beta = self.psd_db[n]
				gamma = self.psd_db[n + 1]
				p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
				self.peak_freq.append(self.f[n] + p * (self.f[n] - self.f[n - 1]))
				self.peak_psd_db.append(beta - 0.25 * (alpha - gamma) * p)
				self.peak_psd_db_baseline.append(1/3 * sum(self.psd_baseline_db[n-1:n+2]))
			# exclude peaks < f1_min
			# must convert list to numpy first
			self.peak_idx = np.asarray(self.peak_idx)
			self.peak_freq = np.asarray(self.peak_freq)
			self.peak_psd_db = np.asarray(self.peak_psd_db)
			new = tuple([self.peak_freq >= f1_min])
			self.peak_idx = self.peak_idx[new]
			self.peak_psd_db = self.peak_psd_db[new]
			self.peak_freq = self.peak_freq[new]

		# check for zero length (no peaks in range were found)
		if len(self.peak_idx) == 0:
			self.peak_idx = np.asarray([0])
			self.peak_freq = np.asarray([0])
			self.peak_psd_db = np.asarray([-100])

		# extract preliminary fundamental frequency
		# (either f1_max or strongest peak in the psd below f1_max)
		self.f1_est = f1_max
		psd = self.peak_psd_db[self.peak_freq <= f1_max]
		if len(psd) > 0:
			idx = np.argmax(psd)
			self.f1_est = self.peak_freq[idx]

	# calculate signal-to-noise ratio for strongest peaks in sprectrogram
	# returns a tuple of s2n and frequency
	def calcS2N(self):
		# must have at least one peak to continue
		if len(self.peak_freq) < 1:
			return (self.fd_power / self.fd_power_baseline, None)

		# get strongest psd peak
		idx = np.argmax(self.peak_psd_db)
		peak_psd_db = self.peak_psd_db[idx]

		# find minima to the left and right
		search_from = 0
		if idx > 0:
			search_from = self.peak_idx[idx - 1]
		search_to = len(self.psd_db)
		if idx < len(self.peak_psd_db)-1:
			search_to = self.peak_idx[idx + 1]
		valley_psd_db = min(self.psd_db[search_from:search_to])

		# finally, compare with baseline, and use maximum
		background_psd_db = max(valley_psd_db, self.psd_baseline_db[self.peak_idx[idx]])

		# now calculate s/n
		return (10 ** ((peak_psd_db - background_psd_db) / 10), self.peak_freq[idx])

		# perform "harmonic analysis" using rules and relationships
	# (fundamental frequency <=> first harmonic)
	def harmonicAnalysis(self, peak_freq, peak_psd_db):
		# the strategy is as follows:
		# 1. If there are less than two peaks, the analysis fails.
		#       - a frequency of zero is returned
		# 2. find fs (strongest psd peak > f1_min, <f1_max
		# 3. hypothesis: this is the first, second, ..., up to ha_n_freq harmonic
		# 4. check if other harmonic frequencies are present under this assumption
		#
		# ht (global parameter) is the tolerance for testing the frequency conditions
		# the median peak frequency spacing is calculated as supplement
		# values of 0 get returned if the analysis fails

		n_harmonics = ha_n_peaks  # number of harmonic peaks to include

		# must convert to numpy array to check frequencies
		freq_values = np.asarray(peak_freq)
		psd_values = np.asarray(peak_psd_db)

		# we only consider frequencies larger than the minimum in the harmonic analysis range
		idx = tuple([freq_values >= f1_min])
		freq_values = freq_values[idx]
		psd_values = psd_values[idx]

		# must have at least one peak to continue
		if len(freq_values) < 1:
			return 0, 0, 0, 'F (0)', 0

		# set up frequency ratios:
		hr = []
		for i in range(ha_n_freq):
			hr.append([(i + 1) / (j + 1) for j in range(n_harmonics)])

		# initialize arrays
		condition = np.full((ha_n_freq, n_harmonics), False)
		f1_estimate = ha_n_freq * [0]
		hits = ha_n_freq * [0]

		# sort peak frequencies by intensity
		sort_index = np.argsort(psd_values)
		freq_values = np.flip(freq_values[sort_index], 0)

		# use strongest peak < f1_max.  If there is no such peak, use strongest, anyway.
		try:
			fs = freq_values[freq_values <= f1_max][0]
		except (IndexError):
			fs = freq_values[0]

		# we no longer need intensity, pick strongest peaks for analysis
		freq_values = np.sort(freq_values[0:n_harmonics])

		# calculate median peak spacing of the strongest peaks
		f_delta = statistics.median(
			[(f2 - f1) for f1, f2 in zip(np.append([0], freq_values[:-1]), freq_values)]
		)

		# loop through the frequencies and count
		# peaks that match the harmonic pattern
		for i, row in enumerate(hr):
			for j, value in enumerate(row):
				if value == 1:
					continue
				for f in freq_values:
					if f == fs: continue
					ratio = f * value / fs
					if 1 - f1_tolerance <= ratio <= 1 + f1_tolerance:
						condition[i, j] = True
						hits[i] += 1
						f1_estimate[i] += f * value

		# process hits and get estimates for f1
		for i in range(0, ha_n_freq):
			hits[i] += 1
			f1_estimate[i] += fs
			f1_estimate[i] = f1_estimate[i] / hits[i] / (i + 1)

		# give  credit for spacing corresponding to calculated fundamental
		# (whether or not there is a peak at the frequency)
		for i in range(0, ha_n_freq):
			if 1 - f1_tolerance <= f_delta / f1_estimate[i] <= 1 + f1_tolerance:
				# if condition[i,0]:
				hits[i] += 1

		# tie breaker: if first and second have the same score, choose second
		# (i.e. lower frequency wins)
		# if hits[0] == hits[1]:
		#	hits[1]+=1

		# find best-performing estimate
		i, n_hits = max(enumerate(hits), key=itemgetter(1))
		f1 = f1_estimate[i]

		# count harmonic peaks in sequence starting with f1;
		# searching in the strongest
		hap_count = 0
		for n in range(1, n_harmonics + 1):
			present = False
			for f in freq_values:  # search for the next frequency in series
				if 1 - f1_tolerance <= f / (n * f1) <= 1 + f1_tolerance:
					hap_count += 1
					present = True
			if n == 1 or present:
				continue
			else:
				break

		# encode results
		score = 0
		verbose = 'F ('
		if n_hits > 1:
			score = 1
			verbose = 'D ('
		if n_hits > 2:
			score = 2
			verbose = 'C ('
		if n_hits > 3:
			verbose = 'B ('
		if n_hits > 4:
			verbose = 'A ('

		return f1, f_delta, score, verbose + str(i + 1) + '/' + str(n_hits) + ')', hap_count

	# harmonic product spectrum (HPS)
	# returns: frequency range, HPS, frequency of maximum
	def getHPS(self, freq, psd, f1_est):
		y = psd.copy()

		# clear values less than f_min
		y[freq <= f_min] = 0

		# determine highest harmonic that can be included,
		# based on an estimate for f1
		N = y.size
		n_harmonics = hps_harmonics
		f_limit = min(f1_max, (1 + hps_tol) * f1_est)
		f_limit = max(f1_min * 2, f_limit)
		while freq[int(np.ceil(N / n_harmonics))] < f_limit:
			n_harmonics -= 1
		smallestLength = int(np.ceil(N / n_harmonics))

		# downsample-multiply
		y = psd[:smallestLength].copy()
		for i in range(2, n_harmonics + 1):
			y *= psd[::i][:smallestLength]

		# only consider psd analysis range
		f = freq[:smallestLength]

		# find maximum (but exclude slope)
		# assumes estimate is first or second harmonic
		start = int(np.ceil((1 - hps_tol) * (f1_est / 2)))
		start = max(start, f1_min)
		idx = np.argmax(y[start:])
		while idx == 0:
			start += 1
			if start >= f1_est: break
			idx = np.argmax(y[start:])
		f1 = freq[idx + start]  # frequency of maximum
		y = y / y[idx + start]  # normalize to 1

		f = f[start:]
		y = y[start:]

		return f, y, f1, n_harmonics

	# return f1 estimate by autocorrelation
	def autocorrelation_result(self, waveform, fs):
		bias = self.baseline_als(waveform)  # determine bias (body signal)
		waveform_unbiased = waveform - bias

		# wavelength range of interest
		start = int(fs / f1_max + .5)
		stop = int (fs / f1_min + 0.5)

		# l = len(waveform) - 1       # perform autocorrelation
		# ac_data = np.correlate(waveform_unbiased, waveform_unbiased, "full")
		# ac_data = ac_data[l:]/ac_data[l]
		ac_data = np.array([1. if l == 0 else
		                    0. if l >= stop else
		                    np.corrcoef(waveform[l:], waveform[:-l])[0][1]
							for l in range(0, len(waveform))])

		# save autocorrelation coefficients for plot
		ac_f = []
		ac_y = []
		start = int(fs / f1_max + .5)
		for i in range(start, stop):
			ac_f.append(fs / i)
			ac_y.append(ac_data[i])

		# peak finding
		ac_peaks = peakutils.indexes(ac_data, ac_thres)         # find peaks
		ac_peaks = ac_peaks[ac_peaks >= int(fs / f1_max + .5)]  # include only range for f1
		n_peaks = min(ac_n_peaks, len(ac_peaks))                # use only first  peaks
		if n_peaks == 0:  # return 0 if no peaks in autocorrelation
			return bias, ac_data, 0, 0, ac_f, ac_y

		ac_peaks_interp = []                # tuples (freq, value)
		for n in ac_peaks[0:n_peaks]:       # interpolate peak position
			alpha = ac_data[n - 1]
			beta = ac_data[n]
			gamma = ac_data[n + 1]
			p = 0.5 * (alpha - gamma) / (alpha - 2 * beta + gamma)
			ac_peaks_interp.append((fs / (n + p), beta - 0.25 * (alpha - gamma) * p))

		# sort by coefficient
		# ac_peaks_interp = sorted (ac_peaks_interp, key=itemgetter(2), reverse=True)

		# # pick the peak with the highest autocorrelation
		f1, coeff = max(ac_peaks_interp, key=operator.itemgetter(1))

		# However, this may be the second harmonic, or at half the fundamental
		# so perform a check
		# for f, value in ac_peaks_interp:
		# 	if value >= ac_min:
		# 		if (f != f1):
		# 			print("Autocorrelation Exception in ", self.filename, f1, f)
		# 			f1 = f
		# 			coeff = value
		# 		break

		return bias, ac_data, f1, coeff, ac_f, ac_y

	# "Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens, 2005
	# parameters determined empirically
	def baseline_als(self, y, lam=500, p=0.002, niter=5):
		L = len(y)
		D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
		w = np.ones(L)
		for i in range(niter):
			W = sparse.spdiags(w, 0, L, L)
			Z = W + lam * D.dot(D.transpose())
			z = spsolve(Z, w * y)
			w = p * (y > z) + (1 - p) * (y < z)
		return z

	# interpolate a harmonic convolution value from a table
	def interpolateHCValue(self, x_table, y_table, x_value):
		if x_value < x_table[0]: return -1
		if x_value > max(x_table): return -1
		f = interpolate.interp1d(x_table, y_table, kind='cubic')
		return f(x_value)

	# interpolate in the frequency domain with 1 Hz resolution
	# this is where we limit the frequency range
	def interpolatePSD(self, freq, y):
		f = interpolate.interp1d(freq, y, kind='cubic')
		freq_limit = 1 + min(f_max, int(freq[len(freq) - 1]))
		freq_new = np.arange(freq[0], freq_limit, 1)
		y_new = f(freq_new)
		return freq_new, y_new

	# plot the recording data into the provided axes
	def plot_rec(self, ax):
		ax.clear()  # clear the previous plot
		ax.set_facecolor('w')
		ax.plot(self.t, self.waveform, label='Waveform')  # original waveform
		ax.plot(self.t, self.baseline, ":", label='Bias')  # bias (body signal)
		ax.plot(self.t, self.ac_data / 4 - 0.75, label="Autocorrelation")  # autocorrelation
		ax.legend(loc='upper right')
		color = 'black'
		if self.td < rec_td_warning: color = 'red'
		ax.set_title(self.displayname[-50:], color=color)
		ax.set_xlabel('Time [ms]')
		ax.set_ylabel('Signal [V]')
		ax.set_xlim(0, self.t_max)
		ax.set_ylim(-1, 1)
		ax.grid(True)

	# plot the wavelet transform into the provided axes
	def plot_cwt(self, ax):
		ax.clear()  # clear the previous plot
		z_data, frequencies = getCWT(self)
		num = len(frequencies)
		ax.imshow(z_data, extent=[0, len(self.waveform)/ self.fs * 1000, cwt_f_max, cwt_f_min],
		          cmap=plt.cm.hot,
		          aspect='auto',
		          origin='upper',
		          vmax = cwt_log_max,
		          vmin = cwt_log_min)
		cv2.imwrite('../cwt.jpg', z_data / np.max(z_data) * 255, [cv2.IMWRITE_JPEG_QUALITY, 100])
		# n_ticks = 20
		# ticks = np.linspace(0, num, n_ticks, endpoint=True).astype(int)
		# tick_labels = ['%d' % freq for freq in frequencies[ticks[:-1]].tolist()]
		# tick_labels.append('%d' % frequencies[-1])
		# ax.set_yticks(ticks)
		# ax.set_yticklabels(tick_labels)
		ax.set_title('Continuous Wavelet Transform (' + ex.cwavelet +')')
		ax.set_xlabel('Time [ms]')
		ax.set_ylabel('Pseudo Frequency [Hz]')

	# plot the psd into the provided axes
	def plot_psd(self, ax, ax2):
		ax.clear()
		ax.set_facecolor('w')
		ax.plot(self.f, self.psd_db, 'b', label='PSD[dB]')
		ax.plot(self.f, self.psd_baseline_db, 'b:')
		# ax.plot(self.mel_f, self.melspec_db, 'c', label='Mel Spectrum')
		# hps_y = 10 * np.asarray(self.hps_y)
		# ax.plot(self.hps_f, hps_y, 'r', label='HPS')
		ac_y = 10 * np.asarray(self.ac_y)
		ax.plot(self.ac_f, ac_y, 'r', label='AC')
		ax.legend(loc='upper right')
		ax.set_title('PSD and MFCC')
		ax.set_xlabel('Fundamental Frequency [Hz]')
		ax.set_ylabel('PSD [V**2/Hz, dB]')
		ax.set_xlim(0, f_max)
		# ax.set_ylim(ex.collection.psd_min, 10)
		ax.set_ylim(-120, 10)
		ax.grid(True)

		# mark the peaks in the PSD
		for [freq, psd] in zip(self.peak_freq, self.peak_psd_db):
			ax.annotate("%4.0f" % freq,
						(freq, psd + 5),
						fontsize=8,
						verticalalignment='bottom', horizontalalignment='center')
			ax.annotate("|",
						(freq, psd),
						fontsize=8,
						verticalalignment='bottom', horizontalalignment='center')

		# now plot the original psd
		ax2.clear()
		ax2.plot(self.freq_interp, 1000 * self.psd_interp, 'g', label='PSD')
		ax2.set_xlim(0, f_max)
		# ax2.set_ylim(-0.2, 1.1)
		ax2.set_ylabel('PSD [V**2/Hz x1000]')
		ax2.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:0.3f}'))
		ax2.legend(loc='lower right')

	# decode and return time stamp and temperature from file names
	def decodeFilename(self, filename):

		name, ext = os.path.splitext(filename)

		try:  # BG-Eye with numeric month, humidity and fractional seconds
			[the_date, the_time, temp, rh, remainder] \
				= name.split('_')
			dt = datetime.datetime.strptime(the_date + ' ' + the_time, '%Y-%m-%d %H.%M.%S.%f')
			temp = float(temp[1:5])
			rh = float(rh[1:5])
			return dt, temp, rh
		except ValueError:
			pass

		try:  # BG-Eye with humidity and fractional seconds
			[the_date, the_time, temp, rh, remainder] \
				= name.split('_')
			dt = datetime.datetime.strptime(the_date + ' ' + the_time, '%Y-%b-%d %H.%M.%S.%f')
			temp = float(temp[1:5])
			rh = float(rh[1:5])
			return dt, temp, rh
		except ValueError:
			pass

		# generic wav file
		dt = datetime.datetime(2019, 1, 1, 0, 0, 0)  # default timestamp
		temp = 25  # default temperatures
		rh = 0  # default humidity

		return dt, temp, rh

	def play(self):
		# the filespec cannot have any spaces
		playsound(self.filespec)


class Features():
	'''Manages the features for machine learning'''

	def __init__(self):
		self.scaler = None
		self.data = None
		self.use_current = None
		self.use_last_dwt = None

	def set(self, reset = False):
		# show temporary status bar message
		ex.statusBar.showMessage('Calculating, this may take a while...')
		QApplication.processEvents()        # this will update the status bar

		if reset:                           # forces recalculation
			self.use_current = None
			self.use_dwt = None

		# updates the feature list if needed
		if self.use == self.use_current:
			X = self.X_current
		elif self.use == self.use_last_dwt:
			X = self.X_last_dwt
		else:
			X = []
			for rec in ex.collection.reclist:
				if self.use == 'PSD':
					feature = rec.psd
				elif self.use == 'MFCC':
					feature = rec.mfcc
				elif self.use == 'f1':
					feature = np.asarray(rec.f1)
				elif self.use == 'RAW':
					feature = rec.rawdata
				else:
					feature = DWT(rec.waveform, self.wavelet).features
				X.append(feature)
			self.X_current = X
			self.use_current = self.use
			if 'dwt' in self.use:
				self.X_last_dwt = X
				self.use_last_dwt = self.use

		# convert to numpy array and reshape
		X = np.asarray(X).reshape(len(ex.collection.features), -1)

		# apply RBM if requested
		if self.apply_RBM:
			n_components = 2 * np.shape(X)[1]
			rbm = BernoulliRBM (n_components=n_components,
			                    learning_rate=0.1,
			                    n_iter=100,
			                    verbose=1)
			X = rbm.fit_transform(X)

		# apply PCA if requested
		if self.apply_PCA:
			n_components = min(pca_components, np.shape(X)[1])
			pca = PCA(n_components=n_components, svd_solver='full')
			X = pca.fit_transform(X)
			print('PCA ratios:', pca.explained_variance_ratio_)

		# now add parameters as requested
		X_par = []
		for rec in ex.collection.reclist:
			feature = []
			if self.include_temp:
				feature = np.append(feature, rec.temp_norm)
			if self.include_sp:
				feature = np.append(feature, rec.fd_power)
			X_par.append(feature)
		X_par = np.asarray(X_par).reshape(len(ex.collection.features), -1)
		self.X_par = X_par

		# combine the two arrays
		X = np.append(X, X_par, axis=1)

		# scale the data
		# X = (X - np.min(X)) / (np.max(X) + 0.0001)  # 0-1 scaling
		if self.scaler_name == global_scalers[0]:
			self.scaler = None
		else:
			if self.scaler_name == global_scalers[1]:
				self.scaler = StandardScaler()
			elif self.scaler_name == global_scalers[2]:
				self.scaler = Normalizer()
			elif self.scaler_name == global_scalers[3]:
				self.scaler = RobustScaler()
			self.scaler.fit(X)
			X = self.scaler.transform(X)

		self.X = X                      # finally, this is the feature matrix

		self.descriptor = self.use
		if self.apply_RBM:
			self.descriptor += '_rbm'
		if self.apply_PCA:
			self.descriptor += '_pca'
		if self.include_temp:
			self.descriptor += '_temp'
		if self.include_sp:
			self.descriptor += '_pwr'
		self.descriptor += '_' + self.scaler_name   # this is the descriptor

		# dismiss temporary status bar messages
		ex.statusBar.clearMessage()
		QApplication.processEvents()

	#retrieve the features
	def get(self):
		return self.descriptor, self.X

	# save the data
	def save(self):
		filespec = os.path.join(ex.cwd, 'X_' + self.descriptor)
		np.save(filespec, self.X)
		filespec = os.path.join(ex.cwd, 'y_' + self.descriptor)
		np.save(filespec, ex.cluster.class_labels)

	def setOptions(self, use_psd, use_mel, use_dwt, wavelet, use_f1, use_raw, include_temp, include_sp, apply_RBM, apply_PCA, scaler):
		if use_psd:
			self.use = 'PSD'
		elif use_mel:
			self.use = 'MFCC'
		elif use_dwt:
			self.use = 'DWT_' + wavelet
			self.wavelet = wavelet
		elif use_f1:
			self.use = 'f1'
		elif use_raw:
			self.use = 'RAW'
		self.include_temp = include_temp
		self.include_sp = include_sp
		self.apply_RBM = apply_RBM
		self.apply_PCA = apply_PCA
		self.scaler_name = scaler


# get continuous wavelet transform (cwt)
def getCWT(recording):
	wavelet = pywt.ContinuousWavelet(ex.cwavelet)
	fc = pywt.central_frequency(wavelet)
	dt = 1 / recording.fs  # sampling interval in seconds
	freqs = np.linspace(cwt_f_min, cwt_f_max, num=cwt_num, endpoint=True)
	scales = fc / (freqs * dt)
	# scale_min = fc * 2    # corresponds to half the sampling frequency
	# scale_max = scale_min * 20
	# scales = np.linspace(scale_min, scale_max, num=cwt_num, endpoint=True)
	signal = recording.waveform
	[coefficients, frequencies] = pywt.cwt(signal, scales, wavelet, dt)
	power = (abs(coefficients)) ** 2
	return np.log10(power), frequencies


class DWT():
	'''Calculates the Discrete Wavelet Transform and parameters'''

	def __init__(self, signal, waveletname='sym9'):
		self.wavelet = pywt.Wavelet(waveletname)
		# set decomposition level based on target duration and wavelet characteristic
		self.level = pywt.dwt_max_level(sr_librosa*wav_segment_dur/1000, self.wavelet)
		self.features = []
		self.list_coeff = pywt.wavedec(signal, self.wavelet, level=self.level)
		for coeff in self.list_coeff:
			self.features += self.get_features(coeff)

	def get_features(self, list_values):
		entropy = self.calculate_entropy(list_values)
		crossings = self.calculate_crossings(list_values)
		statistics = self.calculate_statistics(list_values)
		return [entropy] + crossings + statistics

	def calculate_entropy(self, list_values):
		counter_values = Counter(list_values).most_common()
		probabilities = [elem[1] / len(list_values) for elem in counter_values]
		entropy = scipy.stats.entropy(probabilities)
		return entropy

	def calculate_statistics(self, list_values):
		n5 = np.nanpercentile(list_values, 5)
		n25 = np.nanpercentile(list_values, 25)
		n75 = np.nanpercentile(list_values, 75)
		n95 = np.nanpercentile(list_values, 95)
		median = np.nanpercentile(list_values, 50)
		mean = np.nanmean(list_values)
		std = np.nanstd(list_values)
		var = np.nanvar(list_values)
		rms = np.nanmean(np.sqrt(list_values ** 2))
		return [n5, n25, n75, n95, median, mean, std, var, rms]

	def calculate_crossings(self, list_values):
		zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
		no_zero_crossings = len(zero_crossing_indices)
		mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
		no_mean_crossings = len(mean_crossing_indices)
		return [no_zero_crossings, no_mean_crossings]


class TSNE():
	'''Transforms high-dimensional dataset for visualization'''

	def __init__(self):
		pass

	def update(self):
		ex.statusBar.showMessage('Transforming, this may take a while...')
		QApplication.processEvents()    # this will update the status bar

		# get the data
		self.descriptor, X = ex.features.get()

		# do the transform
		# tsne = manifold.TSNE(n_components=2,
		#                      n_iter=500,
		#                      perplexity=100,
		#                      random_state=101557)
		tsne = manifold.TSNE(n_components=2,
		                     n_iter=500,
		                     init='pca',
		                     random_state=101557,
		                     perplexity=50,
		                     method='exact',
		                     n_jobs=-1)
		self.map = tsne.fit_transform(X)

	def plot(self, the_plot, ax):
		the_plot.update(ax, self.map[:, 0], self.map[:, 1], self.descriptor)


class Model():
	'''Base class for training a model'''

	def __init__(self):
		self.name = 'None'
		self.scaler = None
		self.classifier = None
		self.classes = []
		self.n_classes = 0
		self.trained = False

	def load(self, filepath):
		f = open(filepath, 'rb')
		tmp_dict = pickle.load(f)
		f.close()

		self.__dict__.update(tmp_dict)

	def save(self, cwd):
		filename = self.name + '(' + self.descriptor + ')' + '.wam'
		filepath = os.path.join(cwd, filename)
		f = open(filepath, 'wb')
		pickle.dump(self.__dict__, f)
		f.close()

	def train(self):
		# train the model; allocated 20% of recordings for cross validation
		# show temporary status bar message
		ex.statusBar.showMessage('Training, this may take a while...')
		QApplication.processEvents()        # this will update the status bar

		# copy feature data and information
		self.descriptor, X = ex.features.get()
		self.scaler_name = ex.features.scaler_name
		self.scaler = ex.features.scaler    # copy the scaler

		# copy class labels and info
		y = np.asarray(ex.cluster.class_labels)
		self.class_labels = ex.cluster.group_labels
		self.n_classes = len(set(ex.cluster.class_labels))
		self.n_features = np.shape(X)[1]

		# split the data set
		X_train, X_cv, y_train, y_cv = train_test_split(
			X, y,
			test_size=.20,
			random_state=2017)

		# fit the training set
		self.classifier.fit(X_train, y_train)
		self.trained = True

		# save info
		self.class_labels = ex.cluster.group_labels
		self.n_classes = len(set(ex.cluster.class_labels))
		self.n_features = np.shape(X)[1]

		# evaluate the cross validation set and get the confusion matrix
		y_pred = self.classifier.predict(X_cv)
		self.accuracy_score = accuracy_score(y_cv, y_pred)
		self.confusion_matrix = confusion_matrix(y_cv, y_pred)

		print('Training: ' + str(self.n_features) + ' features; accuracy score = {:0.3f}'.format(self.accuracy_score))

		# dismiss temporary status bar messages
		ex.statusBar.clearMessage()
		QApplication.processEvents()

	def drawConfusionMatrix(self, ax):
		"""
		This function prints and plots the confusion matrix.
		Normalization can be applied by setting `norm=True`.
		"""
		ax.clear()
		ax.set_facecolor('w')

		norm = True
		title = self.name + ' Confusion Matrix (' + ex.features.descriptor + ')'

		if norm:
			cm = self.confusion_matrix.astype('float') / self.confusion_matrix.sum(axis=1)[:, np.newaxis]
			f = '{:.2f}'

		else:
			cm = self.confusion_matrix
			f = '{:d}'

		cmap = plt.cm.Blues
		ax.imshow(cm, interpolation='nearest', cmap=cmap)
		ax.set_title(title)

		n_ticks = np.arange(len(self.class_labels))
		ax.xaxis.set_ticks(n_ticks)
		ax.xaxis.set_ticklabels(self.class_labels)
		ax.yaxis.set_ticks(n_ticks)
		ax.yaxis.set_ticklabels(self.class_labels)

		ax.tick_params(labelrotation=45)

		thresh = cm.max() / 2.
		for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
			ax.text(j, i, f.format(cm[i, j]),
					horizontalalignment="center",
					color="white" if cm[i, j] > thresh else "black")

		ax.set_ylabel('True Label')
		ax.set_xlabel('Predicted Label')


class XGBoost(Model):
	'''Train using XGBoost'''

	def __init__(self):
		super().__init__()
		self.name = 'XGBoost'
		self.classifier = XGBClassifier(
			n_estimators = 1000,
			n_jobs=multiprocessing.cpu_count(),
		)

	def train(self):
		# train the model; allocated 20% of recordings for cross validation
		# show temporary status bar message
		ex.statusBar.showMessage('Training, this may take a while...')
		QApplication.processEvents()        # this will update the status bar

		# copy feature data and information
		self.descriptor, X = ex.features.get()
		self.scaler_name = ex.features.scaler_name
		self.scaler = ex.features.scaler    # copy the scaler

		# copy class labels and info
		y = np.asarray(ex.cluster.class_labels)
		self.class_labels = ex.cluster.group_labels
		self.n_classes = len(set(ex.cluster.class_labels))
		self.n_features = np.shape(X)[1]

		# split the data set
		X_train, X_cv, y_train, y_cv = train_test_split(
			X, y,
			test_size=.20,
			random_state=2017)

		# evaluation
		eval_set = [(X_train, y_train), (X_cv, y_cv)]
		eval_metric = ["error", "logloss"]
		if self.n_classes > 2:
			eval_metric = ["merror", "mlogloss"]

		# fit the training set
		# self.classifier.fit(X_train, y_train)
		self.classifier.fit(X_train, y_train,
		                    early_stopping_rounds=10,
		                    eval_metric=eval_metric,
		                    eval_set=eval_set,
		                    verbose=False)
		self.trained = True

		# retrieve evaluation results
		results = self.classifier.evals_result()
		epochs = len(results['validation_0'][eval_metric[1]])
		x_axis = range(0, epochs)

		# evaluate the cross validation set and get the confusion matrix
		y_pred = self.classifier.predict(X_cv)
		self.accuracy_score = accuracy_score(y_cv, y_pred)
		self.confusion_matrix = confusion_matrix(y_cv, y_pred)

		print('Training: ' + str(self.n_features) + ' features; accuracy score = {:0.3f}'.format(self.accuracy_score))

		self.show_eval_results(x_axis, eval_metric, results)

		# dismiss temporary status bar messages
		ex.statusBar.clearMessage()
		QApplication.processEvents()

	def show_eval_results(self, x_axis, eval_metric, results):
		# set up the figure
		plt.ion()                   # avoids QtCore error message
		fig = plt.figure()
		fig.set_size_inches(12, 8)
		window_flags = fig.canvas.manager.window.windowFlags()
		fig.canvas.manager.window.setWindowFlags(window_flags | Qt.WindowStaysOnTopHint)
		gs = fig.add_gridspec(2, 3)
		ax1 = fig.add_subplot(gs[0, 0])
		ax2 = fig.add_subplot(gs[1, 0])
		ax3 = fig.add_subplot(gs[0, 1])
		ax4 = fig.add_subplot(gs[1, 1])
		ax5 = fig.add_subplot(gs[1, 2])

		# plot log loss
		ax1.plot(x_axis, results['validation_0'][eval_metric[1]], label='Train')
		ax1.plot(x_axis, results['validation_1'][eval_metric[1]], label='Test')
		ax1.legend()
		ax1.set_ylabel('Log Loss')
		ax1.set_title('XGBoost Log Loss')
		ax1.grid(True)
		# plot classification error
		ax2.plot(x_axis, results['validation_0'][eval_metric[0]], label='Train')
		ax2.plot(x_axis, results['validation_1'][eval_metric[0]], label='Test')
		ax2.legend()
		ax2.set_ylabel('Classification Error')
		ax2.set_title('XGBoost Classification Error')
		ax2.grid(True)
		# plot feature importance
		plot_importance(self.classifier, ax = ax3)
		fig.tight_layout()
		# confusion matrix
		self.drawConfusionMatrix(ax4)
		# parameters
		cm = self.confusion_matrix
		tn = cm[0, 0]
		fn = cm[1, 0]
		fp = cm[0, 1]
		tp = cm[1, 1]
		total = tn + fn + tp + fp
		accuracy = (tp + tn) / total
		tpr = tp / (fn + tp)
		fpr = fp / (tn + fp)
		fnr = fn / (fn + tp)
		precision = tp / (fp + tp)
		npv = tn / (tn + fn)
		prevalence = (tp + fn) / total
		rows = ('Accuracy',
		        'TPR True Positive Rate (Recall)',
		        'FPR False Positive Rate',
		        'FNR False Negative Rate',
		        'PPV Positive Predictive Value (Precision)',
		        'NPV Negative Predictive Value',
		        'Prevalence')
		cellText = [['{:0.1%}'.format(accuracy)],
		            ['{:0.1%}'.format(tpr)],
		            ['{:0.1%}'.format(fpr)],
		            ['{:0.1%}'.format(fnr)],
		            ['{:0.1%}'.format(precision)],
		            ['{:0.1%}'.format(npv)],
		            ['{:0.1%}'.format(prevalence)]
		            ]
		colWidths = [0.2]
		ax5.axis('off')
		ax5.table(cellText, rowLabels=rows, colWidths=colWidths, loc='center')

		fig.tight_layout()
		plt.show()


class MLP(Model):
	'''Train using MLP'''

	def __init__(self):
		super().__init__()
		self.name = 'MLP'
		self.classifier = MLPClassifier()

class Cluster():
	'''Base class for clustering
	Data are organized by file group'''

	def __init__(self):
		self.n_clusters = len(ex.collection.groups)  # number of clusters
		self.spec = ['File Group; N =', str(self.n_clusters)]
		self.group_labels = ex.collection.groups

	# perform the clustering operation
	def update(self):
		self.descriptor, self.X = ex.features.get()
		self.class_labels = np.asarray(ex.collection.group_labels)
		self.core_samples_mask = np.full(ex.collection.count, True)  # core samples mask (get plotted larger)

		# metrics
		if self.n_clusters < 2:
			self.silhouette_score = np.NaN
			return
		else:
			self.silhouette_score = metrics.silhouette_score(self.X, self.class_labels)

	# change the label of a point
	def change_label(self, n):
		print('Change class label#', n + 1, 'from', self.class_labels[n], end='')
		self.class_labels[n] += 1
		if self.class_labels[n] >= self.n_clusters:
			self.class_labels[n] = 0
		print(' to', self.class_labels[n])

	# get the mean frequencies
	def f_mean(self, n):
		return np.mean(np.asarray(ex.collection.f1)[self.class_labels == n])

	# get the median frequencies
	def f_median(self, n):
		return np.median(np.asarray(ex.collection.f1)[self.class_labels == n])

	# get the standard deviations
	def f_std(self, n):
		return np.std(np.asarray(ex.collection.f1)[self.class_labels == n])

	# regression of frequency verus temperature
	def f_slope(self, n='all'):
		regression = LinearRegression()
		if n == 'all':
			X = np.asarray(ex.collection.temp).reshape(-1, 1)
			y = np.asarray(ex.collection.f1)
		else:
			X = np.asarray(ex.collection.temp)[self.class_labels == n].reshape(-1, 1)
			y = np.asarray(ex.collection.f1)[self.class_labels == n]
		regression.fit(X, y)
		return regression.coef_[0], regression.score(X, y)

	# count
	def group_counts(self):
		result = []
		unique_labels = set(ex.cluster.class_labels)
		for label in unique_labels:
			result.append(self.class_labels.tolist().count(label))
		return result

	# draw the s/n histogram into the axes
	def draw_s2n_dist(self, ax):
		ax.clear()
		ax.set_facecolor('k')
		ax.set_aspect('auto')

		# set of unique labels (these are not necessarily contiguous and/or start with 0)
		unique_labels = set(self.class_labels)

		# set up the bins
		bins = [0, 10, 20, 50, 100, 200, 500, np.inf]
		bin_labels = ['<10', '20', '50', '100', '200', '500', '>']

		# prepare label colors
		self.colors = []
		for each in np.linspace(0, 1, self.n_clusters):
			self.colors.append(plt.cm.cool(each))

		for i, k in enumerate(unique_labels):
			# use grey for not assigned to cluster
			if k == -1:
				col = [1, 1, 0, 1]
			else:
				col = self.colors[i]

			class_member_mask = (self.class_labels == k)

			# get the distribution, and normalize
			s2n_values = np.asarray(ex.collection.s2n)[class_member_mask]
			events_per_bucket = np.histogram(s2n_values,
											 bins,
											 density=False
											 )[0]
			events_per_bucket = events_per_bucket / (sum(events_per_bucket))

			# plot the distribution
			ax.plot(events_per_bucket,
					color=tuple(col))

		# pretty up the plot
		ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0,8)))
		ax.xaxis.set_major_formatter(ticker.FixedFormatter(bin_labels))
		ax.set_xlim(bins[0], len(bins) - 2)
		ax.set_title('s/n Distribution')
		ax.set_xlabel('s/n Ratio')
		ax.set_ylabel('Proportion')
		ax.grid(True)

	# draw the power histogram into the axes
	def draw_power_dist(self, ax):
		ax.clear()
		ax.set_facecolor('k')
		ax.set_aspect('auto')

		# set of unique labels (these are not necessarily contiguous and/or start with 0)
		unique_labels = set(self.class_labels)

		# set up the bins
		bins = [-100, -50, -45, -40, -35, -30, -25, -20, -15, np.inf]
		bin_labels = ['<-50', '-45', '-40', '-35', '-30', '-25', '-20', '-15', '>']

		# prepare label colors
		self.colors = []
		for each in np.linspace(0, 1, self.n_clusters):
			self.colors.append(plt.cm.cool(each))

		for i, k in enumerate(unique_labels):
			# use grey for not assigned to cluster
			if k == -1:
				col = [1, 1, 0, 1]
			else:
				col = self.colors[i]

			class_member_mask = (self.class_labels == k)

			# get the distribution, and normalize
			power_values = np.asarray(ex.collection.fd_power_db)[class_member_mask]
			events_per_bucket = np.histogram(power_values,
											 bins,
											 density=False
											 )[0]
			events_per_bucket = events_per_bucket / (sum(events_per_bucket))

			# plot the distribution
			ax.plot(events_per_bucket,
					color=tuple(col))

		# pretty up the plot
		ax.xaxis.set_major_locator(ticker.FixedLocator(np.arange(0,10)))
		ax.xaxis.set_major_formatter(ticker.FixedFormatter(bin_labels))
		ax.set_xlim(0, len(bins) - 2)
		ax.set_title('Signal Power Distribution')
		ax.set_xlabel('Signal Power [V**2, dB]')
		ax.set_ylabel('Proportion')
		ax.grid(True)

	# draw the frequency distribution into the axes
	def drawHistogram(self, ax):
		ax.clear()
		ax.set_facecolor('k')

		# get the frequency range and divide into bins
		n_bins = round((f1_max - f1_min) / f1_inc)

		# set of unique labels (these are not necessarily contiguous and/or start with 0)
		unique_labels = set(self.class_labels)

		# get the x positions for the plot
		index = np.arange(n_bins)
		bar_width = 1 / len(unique_labels)

		# prepare label colors
		self.colors = []
		for each in np.linspace(0, 1, self.n_clusters):
			self.colors.append(plt.cm.cool(each))

		for i, k in enumerate(unique_labels):
			# use grey for not assigned to cluster
			if k == -1:
				col = [1, 1, 0, 1]
			else:
				col = self.colors[i]

			class_member_mask = (self.class_labels == k)

			# get the histogram
			freq_values = np.asarray(ex.collection.f1)[class_member_mask]
			events_per_bucket = np.histogram(freq_values,
											 bins=n_bins,
											 range=(f1_min, f1_max)
											 )[0]

			# plot histogram
			ax.plot(np.arange(f1_min, f1_max, f1_inc),
					events_per_bucket,
					color=tuple(col))

			# smoothed data (kernel density estimation)
			Y = freq_values.reshape(-1, 1)
			kde = KernelDensity(kernel='gaussian', bandwidth=f1_inc).fit(Y)
			x_plot = np.arange(f1_min, f1_max, f1_inc / 5).reshape(-1, 1)
			result = np.exp(kde.score_samples(x_plot))
			result[0] = 0               # force the first point to be 0 for fill to work as expected
			result[len(result)-1] = 0   # force the last point to be 0 for fill to work as expected
			# scale to number of events
			y_scale = np.max(events_per_bucket) / np.max(result)
			# plot
			ax.fill(x_plot,
					result * y_scale,
					color=tuple(col),
					alpha=0.6)

			f_min = max(f1_min, min(ex.collection.f1))
			f_max = min(f1_max, max(ex.collection.f1))
			ax.set_xlim(f_min, f_max)

			index = index + bar_width

		# pretty up the plot
		ax.set_title('Frequency Distribution')
		ax.set_xlabel('Frequency [Hz]')
		ax.set_ylabel('Events')
		ax.grid(True)

	# draw the time series into the axes
	def drawTimeseries(self, ax):
		ax.clear()
		ax.set_facecolor('k')

		# we sort all times into buckets for a complete day
		n_bins = 24  # one bucket per hour

		# for each event, get seconds since midnight as well as frequency
		t_event = np.asarray([(t.hour * 3600 + t.minute * 60 + t.second) for t in ex.collection.dt])
		t_range = 86400  # seconds in a day


		# set of unique labels.  These are not necessarily contiguous and/or starting at 0.
		unique_labels = set(self.class_labels)

		# get the x positions for the plot
		offset = datetime.datetime(2017, 1, 1)  # arbitrary date
		start_time = offset
		end_time = offset + datetime.timedelta(hours=24)
		index = [offset + datetime.timedelta(hours=24 * bin / n_bins) for bin in range(0, n_bins)]
		bar_width = datetime.timedelta(hours=24 / n_bins / len(unique_labels))

		# prepare label colors
		self.colors = []
		for each in np.linspace(0, 1, self.n_clusters):
			self.colors.append(plt.cm.cool(each))

		for i, k in enumerate(unique_labels):
			# use grey for not assigned to cluster
			if k == -1:
				col = [1, 1, 0, 1]
			else:
				col = self.colors[i]

			class_member_mask = (self.class_labels == k)

			# get the histogram
			events_per_interval = np.histogram(t_event[class_member_mask],
											   bins=n_bins,
											   range=(0, t_range)
											   )[0]
			ax.plot(index,
					events_per_interval,
					color=tuple(col)
					)

		# index = [(dt + bar_width) for dt in index]

		# format the axis labels
		fmt = mdates.DateFormatter('%H')
		loc = mdates.HourLocator(interval=3)
		ax.xaxis.set_major_formatter(fmt)
		ax.xaxis.set_major_locator(loc)

		ax.set_xlim(start_time, end_time)
		ax.grid(axis='x')

		# label axes
		ax.set_title('Activity Times')
		ax.set_xlabel('Time of Day')
		ax.set_ylabel('Events')


class Predict(Cluster):
	'''Evaluate a trained model'''

	def __init__(self, the_model, mp):
		super().__init__()
		self.n_clusters = the_model.n_classes
		self.min_probability = mp
		self.spec = [the_model.filepath + '; N = ' +
					 '\n(p ≥ {:0.2f})'.format(self.min_probability),
                     str(self.n_clusters)]
		self.model = the_model
		self.group_labels = np.append(the_model.class_labels,'Poor Match')

	def update(self):
		ex.statusBar.showMessage('Classification, this may take a while...')
		QApplication.processEvents()        # this will update the status bar

		# get data and perform scaling
		self.descriptor, X = ex.features.get()
		if self.descriptor != self.model.descriptor:    # do nothing if model doesn't match
			return
		# if self.model.scaler_name != global_scalers[0]: # check of there is a scaler stored with the model
		#	X = self.model.scaler.transform(X)          # if yes, scale the data based on stored scaler

		# do the classification (normal)
		if self.min_probability == 0:
			self.class_labels = self.model.classifier.predict(X)
			self.core_samples_mask = np.full(ex.collection.count, True)  # all are core samples

		else:
			# do the classification (predict probabilities)
			probabilities = self.model.classifier.predict_proba(X)

			self.class_labels = []
			self.core_samples_mask = []

			for i in range(ex.collection.count):
				self.class_labels.append(-1)
				self.core_samples_mask.append(False)
				for j in range(self.n_clusters):
					if probabilities[i, j] >= self.min_probability:
						self.class_labels[i] = j
						self.core_samples_mask[i] = True

			# save classification labels
			self.class_labels = np.asarray(self.class_labels)
			self.core_samples_mask = np.asarray(self.core_samples_mask)

		# metric
		# consider only values with a valid class label
		X = X[self.class_labels != -1, :]
		labels = self.class_labels[self.class_labels != -1]

		# must have at least two class labels for silhouette score
		if len(set(labels)) > 1:
			self.silhouette_score = metrics.silhouette_score(X, labels)
		else:
			self.silhouette_score = np.NaN


class myKMeans(Cluster):
	'''Perform the KMeans Clustering Operation'''

	def __init__(self, n_clusters):
		super().__init__()
		self.n_clusters = n_clusters
		self.spec = ['KMeans; N =', str(n_clusters)]
		# self.group_labels = np.empty(self.n_clusters, dtype=str)
		self.group_labels = [str(n) for n in np.arange(1, n_clusters + 1)]

	def update(self):
		ex.statusBar.showMessage('Clustering, this may take a while...')
		QApplication.processEvents()  # this will update the status bar

		# get data
		self.descriptor, X = ex.features.get()

		# compute KMeans model (we don't need to save it, just the result)
		self.class_labels = KMeans(n_clusters=self.n_clusters,
								   random_state=0,
								   n_jobs=-1). \
			fit_predict(X)

		# no core samples with this algorithm
		self.core_samples_mask = np.full(ex.collection.count, True)  # core samples mask (get plotted larger)

		# metrics
		self.silhouette_score = metrics.silhouette_score(X, self.class_labels)


class Spectral(Cluster):
	'''Perform the Spectral Clustering Operation'''

	def __init__(self, n_clusters):
		super().__init__()
		self.n_clusters = n_clusters
		self.spec = ['Spectral; N =', str(n_clusters)]
		# self.group_labels = np.empty(self.n_clusters, dtype=str)
		self.group_labels = [str(n) for n in np.arange(1, n_clusters + 1)]

	def update(self):
		ex.statusBar.showMessage('Clustering, this may take a while...')
		QApplication.processEvents()  # this will update the status bar

		# get data
		self.descriptor, X = ex.features.get()

		# compute KMeans model (we don't need to save it, just the result)
		self.class_labels = SpectralClustering(n_clusters=self.n_clusters,
											   affinity='nearest_neighbors',
											   n_init=25,
											   n_neighbors=4,
											   random_state=101557,
											   n_jobs=-1). \
			fit_predict(X)

		# no core samples with this algorithm
		self.core_samples_mask = np.full(ex.collection.count, True)  # core samples mask (get plotted larger)

		# metrics
		self.silhouette_score = metrics.silhouette_score(X, self.class_labels)


class DBScan(Cluster):
	'''Perform the DBScan Clustering Operation'''

	def __init__(self, eps):
		super().__init__()
		self.eps = eps
		self.spec = ['DBScan; ɛ =', '{:0.3f}'.format(self.eps)]

	def update(self, metric='euclidean'):
		ex.statusBar.showMessage('Cluster analysis, this may take a while...')
		QApplication.processEvents()  # this will update the status bar

		# get data
		self.descriptor, X = ex.features.get()

		# compute DBSCAN model (we don't need to save it, just the result)
		dbscan = DBSCAN(eps=self.eps, metric=metric).fit(X)

		# save the results
		self.core_samples_mask = np.zeros_like(dbscan.labels_, dtype=bool)
		self.core_samples_mask[dbscan.core_sample_indices_] = True
		self.class_labels = dbscan.labels_

		# Number of clusters in labels, ignoring noise if present.
		self.n_clusters = len(set(self.class_labels)) - (1 if -1 in self.class_labels else 0)
		# self.group_labels = np.empty(self.n_clusters, dtype=str)
		self.group_labels = np.arange(1, n_clusters + 1, dtype=str)

		# metrics
		if self.n_clusters > 1:
			self.silhouette_score = metrics.silhouette_score(X, dbscan.labels_)


class CKMeans(Cluster):
	'''ckmeans
	Sorts 1D data into clusters
	Developed by Wang & Song, The R Journal Vol. 3/2, December 2011
	Modifications by Michael Weber to return indices of data
	'''

	def __init__(self, n_clusters):
		super().__init__()
		self.n_clusters = n_clusters
		self.spec = ['CKMeans; N =', str(n_clusters)]
		# self.group_labels = np.empty(self.n_clusters, dtype=str)
		self.group_labels = [str(n) for n in np.arange(1, n_clusters + 1)]

	def ssq(self, j, i, sum_x, sum_x_sq):
		if (j > 0):
			muji = (sum_x[i] - sum_x[j - 1]) / (i - j + 1)
			sji = sum_x_sq[i] - sum_x_sq[j - 1] - (i - j + 1) * muji ** 2
		else:
			sji = sum_x_sq[i] - sum_x[i] ** 2 / (i + 1)

		return 0 if sji < 0 else sji

	def fill_row_k(self, imin, imax, k, S, J, sum_x, sum_x_sq, N):
		if imin > imax: return

		i = (imin + imax) // 2
		S[k][i] = S[k - 1][i - 1]
		J[k][i] = i

		jlow = k

		if imin > k:
			jlow = int(max(jlow, J[k][imin - 1]))
		jlow = int(max(jlow, J[k - 1][i]))

		jhigh = i - 1
		if imax < N - 1:
			jhigh = int(min(jhigh, J[k][imax + 1]))

		for j in range(jhigh, jlow - 1, -1):
			sji = self.ssq(j, i, sum_x, sum_x_sq)

			if sji + S[k - 1][jlow - 1] >= S[k][i]: break

			# Examine the lower bound of the cluster border
			# compute s(jlow, i)
			sjlowi = self.ssq(jlow, i, sum_x, sum_x_sq)

			SSQ_jlow = sjlowi + S[k - 1][jlow - 1]

			if SSQ_jlow < S[k][i]:
				S[k][i] = SSQ_jlow
				J[k][i] = jlow

			jlow += 1

			SSQ_j = sji + S[k - 1][j - 1]
			if SSQ_j < S[k][i]:
				S[k][i] = SSQ_j
				J[k][i] = j

		self.fill_row_k(imin, i - 1, k, S, J, sum_x, sum_x_sq, N)
		self.fill_row_k(i + 1, imax, k, S, J, sum_x, sum_x_sq, N)

	def fill_dp_matrix(self, data, S, J, K, N):
		sum_x = np.zeros(N, dtype=np.float_)
		sum_x_sq = np.zeros(N, dtype=np.float_)

		# median. used to shift the values of x to improve numerical stability
		shift = data[N // 2]

		for i in range(N):
			if i == 0:
				sum_x[0] = data[0] - shift
				sum_x_sq[0] = (data[0] - shift) ** 2
			else:
				sum_x[i] = sum_x[i - 1] + data[i] - shift
				sum_x_sq[i] = sum_x_sq[i - 1] + (data[i] - shift) ** 2

			S[0][i] = self.ssq(0, i, sum_x, sum_x_sq)
			J[0][i] = 0

		for k in range(1, K):
			if (k < K - 1):
				imin = max(1, k)
			else:
				imin = N - 1

			self.fill_row_k(imin, N - 1, k, S, J, sum_x, sum_x_sq, N)

	def ckmeans(self, data, n_clusters):
		# arguments data must be a list

		if n_clusters <= 0:
			raise ValueError("Cannot classify into 0 or less clusters")
		if n_clusters > len(data):
			raise ValueError("Cannot generate more classes than there are data values")

		# if there's only one value, return it; there's no sensible way to split
		# it. This means that len(ckmeans([data], 2)) may not == 2. Is that OK?
		unique = len(set(data))
		if unique == 1:
			return [data]

		sort_index = np.argsort(data)  # preserves original order of samples
		data.sort()
		n = len(data)
		class_labels = np.zeros(n, int)

		S = np.zeros((n_clusters, n), dtype=np.float_)

		J = np.zeros((n_clusters, n), dtype=np.uint64)

		self.fill_dp_matrix(data, S, J, n_clusters, n)

		clusters = []
		cluster_right = n - 1

		for cluster in range(n_clusters - 1, -1, -1):
			cluster_left = int(J[cluster][cluster_right])
			clusters.append(data[cluster_left:cluster_right + 1])

			class_labels[sort_index[cluster_left:cluster_right + 1]] = cluster

			if cluster > 0:
				cluster_right = cluster_left - 1

		return list(reversed(clusters)), class_labels

	def update(self):

		ex.statusBar.showMessage('Clustering, this may take a while...')
		QApplication.processEvents()  # this will update the status bar

		# get data and convert to list
		self.descriptor, X = ex.features.get()
		data = [x.item() for x in X]

		# perform the clustering
		ignore, self.class_labels = self.ckmeans(data, self.n_clusters)

		# no core samples with this algorithm
		self.core_samples_mask = np.full(ex.collection.count, True)  # core samples mask (get plotted larger)

		# metrics
		self.silhouette_score = metrics.silhouette_score(X, self.class_labels)


class Gaussian(Cluster):
	'''Perform Bayesian Gaussian Mixture analysis'''

	def __init__(self, n_clusters):
		super().__init__()
		self.n_clusters = n_clusters
		self.spec = ['Gaussian Mixture; N =', str(self.n_clusters)]
		# self.group_labels = np.empty(self.n_clusters, dtype=str)
		self.group_labels = [str(n) for n in np.arange(1, n_clusters + 1)]

	def update(self):
		ex.statusBar.showMessage('Cluster analysis, this may take a while...')
		QApplication.processEvents()  # this will update the status bar

		# get data
		self.descriptor, X = ex.features.get()

		# compute Dirichlet Process Gaussian Mixture
		dpgmm = BayesianGaussianMixture(n_components=self.n_clusters,
										random_state=101557,
										covariance_type='full').fit(X)

		# save the results
		self.class_labels = dpgmm.predict(X)

		# Number of clusters in labels, ignoring noise if present.
		self.n_clusters = len(set(self.class_labels)) - (1 if -1 in self.class_labels else 0)
		self.group_labels = np.arange(1, self.n_clusters + 1).astype(str)

		# no core samples with this algorithm
		self.core_samples_mask = np.full(ex.collection.count, True)  # core samples mask (get plotted larger)

		# metrics
		if self.n_clusters > 1:
			self.silhouette_score = metrics.silhouette_score(X, self.class_labels)


# show data in a scatter plot including class labels
class ScatterPlot():

	def __init__(self, title, xlabel, ylabel, grid_visible=False):
		self.title = title
		self.xlabel = xlabel
		self.ylabel = ylabel
		self.grid_visible = grid_visible
		self.lines = []

	def reset(self, ax):
		ax.clear()
		self.lines = []
		self.update(ax, self.x, self.y)
		self.show_highlight(ax)

	def draw(self, ax):         # make copy of plot in new axes
		ax.set_facecolor('k')
		ax.set_aspect('auto')

		cluster = ex.cluster  # this is the current cluster

		# prepare lines
		lines = []

		# set of unique labels.  These are not necessarily contiguous and/or starting at 0.
		unique_labels = set(cluster.class_labels)

		# prepare label colors
		colors = []
		for each in np.linspace(0, 1, cluster.n_clusters):
			colors.append(plt.cm.cool(each))

		for i, k in enumerate(unique_labels):
			# use yellow for noise
			if k == -1:
				col = [1, 1, 0, 1]
			else:
				col = colors[i]

			class_member_mask = (cluster.class_labels == k)

			x = self.x[class_member_mask & cluster.core_samples_mask]
			y = self.y[class_member_mask & cluster.core_samples_mask]
			line, = ax.plot(x, y, 'o',
							markerfacecolor=tuple(col),
							markeredgecolor=tuple(col),
							markersize=2)
			lines.append(line)

			x = self.x[class_member_mask & ~cluster.core_samples_mask]
			y = self.y[class_member_mask & ~cluster.core_samples_mask]
			line, = ax.plot(x, y, 'o',
							markerfacecolor=tuple(col),
							markeredgecolor=tuple(col),
							markersize=1)
			lines.append(line)

		# plot invisible markers.  These are the targets for the picker.
		x = self.x
		y = self.y
		line, = ax.plot(x, y, 'o',
						markerfacecolor='None',
						markeredgecolor='None',
						markersize=4,
						picker=True)
		line.set_pickradius(5)
		lines.append(line)

		if self.show_title:
			if self.descriptor == '':
				ax.set_title(self.title + ' (N = ' + str(self.count) + ')')
			else:
				ax.set_title(self.title + ' (' + self.descriptor + ')')
		ax.set_xlabel(self.xlabel)
		ax.set_ylabel(self.ylabel)
		ax.grid(self.grid_visible)
		ax.tick_params(labelbottom=self.labelbottom)

		return lines

	def update(self, ax, xdata, ydata, descriptor = '', show_title='True', labelbottom=True):
		self.descriptor = descriptor
		self.show_title = show_title
		self.labelbottom = labelbottom

		# clear the plot
		for line in self.lines:
			try:
				line.remove()
			except ValueError:
				ax.clear
				break

		# clear the highlight, also
		try:
			self.highlight.remove()
		except:
			pass

		# set up for drawing.
		# convert data to array if necessary
		self.x = np.asarray(xdata)
		self.y = np.asarray(ydata)
		self.count = len(self.x)    # number of datapoints

		self.lines = self.draw(ax)

	# highlight the current recording
	def show_highlight(self, ax):
		try:
			self.highlight.remove()
		except:
			pass
		self.highlight, = ax.plot(self.x[ex.collection.latest],
								  self.y[ex.collection.latest],
								  marker='o',
								  markerfacecolor='None', markeredgecolor='orange',
								  markersize=6)


# MAIN PROGRAM STARTS HERE
# ========================

#######################################################################
# Global Parameters                                                   #
#######################################################################

#######################################################################
# Save feature file                                                   #
#######################################################################
save_X = True

#######################################################################
# Standard Sampling Rate for Librosa wav file read                    #
#######################################################################
sr_librosa = 8000      # 8kHz

#######################################################################
# wav File Scale Factor for Display                                   #
#######################################################################
wavScaleFactor = -1

#######################################################################
# Target wav file length or segment for long WAV file conversion      #
#######################################################################
wav_segment_dur = 200   # ms per segment

#######################################################################
# Recording Duration                                                  #
#######################################################################
dur_noisefloor = 0.001  # noise floor (least significant bit)
dur_factor = 10         # factor above noise
dur_hw = 2              # half window for power averaging [ms]
rec_td_warning = 0.5    # warning: recording very close to previous

#######################################################################
# PSD Calculation Parameters                                          #
#######################################################################
min_rec_length = 256    # minimum length of a recording
welch_nperseg = 256     # segment length for Welch psd
welch_noverlap = 128    # overlap for Welch psd
welch_nfft = 256        # length of fft for Welch psd
welch_window = 'hann'   # window for fft
#welch_window = 'boxcar' # window for fft
welch_epsilon = 1E-10   # add to PSD to avoid zero values

#######################################################################
# PSD Analysis Parameters                                             #
#######################################################################
f_min = 200             # minimum frequency
f_max = 4000            # maximum frequency
n_mels = 26             # number of mel frequency bands
n_mfcc = 13             # number of coefficients
psd_pf_threshold = 0.25 # threshold for psd peak finding

#######################################################################
# Continuous Wavelet List                                             #
#######################################################################
cwt_wavelets = ['fbsp10-0.5-1', 'shan0.5-1', 'morl', 'mexh',
                'gaus1', 'gaus2', 'gaus3', 'gaus4', 'gaus5',
                'gaus6', 'gaus7', 'gaus8']
cwt_default_cwavelet = cwt_wavelets[0]
cwt_f_delta = 10
cwt_f_min = f_min
cwt_f_max = f_max
cwt_num = int((cwt_f_max - cwt_f_min) / cwt_f_delta + 1 + 0.5)
cwt_log_min = np.log10(0.0005)
cwt_log_max = np.log10(5)

#######################################################################
# Harmonic Spectral Product (HPS)                                     #
#######################################################################
hps_offset = 25         # offset for finding maximum
hps_harmonics = 6       # max harmonics (use f1_max to calculate)
hps_tol = 0.05          # tolerance to be added to f1 estimate

#######################################################################
# Autocorrelation (AC)                                                #
#######################################################################
ac_min = 0.5            # minimum value
ac_thres = 0.1          # peak finder threshold
ac_n_peaks = 2          # number of peaks to evaluate

#######################################################################
# Frequency Analysis Parameters                                       #
#######################################################################
ha_n_freq = 3           # number of harmonic frequencies to include
ha_n_peaks = 5          # number of harmonic peaks to include
f1_tolerance = 0.100    # tolerance for checking harmonic frequencies
f1_min = 100            # minimum fundamental frequency / delta
f1_max = 1000           # maximum fundamental frequency to look for
f1_inc = 10             # increment for plotting histogram
#######################################################################

#######################################################################
# Feature encoding                                                    #
#######################################################################
temp_min = 5            # minimum temperature
temp_max =  55          # maximum temperature
temp_step = 0.5         # temperature step for one-hot encoding
temp_weight = 12.5      # weight for temperature value
pca_components = 5      # number of components to use in PCA
rbm_components = 50     # number of components to use in RBM
global_scalers = ['NoScaler',
                  'StandardScaler',
                  'Normalizer',
                  'RobustScaler']   # Selection of scalers
#######################################################################


# if platform() == 'Darwin':  # How Mac OS X is identified by Python
#    system('''/usr/bin/osascript -e 'tell app "Finder" to set frontmost of process "python" to true' ''')

# create and run the interactive application
if __name__ == '__main__':
	QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
	app = QApplication(sys.argv)
	app.setStyle(QStyleFactory.create('Fusion'))
	ex = App()          # create the application window

	# show the main window, trigger the file dialog, and enter the GUI loop
	ex.show()
	ex.actionLoadDirectory.triggered.emit()

	sys.exit(app.exec_())
