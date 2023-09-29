# -*- coding: utf-8 -*-
"""
Functions and classes to support GUI interaction for scripts.

Makes use of Qt4.
"""

"""
Created on Fri Jul  3 19:41:35 2015

@author: mpalmer
"""

import sys
import pydicom as dicom
from pydicom.filereader import read_dicomdir
import os.path
from PyQt5 import QtGui, QtCore, QtWidgets
from PyQt5.QtCore import QCoreApplication

package_dir = os.path.split(os.path.realpath(__file__))[0]
icon_dir = os.path.join(package_dir, 'icons')

COPY_ICON_FN = os.path.join(icon_dir, 'copy.png')
SAVEAS_ICON_FN = os.path.join(icon_dir, 'saveas.png')

def attach_ctc(fig, name=''):
    """
    Attach a copy-to-clipboard function to the matplotlib toolbar.
    
    Parameters
    ----------
    fig : FigureCanvas
        Matplotlib's figure to attach. Returned by figure().
    name : str
        Optional name to set default filename on save-as dialog.
        
    """
    fig.attach_ctc = EnableCopyToClipboard(fig.canvas, name)
    
    
def _get_or_set_QApplication():
    """
    Utility routine to return the PyQt QApplication object if it's been
    created or create one if it doesn't yet exist.
    
    Parameters
    ----------
    QApplication : PyQt4.QtGui.QApplication
        if set then simply return as is.
        
    Returns
    -------
    QApplication : PyQt4.QtGui.QApplication
        object now in exisitence.
    """
    
    QApplication = QCoreApplication.instance()
    if QApplication == None:
        QApplication = QtWidgets.QApplication(sys.argv)
    return QApplication

class EnableCopyToClipboard():
    """
    Attach copy-to-clipboard action for the Matplotlib figure.
    
    Attaches an action to existing figure to copy the entire canvas to the 
    system clipboard as a graphical object.
    
    Parameters
    ----------
    canvas : QObject
        object to attach the copy signal.
    QApplication : PyQt4.QtGui.QApplication
        QApplication object.  If None, then look it up or create it.
        
    Methods
    -------
    action_callback :
        processes the copy canvas request
    """

    def __init__(self, canvas, filename = ''):
        self.canvas = canvas
        self.QApplication = _get_or_set_QApplication()        

        tb = canvas.toolbar

        tb.removeAction(tb.actions()[-1])

        copy_iconfn = os.path.join(sys.path[0], COPY_ICON_FN)
        copy_act = QtWidgets.QAction(QtGui.QIcon(copy_iconfn), 'Copy', canvas)
        copy_act.setShortcut('Ctrl+C')
        copy_act.setToolTip('Copy to clipboard')
#        tb.insertAction(tb.actions()[-1], act)
        tb.addAction(copy_act)

        copy_act.triggered.connect(self.copy_callback)
        self.canvas.addAction(copy_act)

        """
        saveas_iconfn = os.path.join(sys.path[0], SAVEAS_ICON_FN)
        saveas_act = QtWidgets.QAction(QtGui.QIcon(saveas_iconfn), 'Save image file', canvas)
        saveas_act.setShortcut('Ctrl+S')
        saveas_act.setToolTip('Save as...')
        #        t.insertAction(t.actions()[-1], act)
        tb.addAction(saveas_act)

        self.save_image_dir = '.'
        self.filename = filename

        saveas_act.triggered.connect(self.saveas_callback)
        self.canvas.addAction(saveas_act)
        """

    def copy_callback(self, ec):
        # Qt4: pixmap = QtGui.QPixmap.grabWidget(self.canvas)
        pixmap = QtWidgets.QWidget.grab(self.canvas)
        self.QApplication.clipboard().setPixmap(pixmap)

    def saveas_callback(self, ec):
        fn, _ = QtWidgets.QFileDialog.getSaveFileName(self.canvas, 'Save Image',
                                                      os.path.join(self.save_image_dir, self.filename + '.png'),
                                                      'PNG (*.png);;JPG (*.jpg)')
        if fn is None or len(fn) == 0:
            return
        self.save_image_dir = os.path.dirname(fn)
        pixmap = QtWidgets.QWidget.grab(self.canvas)
        pixmap.save(fn)


#class DicomDirWindow(QtGui.QMainWindow):
class DicomDirWindow(QtWidgets.QDialog):
    """
    Implements a structured layout for displaying and interacting with 
    contents of DICOMDIR.  Used by gui_dicomdir.
    """
    def __init__(self):
        super().__init__()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.resize(800, 240)
        self.setModal(True)

        self.main_widget = QtWidgets.QWidget(self)

        layout = QtWidgets.QVBoxLayout(self.main_widget)

        box_layout = QtWidgets.QHBoxLayout()
        
        self.ptbox = QtWidgets.QListWidget(self.main_widget)
        self.stbox = QtWidgets.QListWidget(self.main_widget)
        self.sebox = QtWidgets.QListWidget(self.main_widget)
        box_layout.addWidget(self.ptbox)
        box_layout.addWidget(self.stbox)
        box_layout.addWidget(self.sebox)
        
        but_layout = QtWidgets.QHBoxLayout()
        self.gobutton = QtWidgets.QPushButton('Go')
        but_layout.addWidget(self.gobutton)
        but_layout.addStretch(1)
        

        layout.addLayout(box_layout)    
        layout.addLayout(but_layout)

        self.main_widget.setFocus()
#        self.setCentralWidget(self.main_widget)



class BoxIF():
    def __init__(self, base_dir, dcmdir, aw):
        
        self.base_dir = base_dir
        self.dcmdir = dcmdir
        self.ptbox = aw.ptbox
        self.stbox = aw.stbox
        self.sebox = aw.sebox
        self.gobutton = aw.gobutton
        
        self.complete = aw.close

        self.patrec = None
        self.study = None
        self.series = None
        self.image_filenames = []

        self.ptbox.itemClicked.connect(self.pt_select)
        self.stbox.itemClicked.connect(self.st_select)
        self.sebox.itemClicked.connect(self.se_select)

        self.gobutton.clicked.connect(self.go_pressed)

#initial population
    
        for patrec in dcmdir.patient_records:
            self.ptbox.addItem("{0.PatientID}: {0.PatientName}".format(patrec))
            
    def pt_select(self, item):
#        print(dir(item))

    
        self.patrec = self.dcmdir.patient_records[self.ptbox.row(item)]
 
        studies = self.patrec.children

        self.stbox.clear()     
        self.sebox.clear()
        self.study = None
        self.series = None
        
        for study in studies:
            sid = getattr(study, 'studyID', '<>')
            std = getattr(study, 'StudyDate', '<>')
            sde = getattr(study, 'StudyDescription', '<>')
            self.stbox.addItem('%s: %s: %s' % (sid, std, sde))
            #self.stbox.addItem(("{0.StudyID}: {0.StudyDate}:"
            #      " {0.StudyDescription}".format(study)))

#        print(patrec)       
#        print(item.listWidget.text())

    def st_select(self, item):

        studies = self.patrec.children
        self.study = studies[self.stbox.row(item)]
        
        self.sebox.clear()
        self.series = None

        all_series = self.study.children
        for series in all_series:
            image_count = len(series.children)
            plural = ('', 's')[image_count > 1]

            # Write basic series info and image count

            # Put N/A in if no Series Description
            if 'SeriesDescription' not in series:
                series.SeriesDescription = "N/A"
            self.sebox.addItem((" " * 8 + "Series {0.SeriesNumber}:  {0.Modality}: {0.SeriesDescription}"
                  " ({1} image{2})".format(series, image_count, plural)))
                  
    def se_select(self, item):
        all_series = self.study.children
        self.series = all_series[self.sebox.row(item)]

#        for image_filename in image_filenames:
#            print(image_filename)
            
    def go_pressed(self):
        if not all([self.patrec, self.study, self.series]):
            return

        image_records = self.series.children
        self.image_filenames = [os.path.join(self.base_dir, *image_rec.ReferencedFileID)
                           for image_rec in image_records]

        self.complete()



def gui_dicomdir(dfilename):
    """
    GUI interface to DICOMDIR for series selection.
    
    Puts up an Qt object containing three list boxes and a "go" button.
    Can be called inside a Qt application loop, or can be called from non-Qt
    application in which case one will be initiated for the duration of this
    dialog and then terminated. If user quits then system exit is called.
    
    Parameters
    ----------
    dfilename : str
        path to DICOMDIR file.
        
    Returns
    -------
    image_filenames : list of strings
        fully-qualified paths to individual files in the selected series. 
    """
    

    qApp = QCoreApplication.instance()
    if qApp == None:
        stand_alone = True
        qApp = QtWidgets.QApplication(sys.argv)
    else:
        stand_alone = False    

    aw = DicomDirWindow()

    dcmdir = read_dicomdir(dfilename)
    base_dir = os.path.dirname(dfilename)
    
    pb = BoxIF(base_dir, dcmdir, aw)

    aw.setWindowTitle(sys.argv[0] + ' - ' + dfilename)
    aw.show()

    if stand_alone:
        qApp.exec_()
        if len(pb.image_filenames) == 0:
            sys.exit()
    else:
        aw.exec_()

    return pb.image_filenames  
    


