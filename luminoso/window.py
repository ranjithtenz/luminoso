from __future__ import with_statement
from PyQt4 import QtCore, QtGui
from PyQt4.QtCore import Qt

from luminoso.study import StudyDirectory, Study, StudyLoadError
from luminoso.ui import LuminosoUI
from luminoso.batch import progress_reporter

from luminoso.whereami import package_dir, get_icon
from luminoso.simplethread import ThreadRunner

from luminoso.csv_reader import CSVFile, CSVReader

from webbrowser import open as webo

import sys, os, time

import logging
logger = logging.getLogger('luminoso')
#formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

logger.setLevel(logging.INFO)

VERSION = "1.3.2"
DEFAULT_MESSAGE = """
<h2>Luminoso %(VERSION)s</h2>
<p>Choose "New Study", "Open Study" or "Import CSV File" to begin.</p>
""" % globals()

class MainWindow(QtGui.QMainWindow):
    """
    The main Luminoso window.
    """
    def __init__(self):
        QtGui.QMainWindow.__init__(self)

        self.ui = LuminosoUI(self)
        self.toolbar = self.addToolBar("Toolbar")
        self.toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        #self.toolbar.setIconSize(QtCore.QSize(24, 24))
        self.setUnifiedTitleAndToolBarOnMac(True)
        self.setCentralWidget(self.ui)

        self.study = None
        self.results = None
        self.already_closed = False

        self.menus = {}
        self.actions = {}

        # A silly trick to make instance methods available from the console.
        # "The biggest self of self is self." -- Gov. Mark Sanford
        self.self = self

        self.setWindowTitle("Luminoso %s" % VERSION)
    
    def setup(self):
        """
        This is stuff that for some inscrutable reason doesn't go in __init__.
        """
        self.path = os.curdir

        
        # Set up the embedded console.
        if '--console' in sys.argv:
            from spyderlib.plugins.console import Console
            self.console = Console(self, commands=[], namespace=self.__dict__)
            self.ui.tab_stack.addTab(self.console, "Console")
            self.console.setVisible(True)
            self.console_font = QtGui.QFont("Monaco", 10)
            self.console.shell.set_font(self.console_font)
            self.console.shell.execute_command("cls")
        
        logger.addHandler(logging.StreamHandler(sys.stdout))
        logger.addHandler(logging.StreamHandler(sys._stdout))

        # Connect the tree to the filesystem
        self.dir_model = QtGui.QFileSystemModel()
        self.ui.tree_view.setModel(self.dir_model)
        self.ui.show_info(DEFAULT_MESSAGE)
        
        # Set up signals
        self.ui.tree_view.clicked.connect(self.select_document)
        self.ui.axes_spinbox.valueChanged.connect(self.set_num_axes)
        self.ui.cutoff_spinbox.valueChanged.connect(self.set_concept_cutoff)
        self.ui.svdview_panel.svdSelectEvent.connect(self.svdview_select)

        self.setup_menus()
        # Disable elements that require a study loaded.
        self.study_loaded(False)

        self.ui.search_box.returnPressed.connect(self.toolbar_search)
        self.ui.search_button.clicked.connect(self.toolbar_search)
        self.toolbar.addWidget(self.ui.search_panel)

    def __del__(self):
        # De-Sanfordize so that the garbage collector can do its job.
        del self.self
    
    def svdview_select(self):
        #Figure out which concept has been selected
        label = self.ui.svdview_panel.get_selected_label()

        if label is not None:
            self.selected_label(label)

    def selected_label(self, label):
        index = self.ui.tree_view.find_document_index(label)
        if index is not None:
            self.ui.tree_view.setSelection(self.ui.tree_view.visualRect(index), self.ui.tree_view.selectionModel().ClearAndSelect)

            text = self.ui.tree_view.get_file_contents_at(index)
            self.ui.show_document_info(label, text)
        elif self.results is not None:
            # we might be able to show some information here
            info = self.results.get_concept_info(label)
            if info is not None: self.ui.show_info(info)

    @QtCore.pyqtSlot()
    def toolbar_search(self):
        self.search(self.ui.search_box.text())

    def search(self, text):
        self.ui.svdview_panel.find_point(text)

    def select_document(self, modelIndex):
        filename = self.ui.tree_view.get_filename_at(modelIndex)
        self.ui.svdview_panel.find_point(filename)
        if filename == u'NotImplemented': filename = ''

        #Changes the text in the info panel
        text = self.ui.tree_view.get_file_contents_at(modelIndex)
        self.ui.show_document_info(filename, text)

    def set_num_axes(self, axes):
        if self.study is None: return
        else: self.study_dir.set_setting('axes', axes)

    def set_concept_cutoff(self, cutoff):
        if self.study is None: return
        else: self.study_dir.set_setting('concept_cutoff', cutoff)

    def add_action(self, menu, name, func, shortcut=None, toolbar_icon=None):
        """
        Define an action that can be taken using the GUI, and associate it with
        a slot. Add it to a menu, optionally to the toolbar, and optionally
        associate it with a keyboard shortcut.
        """
        if menu not in self.menus:
            self.menus[menu] = self.menuBar().addMenu(menu)
        if toolbar_icon is not None:
            action = QtGui.QAction(QtGui.QIcon(get_icon(toolbar_icon)), name, self)
        else:
            action = QtGui.QAction(name, self)
        self.actions[name] = action
        action.triggered.connect(func)
        self.menus[menu].addAction(action)
        if shortcut is not None:
            action.setShortcut(shortcut)
        if toolbar_icon is not None:
            self.toolbar.addAction(action)
        return action

    def setup_menus(self):
        self.add_action("&File", "&New study...", self.new_study_dialog, "Ctrl+N", 'actions/document-new.png')
        self.add_action("&File", "&Open study...", self.load_study_dialog, "Ctrl+O", 'actions/document-open.png')
        self.add_action("&File", "&Edit study...", self.edit_study, "Ctrl+E", 'actions/document-properties.png')
        self.add_action("&File", "&Import CSV File...", self.csv_study, "Ctrl+I", 'actions/csv_file.png')
        self.add_action("&Analysis", "&Analyze", self.analyze, "Ctrl+A", 'actions/go-next.png')
        self.add_action("&Analysis", "Show Study &Info", self.show_info, "Ctrl+I", "actions/edit-find.png")
        self.add_action("&Viewer", "&Reset view", self.ui.svdview_panel.reset_view, "Ctrl+R")
        self.toolbar.addSeparator()
        self.add_action("&Viewer", "&Previous axis", self.ui.svdview_panel.prev_axis, "Ctrl+Left", 'actions/media-seek-backward.png')
        self.add_action("&Viewer", "&Next axis", self.ui.svdview_panel.next_axis, "Ctrl+Right", 'actions/media-seek-forward.png')
        self.add_action("&Viewer", "Save as &SVG...", self.save_svg, "Ctrl+G")
        self.toolbar.addSeparator()
        self.add_action("&Help", "&About...", self.info_luminoso)
        self.add_action("&Help", "&Documentation...", self.doc_luminoso)
    
    def save_svg(self):
        filename = QtGui.QFileDialog.getSaveFileName(self, "Choose where to save this SVG", package_dir)
        if filename:
            self.ui.svdview_panel.write_svg(filename)

    def new_study_dialog(self):
        dirname = QtGui.QFileDialog.getSaveFileName(self, "Choose where to save this study", package_dir)
        if dirname:
            study = StudyDirectory.make_new(unicode(dirname))
            self.load_study(unicode(dirname))

    def csv_study(self):
        csv_file = QtGui.QFileDialog.getOpenFileNames(None,"Select a CSV File to open.","/home")
        filename = csv_file[0]
        if filename and filename.endsWith(QtCore.QString('.csv')):
            csv = CSVFile(unicode(filename))
            self.ui.show_info("<h3>Creating Study...</h3><p>Study being located at "
                              +str(csv.study_path)+"</p><p>(this may take a few minutes)</p>")
            reader = CSVReader(csv)
            reader.read_csv()
            self.load_study(unicode(csv.study_path))

    def info_luminoso(self):
        webo('http://csc.media.mit.edu/analogyspace/luminoso')

    def doc_luminoso(self):
        webo('http://csc.media.mit.edu/docs/luminoso/index.html')

    def load_study_dialog(self):
        dir = QtGui.QFileDialog.getExistingDirectory()
        if dir:
            self.load_study(unicode(dir))

    def load_study(self, dir):
        """
        Loads a specified study into the file browser and the SVDview.
        """
        self.set_study_dir(dir)
        self.ui.show_info("<h3>Loading...</h3>")
        try:
            with progress_reporter(self, 'Loading study %s' % dir, 7) as progress:
                progress.set_text('Loading.')
                self.study = self.study_dir.get_study()
                self.study.status_callback = progress.tick
                progress.set_text('Loading analysis.')
                results = self.study_dir.get_existing_analysis()
                progress.tick('Updating view.')
                self.update_svdview(results)
                progress.tick('Updating options.')
                self.update_options()
                self.study.status_callback = None

            self.results = results
            self.show_info()
            self.study_loaded() # TODO: Make it a slot.
        except StudyLoadError:
            self.ui.show_info("%s is not a valid study directory." % dir)

    def study_loaded(self, loaded=True):
        '''
        Enable study actions when the study is loaded.
        '''
        for action_name in ['&Edit study...', "Show Study &Info", '&Analyze',
        '&Next axis', '&Previous axis']:
            self.actions[action_name].setEnabled(loaded)

    
    def edit_study(self):
        #TODO: Fix when no study loaded
        dir = QtCore.QDir.toNativeSeparators(self.dir_model.rootPath())
        platforms = {"darwin" : "open " , 
                     "win32" : "explorer ", 
                     "cygwin" : "explorer ",
                     "linux2" : "xdg-open "}

        system = sys.platform

        if dir:
            if (system in platforms):
                os.system(platforms[system] + str(dir))
            else:
                print "OS not supported"
    
    def update_options(self):
        axes = self.study_dir.settings.get('axes')
        cutoff = self.study_dir.settings.get('concept_cutoff')
        if axes is not None:
            self.ui.set_num_axes(axes)
        if cutoff is not None:
            self.ui.set_concept_cutoff(cutoff)
    
    def get_svdview(self):
        """
        Get the actual SVDView component, assuming it exists.
        """
        return self.ui.svdview_panel.viewer

    def update_svdview(self, results):
        """
        Let the SVDView component know that it should load new data.
        """
        if results is None:
            self.ui.svdview_panel.deactivate()
        else:
            self.ui.svdview_panel.activate(results.docs, results.projections,
                                           results.magnitudes,
                                           results.canonical_filenames)
    
    def show_info(self):
        if self.results is not None:
            self.ui.show_info(self.results.get_info())
        else:
            self.ui.show_info("Click <b>Analyze</b> to analyze this study.")

    def analyze(self):
        """
        Called when the Analyze button is clicked.

        This is meant to be used as a slot, but someone could also type
        `self.analyze()` from the console if they wanted.
        """
        logger.info('Start analysis')
        self.ui.svdview_panel.deactivate()
        self.ui.show_info("<h3>Analyzing...</h3><p>(this may take a few minutes)</p>")
        
        with progress_reporter(self, 'Analyzing...', 8) as progress:
            self.study.status_callback = progress.tick
            results = self.study_dir.analyze()
            logger.info('Analysis finished.')
            progress.tick('Updating view')
            self.update_svdview(results)
            self.results = results
            self.show_info()
            self.study.status_callback = None

    def set_study_dir(self, dir):
        self.dir_model.setRootPath(dir)
        index = self.dir_model.index(dir)
        self.ui.tree_view.setRootIndex(index)
        self.ui.tree_view.hideColumn(1)
        self.ui.tree_view.hideColumn(2)
        self.ui.tree_view.hideColumn(3)

        self.study_dir = StudyDirectory(dir)
        
        # Expand the trees that should be initially visible
        self.ui.tree_view.expand(self.dir_model.index(self.dir_model.rootPath()+'/Canonical'))
        self.ui.tree_view.expand(self.dir_model.index(self.dir_model.rootPath()+'/Documents'))
        self.ui.tree_view.expand(self.dir_model.index(self.dir_model.rootPath()+'/Matrices'))

    #def sizeHint(self):
    #    """
    #    If Qt is nice enough to ask, how big should our window be?
    #    """
    #    return QtCore.QSize(1000, 800)
