import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt

from fault_profile_tool.profiles.profile import ProfileFault, ProfileLayout
from fault_profile_tool.displacement_distribution.dialog_options import SpinBoxCaption
from fault_profile_tool.displacement_distribution.graph_stack import GraphStack
from fault_profile_tool.profiles.profile import GraphHolder

# Generate test_data
test_x = np.arange(-100, 100)
test_y = np.random.rand(len(test_x))


class NavigationBox(QtWidgets.QGroupBox):
    def __init__(self, parent):
        super(NavigationBox, self).__init__()
        self.parent = parent
        layout = QtWidgets.QHBoxLayout()
        self.previous = QtWidgets.QPushButton("&Previous")
        self.next = QtWidgets.QPushButton("&Next")
        self.exit = QtWidgets.QPushButton("&Exit")
        self.save_button = QtWidgets.QPushButton("&Save GMT")
        self.exit.setDefault(False)
        self.exit.setAutoDefault(False)
        self.number = self.number_box()
        layout.addWidget(self.exit)
        layout.addWidget(self.save_button)
        layout.addWidget(self.previous)
        layout.addWidget(self.number)
        layout.addWidget(self.next)
        layout.setContentsMargins(0,0,0,0)

        self.exit.clicked.connect(self.parent.close)
        self.next.clicked.connect(self.parent.next_plot)
        self.previous.clicked.connect(self.parent.previous_plot)
        self.save_button.clicked.connect(self.parent.save_gmt)
        self.number.spin.valueChanged.connect(self.parent.spin_choose)
        self.setLayout(layout)

    def number_box(self):
        max_profiles = self.parent.number_profiles
        number = SpinBoxCaption(text=" of {:d}".format(max_profiles),
                                default=1, increment=1, double=False)
        number.spin.setMaximum(max_profiles)
        number.spin.setMinimum(1)

        return number


class ApplicationWindow(QtWidgets.QWidget):

    def __init__(self, fault: ProfileFault, save_file: str):
        super(ApplicationWindow, self).__init__()
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setWindowTitle("Profile drawing")
        self.setStyleSheet("QPushButton, QRadioButton, QCheckBox, QLabel {font: 11pt Arial}")
        self.fault = fault
        self.save_file = save_file

        self.graphs = [ProfileLayout(profile) for profile in self.fault.profiles]
        self.profile_stack = GraphStack(self.graphs)
        self.navigation = NavigationBox(parent=self)

        self.graphs = None
        self.layout = None
        self.profile_number = 0

        self.main_widget = QtWidgets.QWidget(self)
        self.layout = QtWidgets.QVBoxLayout(self.main_widget)
        self.layout.addWidget(self.profile_stack)
        self.layout.addWidget(self.navigation)
        self.layout.setSpacing(0)
        self.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
        self.disable_buttons()

        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        if self.fault.gmt_directory is None:
            self.navigation.save_button.setDisabled(True)

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_X:
            exclude_button = self.current_options.point_selection.exclude
            exclude_button.click()
        elif key == Qt.Key_I:
            include_button = self.current_options.point_selection.include
            include_button.click()

        elif key == Qt.Key_R:
            rect_button = self.current_options.point_selection.rect_button
            rect_button.toggle()

        elif key == Qt.Key_A:
            arrow_button = self.current_options.point_selection.arrow_button
            arrow_button.toggle()

        elif key == Qt.Key_L:
            lasso_button = self.current_options.point_selection.lasso_button
            lasso_button.toggle()

        elif key == Qt.Key_Z:
            zoom_button = self.current_options.size_adjust.rect_zoom_button
            zoom_button.click()

        elif key == Qt.Key_Right:
            self.next_plot()

        elif key == Qt.Key_Left:
            self.previous_plot()

    def closeEvent(self, QCloseEvent):
        quit_msg = "Are you sure you want to exit the program?"
        reply = QtWidgets.QMessageBox.question(self, 'Message',
                                               quit_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            self.fault.write_data(self.save_file)
            self.fault.save_all_profiles()
            self.save_gmt()
            QCloseEvent.accept()
        else:
            QCloseEvent.ignore()

    @property
    def number_profiles(self):
        return len(self.fault.profiles)

    @property
    def current_options(self):
        return self.profile_stack.currentWidget().profile.current_options

    def plot_profile(self, number):
        assert number in range(self.number_profiles)
        new_graphs = GraphHolder(self.fault.profiles[number])
        if self.graphs is not None:
            self.layout.replaceWidget(self.graphs, new_graphs, )
        self.graphs = new_graphs
        self.disable_buttons()

    def next_plot(self):
        if self.profile_number < len(self.fault.profiles) - 1:
            number = self.profile_number + 1
            self.choose_plot(number)

    def previous_plot(self):
        if self.profile_number > 0:
            number = self.profile_number - 1
            self.choose_plot(number)

    def spin_choose(self, number: int):
        self.choose_plot(number - 1)

    def choose_plot(self, number: int):
        self.fault.write_data(self.save_file)
        self.fault.save_all_profiles()
        self.profile_stack.setCurrentWidget(self.profile_stack.graphs[number])
        self.profile_number = number
        self.navigation.number.spin.setValue(number + 1)
        self.disable_buttons()

    def save_gmt(self):
        if self.fault.gmt_directory is not None:
            self.fault.write_all_gmt()

    def disable_buttons(self):
        if self.profile_number == len(self.fault.profiles)-1:
            self.navigation.next.setDisabled(True)
        else:
            self.navigation.next.setDisabled(False)

        if self.profile_number == 0:
            self.navigation.previous.setDisabled(True)
        else:
            self.navigation.previous.setDisabled(False)


class MyApp(QtWidgets.QApplication):
    def __init__(self, args):
        super(MyApp, self).__init__(args)

