from PyQt5 import QtWidgets, QtGui
# from icp_error.profiles.manual_slip import Profile, ComponentGraph
# from icp_error.displacement_distribution.profile_operations import ProfileDirection
from fault_profile_tool.displacement_distribution.canvas import MapCanvas
from fault_profile_tool.displacement_distribution.graph_stack import GraphStack, MapStack
from typing import Union


class CheckableButton(QtWidgets.QToolButton):
    def __init__(self, name: str):
        super(CheckableButton, self).__init__()
        self.setText(name)
        self.setCheckable(True)
        # self.setAutoExclusive(True)


class OnePushButton(QtWidgets.QToolButton):
    def __init__(self, name: str):
        super(OnePushButton, self).__init__()
        self.setText(name)


class SpinBoxCaption(QtWidgets.QWidget):
    def __init__(self, text: str, default: Union[int, float], increment: Union[int, float],
                 double: bool = True, min_value: int = -20000, max_value: int = 20000):
        super(SpinBoxCaption, self).__init__()
        # Layout
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        # spin box widget
        if double:
            self.spin = QtWidgets.QDoubleSpinBox()
        else:
            self.spin = QtWidgets.QSpinBox()
            assert all([isinstance(x, int) for x in (default, increment)])
        self.spin.setMinimum(min_value)
        self.spin.setMaximum(max_value)
        self.spin.setFixedWidth(70)
        self.spin.setFont(QtGui.QFont("Arial", 11))

        self.spin.setValue(default)
        self.spin.setSingleStep(increment)

        # Add to layout
        self.layout.addWidget(self.spin)
        # Label
        self.text = QtWidgets.QLabel(text)
        self.text.setFont(QtGui.QFont("Arial", 11))
        self.layout.addWidget(self.text)
        self.setLayout(self.layout)
        self.setFixedWidth(280)


class SideOptions(QtWidgets.QWidget):
    def __init__(self, profile):
        super(SideOptions, self).__init__()
        self.profile = profile
        num_components = self.profile.num_components
        if num_components == 1:
            component_names = ["Displacements"]
        else:
            component_names = ["East", "North", "Vertical"]
        profile_components = (self.profile.x1, self.profile.x2,
                              self.profile.x3)
        graph_components = self.profile.graphs.components
        self.components = []
        for name, profile_component, graph_component in zip(component_names[:num_components],
                                                            profile_components[:num_components],
                                                            graph_components[:num_components]):
            component = ComponentOptions(self.profile, profile_component, self,
                                         name, graph_component)
            self.components.append(component)

        self.x1, self.x2, self.x3 = self.profile.return_excess_nones(self.components)
        self.component_options = GraphStack(self.components)
        self.transferable = TransferableOptions(profile=self.profile, options=self)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.transferable)
        layout.addWidget(self.component_options)
        self.setLayout(layout)


class ComponentOptions(QtWidgets.QGroupBox):
    def __init__(self, profile, component, options: SideOptions,
                 title: str, graph,
                 min_x: float = None, max_x: float = None,
                 min_y: float = None, max_y: float = None):
        super(ComponentOptions, self).__init__()
        # self.setStyleSheet("QCheckBox {font: 11pt Arial}")
        self.profile = profile
        self.all_options = options
        self.component = component
        self.graph = graph
        self.graph.options = self
        self.title = title
        self.setTitle(title)
        self.button = None
        if any([x is None for x in (min_x, max_x)]):
            self.min_x_num, self.max_x_num = (-profile.length/2, profile.length/2)
        if any([x is None for x in (min_y, max_y)]):
            self.min_y_num, self.max_y_num = (min(component.displacements),
                                              max(component.displacements))

        self.fit_display = FitDisplay(self.profile)

        self.size_adjust = SizeAdjuster(self.graph, self.all_options, self.min_x_num, self.min_y_num,
                                        self.max_x_num, self.max_y_num)
        self.point_selection = PointSelection(profile, self.graph, self.all_options, self)
        layout = QtWidgets.QVBoxLayout()
        for widget in (self.fit_display, self.size_adjust,
                       self.point_selection):
            layout.addWidget(widget)
        layout.setSpacing(0)
        self.setLayout(layout)

    @property
    def fit_flag(self):
        return self.fit_display.fit_flag


class TransferableOptions(QtWidgets.QGroupBox):
    def __init__(self, profile, options: SideOptions):
        super(TransferableOptions, self).__init__()
        self.profile = profile
        self.options = options
        self.setFixedSize(270, 150)
        layout = QtWidgets.QVBoxLayout()
        self.radio = RadioButtons(profile, options)
        if profile.num_components > 1:
            for component, button in zip(self.options.components, self.radio.components):
                component.button = button
        else:
            self.radio.connect_x1()

        self.adjuster = AdjustProfile(profile)
        self.exclude_radios = ExcludeRadios()

        layout.addWidget(self.radio)
        layout.addWidget(self.adjuster)
        layout.addWidget(self.exclude_radios)

        self.setLayout(layout)


class RadioButtons(QtWidgets.QWidget):
    def __init__(self, profile, options: SideOptions):
        super(RadioButtons, self).__init__()

        self.profile = profile
        self.options = options
        if self.profile.num_components == 1:
            self.setFixedSize(250, 5)
            self.layout = QtWidgets.QHBoxLayout()
            self.setLayout(self.layout)
        else:
            self.setFixedSize(250, 40)
            self.layout = QtWidgets.QHBoxLayout()
            self.x1 = QtWidgets.QRadioButton("East")
            self.x2 = QtWidgets.QRadioButton("North")
            if self.profile.num_components == 2:
                for button in (self.x1, self.x2):
                    self.layout.addWidget(button)
                    self.components = (self.x1, self.x2)

            else:
                self.x3 = QtWidgets.QRadioButton("Vertical")
                self.components = (self.x1, self.x2, self.x3)

            for button in self.components:
                self.layout.addWidget(button)

            self.setLayout(self.layout)
            self.x1.toggle()

            all_functions = [self.connect_x1, self.connect_x2, self.connect_x3]

            for button, function_i in zip(self.components,
                                          all_functions[:self.profile.num_components]):

                button.toggled.connect(function_i)

            self.connect_x1()

    def connect_x1(self):
        self.profile.current_graph = self.profile.graphs.x1
        self.set_current_options(self.options.x1)
        self.set_current_map(self.profile.maps.x1)
        self.profile.redraw()

    def connect_x2(self):
        self.profile.current_graph = self.profile.graphs.x2
        self.set_current_options(self.options.x2)
        self.set_current_map(self.profile.maps.x2)
        self.profile.redraw()

    def connect_x3(self):
        self.profile.current_graph = self.profile.graphs.x3
        self.set_current_options(self.options.x3)
        self.set_current_map(self.profile.maps.x3)
        self.profile.redraw()

    def set_current_options(self, options_box: ComponentOptions):
        self.profile.current_options = options_box
        self.options.component_options.setCurrentWidget(options_box)

    def set_current_map(self, map_canvas: MapCanvas):
        self.profile.current_map = map_canvas
        self.profile.maps.stack.setCurrentWidget(map_canvas)
        map_canvas.draw()


class ExcludeRadios(QtWidgets.QWidget):
    def __init__(self, num_components: int = 3):
        super(ExcludeRadios, self).__init__()
        self.setFixedSize(250, 40)
        self.num_components = num_components
        self.layout = QtWidgets.QHBoxLayout()
        self.ignore = QtWidgets.QCheckBox("Ignore profile")
        self.layout.addWidget(self.ignore)
        if num_components > 1:
            self.all = QtWidgets.QRadioButton("All")
            self.vertical = QtWidgets.QRadioButton("V. only")
            self.horizontal = QtWidgets.QRadioButton("H. only")
            self.layout.addWidget(self.all)
            self.layout.addWidget(self.vertical)
            self.layout.addWidget(self.horizontal)

        self.setLayout(self.layout)

        if num_components > 1:
            self.all.setChecked(True)

    def set_measurement_type(self, measurement_type: str):
        assert(measurement_type in ["All", "Vertical", "Horizontal"])
        if measurement_type == "All":
            self.all.setChecked(True)
        elif measurement_type == "Vertical":
            self.vertical.setChecked(True)
        else:
            self.horizontal.setChecked(True)

    @property
    def selection(self):
        if self.num_components > 1:
            if self.all.isChecked():
                return "All"
            elif self.vertical.isChecked():
                return "Vertical"
            else:
                return "Horizontal"
        else:
            return None


class AdjustProfile(QtWidgets.QWidget):
    def __init__(self, profile):
        super(AdjustProfile, self).__init__()
        self.profile = profile
        self.setFixedSize(250, 50)
        self.layout = QtWidgets.QHBoxLayout()
        self.centre_button = SpinBoxCaption("Cen. offset (m)", 0, 5)
        self.strike_button = SpinBoxCaption("Strike offset (deg)", 0, 1)
        self.centre_button.spin.valueChanged.connect(self.change_centre)
        self.strike_button.spin.valueChanged.connect(self.change_strike)
        self.layout.addWidget(self.centre_button)
        self.layout.addWidget(self.strike_button)
        self.setLayout(self.layout)

    def change_centre(self):
        self.profile.centre_change(self.centre_button.spin.value())

    def change_strike(self):
        self.profile.strike_change(self.strike_button.spin.value())


class FitDisplay(QtWidgets.QGroupBox):
    def __init__(self, profile):
        super(FitDisplay, self).__init__()
        self.profile = profile
        self.setTitle("Best fit")
        if profile.num_components == 1:
            self.setFixedSize(250, 105)
        else:
            self.setFixedSize(250, 80)
        self.layout = QtWidgets.QGridLayout()
        self.spin_layout, self.spin, self.checkbox = None, None, None
        self.best_offset = QtWidgets.QLabel("Offset: ")
        self.best_error = QtWidgets.QLabel("Error: ")
        if profile.num_components == 1:
            self.lslope = QtWidgets.QLabel("LSlope: ")
            self.rslope = QtWidgets.QLabel("RSlope: ")
        self.show_fit = QtWidgets.QCheckBox("Show best fit")
        self.show_fit.setChecked(True)
        self.show_fit.toggled.connect(self.profile.redraw)

        # Add widgets to group box
        self.layout.addWidget(self.best_offset, 1, 0)
        self.layout.addWidget(self.best_error, 1, 1)
        if profile.num_components == 1:
            self.layout.addWidget(self.lslope, 2, 0)
            self.layout.addWidget(self.rslope, 2, 1)
            self.layout.addWidget(self.show_fit, 3 , 0)
        else:
            self.layout.addWidget(self.show_fit, 2, 0)

        self.setLayout(self.layout)

    @property
    def fit_flag(self):
        return self.show_fit.isChecked()


class SizeAdjuster(QtWidgets.QGroupBox):
    def __init__(self, figure, all_options: SideOptions, min_x: float, min_y: float,
                 max_x: float, max_y: float, x_increment: float = 10.,
                 y_increment: float = 1.):
        super(SizeAdjuster, self).__init__()
        self.figure = figure
        self.all_options = all_options
        self.setFixedSize(250, 150)
        self.setTitle("Adjust size")
        layout = QtWidgets.QGridLayout()
        self.min_x_spin = self.linked_spin_box("Min. X", min_x, x_increment)
        self.max_x_spin = self.linked_spin_box("Max. X", max_x, x_increment)
        self.min_y_spin = self.linked_spin_box("Min. Y", min_y, y_increment)
        self.max_y_spin = self.linked_spin_box("Max. Y", max_y, y_increment)
        self.change_check = QtWidgets.QCheckBox("Scale all X")

        self.rect_zoom_button = CheckableButton("Rect. zoom")
        self.rect_zoom_button.clicked.connect(self.connect_zoom)

        self.zoom_extents_button = OnePushButton("Zoom extents")
        self.zoom_extents_button.clicked.connect(self.figure.zoom_all)
        self.zoom_included_button = OnePushButton("Zoom included")
        self.zoom_included_button.clicked.connect(self.figure.zoom_included)

        # self.max_x_spin.spin.valueChanged.connect()

        layout.addWidget(self.min_x_spin, 0, 0)
        layout.addWidget(self.max_x_spin, 0, 1)
        layout.addWidget(self.min_y_spin, 1, 0)
        layout.addWidget(self.max_y_spin, 1, 1)
        layout.addWidget(self.change_check, 2, 0)
        layout.addWidget(self.rect_zoom_button, 2, 1)
        layout.addWidget(self.zoom_extents_button, 3, 0)
        layout.addWidget(self.zoom_included_button, 3, 1)
        self.setLayout(layout)

    @property
    def zoom_flag(self):
        return self.rect_zoom_button.isChecked()

    @property
    def x_min(self):
        return self.min_x_spin.spin.value()

    @x_min.setter
    def x_min(self, value: float):
        self.min_x_spin.spin.setValue(value)
        self.figure.change_limits()

    @property
    def x_max(self):
        return self.max_x_spin.spin.value()

    @x_max.setter
    def x_max(self, value: float):
        self.max_x_spin.spin.setValue(value)
        self.figure.change_limits()

    @property
    def y_min(self):
        return self.min_y_spin.spin.value()

    @y_min.setter
    def y_min(self, value: float):
        self.min_y_spin.spin.setValue(value)
        self.change_limits()

    @property
    def y_max(self):
        return self.max_y_spin.spin.value()

    @y_max.setter
    def y_max(self, value: float):
        self.max_y_spin.spin.setValue(value)
        self.figure.data_refresh()

    @property
    def change_all(self):
        return self.change_check.isChecked()

    @property
    def other_component_sizes(self):
        other_options = self.all_options.profile.other_options
        other_sizes = [options.size_adjust for options in other_options]
        return other_sizes

    def linked_spin_box(self, caption: str,
                        default_value: float, increment: Union[int, float]):
        spin = SpinBoxCaption(caption, default_value, increment)

        spin.spin.valueChanged.connect(self.change_limits)
        return spin

    def change_limits(self):
        x_lim = (self.x_min, self.x_max)
        y_lim = (self.y_min, self.y_max)
        self.figure.change_limits(x_lim, y_lim)

        if self.change_all:
            for size in self.other_component_sizes:
                size.x_min = self.x_min
                size.x_max = self.x_max

        self.figure.data_refresh()

    def connect_zoom(self):
        if all([self.rect_zoom_button.isChecked()]):
            point_selection = self.all_options.transferable.profile.current_options.point_selection
            for button in point_selection.buttons:
                button.setChecked(False)
            self.rect_zoom_button.setChecked(True)
            self.figure.rectangle_zoom()
        else:
            for cid in self.figure.zoom_cid:
                self.figure.mpl_disconnect(cid)


class PointSelection(QtWidgets.QGroupBox):
    def __init__(self, profile, graph, all_options: SideOptions,
                 component_options: ComponentOptions,
                 click_tolerance=1):
        super(PointSelection, self).__init__()
        self.profile = profile
        self.graph = graph
        self.all_options = all_options
        self.component_options = component_options
        self.click_tolerance = click_tolerance
        self.setTitle("Point operations")
        self.setFixedSize(250, 150)
        self.toggle_list = None

        layout = QtWidgets.QGridLayout()

        # Point selection part of dialog
        self.point_label = QtWidgets.QLabel("Select")
        layout.addWidget(self.point_label, 0, 0)
        self.rect_button = CheckableButton("Rect.")
        self.arrow_button = CheckableButton("Arrow")
        self.lasso_button = CheckableButton("Lasso")

        self.buttons = [self.rect_button, self.arrow_button, self.lasso_button]

        for i, widget in enumerate((self.rect_button, self.arrow_button,
                                    self.lasso_button)):
            layout.addWidget(widget, 1, i)

        # Assign points to be included or excluded
        self.assign_label = QtWidgets.QLabel("Assign", )
        layout.addWidget(self.assign_label, 2, 0)
        self.exclude, self.include, self.clear_selection = [OnePushButton(name) for name in ("Exclude", "Include",
                                                                                             "Clear")]
        self.exclude.clicked.connect(profile.exclude_points)
        self.include.clicked.connect(profile.include_points)
        self.clear_selection.clicked.connect(self.clear_selected_points)

        assign_buttons = [self.exclude, self.include, self.clear_selection]
        for i, widget in enumerate(assign_buttons):
            layout.addWidget(widget, 3, i)
        self.setLayout(layout)

        # Connect functions
        for button in (self.rect_button, self.arrow_button,
                       self.lasso_button):
            button.toggled.connect(self.connect_function)

    @property
    def arrow(self):
        return self.arrow_button.isChecked()

    @property
    def rectangle(self):
        return self.rect_button.isChecked()

    @property
    def lasso(self):
        return self.lasso_button.isChecked()

    def clear_buttons(self, buttons: Union[list, tuple]):
        pass

    def disconnect_zoom(self):
        if self.component_options.size_adjust.zoom_flag:
            self.component_options.size_adjust.rect_zoom_button.setChecked(False)
            for cid in self.graph.zoom_cid:
                self.graph.mpl_disconnect(cid)

    def connect_function(self):
        if self.component_options.size_adjust.zoom_flag:
            self.component_options.size_adjust.rect_zoom_button.setChecked(False)
            for cid in self.graph.zoom_cid:
                self.graph.mpl_disconnect(cid)
        for cid in self.graph.connected_functions:
            self.graph.mpl_disconnect(cid)
        cid1 = self.graph.mpl_connect("button_press_event", self.graph.double_click)
        self.graph.connected_functions.append(cid1)

        pre_list = [self.rectangle, self.arrow, self.lasso]
        buttons = (self.rect_button, self.arrow_button, self.lasso_button)
        if self.toggle_list is not None:
            for button, before, after in zip(buttons, self.toggle_list, pre_list):
                self.reset_button_status(button, before, after)

        if sum(pre_list):
            self.profile.find_control()
        self.toggle_list = pre_list

    def clear_selected_points(self):
        self.profile.selected = set()
        self.profile.redraw()

    @staticmethod
    def reset_button_status(button: CheckableButton, before: bool, after: bool):
        if all([before, before == after]):
            button.setChecked(False)
            return
        else:
            return



