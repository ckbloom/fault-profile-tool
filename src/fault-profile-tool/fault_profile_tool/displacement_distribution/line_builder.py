import numpy as np


class LineStore:
    def __init__(self, axis, drawn_colour: str = "r", best_colour: str = "c"):
        """
        To plot lines
        :param axis:
        :param drawn_colour:
        :param best_colour:
        """
        self.axis = axis
        self.drawn, = self. axis.plot([0], [0], drawn_colour)
        self.drawn_extra, = self.axis.plot([0], [0], "{}:".format(drawn_colour))
        self.best, = self. axis.plot([0], [0], best_colour)
        self.best_extra, = self.axis.plot([0], [0], "{}:".format(best_colour))


class LineBuilder:
    """
    To plot lines and calculate gradient, error and offset
    """
    def __init__(self, lines: LineStore, pair, positive: bool = True, alternative_error: bool = False):
        """
        Generated in init method of PairedLines class
        :param lines: LineStore object to plot
        :param pair: PairedLines object to give matching line
        :param positive: RHS or LHS of fault line
        :param alternative_error: Method for calculation of offset
        """
        self.pair = pair
        self.figure = self.pair.figure
        self.axis = self.pair.axis
        self.lines = lines
        self.first = True
        self.dot = None
        self.intercept = None
        self.positive = positive
        self.alternative_error = alternative_error
        self.sign = (0, 1) if positive else (0, -1)

        self.opposite = None

        self.xs = list(lines.drawn.get_xdata())
        self.ys = list(lines.drawn.get_ydata())
        self.previous_xs, self.previous_ys = [[], []]
        self.cid = lines.drawn.figure.canvas.mpl_connect('button_press_event', self)

    def __call__(self, event):
        if any([(event.inaxes != self.lines.axis),
                (np.sign(event.xdata) not in self.sign)]):
            return
        self.line_actions(event)

    def save_old(self):
        while len(self.previous_xs) >= 10:
            self.previous_xs.pop(0)
            self.previous_ys.pop(0)
        self.previous_xs.append(self.xs)
        self.previous_ys.append(self.ys)

    def line_actions(self, event):
        """
        For manual drawing of lines. Not currently used 29/3/21
        :param event:
        :return:
        """
        if len(self.xs) == 2:
            self.save_old()
            self.first = True

        if self.first:
            self.xs = [event.xdata]
            self.ys = [event.ydata]
            self.first = False
            self.dot, = self.axis.plot(event.xdata, event.ydata, 'r.')
        else:
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            self.intercept = self.gradient_intercept()
            self.dot.remove()
            self.lines.drawn.set_data(self.xs, self.ys)

            furthest_index = np.argmax([abs(x) for x in self.xs])
            furthest_point = [self.xs[int(furthest_index)], self.ys[int(furthest_index)]]
            self.lines.drawn_extra.set_data([furthest_point[0], 0], [furthest_point[1], self.intercept])
            self.lines.drawn_extra.figure.canvas.draw()
        if len(self.xs) == 2:
            self.pair.calculate_offset()
            if self.alternative_error:
                self.pair.calculate_error_alt()
            else:
                self.pair.calculate_error()

        self.lines.drawn.figure.canvas.draw()

    def gradient_intercept(self):
        assert len(self.xs) == 2
        gradient = (self.ys[1] - self.ys[0])/(self.xs[1] - self.xs[0])
        intercept = self.ys[0] - gradient * self.xs[0]
        return intercept

    def error_estimate(self):
        x, y = self.find_x_indices()
        try:
            [gradient, _] = np.polyfit(x, y, deg=1)
            detrended_y = y - x * gradient
            standard_deviation = np.std(detrended_y)
        except TypeError:
            standard_deviation = np.NaN
        return standard_deviation

    def error_estimate_alt(self):
        """
        Using method at https://stackoverflow.com/questions/27634270/how-to-find-error-on-slope-and-intercept-using-numpy-polyfit
        :return:
        """
        x, y = self.find_x_indices()
        try:
            [_, cov] = np.polyfit(x, y, deg=1, cov=True)
            intercept_error = np.sqrt(cov[1][1])
        except TypeError:
            intercept_error = np.NaN
        return intercept_error

    def best_fit(self):
        """
        Find best fit using numpy polyfit
        :return:
        """
        x, y = self.find_x_indices()
        try:
            [gradient, intercept] = np.polyfit(x, y, deg=1)
        except TypeError:
            return

        best_ys = intercept + gradient * np.array(self.xs)
        self.lines.best.set_data(self.xs, best_ys)
        furthest_index = np.argmax([abs(x) for x in self.xs])
        furthest_point = [self.xs[int(furthest_index)], best_ys[int(furthest_index)]]
        self.lines.best_extra.set_data([furthest_point[0], 0], [furthest_point[1], intercept])
        self.lines.best_extra.figure.canvas.draw()
        self.lines.best.figure.canvas.draw()

    def find_x_indices(self):
        x_data, y_data = self.figure.x_data, self.figure.y_data
        interest = np.where(np.logical_and(x_data > min(self.xs), x_data <= max(self.xs)))
        return x_data[interest], y_data[interest]

    def detrend(self):
        pass


class PairedLines:
    def __init__(self, axis, figure, positive_colour: str="r", negative_colour: str = "b",
                 alternative_error: bool = False):
        self.axis = axis
        self.figure = figure
        self.offset, self.error = None, None
        self.alternative_error = alternative_error
        self.rhs = LineStore(self.axis, drawn_colour=positive_colour)
        self.lhs = LineStore(self.axis, drawn_colour=negative_colour)

        self.line_builder1 = LineBuilder(self.rhs, pair=self, positive=True, alternative_error=alternative_error)
        self.line_builder2 = LineBuilder(self.lhs, pair=self, positive=False, alternative_error=alternative_error)

        self.line_builder1.opposite = self.line_builder2
        self.line_builder2.opposite = self.line_builder1

    def calculate_offset(self):
        if all([x.intercept is not None for x in [self.line_builder1, self.line_builder2]]):
            self.offset = abs(self.line_builder1.intercept - self.line_builder2.intercept)
            # self.figure.options.fit_display.offset_label.setText("Offset: {:.2f} m".format(self.offset))

    def calculate_error(self):
        if all([len(x.xs) == 2 for x in [self.line_builder1, self.line_builder2]]):
            self.error = np.linalg.norm(np.array([self.line_builder2.error_estimate(),
                                                  self.line_builder1.error_estimate()]))

    def calculate_error_alt(self):
        if all([len(x.xs) == 2 for x in [self.line_builder1, self.line_builder2]]):
            self.error = np.linalg.norm(np.array([self.line_builder2.error_estimate_alt(),
                                                  self.line_builder1.error_estimate_alt()]))

    def disconnect(self):
        for line in (self.line_builder1, self.line_builder2):
            self.figure.mpl_disconnect(line.cid)

    def reconnect(self):
        for line in (self.line_builder1, self.line_builder2):
            line.cid = self.figure.mpl_connect('button_press_event', line)


class PointSelector:
    def __init__(self):
        pass



