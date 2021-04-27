from PyQt5 import QtWidgets
from typing import Iterable


class GraphStack(QtWidgets.QStackedWidget):
    def __init__(self, graphs: Iterable):
        super(GraphStack, self).__init__()
        self.graphs = {}
        for i, graph in enumerate(graphs):
            self.addWidget(graph)
            self.graphs[i] = graph


class MapStack(QtWidgets.QStackedWidget):
    def __init__(self, maps: Iterable):
        super(MapStack, self).__init__()
        self.maps = {}
        for i, map_i in enumerate(maps):
            self.addWidget(map_i)
            self.maps[i] = map_i

