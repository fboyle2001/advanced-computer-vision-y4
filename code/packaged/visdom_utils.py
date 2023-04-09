class MultiLinePlot:
    def __init__(self, vis, title, xlabel, ylabel):
        self.vis = vis
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
    
    def append_values(self, x, labelled_ys):
        names = list(labelled_ys.keys())
        xs = [[x] for _ in range(len(labelled_ys))]
        ys = [[y] for y in labelled_ys.values()]

        for (x, y, name) in zip(xs, ys, names):
            self.vis.line(
                X=x,
                Y=y,
                opts={
                    "title": self.title,
                    "legend": names,
                    "xlabel": self.xlabel,
                    "ylabel": self.ylabel
                },
                name=name,
                win=self.title,
                update="append"
            )
