"""
progressbar
Written by Eirik Vesterkjær, 2019
Apache License

project-specific wrapper for progressbar
"""

import progressbar

class DisplayBar():
    def __init__(self, max_value: int = None, start_epoch: int = 0, start_it: int = 0):

        self.progressbar_it_epoch = progressbar.FormatCustomText(
        "[It: %(it)6d] [Epoch: %(epoch)4d (%(current_value)5d/%(max_value)5d)]",
        dict(
            epoch=start_epoch,
            it=start_it,
            current_value=0,
            max_value=max_value
        ),
        )

        self.progressbar_widgets = [
            " [", progressbar.AnimatedMarker(), "] ",
            self.progressbar_it_epoch, " ",
            progressbar.Bar(), " ",
            progressbar.Timer(format='T: %(elapsed)s'), " ", progressbar.ETA()
        ]

        self.bar = progressbar.ProgressBar(max_value=max_value,
                                           widgets=self.progressbar_widgets)

    def update(self, current_value: int, current_epoch: int, current_it: int):
        self.progressbar_it_epoch.update_mapping(epoch=current_epoch,
                                                 it=current_it,
                                                 current_value=current_value)
        self.bar.update(current_value)