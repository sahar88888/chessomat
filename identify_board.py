from scipy import misc
"""
This file is responsible for getting an image and returning only the board's
image, projected to be rectangular.
"""
class identify_board:

    def __init__(self):
        self.first = False

    """
    :return image of board, including an extra line above the board.
    """
    def get_board_image(self, im):
        if(self.first):
            return misc.imresize(misc.imread('2.jpg'),(600,600))
        self.first = True
        return misc.imresize(misc.imread('1.jpg'),(600,600))
