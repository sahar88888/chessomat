import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import chess_helper
from PIL import Image
import chess

BLACK = (0.0, 0.0, 0.0)
BLACK_NUM = 1
WHITE_NUM = 2
ME_NUM = 4
HIM_NUM = 8
RELEVANT_CHANGES_ME_BLACK = [abs(ME_NUM - BLACK_NUM), abs(ME_NUM - HIM_NUM)]
RELEVANT_CHANGES_ME_WHITE = [abs(ME_NUM - WHITE_NUM), abs(ME_NUM - HIM_NUM)]
RELEVANT_CHANGES_HIM_BLACK = [abs(HIM_NUM - BLACK_NUM), abs(HIM_NUM - ME_NUM)]
RELEVANT_CHANGES_HIM_WHITE = [abs(HIM_NUM - WHITE_NUM), abs(HIM_NUM - ME_NUM)]
IM_NAME = 'up.jpg'
IM_NAME_1 = '1.jpg'
IM_NAME_2 = '2.jpg'

class filter_colors:
    """
    This file is responsible for filtering and cataloguing the colors in the
    picture.
    """

    def __init__(self, im, chess_helper):
        self.chess_helper = chess_helper
        self.initialize_colors(im)

    """
    gets all 4/3/2 colors in the board.
    :im initial image, that contains the relevant colors on the board.
    :return nothing
    """
    def initialize_colors(self,im):
        self.main_colors = self.get_main_colors(im)
        self.set_prev_im(im)


    """
    :return an image with 4/3/2 colors only.
    """
    def catalogue_colors(self,im):
        cat_im = self.fit_colors(im, self.main_colors)
        return cat_im


    """
    sets previous image - in rgb format.
    """
    def set_prev_im(self,im):
        self.prev_im = im


    """
    :im image of square after turn NOT CATALOGUED
    :square_loc location of square, in uci format
    :return binary image of relevant differences only (according to
    player/square color)
    """
    def get_square_diff(self,im, square_loc):
        before_square = self.get_square_image(self.prev_im, square_loc)
        after_square = self.get_square_image(im, square_loc)
        before_square = self.catalogue_colors(before_square)
        after_square = self.catalogue_colors(after_square)
        is_white = self.chess_helper.square_color(square_loc) == chess.WHITE
        square_diff = self.make_binary_relevant_diff_im(before_square,
                                                        after_square, is_white)
        return square_diff

    def make_binary_relevant_diff_im(self,im1, im2, is_white):
        curr_player = self.chess_helper.curr_player
        if(curr_player == chess_helper.chess_helper.ME):
            relevant_changes_white = RELEVANT_CHANGES_ME_WHITE
            relevant_changes_black = RELEVANT_CHANGES_ME_BLACK
        else:
            relevant_changes_white = RELEVANT_CHANGES_HIM_WHITE
            relevant_changes_black = RELEVANT_CHANGES_HIM_BLACK

        binary_im = []
        for rowidx in range(len(im1)):
            binary_im.append([])
            for pixidx in range(len(im1[0])):
                if is_white:
                    if abs(im1[rowidx][pixidx] - im2[rowidx][
                        pixidx]) in relevant_changes_white:
                        binary_im[rowidx].append(1)
                    else:
                        binary_im[rowidx].append(0)
                else:
                    if abs(im1[rowidx][pixidx] - im2[rowidx][
                        pixidx]) in relevant_changes_black:
                        binary_im[rowidx].append(1)
                    else:
                        binary_im[rowidx].append(0)
        return binary_im

    """
    returns subimage of a square in the board.
    """
    def get_square_image(self, im, loc):
        locidx = self.chess_helper.ucitoidx(loc)
        sq_sz = len(im[0]) // 8
        x = locidx[0]
        y = locidx[1]
        area = (x * sq_sz, y * sq_sz, (x + 1) * sq_sz, (y + 1) * sq_sz)
        sqr_im = im[area[0]:area[2], area[1]:area[3]]
        return sqr_im


    """
    gets 2 primary colors from board image.
    """
    def get_board_colors(self,im):
        ar = np.asarray(im)
        ar_sz = len(ar)
        ar = ar[round(ar_sz / 4):round(3 * ar_sz / 4)]
        shape = ar.shape
        ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
        codes, dist = scipy.cluster.vq.kmeans(ar, 2)
        vecs, dist = scipy.cluster.vq.vq(ar, codes)  # assign codes
        counts, bins = scipy.histogram(vecs, len(codes))  # count occurrences
        indices = [i[0] for i in
                   sorted(enumerate(-counts), key=lambda x: x[1])]
        new_indices = []
        new_codes = []
        for i in indices:
            new_codes.append(codes[i])
        if self.color_dist(new_codes[0], BLACK) < self.color_dist(new_codes[1],
                                                        BLACK):
            new_indices.append(indices[0])
            new_indices.append(indices[1])
        else:
            new_indices.append(indices[1])
            new_indices.append(indices[0])
        return [codes[i] for i in new_indices]


    """
    gets player's colors from the board image.
    """
    def get_player_color(self,im, board_colors, is_up):
        ar = np.asarray(im)
        ar_sz = len(ar)
        if is_up:
            ar = ar[:round(ar_sz / 4)]
        else:
            ar = ar[3 * round(ar_sz / 4):]
        shape = ar.shape
        ar = ar.reshape(scipy.product(shape[:2]), shape[2]).astype(float)
        codes, dist = scipy.cluster.vq.kmeans(ar, 3)
        for i in range(3):
            color_dist_1 = self.color_dist(codes[i], board_colors[0])
            color_dist_2 = self.color_dist(codes[i], board_colors[1])
            if i == 0:
                max_dist = min(color_dist_1, color_dist_2)
                max_dist_index = i
            else:
                dist_i = min(color_dist_1, color_dist_2)
                if dist_i > max_dist:
                    max_dist = dist_i
                    max_dist_index = i
        return codes[max_dist_index]

    """
    :return black,white,my soldier color, rival soldier color.
    """
    def get_main_colors(self,im):
        im_resize = scipy.misc.imresize(im,(400,400))
        board_colors = self.get_board_colors(im_resize)
        down_color = self.get_player_color(im_resize, board_colors, False)
        up_color = self.get_player_color(im_resize, board_colors, True)
        main_colors = board_colors
        main_colors.append(down_color)
        main_colors.append(up_color)
        ### DEBUG ###
        print("main colors are:")
        print(main_colors)
        return main_colors

    """
    :return color difference in Lightness.
    """
    def color_dist(self,color1, color2):
        light1 = max(color1) / 2 + min(color1) / 2
        light2 = max(color2) / 2 + min(color2) / 2
        return abs(light1 - light2)


    """
    :return image fit to 4 main colors.
    """
    def fit_colors(self, im, main_colors):
        new_im = []
        for rowidx in range(len(im)):
            i = 0
            row = im[rowidx]
            new_im.append([])
            for pix in row:
                new_im[rowidx].append(BLACK_NUM)
                min_dist = self.color_dist(pix, main_colors[0])
                if self.color_dist(pix, main_colors[1]) < min_dist:
                    min_dist = self.color_dist(pix, main_colors[1])
                    new_im[rowidx][i] = WHITE_NUM
                if self.color_dist(pix, main_colors[2]) < min_dist:
                    min_dist = self.color_dist(pix, main_colors[2])
                    new_im[rowidx][i] = ME_NUM
                if self.color_dist(pix, main_colors[3]) < min_dist:
                    new_im[rowidx][i] = HIM_NUM
                i += 1
        return new_im

