import numpy as np
import scipy
import scipy.misc
import scipy.cluster
import chess_helper
from PIL import Image
import chess
from scipy import misc

BLACK = (0.0, 0.0, 0.0)
BLACK_NUM = 1
WHITE_NUM = 2
ME_NUM = 4
HIM_NUM = 8
BLACK_SHOW = (0.0, 0.0, 0.0)
WHITE_SHOW = (1.0, 1.0, 1.0)
ME_SHOW = (0.6, 0.8, 0.8)
HIM_SHOW = (0.5, 0.2, 0.2)
RELEVANT_CHANGES_ME_BLACK = [abs(ME_NUM - BLACK_NUM), abs(ME_NUM - HIM_NUM)]
RELEVANT_CHANGES_ME_WHITE = [abs(ME_NUM - WHITE_NUM), abs(ME_NUM - HIM_NUM)]
RELEVANT_CHANGES_HIM_BLACK = [abs(HIM_NUM - BLACK_NUM), abs(HIM_NUM - ME_NUM)]
RELEVANT_CHANGES_HIM_WHITE = [abs(HIM_NUM - WHITE_NUM), abs(HIM_NUM - ME_NUM)]

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

    def initialize_colors(self, im):
        self.main_colors = self.get_main_colors(im)
        self.set_prev_im(im)

    """
    :return an image with 4/3/2 colors only.
    """

    def catalogue_colors(self, im, is_white):
        main_colors_white = self.main_colors[1:]
        main_colors_black = [self.main_colors[0]] + self.main_colors[2:]
        if is_white:
            cat_im = self.fit_colors(im, main_colors_white, is_white)
        else:
            cat_im = self.fit_colors(im, main_colors_black, is_white)
        return cat_im

    def catalogue_colors_show(self, im, is_white):
        main_colors_white = self.main_colors[1:]
        main_colors_black = [self.main_colors[0]] + self.main_colors[2:]
        if is_white:
            cat_im = self.fit_colors_show(im, main_colors_white, is_white)
        else:
            cat_im = self.fit_colors_show(im, main_colors_black, is_white)
        return cat_im

    """
    sets previous image - in rgb format.
    """

    def set_prev_im(self, im):
        self.prev_im = im

    """
    :im image of square after turn NOT CATALOGUED
    :square_loc location of square, in uci format
    :return binary image of relevant differences only (according to
    player/square color)
    """

    def get_square_diff(self, im, square_loc):
        is_white = self.chess_helper.square_color(square_loc) == chess.WHITE
        before_square = self.get_square_image(self.prev_im, square_loc,
                                              self.chess_helper.user_starts)
        after_square = self.get_square_image(im, square_loc,self.chess_helper.user_starts)
        before_square = self.catalogue_colors(before_square,is_white)
        after_square = self.catalogue_colors(after_square,is_white)
        square_diff = self.make_binary_relevant_diff_im(before_square,
                                                        after_square, is_white)
        return square_diff

    def make_binary_relevant_diff_im(self, im1, im2, is_white):
        curr_player = self.chess_helper.curr_player
        if (curr_player == chess_helper.chess_helper.ME):
            if is_white:
                relevant_changes = RELEVANT_CHANGES_ME_WHITE
            else:
                relevant_changes = RELEVANT_CHANGES_ME_BLACK
        else:
            if is_white:
                relevant_changes = RELEVANT_CHANGES_HIM_WHITE
            else:
                relevant_changes = RELEVANT_CHANGES_HIM_BLACK
        binary_im = []
        for rowidx in range(len(im1)):
            binary_im.append([])
            for pixidx in range(len(im1[0])):
                if abs(im1[rowidx][pixidx] - im2[rowidx][
                    pixidx]) in relevant_changes:
                    binary_im[rowidx].append(1)
                else:
                    binary_im[rowidx].append(0)
        return binary_im

    """
    returns subimage of a square in the board.
    """


    def get_square_image(self, im, loc,did_I_start):
        locidx = self.chess_helper.ucitoidx(loc)
        sq_sz = len(im[0]) // 8
        x = locidx[0]
        if did_I_start:
            y = 8 - locidx[1]
        else:
            y = locidx[1]
        area = (x * sq_sz, y * sq_sz, (x + 1) * sq_sz, (y + 1) * sq_sz)
        sqr_im = im[area[1]:area[3], area[0]:area[2]]
        return sqr_im

    """
    gets 2 primary colors from board image.
    """


    def get_board_colors(self, im):
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

    def get_player_color(self, im, board_colors, is_up):
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

    def get_main_colors(self, im):
        im_resize = scipy.misc.imresize(im, (400, 400))
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

    def color_dist(self, color1, color2):
        light1 = max(color1) / 2 + min(color1) / 2
        light2 = max(color2) / 2 + min(color2) / 2
        return abs(light1 - light2)

    """
    :return image fit to 4 main colors.
    """

    def fit_colors(self, im, main_colors, is_white):
        new_im = []
        for rowidx in range(len(im)):
            i = 0
            row = im[rowidx]
            new_im.append([])
            for pix in row:
                min_dist = self.color_dist(pix, main_colors[0])
                if is_white:
                    new_im[rowidx].append(WHITE_NUM)
                else:
                    new_im[rowidx].append(BLACK_NUM)
                if self.color_dist(pix, main_colors[1]) < min_dist:
                    min_dist = self.color_dist(pix, main_colors[1])
                    new_im[rowidx][i] = ME_NUM
                if self.color_dist(pix, main_colors[2]) < min_dist:
                    new_im[rowidx][i] = HIM_NUM
                i += 1
        return new_im

    def fit_colors_show(self, im, main_colors, is_white):
        new_im = []
        for rowidx in range(len(im)):
            i = 0
            row = im[rowidx]
            new_im.append([])
            for pix in row:
                min_dist = self.color_dist(pix, main_colors[0])
                if is_white:
                    new_im[rowidx].append(WHITE_SHOW)
                else:
                    new_im[rowidx].append(BLACK_SHOW)
                if self.color_dist(pix, main_colors[1]) < min_dist:
                    min_dist = self.color_dist(pix, main_colors[2])
                    new_im[rowidx][i] = ME_SHOW
                if self.color_dist(pix, main_colors[2]) < min_dist:
                    new_im[rowidx][i] = HIM_SHOW
                i += 1
        return new_im

    """
    gets main_im (for the main colors), and 2 ims of the same square in 2
    different moves
    image.
    """

"""
def tester(main_im_name, im1_name, im2_name, is_white, is_my_turn):
    main_im = scipy.misc.imread(main_im_name)
    im1 = scipy.misc.imread(im1_name)
    im2 = scipy.misc.imread(im2_name)
    if is_my_turn:
        chesshelper = chess_helper.chess_helper(chess_helper.chess_helper.ME)
    else:
        chesshelper = chess_helper.chess_helper(
            chess_helper.chess_helper.RIVAL)
    colorfilter = filter_colors(main_im, chesshelper)
    im1_cat = colorfilter.catalogue_colors(im1, is_white)
    im2_cat = colorfilter.catalogue_colors(im2, is_white)
    im1_cat_show = colorfilter.catalogue_colors_show(im1, is_white)
    im2_cat_show = colorfilter.catalogue_colors_show(im2, is_white)

    scipy.misc.imsave('cat1.JPEG', im1_cat_show)
    scipy.misc.imsave('cat2.JPEG', im2_cat_show)
    diff_im = colorfilter.make_binary_relevant_diff_im(im1_cat, im2_cat,
                                                       is_white)
    scipy.misc.imsave('cat_diff.JPEG', diff_im)
    return
"""

def tester(main_im_name, im1_name, im2_name, is_my_turn,loc):
    main_im = misc.imresize(misc.imread(main_im_name), (600, 600))
    im1 = misc.imresize(misc.imread(im1_name), (600, 600))
    im2= misc.imresize(misc.imread(im2_name), (600, 600))
    if is_my_turn:
        chesshelper = chess_helper.chess_helper(chess_helper.chess_helper.ME)
    else:
        chesshelper = chess_helper.chess_helper(
            chess_helper.chess_helper.RIVAL)
    colorfilter = filter_colors(main_im, chesshelper)
    colorfilter.set_prev_im(im1)
    im1_square = colorfilter.get_square_image(im1, loc,is_my_turn)
    im2_square = colorfilter.get_square_image(im2, loc,is_my_turn)

    is_white = chesshelper.square_color(loc)

    im1_cat_show = colorfilter.catalogue_colors_show(im1_square, is_white)
    im2_cat_show = colorfilter.catalogue_colors_show(im2_square, is_white)

    scipy.misc.imsave('cat1.JPEG', im1_cat_show)
    scipy.misc.imsave('cat2.JPEG', im2_cat_show)
    scipy.misc.imsave('im1_square.JPEG', im1_square)
    scipy.misc.imsave('im2_square.JPEG', im2_square)
    square_diff = colorfilter.get_square_diff(im2,loc)
    scipy.misc.imsave('square_diff.JPEG', square_diff)
    return

tester('main_im_b.jpg', 'im1_b.jpg', 'im2_b.jpg', True, 'g1')
