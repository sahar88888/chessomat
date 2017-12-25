import numpy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


"""
This file is responsible for identifying if a move has been made in a square.
"""


mindensity=0.45
maxdensity=3.5
minyahas=0.5
maxyahas=1
parameterOfSize=0.45 ##the minimal size of white that should be in an image
#if its lower than that, we cancel the image

WHITE = 1
BLACK = 0

CROP_RT = 40
INIT_HI_RT = 100

MAX_RT = 1.5
WINDOW_RT = 40
WHITE_DEF_RT = 2
MIN_WHITE_RT = 5


class find_move:


    def __init__(self, chess_helper):
        self.chess_helper = chess_helper

    """
    :arg square_im a binary image of changes in the square
    :return whether there's been a move on the square, below it, or none,
    and the score (if there's a move)
    """
    def check_move(self,square_im):
        return [find_move.MOVE_NONE, 0]


    """

    :return likeliest move
    """
    def get_move(self,sources_place, sources_self, sources_above,
                               targets_place, targets_self, targets_above):
        sources_rank = self.check_squares(sources_self,
                                     sources_above)

        targets_rank = self.check_squares(targets_self,
                                     targets_above)

        for i in range(len(sources_self)):
            plt.imsave(sources_place[i]+'.png', sources_self[i],cmap=cm.gray)
        for i in range(len(targets_self)):
            plt.imsave(targets_place[i]+'.png', targets_self[i],cmap=cm.gray)
        ### DEBUG ###
        print("sources : ")
        print(sources_place)
        print("ranking : ")
        print(sources_rank)
        print("dests : ")
        print(targets_place)
        print("ranking : ")
        print(targets_rank)

        return self.choose_pair(sources_place, targets_place, sources_rank,
                           targets_rank)

    '''
    receives a list of square images, and list of square images above them and
    returns a list of rank with the corresponding indexes
    '''
    def check_squares(self, squares, above_squares):
        rank_lst = []
        for i in range(len(squares)):
            rank_lst.append(max(self.check_square_below(above_squares[i]),
                              (self.check_square(
                                  squares[i]) - self.check_square_below(
                                  squares[i]))))
            # a metric that consider both the square itself and the square above checks
        return rank_lst


    '''
    receives one square of source, all the targets(their places and their ranks), and the chess board and return the
    best target for this source, using the check squares method
    '''
    def best_target_for_given_source(self, source_place, targets_place,
                                     target_ranks):
        target_img_dict = dict(zip(targets_place, target_ranks))
        inv_target_img_dict = dict(zip(target_ranks, targets_place))
        matches = self.chess_helper.square_dests(source_place)  # all the
        # natches of a given square
        best_match_rank = 0
        for match in matches:
            best_match_rank = max(target_img_dict[match], best_match_rank)
        best_match_place = inv_target_img_dict[best_match_rank]

        return best_match_place, best_match_rank

    '''
    receives a sources ranks and places, targets ranks and places, and the chess board, and returns the best pair.
    this is done by find the best match of each source (using best_target_for_given_source method),
    and comparing between all theses matches
    '''
    def choose_pair(self,sources_place, targets_place, source_ranks,
                    target_ranks):
        best_targets_per_source = []
        pairs_rank = []

        for i in range(len(source_ranks)):
            tmp, best_match_rank = \
                self.best_target_for_given_source(sources_place[i], targets_place, target_ranks)
            best_targets_per_source.append(tmp)
            pairs_rank.append(source_ranks[i] + best_match_rank)
        best_pair_idx = [i for i in range(len(pairs_rank)) if pairs_rank[i] == max(pairs_rank)][0]
            #the idx of the pair with the highest rank

        return sources_place[best_pair_idx], targets_place[best_pair_idx]


    '''
    This function receive a vector that represent one row, and returns true if the vector pass the requirements
    The method is to run with a window on the vector and for each position of the windows check if it is white or not.
    if the longest sequence is in the acceptable size this vector passes
    '''
    def check_vector(self, vector, max1):
        min_white = len(vector)//MIN_WHITE_RT
        window_len = len(vector)//WINDOW_RT
        w_counter = 0
        w_max = 0
        sum1 = sum(vector[0:window_len-1])

        for i in range(len(vector)-window_len):
            sum1 = sum1 + vector[i+window_len] - vector[i]  #the current window whiteness.
            if sum1 > window_len//WHITE_DEF_RT:
                w_counter += 1
                w_max = max(w_max, w_counter)
            else:
                w_counter = 0

        return min_white < w_max < max1, w_max  #T/F, longest sequence

    """
    This function is the main function of the file. it receives an image and return a boolean.
    it use the check_vector function and decide if an image pass or not according to the sequence of vectors that passes,
    and are in the relevant place of the image (according to the ratios).
    """

    def check_square_below(self, img):
        crop_length = len(img) // CROP_RT
        init_hi = len(img) // INIT_HI_RT
        false_counter = 0
        true_counter = 0

        crop_img = img[crop_length - crop_length + init_hi: len(img) - init_hi]
        d_vector = img[len(crop_img) - 1]
        d_b, d_len = self.check_vector(d_vector, len(crop_img))
        max1 = d_len * MAX_RT

        for i in range(len(crop_img)):
            v_b, v_len = self.check_vector(img[len(img) - i - 1], max1)
            if not v_b:
                if true_counter > 1:
                    return 0
                true_counter = 0
                false_counter += 1
            else:
                true_counter += 1
                false_counter = 0

        if false_counter <= 1:
            return 1
        else:
            return 0

    """
    a code that receives an image of chess square, returns TRUE whether a
    chess soldier sits on the square and FALSE  if not.
    the classification is done in that way:
    1. Calculate the center of mass of the white and black squares.
    2. Calculate the density of white squares in the black squares.
    3. Cancel the squares that their densities are to low.
    4. Calculate the mean distance between a pixel in the fitting color to its
    center of mass
    5. the mean distance of white should be lower
    6. return true if the mean distance of the white from CM is lower that the
    black one.
    """

    # the number of pixels in certain color in an image
    def numofColor(self, img, color):
        count = 0
        for i in range(len(img)):
            for j in range(len(img[0])):
                if img[i][j] == color:
                    count += 1
        return count

    # return the density of white squares in black squares
    def checkDensity(self,img):
        numofWhite = self.numofColor(img, WHITE)
        numOfBlack = self.numofColor(img, BLACK)
        density = numofWhite * 1.0 / numOfBlack
        if (density < mindensity or density > maxdensity):
            return 0
        return -0.5 * (density - mindensity) * (density - maxdensity) + 0.2

    # return the X,Y of CM of the color given to the function
    def centerMass(self, img, color):
        sumx = 0
        count = 0
        sumy = 0
        for i in range(len(img)):
            for j in range(len(img[0])):
                if (img[i][j] == color):
                    sumx = sumx + i
                    sumy = sumy + j
                    count = count + 1
        if (count == 0):
            return (0, 0)
        Xcm = sumx / count
        Ycm = sumy / count
        return (Xcm, Ycm)

    # returns the mean distance of the color given to the function from its
    # center of mass
    def meanDist(self, img, color, Xcm, Ycm):
        countdist = 0
        counter = 0
        for i in range(len(img)):
            for j in range(len(img[0])):
                if (img[i][j] == color):
                    dist = numpy.sqrt((i - Xcm) ** 2 + (j - Ycm) ** 2)
                    countdist = countdist + dist
                    counter = counter + 1
        if (counter == 0):
            return 0
        return (countdist / counter)

    # gets image in RGB and return 1,0 image
    def makeoneZero(self,img):
        newimg = [[0 for x in range(len(img[0]))] for y in range(len(img))]
        for i in range(len(img)):
            for j in range(len(img[0])):
                if (img[i][j][0] > 100 and img[i][j][1] > 100 and img[i][j][
                    2] > 100):
                    newimg[i][j] = WHITE
                else:
                    newimg[i][j] = BLACK
        return newimg

    ##receives image in RGB, returns true if there is a soldies on that square
    # and false otherwise.
    def check_square(self,imgOneZero):

        WXcm, WYcm = self.centerMass(imgOneZero, WHITE)
        BXcm, BYcm = self.centerMass(imgOneZero, BLACK)
        yahas = self.meanDist(imgOneZero, WHITE, WXcm, WYcm) / self.meanDist(imgOneZero,
                                                                   BLACK, BXcm,
                                                                   BYcm)
        if (yahas < minyahas or yahas > maxyahas):
            return 0
        yahasmetrics = -(yahas - minyahas) * (yahas - maxyahas) + 0.9

        return self.checkDensity(imgOneZero) * yahasmetrics



