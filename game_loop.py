import identify_board as boardid
import filter_colors
import hardware as hw
import chess_helper as ch
import find_move as fm

"""
Main logic file.
"""

class game_loop:
    def __init__(self):
        self.hardware = hw.hardware()
        self.boardid = boardid.identify_board()
        board_im = self.hardware.get_image()
        # TODO : get starting player type by color!
        cut_board_im = self.boardid.get_board_image(board_im)
        self.chesshelper = ch.chess_helper(ch.chess_helper.ME)
        self.colorfilter = filter_colors.filter_colors(cut_board_im, self.chesshelper)
        self.movefinder = fm.find_move(self.chesshelper)


    def get_new_move(self):
        new_board_im = self.hardware.get_image()
        cut_board_im = self.boardid.get_board_image(new_board_im)
        relevant_squares = self.chesshelper.get_relevant_locations()
        sources = relevant_squares[0]
        dests = relevant_squares[1]
        sourcesims = []
        destsims = []
        sourcesabvims = []
        destsabvims = []
        ### DEBUG ###
        print("sources are:")
        print(sources)
        for src in sources:
            sourcesims.append(self.colorfilter.get_square_diff(cut_board_im,
                             src))

        print("destinations are:")
        print(dests)
        for dest in dests:
            destsims.append(self.colorfilter.get_square_diff(cut_board_im,
                          dest))

        sourcesabvims = sourcesims
        destsabvims = destsims

    #    for src in sources:
    #        #srcabv = [src[0], src[1]-1]
    #        srcabv = src
    #        sourcesabvims.append(self.colorfilter.get_square_diff(
            # cut_board_im,
    #                                    srcabv))

    #    for dest in dests:
    #        #destabv = [dest[0], dest[1]-1]
    #        destabv = dest
    #        destsabvims.append(self.colorfilter.get_square_diff(cut_board_im,
    #                                                           destabv))



        move = self.movefinder.get_move(sources,sourcesims,sourcesabvims,
                                       dests, destsims, destsabvims)
        print('move')
        ### save prev picture ###
        self.colorfilter.set_prev_im(cut_board_im)
        return move


##### TEST: #####
gameloop = game_loop()
print(gameloop.get_new_move())