import os

import numpy as np

import automata

BASE_PATH = os.path.dirname(__file__)

def test_sandpile():
    # test with given file
    initial64 = np.load(os.sep.join((BASE_PATH, 'pile_64x64_init.npy')))
    final64 = np.load(os.sep.join((BASE_PATH, 'pile_64x64_final.npy')))

    assert (automata.sandpile(initial64) == final64).all()

    # test with given file
    initial128 = np.load(os.sep.join((BASE_PATH, 'pile_128x128_init.npy')))
    final128 = np.load(os.sep.join((BASE_PATH, 'pile_128x128_final.npy')))

    assert (automata.sandpile(initial128) == final128).all()

    # test with 5x5 matrix with value of 5 in the middle of the matrix (similar condition in assesment.pdf)
    initial = np.zeros((5, 5), dtype=int)
    initial[2, 2] = 5
    final = np.zeros((5, 5), dtype=int)
    final[1, 2] = 1
    final[2, 1:4] = 1
    final[3, 2] = 1

    assert (automata.sandpile(initial) == final).all()

def test_life():
    #test with beacon oscilattor pattern (n=1)
    initial_1 = np.zeros((6, 6), dtype=bool)
    initial_1[1, 1:3] = 1
    initial_1[2, 1] = 1
    initial_1[3, 4] = 1
    initial_1[4, 3:5] = 1

    final_1 = np.zeros((6, 6), dtype=bool)
    final_1[1:3, 1:3] = 1
    final_1[3:5, 3:5] = 1

    assert (automata.life(initial_1, 1, False) == final_1).all()

    #test with blinker oscilattor pattern (n=1) with Periodic
    initial_2 = np.zeros((5, 5), dtype=bool)
    initial_2[0:3, 4] = 1
 
    final_2 = np.zeros((5, 5), dtype=bool)
    final_2[1, 0] = 1
    final_2[1, -2:] = 1

    assert (automata.life(initial_2, 1, True) == final_2).all()
    
    #test with glider oscilattor pattern (n=4)
    initial_3 = np.zeros((5, 5), dtype=bool)
    initial_3[1, 1:4] = 1
    initial_3[2, 3] = 1
    initial_3[3, 2] = 1
 
    final_3 = np.zeros((5, 5), dtype=bool)
    final_3[0, -3:] = 1
    final_3[1, 4] = 1
    final_3[2, 3] = 1

    assert (automata.life(initial_3, 4, True) == final_3).all()

def test_lifetri():
    #test triangle glider(n=3)
    initial_4 = np.zeros((10, 10), dtype=bool)
    initial_4[4:6, 3:7] = 1

    final_4 = np.zeros((10, 10), dtype=bool)
    final_4[4:6, 5:9] = 1

    assert (automata.lifetri(initial_4, 3, True) == final_4).all()

def test_life_generic():
    #test with 2D blinker in 1D array input (n=1)
    matrix_1 = np.array([[0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1],
                         [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1, 1],
                         [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0, 1],
                         [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0],
                         [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0],
                         [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1],
                         [1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0]])

    initial_5 = np.zeros((25), dtype=bool)
    initial_5[11:14] = 1

    final_5 = np.zeros((25), dtype=bool)
    final_5[7] = 1
    final_5[12] = 1
    final_5[17] = 1
    assert (automata.life_generic(matrix_1, initial_5, 1, {2, 3}, {3}) == final_5).all()