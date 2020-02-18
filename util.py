import sys
import os
import json
import numpy as np
import torch

def print_statement(statement, isCenter=False, symbol='=', number=15, newline=False, verbose=None):
    '''
    Print required statement in a given format.
    '''
    if verbose is not None:
        if verbose > 0:
            if newline:
                print()
            if number > 0:
                prefix = symbol * number + ' '
                suffix = ' ' + symbol * number
                statement = prefix + statement + suffix
            if isCenter:
                print(statement.center(os.get_terminal_size().columns))
            else:
                print(statement)
        else:
            pass
    else:
        if newline:
            print()
        if number > 0:
            prefix = symbol * number + ' '
            suffix = ' ' + symbol * number
            statement = prefix + statement + suffix
        if isCenter:
            print(statement.center(os.get_terminal_size().columns))
        else:
            print(statement)