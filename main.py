from colorama import Fore
import argparse
import os
import sys
from Dynamic import Sandbox
from Static import buildNET

'''
UNDER CONSTRUCTION 
=============
author: amartya mannava
'''

# TODO : print results

parser = argparse.ArgumentParser(prog='mal', description= 'intelligent malware detector')
parser.add_argument('MODE', metavar='mode of analysis', type=str, help='D for dynamic | S for static')
parser.add_argument('PATH', metavar="path", type=str, help='path to exe')

args= parser.parse_args()
exe_path= args.PATH
mode= args.MODE
mode=mode.strip().upper()

if not os.path.isdir(exe_path):
    print('The path specified does not exist')
    sys.exit()

if args.MODE == 'D':
    Sandbox.init_sandbox(exe_path)
elif args.MODE == 'S':
    buildNET(exe_path)
else:
    print("Invalid Mode")
    sys.exit(0)


def banner_print():
   print(Fore.RED + """ 
 ooo        ooooo       .o.       ooooo        
`88.       .888'      .888.      `888'        
 888b     d'888      .8"888.      888         
 8 Y88. .P  888     .8' `888.     888         
 8  `888'   888    .88ooo8888.    888         
 8    Y     888   .8'     `888.   888       o 
o8o        o888o o88o     o8888o o888ooooood8 
                                              
   """)
   print("==============================================================")
   print("Tool for malware detection and analysis")
   print("=============")
   print("author: Amartya Mannava ||https://github.com/rootwipe")


if __name__ == '__main__':
  banner_print()
