import random 
import shutil
import os

import numpy as np
from tqdm import tqdm

def copy_tile_files(src_filename, tile_ref, trdst_path, valdst_path):
    
    filename = os.path.basename(src_filename)
    
    
    if tile_ref in filename:
        shutil.copy2(src_filename, valdst_path)
    else:
        shutil.copy2(src_filename, trdst_path)


def target_filename(input_path, target_path):
    
    filename = os.path.basename(input_path)
    target_filename = os.path.join(target_path, filename[:-3] + '.tif')
    
    return target_filename
    

def copy_files(input_path, target_path, filename_index, dst_path):
    
 
    dst_inputs = [os.path.join(dst_path, f'{folder_type}_input') for folder_type in ['training', 'validation']]
    dst_target = [os.path.join(dst_path, f'{folder_type}_target')  for folder_type in ['training', 'validation']]
    
    for inputpath, targetpath in zip(dst_inputs, dst_target):
        if not os.path.exists(inputpath): os.mkdir(inputpath)
        if not os.path.exists(targetpath): os.mkdir(targetpath)
    
    
    listallinput = [os.path.join(input_path,f) for f in os.listdir(input_path) if f.endswith('.nc')]
    
    for input_pathdir in tqdm(listallinput):
        target_pathdir = target_filename(input_pathdir, target_path)
    
        if os.path.exists(target_pathdir):
            # input
    
            copy_tile_files(input_pathdir, filename_index, *dst_inputs)
            # target
            copy_tile_files(target_pathdir, filename_index, *dst_target)


def main():
    total_tiles = 20
    all_input_data_path = 'hls_data96/all_filtered'
    target_path = 'hls_data96/target_data'
    val_ntiles = int(total_tiles*0.15)
    n = []
    random.seed(123)
    while len(n)<val_ntiles:
        val = random.randint(1,20)
        if val not in n:
            n.append(val)

    n_tile = n[0]
    for n_tile in n:
        print(f'tile {n_tile}: ')
        tile_ref = f'tile_{n_tile}_'

        copy_files(all_input_data_path, target_path=target_path, filename_index=tile_ref,dst_path= 'hls_data96')
    

if __name__ == '__main__':
    main()
    
        
    
    
    
    
    
