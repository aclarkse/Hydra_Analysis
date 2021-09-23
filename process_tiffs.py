import os
import math
import argparse
import nibabel as nib
import skimage.io as skio
import numpy as np
import tqdm as tqdm

def process_stack(filename, step_size):
    """ Processes a stack of tiff files by extracting slices 
        by a specified step size and saves them as NIfTI files.

        Arguments:
            - filename [string]: the filename of the tiff stack to process
            - step_size [int]: the desired increment with which to extract tiff slices
    """

    tiff_stack = skio.imread(filename, plugin="tifffile")
    n_frames = tiff_stack.shape[0]
    h = tiff_stack.shape[1]
    w = tiff_stack.shape[2]
    print(f"Tiff stack loaded. Found {n_frames} images.")

    # compute the number of frames in the reduced array
    rd_frames = math.floor(n_frames/step_size) + 1
    tiff_arr = np.zeros((rd_frames, h, w))
    i = 0
    for frame in tqdm.tqdm(range(n_frames)):
        if frame % step_size == 0:
            tiff_arr[i, :, :] = tiff_stack[frame,:,:]
            i+= 1 

    # convert to NIfTI image
    img = nib.Nifti1Image(tiff_arr, affine=np.eye(4))
    fn = filename.split('.')[0]
    nib.save(img, os.path.join(f'{fn}_sz_{step_size}.nii.gz'))

    processed_imgs = math.floor(n_frames/step_size) + 1
    print(f"Finished processing {processed_imgs} images!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Loads a tiff stack and extracts slices to save as NIfTI files.')
    parser.add_argument('filename', metavar='fn', type=str,
                        help='The filename of the tiff stack you wish to process.')
    filename = parser.parse_args().filename                   

    process_stack(filename, 5)

