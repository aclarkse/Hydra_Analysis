import os
import argparse
import nibabel as nib
import skimage.io as skio
import numpy as np
import tqdm as tqdm

def process_stack(filename, step_size, new_dir):
    """ Processes a stack of tiff files by extracting slices 
        by a specified step size and saves them as NIfTI files.

        Arguments:
            - filename [string]: the filename of the tiff stack to process
            - step_size [int]: the desired increment with which to extract tiff slices
            - new_dir [string]: the name of the directory in which to save slices
    """

    tiff_stack = skio.imread(filename, plugin="tifffile")
    n_frames = tiff_stack.shape[0]
    print(f"Tiff stack loaded. Found {n_frames} images.")
    for frame in tqdm.tqdm(range(n_frames)):
        if frame % step_size == 0:
            if len(str(frame)) == 2:
                # for formatting the filename, prepend extra 0s
                frame_name = "0" + str(frame)
            elif len(str(frame)) == 1:
                frame_name = "00"+ str(frame)
            else:
                frame_name = frame
            # convert to NIfTI type
            img = nib.Nifti1Image(tiff_stack[frame,:,:], np.eye(4))
            nib.save(img, os.path.join(new_dir, '{}.nii.gz'.format(frame_name)))
    print("Finished processing images!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Loads a tiff stack and extracts slices.')
    parser.add_argument('filename', metavar='fn', type=str,
                        help='The filename of the tiff stack you wish to process.')
    filename = parser.parse_args().filename                   

    # create a new directory for saving slices
    parent_dir = os.getcwd()
    new_dir = os.path.join(parent_dir, "{}_slices".format(filename.split('.')[0]))
    if not os.path.exists(new_dir):
        os.mkdir(os.path.join(parent_dir, new_dir))
        print(f"Directory \'{new_dir}/' created")

    process_stack(filename, 10, new_dir)

