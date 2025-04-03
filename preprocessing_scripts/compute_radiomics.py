from radiomics.firstorder import RadiomicsFirstOrder
from radiomics.shape import RadiomicsShape
from radiomics.glcm import RadiomicsGLCM
from tqdm import tqdm
import numpy as np
from scipy.ndimage import center_of_mass
from SimpleITK import GetImageFromArray
from scipy.ndimage import find_objects
import argparse

def compute_bounding_box(mask):
    # find the bounding box of the binary mask
    mask = mask.astype(np.uint8)
    slices = find_objects(mask)
    
    # convert the slices to a tuple of (start, stop) pairs
    bbox = []
    for s in slices:
        start = (s[0].start, s[1].start, s[2].start)
        stop = (s[0].stop, s[1].stop, s[2].stop)
        bbox.append((start, stop))

    # return the bounding box
    return bbox

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data-path', type=str, required=True, help='Path to the preprocessed data (in npy format)')
    parser.add_argument('-s', '--save-path', type=str, required=True, help='Path to save the radiomics')
    args = parser.parse_args()

    data = np.load(args.data_path, allow_pickle=True).item()['images']
    volumes, masks = data[:, 0, ...], data[:, -1, ...]

    # transform volume and mask to niftii without saving but just for radiomics
    radiomics = {
        'x': [], 'y': [], 'z': [], # position features
        'voxel_volume': [], 'sphericity': [] # shape features
    }

    for idx in tqdm(range(volumes.shape[0]), position=0, leave=True):
        volume, mask = volumes[idx], masks[idx]

        if mask.min() < 0:
            mask[mask < 0] = 0

        # position relate features
        x, y, z = center_of_mass(mask)
        (x_a, y_a, z_a), (x_b, y_b, z_b) = compute_bounding_box(mask)[0]
        # w, h, d = x_b - x_a, y_b - y_a, z_b - z_a
        
        volume = GetImageFromArray(volume)
        mask = GetImageFromArray(mask)

        shape_radiomics = RadiomicsShape(volume, mask)
        glcm_radiomics = RadiomicsGLCM(volume, mask)
        
        # get shape radiomics
        voxel_volume = shape_radiomics.getVoxelVolumeFeatureValue()
        # surface_area = shape_radiomics.getSurfaceAreaFeatureValue()
        sphericity = shape_radiomics.getSphericityFeatureValue()

        # put all together
        for feature_key, feature in zip(
            radiomics.keys(),
            [x, y, z, voxel_volume, sphericity]
        ):
            radiomics[feature_key].append(feature)

    # convert to numpy array
    for key in radiomics.keys():
        radiomics[key] = np.array(radiomics[key])

    # normalizing features into classes
    # categorize each position feature {x, y, z} into 2 classes, dividing the image into 2 parts across each axis
    for idx, key in enumerate(['x', 'y', 'z']):
        radiomics[key] = (radiomics[key] > (masks.shape[idx + 1] // 2)).astype(np.int32)

    # for volume and spherecity, we will divide the values into 3 classes (small, medium, large)
    radiomics['voxel_volume'] = np.digitize(
        radiomics['voxel_volume'], 
        np.linspace(radiomics['voxel_volume'].min(), radiomics['voxel_volume'].max() + 1, 5)
    ) - 1
    radiomics['voxel_volume'][radiomics['voxel_volume'] == 3] = 2 # mergin 3rd and 4th classes

    # sphericity is already in the range of [0, 1]
    radiomics['sphericity'] = np.digitize(
        radiomics['sphericity'], 
        np.linspace(radiomics['sphericity'].min(), radiomics['sphericity'].max() + 0.01, 4)
    ) - 1

    # saving
    np.save('{}/radiomics.npy'.format(args.save_path), np.array(radiomics))
    print('Done!')

