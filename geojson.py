import os
# Provide the OpenSlide path if you are running OpenSlide from Windows !
openslide_path = 
os.environ['PATH'] = openslide_path + ";" + os.environ['PATH']
from openslide import OpenSlide
import numpy as np
import matplotlib.pyplot as plt
import cv2
import geojson
from skimage import io



file_geojson = # path to geojson

def get_labeled_zone_from_geojson(filename):
    data = geojson.load(open(filename))
    
    # A voir si on en a besoin ou si c'est toujours le même
    if "features" in data.keys():
        data = data["features"]
    elif "geometries" in data.keys():
        data = data["geometries"]
        
    regions = []
    regions_labels = []
    for i in range(len(data)):
        coordinates = data[i]['geometry']['coordinates'][0]
        res = []
        for a,b in coordinates:
            res.append([int(a), int(b)])
        regions_labels.append(data[i]['properties']['classification']['name'])
        regions.append(res)
    return regions, regions_labels


dict_labels = {"NERVE": 3, "ISLET": 2, "TISSUE FOR CLASSIF": 1, "Other": 0}

def compute_label_grid_geojson(regions, regions_labels, slide):
    matrix = np.zeros((slide.level_dimensions[0][1], slide.level_dimensions[0][0]),dtype=np.byte)
    for arrays, labels in zip(regions, regions_labels):
        # Cas Nerf, Ilot, Other
        if dict_labels[labels] != 1:
            continue
        contours = np.array([arrays])
        cv2.drawContours(matrix, contours, -1, (1), thickness=-1)
    
    matrix -= 1
    
    for arrays, labels in zip(regions, regions_labels):
        # Cas Nerf, Stroma, Other
        if dict_labels[labels] != 2:
            continue
        contours = np.array([arrays])
        cv2.drawContours(matrix, contours, -1, (1), thickness=-1)
        
    matrix -= 1    
    
    for arrays, labels in zip(regions, regions_labels):
        # Cas Ilot, Stroma, Other
        if dict_labels[labels] != 3:
            continue
        contours = np.array([arrays])
        cv2.drawContours(matrix, contours, -1, (1), thickness=-1)

    matrix += 2
    return matrix


# Compute bounding box around labeled zones
def create_annotation_zone_square(regions, regions_labels):
    res = []
    for arrays, labels in zip(regions, regions_labels):
        if labels == 'TISSUE FOR CLASSIF':
            min_x = arrays[0][0]
            max_x = arrays[0][0]
            min_y = arrays[0][1]
            max_y = arrays[0][1]
            for i in arrays:
                if min_x > i[0]:
                    min_x = i[0]
                if max_x < i[0]:
                    max_x = i[0]
                if min_y > i[1]:
                    min_y = i[1]
                if max_y < i[1]:
                    max_y = i[1]
            res.append([min_x, max_x, min_y, max_y])
    return res

  
# Generate dataset 
def create_all_files_for_training(images_path, new_images_path, patch_size):
    image_nb = 0
    temp = 0
    for filename in os.listdir(images_path):
        if filename.endswith('.svs'):
            svs_file_name = filename[:-4] + '.svs'
            label_file_name = filename[:-4] + '.geojson'

            regions, regions_labels = get_labeled_zone_from_geojson(os.path.join(images_path, label_file_name))

            label_zones_coords = create_annotation_zone_square(regions, regions_labels)

            slide = OpenSlide(os.path.join(images_path, svs_file_name))
            matrix = compute_label_grid_geojson(regions, regions_labels, slide)
            print('temp = ' + str(temp))
            temp += 1
            print(label_zones_coords)
            
            zone_nb = 0
            for i in label_zones_coords:
                min_x = i[0]
                max_x = i[1]
                min_y = i[2]
                max_y = i[3]

                for y in range (min_y, max_y - patch_size, patch_size):
                    for x in range (min_x, max_x - patch_size, patch_size):
                        #TODO CHECK IF DIMS BELOW ARE REVERSED OR NOT
                        new_matrix = matrix[y:y+patch_size, x:x+patch_size]
                        new_matrix[new_matrix != 0] -= 1
                        unique, counts = np.unique(new_matrix, return_counts=True)
                        my_dict = dict(zip(unique, counts))
                        text_output = max(my_dict, key=my_dict.get)
                        
                        text_file = open(os.path.join(new_images_path, filename[:-4] + '_zone_' + str(zone_nb) + '_' + str(image_nb) + '.txt'), "w")
                        text_file.write(str(text_output))
                        text_file.close()

                        io.imsave(os.path.join(new_images_path, filename[:-4] + '_zone_' + str(zone_nb) + '_' + str(image_nb) + '.tiff'), new_matrix, check_contrast=False)
                        new_image = np.asarray(slide.read_region((x, y), 0, (patch_size, patch_size)))[..., :3]
                        io.imsave(os.path.join(new_images_path, filename[:-4] + '_zone_' + str(zone_nb) + '_' + str(image_nb) + '.tiff'), new_image, check_contrast=False)
                        image_nb += 1
                zone_nb += 1  
  
tile_size = 1024# tile size
images_path = # Input Images
new_images_path # Output generated tiles of size tile size
create_all_files_for_training(images_path, new_images_path, tile_size)
