#! /usr/bin/env python

#Face detection software based on several HSFA or HiGSFA Networks
#This program implements a special purpose hierarchical network pipeline for image processing
#Each network is a hierarchical implementation of Slow Feature Analysis (dimensionality reduction) followed by a regression algorithm
#Now with performance analysis based on a coordinate file with ground-truth, work in progress
#Now with extensive error measurements
#By Alberto Escalante. Alberto.Escalante@neuroinformatik.ruhr-uni-bochum.de First Version 7 Juni 2010
#Ruhr-University-Bochum, Institute of Neural Computation, Group of Prof. Dr. Wiskott

import numpy
import scipy
import mkl
mkl.set_num_threads(12)

display_plots = True and False

import PIL
from PIL import Image
import mdp
import more_nodes
import patch_mdp

#import svm as libsvm
import object_cache as cache
import os, sys
import glob
import random
import sfa_libs
from sfa_libs import (scale_to, distance_squared_Euclidean, str3, wider_1Darray, ndarray_to_string, cutoff)
 
import system_parameters
from imageLoader import *
import classifiers_regressions as classifiers
import network_builder
import benchmarking
#import time
#from matplotlib.ticker import MultipleLocator
import copy
import string
import xml_frgc_tools as frgc
import getopt
import face_normalization_tools 


command_line_interface = True #and False
display_only_diagonal = False #or True
adaptive_grid_coords = True
adaptive_grid_scale = True
smallest_face = 0.20 # 20% of image size
grid_centering = True
plots_created = False #Move, this is not a configuration
write_results = True
if not command_line_interface: #WARNING!
    write_results = False
right_screen_eye_first = False
verbose_networks = True and False
true_coordinates = None
display_errors = True #and False
skip_existing_output = False
save_patches = False #or True
save_patches_base_dir = "./saved_patches"
network_figures_together = True #and False
cut_offs_face = [0.99, 0.95, 0.85, 0.8, 0.7, 0.6, 0.5, 0.45, 0.10, 0.05] #[0.99995,0.999,0.99997,0.9,0.8]
last_cut_off_face = -1
write_age_gender_confidence = True #and False
show_confidences = True
show_final_detection = False or True
camera_enabled = False
track_single_face= False
pygame_display = False
screen_width = 640 # 640
screen_height = 400 #400
save_normalized_face_detections = False

patch_overlap_sampling = 1.1 #1.0 = no overlap, 2.0 each image region belongs to approx 2 patches
patch_overlap_posx_posy = 1.1 #1.0 = no overlap, 2.0 each image region belongs to approx 4 patches (twice overlap in each direction)
tolerance_scale_deviation = 1.1 #1.4 # 1.2  #WARNING!!! 
tolerance_angle_deviation = 1.1
tolerance_posxy_deviation = 1.1 #1.0 = no tolerance besides invariance originally learned

estimate_age = True
estimate_gender = True
estimate_race = True

image_prescaling = True #and False
prescale_size = 1000
prescale_factor = 1.0

interpolation_formats = [Image.NEAREST]*10 #[Image.NEAREST]*1 + [Image.BILINEAR]*1 + [Image.BICUBIC]*8
                    
#sys.path.append("/home/escalafl/workspace/hiphi/src/hiphi/utils")
#list holding the benchmark information with entries: ("description", time as float in seconds)

benchmark = benchmarking.Benchmark(enabled=False)

executable_path = os.path.abspath(os.path.dirname(__file__))
print "executable path is", executable_path

pipeline_base_dir = executable_path + "/Pipelines"
#images_base_dir = "/local/escalafl/Alberto/ImagesFaceDetection"
networks_base_dir = executable_path + "/SavedNetworks"
classifiers_base_dir = executable_path + "/SavedClassifiers"
coordinates_filename = None

#scheduler = mdp.parallel.ThreadScheduler(n_threads=n_parallel)
scheduler = None

cache_obj = cache.Cache("", "")

#Returns the eye coordinates in the same scale as the box, already considering correction face_sampling
#First left eye, then right eye. Notice, left_x > right_x and eye_y < center_y
#Given approximate box coordinates, corresponding to a box with some face_sampling, approximates the
#positions of the eyes according to the normalization criteria.
#face_sampling < 1 means that the face is larger inside the box
def compute_approximate_eye_coordinates(box_coordinates, face_sampling=0.825, leftscreen_on_left=True):
    x0, y0, x1, y1 = box_coordinates
    fc_x = (x0+x1)/2.0
    fc_y = (y0+y1)/2.0

    if leftscreen_on_left == True:
        factor = 1
    else:
        factor = -1
    #eye deltas with respect to the face center
    eye_dx = 37.0/2.0 * numpy.abs(x1-x0)/ 128 / face_sampling
    eye_dy = 42.0/2.0 * numpy.abs(y1-y0)/ 128 / face_sampling
    eye_left_x = fc_x - factor*eye_dx
    eye_right_x = fc_x + factor*eye_dx
    eye_y = fc_y - eye_dy

    return numpy.array([eye_left_x, eye_y, eye_right_x, eye_y])

#In addition to the eye coordinates, it gives two boxes containing the left and right eyes
def compute_approximate_eye_boxes_coordinates(box_coordinates, face_sampling=0.825, eye_sampling=2.3719, leftscreen_on_left=True, rot_angle=None):
    x0, y0, x1, y1 = box_coordinates
    fc_x = (x0+x1)/2.0
    fc_y = (y0+y1)/2.0
    
    if leftscreen_on_left == True:
        mirroring_factor = 1
    else:
        mirroring_factor = -1
    
    if rot_angle is None:
        rot_angle = 0
        
    #eye deltas with respect to the face center in original image coordinates
    eye_dx = (37.0/2.0) * (numpy.abs(x1-x0) / 64.0) / (2 * 0.825)  # *0.825 ### 37.0/2.0 * numpy.abs(x1-x0) * 0.825 / 128 / face_sampling
    eye_dy = (42.0/2.0) * (numpy.abs(y1-y0) / 64.0) / (2 * 0.825) 
    
    (numpy.abs(x1-x0) / 64.0) * 64 * (2 *0.825)

    box_width = (numpy.abs(x1-x0) / (64.0 * 2 * 0.825)) * (64 * 2.3719 / 2) # 64 * numpy.abs(x1-x0) / 128 * eye_sampling / face_sampling
    box_height = box_width + 0.0
    #box_width = 64 * numpy.abs(x1-x0) / 128 * eye_sampling / face_sampling # 64 * numpy.abs(x1-x0) / 128 * eye_sampling / face_sampling
    #box_height = 64 * numpy.abs(y1-y0) / 128 * eye_sampling / face_sampling  
    rot_angle_radians = rot_angle * numpy.pi / 180

    eyeR_dx_rotated = eye_dx * numpy.cos(rot_angle_radians) - eye_dy * numpy.sin(rot_angle_radians)
    eyeR_dy_rotated = eye_dy * numpy.cos(rot_angle_radians) + eye_dx * numpy.sin(rot_angle_radians)
    eyeL_dx_rotated = (-1 * eye_dx) * numpy.cos(rot_angle_radians) - eye_dy * numpy.sin(rot_angle_radians)
    eyeL_dy_rotated = eye_dy * numpy.cos(rot_angle_radians) + (-1 * eye_dx) * numpy.sin(rot_angle_radians)
        
    eye_left_x = fc_x + mirroring_factor * eyeL_dx_rotated
    eye_right_x = fc_x + mirroring_factor * eyeR_dx_rotated
    eye_left_y = fc_y - eyeL_dy_rotated
    eye_right_y = fc_y - eyeR_dy_rotated
    box_left_x0 = eye_left_x - box_width/2.0
    box_left_x1 = eye_left_x + box_width/2.0
    box_right_x0 = eye_right_x - box_width/2.0
    box_right_x1 = eye_right_x + box_width/2.0
    box_left_y0 = eye_left_y - box_height/2.0
    box_left_y1 = eye_left_y + box_height/2.0
    box_right_y0 = eye_right_y - box_height/2.0
    box_right_y1 = eye_right_y + box_height/2.0
    
    # [coordinates of both eyes], [left eye box], [right eye box] 
    return numpy.array([eye_left_x, eye_left_y, eye_right_x, eye_right_y]), numpy.array([box_left_x0, box_left_y0, box_left_x1, box_left_y1]), numpy.array([box_right_x0, box_right_y0, box_right_x1, box_right_y1])

#Face midpoint is the average of the point between the eyes and the mouth
def compute_face_midpoint(eye_left_x, eye_left_y, eye_right_x, eye_right_y, mouth_x, mouth_y):
    eye_center_x = (eye_left_x+eye_right_x)/2.0
    eye_center_y = (eye_left_y+eye_right_y)/2.0
    midpoint_x = (eye_center_x + mouth_x)/2.0
    midpoint_y = (eye_center_y + mouth_y)/2.0
    return midpoint_x, midpoint_y

#Error in the (Euclidean) distance relative to the distance between the eyes
def relative_error_detection(app_eye_coords, eye_coords):
    dist_left = eye_coords[0:2]-app_eye_coords[0:2] #left eye
    dist_left = numpy.sqrt((dist_left**2).sum())
    dist_right = eye_coords[2:4]-app_eye_coords[2:4] #right eye
    dist_right = numpy.sqrt((dist_right**2).sum())
    dist_eyes = eye_coords[0:2]-eye_coords[2:4]
    dist_eyes = numpy.sqrt((dist_eyes**2).sum())
    return max(dist_left, dist_right) / dist_eyes
    
def face_detected(app_eye_coords, eye_coords, factor=0.25):
    rel_error = relative_error_detection(app_eye_coords, eye_coords)
    if rel_error < factor:
        return True
    else:
        return False

def FAR(faces_wrongly_detected, total_nofaces):
    return faces_wrongly_detected * 1.0 / total_nofaces

def FRR(faces_wrongly_rejected, total_faces):
    return faces_wrongly_rejected * 1.0 / total_faces

#TODO:USE SCORES TO SEE WHICH DETECTIONS ARE RETAINED AND WHICH ELIMINATED
#Detection confidence: 0.0=most likely a detection, 1.0=unlikely a detection
def purgueDetectedFacesEyesConfidence(detected_faces_eyes_confidences, weight_confidences_by_area=True):
#detected_faces_eyes might also contain confidence values or other information
    detected_faces_eyes_confidences = numpy.array(detected_faces_eyes_confidences)

    if len(detected_faces_eyes_confidences) > 1:
        detection_confidences = detected_faces_eyes_confidences[:,-1]
        if weight_confidences_by_area:
            #Eye distance is used to compute an "area"
            detection_areas = ((detected_faces_eyes_confidences[:,6]-detected_faces_eyes_confidences[:,4])**2+
                               (detected_faces_eyes_confidences[:,7]-detected_faces_eyes_confidences[:,5])**2)**0.5
            weighted_confidences = (1.0-detection_confidences) * detection_areas
            weighted_confidences = weighted_confidences / weighted_confidences.max()
        else:
            weighted_confidences = detection_confidences.copy()
        ordering = numpy.argsort(weighted_confidences)[::-1]
        #print "original confidences =", detected_faces_eyes_confidences[:,-1]
        detected_faces_eyes_confidences = detected_faces_eyes_confidences[ordering,:]
        
        #print "ordering=", ordering
        #print "ordered confidences =", detected_faces_eyes_confidences[:,-1]
        print "sorted detected_faces_eyes_confidences", detected_faces_eyes_confidences[:,-1]
        print "sorted weighted confidences:", weighted_confidences[ordering]
        
        unique_faces_eyes_confidences = []
        unique_faces_eyes_confidences.append(detected_faces_eyes_confidences[0])
        for face_eye_coords in detected_faces_eyes_confidences:
            min_d = 10000
            for face_eye_coords2 in unique_faces_eyes_confidences:
                error = relative_error_detection(face_eye_coords[4:8], face_eye_coords2[4:8])
                if error < min_d:
                    min_d = error
            if min_d > 0.25: #25: #entries are different enough
                unique_faces_eyes_confidences.append(face_eye_coords)
        return unique_faces_eyes_confidences
    else:
        return detected_faces_eyes_confidences.copy()

def read_batch_file(batch_filename):
    batch_file = open(batch_filename, "rb")
    lines = batch_file.readlines()
    batch_file.close()

#    if len(lines)%2 != 0:
#        print "Incorrect (odd) number of entries in batch file:"
#        print "Each line in the batch file should be an input image_filename followed with another line containing the corresponding output_filename"
#        exit(0)

    image_filenames = []
    output_filenames = []

    for i in range(len(lines)/2):
        image_filename = lines[2*i].rstrip()
        output_filename = lines[2*i+1].rstrip()
        image_filenames.append(image_filename)
        output_filenames.append(output_filename)
    return image_filenames, output_filenames




def load_true_coordinates(base_dir, true_coordinates_file): 
    #true_coordinates_file is a "normalization" file, with the following
    #FILE STRUCTURE. For each image: filename \n le_x le_y re_x re_y m_x m_y
    normalization_file = open(true_coordinates_file, "r")
    count = 0
    working = 1
    max_count = 200000
    image_filenames = []
    coordinates_dir = {}
    while working==1 and count < max_count:
        filename = normalization_file.readline().rstrip()
        if filename == "":
            working = 0
        else:
            coords_str = normalization_file.readline()
            coords = string.split(coords_str, sep=" ")
            float_coords = map(float, coords)

            if len(float_coords) == 8:            
                LeftEyeCenter_x, LeftEyeCenter_y, RightEyeCenter_x, RightEyeCenter_y, Nose_x, Nose_y, Mouth_x, Mouth_y = float_coords
            else: #Should be 6
                LeftEyeCenter_x, LeftEyeCenter_y, RightEyeCenter_x, RightEyeCenter_y, Nose_x, Nose_y = float_coords
                #Approximating mouth position as if face were perfectly vertical.
                Mouth_x = (LeftEyeCenter_x+RightEyeCenter_x)/2
                Mouth_y = (LeftEyeCenter_y+RightEyeCenter_y)/2 + (RightEyeCenter_x-LeftEyeCenter_x)*42.0/37.0 

            eyes_x_m = (RightEyeCenter_x + LeftEyeCenter_x) / 2.0
            eyes_y_m = (RightEyeCenter_y + LeftEyeCenter_y) / 2.0
            midpoint_eyes_mouth_x = (eyes_x_m + Mouth_x) / 2.0
            midpoint_eyes_mouth_y = (eyes_y_m + Mouth_y) / 2.0    
            dist_eyes = numpy.sqrt((LeftEyeCenter_x - RightEyeCenter_x)**2 + (LeftEyeCenter_y - RightEyeCenter_y)**2) 
    
            height_triangle = numpy.sqrt((eyes_x_m - Mouth_x)**2 + (eyes_y_m - Mouth_y)**2) 
      
            current_area = dist_eyes * height_triangle / 2.0
            desired_area = (37.0 / 0.825) * (42.0 / 0.825) / 2.0

            # if normalization_method == "mid_eyes_mouth":
            scale_factor =  numpy.sqrt(current_area / desired_area )
            #WARNING!!!
            ori_width = 128 * scale_factor #/ 0.825 #regression_width*scale_factor * 0.825
            ori_height = 128 * scale_factor #/ 0.825 #regression_height*scale_factor * 0.825

            box_x0 = midpoint_eyes_mouth_x-ori_width/2.0
            box_x1 = midpoint_eyes_mouth_x+ori_width/2.0
            box_y0 = midpoint_eyes_mouth_y-ori_height/2.0
            box_y1 = midpoint_eyes_mouth_y+ori_height/2.0

            # 8 coordinates +  6 coordinates
            all_coordinates = (LeftEyeCenter_x, LeftEyeCenter_y, RightEyeCenter_x, RightEyeCenter_y, Nose_x, Nose_y, Mouth_x, Mouth_y, midpoint_eyes_mouth_x, midpoint_eyes_mouth_y, box_x0, box_y0, box_x1, box_y1)
            if max_count != None and count > max_count:
                break

            if base_dir != None and base_dir != "":
                full_image_filename = base_dir + "/" + filename
            else:
                full_image_filename = filename
            image_filenames.append(full_image_filename)
            count += 1
            coordinates_dir[full_image_filename] = numpy.array(all_coordinates)
    return image_filenames, coordinates_dir

def pygame_sourface_to_PIL_image(im_pygame):
    imgstr = pygame.image.tostring(im_pygame, 'RGB')
    return Image.frombytes('RGB', im_pygame.get_size(), imgstr, "raw")
    
def image_array_contrast_normalize_avg_std(subimages_array, mean=0.0, std=0.2):
    #print "XXXX ", subimages_array[0].mean(), subimages_array[0].std(), subimages_array[0].min(), subimages_array[0].max()
    subimages_array -= subimages_array.mean(axis=1).reshape((-1,1)) 
    subimages_array /= (subimages_array.std(axis=1).reshape((-1,1)) / std) + 0.00000001 # std ends up being multiplied, division over zero is avoided
    #print "min =", subimages_array.min(axis=1).mean()
    subimages_array += mean 
    #print "after mean addition: min =", subimages_array.min(axis=1).mean()
    numpy.clip(subimages_array, 0.0, 255.0, subimages_array)
    #print "after clip: min =", subimages_array.min(axis=1).mean()
    #print "XXXX ", subimages_array[0].mean(), subimages_array[0].std(), subimages_array[0].min(), subimages_array[0].max()

def map_real_gender_labels_to_strings(gender_label_array, long_text=True):
    strings = []
    #label = -1 => Male, label = 1 => Female
    for label in gender_label_array:
        if label <= 0:
            if long_text:
                strings.append("Male")
            else:
                strings.append("M")
        else:
            if long_text:
                strings.append("Female")
            else:
                strings.append("F")

        if label < -1.000001 or label > 1.000001:
            er = "Unrecognized label: "+str(label)
            raise Exception(er) 
    return strings

def map_real_race_labels_to_strings(race_label_array, long_text=True):
    strings = []
    # label==-2: "Black", label == 2: "White"  ##NOTE:HERE LABELS ARE USED, NOT CLASSES
    for label in race_label_array:
        if label <= 0.0:
            if long_text:
                strings.append("Black")
            else:
                strings.append("B")
        else:
            if long_text:
                strings.append("White")
            else:
                strings.append("W")
        if label < -2.000001 or label > 2.000001:
            er = "Unrecognized label: "+str(label)
            raise Exception(er)
    return strings

numpy.random.seed(12345600)

verbose_pipeline = False

if verbose_pipeline:
    print "LOADING PIPELINE DESCRIPTION FILE"        

pipeline_filenames = cache.find_filenames_beginning_with(pipeline_base_dir, "Pipeline", recursion=False, extension=".txt")
if verbose_pipeline:
    print "%d pipelines found:"%len(pipeline_filenames), pipeline_filenames

if len(pipeline_filenames) <= 0:
    print "ERROR: No pipelines found in directory", pipeline_base_dir
    quit()

enable_select_pipeline = False #or True
for i, pipeline_filename in enumerate(pipeline_filenames):
    pipeline_base_filename = string.split(pipeline_filename, sep=".")[0] #Remove extension          
#    (NetworkType, Imsize) = pipeline_info = cache_read.load_obj_from_cache(base_dir="/", base_filename=pipeline_base_filename, verbose=True)
    if verbose_pipeline or enable_select_pipeline:
        print "Pipeline %d: %s"%(i, pipeline_base_filename)
                    
if enable_select_pipeline ==True:
    selected_pipeline = int( raw_input("Please select a pipeline [0--%d]:"%(len(pipeline_filenames)-1) ))
else:
    selected_pipeline = 0

if verbose_pipeline:
    print "Pipeline %d was selected"%selected_pipeline           

pipeline_filename = pipeline_filenames[selected_pipeline]
#pipeline_base_filename = string.split(pipeline_filename, sep=".")[0]

enable_eyes = True

benchmark.update_start_time()

pipeline_file = open(pipeline_filename, "rb")

num_networks = int(pipeline_file.readline())
if verbose_pipeline:
    print "Pipeline contains %d network/classifier pairs"%num_networks
#num_networks = 1

#The first line describes the face detection networks
tmp_string = pipeline_file.readline()
tmp_strings = string.split(tmp_string, " ")
net_Dx = int(tmp_strings[0])
net_Dy = int(tmp_strings[1])
net_Dang = float(tmp_strings[2])
net_mins = float(tmp_strings[3])
net_maxs = float(tmp_strings[4])

#Now read data for eye networks
#This is the scale in which the image patches are generated from the input image (usually 64x64)
#Pixel functions use this scale
subimage_width = int(tmp_strings[5])
subimage_height = int(tmp_strings[6])

#This is the scale in which the labels are given (usually 128x128)
#Functions related to regression/classification use this scale
regression_width = int(tmp_strings[7])
regression_height = int(tmp_strings[8])

#The second line describes the eye detection networks
tmp_string = pipeline_file.readline()
tmp_strings = string.split(tmp_string, " ")
eye_Dx = int(tmp_strings[0])
eye_Dy = int(tmp_strings[1])
eye_mins = float(tmp_strings[2])
eye_maxs = float(tmp_strings[3])

#This is the scale in which the image patches are generated from the input image (usually 32x32 or 64x64)
#Pixel functions use this scale
eye_subimage_width = int(tmp_strings[4])
eye_subimage_height = int(tmp_strings[5])

#This is the scale in which the labels are given (usually 128x128 or 64x64)
#Functions related to regression/classification use this scale
eye_regression_width = int(tmp_strings[6])
eye_regression_height = int(tmp_strings[7])

#regression_width = regression_height = 128 #Regression data assumes subimage has this size

#The third line describes the age estimation network
tmp_string = pipeline_file.readline()
tmp_strings = string.split(tmp_string, " ")
age_Dx = int(tmp_strings[0])
age_Dy = int(tmp_strings[1])
age_mins = float(tmp_strings[2])
age_maxs = float(tmp_strings[3])
age_subimage_width = int(tmp_strings[4]) #Size of image patches
age_subimage_height = int(tmp_strings[5])
age_regression_width = int(tmp_strings[6]) #Original size of image patches, with original scale of labels 
age_regression_height = int(tmp_strings[7])


network_types = []
network_filenames = []
classifier_filenames = [] 
for i in range(num_networks):
    network_type = pipeline_file.readline().rstrip()
    network_types.append(network_type)
    network_filename = pipeline_file.readline().rstrip()[0:-5]
    network_filenames.append(network_filename)    
    classifier_filename = pipeline_file.readline().rstrip()[0:-5]
    classifier_filenames.append(classifier_filename)

network_types = network_types + ["None"]*(18-len(network_types))
if verbose_networks:    
    print "network types:", network_types
    print "networks:", network_filenames
    print "classifiers:", classifier_filenames

networks = []
for network_filename in network_filenames:
    #load_obj_from_cache(self, hash_value=None, base_dir = None, base_filename=None, verbose=True)
    #[flow, layers, benchmark, Network]
    print "loading network or flow:", network_filename, "...",
    if network_filename !="None0":
        all_data = cache_obj.load_obj_from_cache(None, base_dir=networks_base_dir, base_filename=network_filename, verbose=True) 

        #for layer_node in all_data.flow:
        #    if isinstance(layer_node, mdp.hinet.Layer):
        #        for node in layer_node.nodes:
        #            if isinstance(node, mdp.nodes.IEVMLRecNode):
        #                print "deleting unnecessary data"
        #                if "cov_mtx" in node.sfa_node.__dict__:
        #                    del node.sfa_node.cov_mtx
        #                    del node.sfa_node.dcov_mtx
        #cache_obj.update_cache(all_data, None, networks_base_dir, network_filename+"OUT", overwrite=True, use_hash=None, verbose=True)         
    else:
        all_data = None
    
    if isinstance(all_data, (list, tuple)):
        print "Network flow was in a tupple"
        networks.append(all_data[0]) #Keep only the flows
    else:
        print "Network flow was not in a tupple"
        networks.append(all_data) #It is only a flow
    print "done"
#quit() 
    
classifiers = []
for classifier_filename in classifier_filenames:
    #load_obj_from_cache(self, hash_value=None, base_dir = None, base_filename=None, verbose=True)
    classifier = cache_obj.load_obj_from_cache(None, base_dir=classifiers_base_dir, base_filename=classifier_filename, verbose=True) 
    classifiers.append(classifier)

benchmark.add_task_from_previous_time("Loading of all networks and classifiers")

#if command_line_interface:
#    load_FRGC_images=False


#This is used for speed benchmarking, for detection accuracy see below
fr_performance = {}

true_positives = numpy.zeros(num_networks, dtype='int')
active_boxes = numpy.zeros(num_networks, dtype='int')
num_boxes = numpy.zeros(num_networks, dtype='int')
false_positives = numpy.zeros(num_networks, dtype='int')
false_negatives = numpy.zeros(num_networks, dtype='int')
offending_images = []
for i in range(num_networks):
    offending_images.append([])

def usage():
    usage_txt = "\n ********************** \n USAGE INFORMATION \n \
    FaceDetect: A program for face detection from frontal images  \n \
    Program Usage (either A, B or C): \n \
    A) python FaceDetect.py image_filename results_filename \n \
    example: $python FaceDetect.py sample_images/image0000.jpg results/output0000.txt \n \
    Many image formats are supported. The output file is a text file that has \n \
    zero or more lines of the form: left, top, right, bottom, xl, yl, xr, yr, \n \
    where each entry is an INTEGER value. \n \n \
    B) (batch mode) python FaceDetect.py --batch=batch_filename \n \
    where batch_filename is a text file containing many pairs \n \
    image_filename/results_filename (in different lines). \n \
    example $python FaceDetect.py --batch=sample_batchfile.txt \n \
    where sample_batchfile.txt might contain: \n \
    sample_images/image0000.jpg \n \
    output/output0000.txt \n \
    sample_images/image0001.jpg \n \
    output/output0001.txt \n \
    \n \
    Batch mode is much faster than the one-filename approach because the \n \
    software modules are loaded only once. It is important not to add \n \
    any white line or space after the last entry. See the provided \n \
    sample_batchfile.txt for an example of a batch file. \n \
    \n \
    C) Instead of excecuting Python directly, the bash script FaceDetect can be used: \n \
    $./FaceDetect image_filename results_filename can be used \n \
    (make sure FaceDetect is executable: $chmod +x FaceDetect) \n \
    \n \
    \
    Switches: \n \
    IMPORTANT: all switches must appear BEFORE the input and output filenames, if any. \n \
    \n \
        --smallest_face=k allows to specify the (approximate) size of the \n \
    smallest face that might appear in terms of the \n \
    size of the corresponding input image. 0.0<k<0.5 \n \
    example: $python FaceDetect.py  --smallest_face=0.15 sample_images/image000.jpg output/output000.txt \n \
    means that the smallest detected faces will have a size of at least \n \
    0.15*min(image_width, image_height) pixels. \n \
    The default value is 0.15. \n \
    \n \
        --right_screen_eye_first inverts the normal ordering of the eyes when writing the output file. \n \
    example: $python FaceDetect.py sample_images/AlbertoEscalante.jpg output1.txt \n \
    writes 27, 48, 82, 102, 43, 64, 63, 64 to the file output1.txt, while \n \
    $python FaceDetect.py --right_screen_eye_first sample_images/AlbertoEscalante.jpg output1.txt\n \
    writes 27, 48, 82, 102, 63, 64, 43, 64 to the file output2.txt \n \n \
    \n \
    Bugs/Suggestions/Comments/Questions: please write to alberto.escalante@ini.rub.de \n \
    I will be glad to help you \n"
    print usage_txt


image_filenames = []
output_filenames = []
image_numbers = []
if command_line_interface:
    argv = None
    if argv is None:
        argv = sys.argv
    if len(argv) >= 2:
        print "argv=", argv
        try:
            opts, args = getopt.getopt(argv[1:], "b:", ["batch=","smallest_face=","right_screen_eye_first", "display_errors=", "display_plots=", 
                                                        "coordinates_filename=", "true_coordinates_file=", "skip_existing_output=", "write_results=", 
                                                        "adaptive_grid_scale=", "adaptive_grid_coords=", "save_patches=","network_figures_together=", 
                                                        "last_cut_off_face=", "cut_offs_face=", "write_age_gender_confidence=", "show_final_detection=",
                                                        "camera_enabled=", "track_single_face=", "pygame_display=", "estimate_age_race_gender=",
                                                        "image_prescaling=", "save_normalized_face_detections="])
            files_set=False
            print "opts=", opts
            print "args=", args
            if len(args)==2:
                input_image = args[0]
                image_filenames = [input_image]
                image_numbers = numpy.arange(1)
                output_file = args[1]
                output_filenames = [output_file]
                print "Input image filename:", input_image
                print "Results filename:", output_file
                files_set=True
            elif len(args)==1 or len(args) > 2:
                print "Error: Wrong number of filenames: %s \n"%args 
                usage()
                sys.exit(2)
                               
            for opt, arg in opts:
#                print "opt=", opt
#                print "arg=", arg
                if opt in ('-b', '--batch'):
                    print "batch processing using file:", arg
                    if files_set:
                        print "Error: input image / output file was already set: ", input_image, output_file
                        usage()
                        sys.exit(2)

                    image_filenames, output_filenames = read_batch_file(arg)
                    image_numbers = numpy.arange(len(image_filenames))
                    print image_filenames
                    print output_filenames

                elif opt in ('--smallest_face'):
                    smallest_face = float(arg) 
                    print "changing default size of smallest face to be found to %f * min(image_height, image_width)"%smallest_face
                elif opt in ('--right_screen_eye_first'):
                    right_screen_eye_first = bool(int(arg))
                    print "changing default eye ordering. Now the eye most to the right on the screen appears on the output before the other eye"
                elif opt in ('--true_coordinates_file'):
                    true_coordinates_file = arg
                    image_filenames, true_coordinates = load_true_coordinates("",true_coordinates_file)
                    image_numbers = numpy.arange(len(image_filenames))
                    print "Loaded true coordinates file with %d entries"%(len(true_coordinates.keys()))
                elif opt in ('--display_errors'):    
                    display_errors = int(arg)
                    print "Changing display_errors to %d"%display_errors
                elif opt in ('--display_plots'):    
                    display_plots = bool(int(arg))
                    print "Changing display_plots to %d"%display_plots
                elif opt in ('--coordinates_filename'):    
                    coordinates_filename = arg
                    print "Setting coordinates file to %s"%coordinates_filename
                elif opt in ('--skip_existing_output'):
                    skip_existing_output=bool(int(arg))
                    print "Setting skip_existing_output to", skip_existing_output
                elif opt in ('--write_results'):
                    write_results = bool(int(arg))
                    print "Setting write_results to", write_results
                elif opt in ('--adaptive_grid_scale'):
                    adaptive_grid_scale = bool(int(arg))
                    print "Setting adaptive_grid_scale to", adaptive_grid_scale
                elif opt in ('--adaptive_grid_coords'):
                    adaptive_grid_coords = bool(int(arg))
                    print "Setting adaptive_grid_coords to", adaptive_grid_coords
                elif opt in ('--save_patches'):
                    save_patches = bool(int(arg))
                    print "Setting save_patches to", save_patches
                elif opt in ('--network_figures_together'):
                    network_figures_together = bool(int(arg))
                    print "Setting network_figures_together to", network_figures_together
                elif opt in ('--last_cut_off_face'):
                    last_cut_off_face = float(arg)
                    print "Setting last_cut_off_face to", last_cut_off_face
                elif opt in ('--cut_offs_face'):
                    cut_offs_face = string.split(arg,",")
                    cut_offs_face = map(float, cut_offs_face)
                    if len(cut_offs_face) != 10:
                        er = "Number of cut_off values should be 10 and separated by commas."
                        raise Exception(er)
                    print "Setting cut_off_faces to", cut_offs_face #," Last cut_off not changed."
                elif opt in ('--write_age_gender_confidence'):
                    write_age_gender_confidence = bool(int(arg))
                    print "Setting write_age_gender_confidence to", write_age_gender_confidence
                elif opt in ('--show_final_detection'):
                    show_final_detection = bool(int(arg))
                    print "Setting show_final_detection to", show_final_detection
                elif opt in ('--camera_enabled'):
                    camera_enabled = bool(int(arg))
                    print "Setting camera_enabled to", camera_enabled
                elif opt in ('--track_single_face'):
                    track_single_face = bool(int(arg))
                    print "Setting track_single_face to", track_single_face      
                elif opt in ('--pygame_display'):
                    pygame_display = bool(int(arg))
                    print "Setting pygame_display to", pygame_display  
                elif opt in ('--estimate_age_race_gender'):
                    estimate_age = estimate_race = estimate_gender = bool(int(arg))
                    print "Setting estimate_age,estimate_race, estimate_gender to", estimate_age, estimate_race, estimate_gender  
                elif opt in ('--image_prescaling'):
                    image_prescaling = bool(int(arg))
                    print "Setting image_prescaling to", image_prescaling
                elif opt in ('--save_normalized_face_detections'):
                    save_normalized_face_detections = bool(int(arg))
                    print "Setting save_normalized_face_detections to", save_normalized_face_detections
                else:
                    print "Option not handled:", opt

        except getopt.GetoptError:
            print "Error parsing the arguments"+ str(getopt.GetoptError)
            usage()
            sys.exit(2)
    else:
            usage()
            sys.exit(2)
    #quit()
        
else: #Use FRGC images
    print "Images:", image_filenames
    num_images = len(image_filenames)
    if num_images <= 0:
        raise Exception("No images Found")
    image_numbers = [34, 45, 47, 48, 49, 61, 74, 77]
    image_numbers = [762, 773, 777, 779, 850, 852, 871, 920, 921, 984]
    image_numbers = [871, 920, 921, 984]

#offending net 12/1000
#    image_numbers = [45, 47, 48, 49, 61, 102, 103, 104, 105, 136, 149, 150, 152, 153, 173, 175, 193, 196, 206, 230, 245, 261, 272, 282, 284, 292, 338, 380, 381, 411, 426, 427, 428, 437, 445, 489, 493, 499, 566, 591, 635, 636, 651, 661, 741, 750, 758, 762, 773, 777, 779, 850, 852, 871, 920, 921, 968, 984, 986]
    image_numbers = numpy.arange(0, 1000) #7,8,9
#image_numbers= [87]

benchmark.add_task_from_previous_time("Parsing of command line options")
    
if last_cut_off_face >= 0:
    print "Updating last cut_off to", last_cut_off_face
    cut_offs_face[9] = last_cut_off_face #There are exactly 10 possible cut_offs, not all necessarily used, except for the last one
    print "cut_offs_face=", cut_offs_face

if camera_enabled or pygame_display:
    import pygame
    import pygame.image
    #import pygame.freetype

   
if camera_enabled:
    import pygame.camera

    pygame.init()
    pygame.camera.init()
        
    cameras = pygame.camera.list_cameras()
    print "cameras available=", cameras
    if len(cameras) == 0:
        ex = "No cameras found"
        raise Exception(ex)

    print "Using camera %s ..." % cameras[0]

    webcam = pygame.camera.Camera(cameras[0])
    webcam.start()

    image = webcam.get_image()
    image_numbers = numpy.arange(0, 1000)
    screen_width = image.get_width()
    screen_height = image.get_height()

if pygame_display:
    screen = pygame.display.set_mode(( screen_width, screen_height))
    pygame.display.set_caption("Pygame/Camera View")
    if camera_enabled:
        screen.blit(image, (0,0))
        pygame.display.update() #pygame.display.flip()

if pygame_display and (estimate_age or estimate_race or estimate_gender):
    pygame.font.init()
    myfont = pygame.font.SysFont(None, 24) #pygame.font.SysFont("monospace", 15)
    #myfont = pygame.freetype.SysFont('Ubuntu Mono', 13)

benchmark.add_task_from_previous_time("Loaded and Started pygame (if enabled)")

database_original_points = {}
if coordinates_filename != None:
    coordinates_file = open(coordinates_filename, "r")
    working = 1
    while working==1:
        filename = coordinates_file.readline().rstrip()
        if filename == "":
            working = 0
        else: 
            coords_str = coordinates_file.readline()
            coords = string.split(coords_str, sep=" ")
            float_coords = map(float, coords)
            #Here person based coordinate system for the eyes
            RightEyeCenter_x, RightEyeCenter_y, LeftEyeCenter_x, LeftEyeCenter_y, Mouth_x, Mouth_y = float_coords

            n_x = 0
            n_y = 0 #we dont know much about the nose...
            
            eyes_x_m = (RightEyeCenter_x + LeftEyeCenter_x) / 2.0
            eyes_y_m = (RightEyeCenter_y + LeftEyeCenter_y) / 2.0

            midpoint_eyes_mouth_x = (eyes_x_m + Mouth_x) / 2.0
            midpoint_eyes_mouth_y = (eyes_y_m + Mouth_y) / 2.0
            
            dist_eyes = numpy.sqrt((LeftEyeCenter_x - RightEyeCenter_x)**2 + (LeftEyeCenter_y - RightEyeCenter_y)**2) 
    
            #Triangle formed by the eyes and the mouth.
            height_triangle = numpy.sqrt((eyes_x_m - Mouth_x)**2 + (eyes_y_m - Mouth_y)**2) 
          
            #Assumes eye line is perpendicular to the line from eyes_m to mouth
            current_area = dist_eyes * height_triangle / 2.0
            desired_area = 37.0 * 42.0 / 2.0
    
            # if normalization_method == "mid_eyes_mouth":
            scale_factor =  numpy.sqrt(current_area / desired_area )
            #Warning, is it subimage or regression???
            #regression is fine: subimage is used only for the physical sampling of the box, but its logical size is given by regression
    #        ori_width = subimage_width*scale_factor 
    #        ori_height = subimage_height*scale_factor
            ori_width = regression_width*scale_factor * 0.825
            ori_height = regression_height*scale_factor * 0.825

    #WARNING, using subpixel coordinates!   
    #        box_x0 = int(midpoint_eyes_mouth_x-ori_width/2)
    #        box_x1 = int(midpoint_eyes_mouth_x+ori_width/2)
    #        box_y0 = int(midpoint_eyes_mouth_y-ori_height/2)
    #        box_y1 = int(midpoint_eyes_mouth_y+ori_height/2)
            box_x0 = midpoint_eyes_mouth_x-ori_width/2
            box_x1 = midpoint_eyes_mouth_x+ori_width/2
            box_y0 = midpoint_eyes_mouth_y-ori_height/2
            box_y1 = midpoint_eyes_mouth_y+ori_height/2
            #############

            coordinates = (LeftEyeCenter_x, LeftEyeCenter_y, RightEyeCenter_x, RightEyeCenter_y, n_x, n_y, Mouth_x, Mouth_y)          
            more_coordinates = (midpoint_eyes_mouth_x, midpoint_eyes_mouth_y, box_x0, box_y0, box_x1, box_y1)
            all_coordinates = list(coordinates)+list(more_coordinates) # 8 coordinates +  6 coordinates
            
#            print "all_coords=", all_coordinates
            if filename in database_original_points.keys():
                database_original_points[filename].append(all_coordinates)
            else:
                database_original_points[filename] = [all_coordinates]                
    coordinates_file.close() 

    database_image_coordinates = []
    for i, filename in enumerate(image_filenames):
        database_image_coordinates.append(database_original_points[filename][0])
    database_image_coordinates = numpy.array(database_image_coordinates)

benchmark.add_task_from_previous_time("Loaded provided coordinate file (if any)")


if display_plots or show_final_detection:
    import matplotlib as mpl
    mpl.use('Qt4Agg')
    import matplotlib.pyplot as plt

benchmark.add_task_from_previous_time("Imported matplotlib (if enabled)")

#Sampling values with respect to regression_width and height values
#sampling_values = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
default_sampling_values = [0.475, 0.95, 1.9, 3.8]
default_sampling_values = [3.0] # [2.0, 3.8] # 2.0 to 4.0 for FRGC, 1.6

#Error measures for each image, sampling_value, network_number, box_number
#Include rel_ab_error, rel_scale_error, rel_eyepos_error, rel_eye_error

#plt.show(block=False) #WARNING!!!!

#Perhaps exchange sampling and image???
face_has_been_found = False
t_previous_capture = False
max_num_plots=10
num_plots=0
number_saved_image_age_estimation=0

finish_capture = False
for im_number in image_numbers:
    if finish_capture:
        break
    benchmark.update_start_time()
#    if load_FRGC_images:
#        frgc_image_coordinates = frgc_original_coordinates[im_number]
#
#    detected_faces_eyes = []
    if skip_existing_output:
        if os.path.lexists(output_filenames[im_number]):
            print "skipping image %d: %s" %(im_number, image_filenames[im_number])
            continue
    if coordinates_filename:
        database_current_image_coordinates = database_image_coordinates[im_number]
#        print "db_im_c=", database_current_image_coordinates
    #TODO: sampling values should be a function of the image size!!!
    
    
    #for now images has a single image
    print "I",
    if not camera_enabled:
        images = load_images([image_filenames[im_number]], image_format="L")
        images[0].load()
        images_rgb = load_images([image_filenames[im_number]], image_format="RGB")
        images_rgb[0].load()
    else:
        for i in range(10): #hoping to ensure the image is fresh
            im_pygame = webcam.get_image()

        im = pygame_sourface_to_PIL_image(im_pygame)
        
        images = [im.convert("L")]
        images_rgb = [im]
        print "images[0]=", images[0]
        t = time.time()
        if t_previous_capture is not None:
            print "%.5f Frames per second ##############################"%(1.0/(t-t_previous_capture))
        t_previous_capture = t

    if image_prescaling:
        prescaling_factor = max(images[0].size[0]*1.0/prescale_size, images[0].size[1]*1.0/prescale_size)
        if prescaling_factor > 1.0:
            prescaled_width = int(images[0].size[0]/prescaling_factor)
            prescaled_height= int(images[0].size[1]/prescaling_factor)
            images[0] = images[0].resize((prescaled_width, prescaled_height), Image.BILINEAR) # #[Image.NEAREST]*1 + [Image.BILINEAR]*1 + [Image.BICUBIC]*8
            images_rgb[0] = images_rgb[0].resize((prescaled_width, prescaled_height), Image.BILINEAR) # ANTIALIAS
        else:
            prescaling_factor = 1.0
            
    benchmark.add_task_from_previous_time("Image loaded or captured")

    im_height = images[0].size[1]
    im_width = images[0].size[0]
    
    if pygame_display:
        if (screen_width != im_width) or (screen_height != im_height):
            screen_width = im_width
            screen_height = im_height
            screen = pygame.display.set_mode(( screen_width, screen_height))
            pygame.display.set_caption("Pygame/Camera View")
        
        if not camera_enabled:
            im_pygame =  pygame.image.fromstring(images_rgb[0].tobytes(), images_rgb[0].size, images_rgb[0].mode)
        screen.blit(im_pygame, (0,0))
        ####pygame.display.flip()
        for e in pygame.event.get() :
            if e.type == pygame.QUIT :
                finish_capture = True #quit()


    #print images
    #quit()
    


    detected_faces_eyes_confidences = []
    t2 = time.time()
#    benchmark.append(("Image loading a sampling value %f"%sampling_value, t2-t1))

    if face_has_been_found and track_single_face:
        b_x0 = tracked_face[0]
        b_y0 = tracked_face[1]
        b_x1 = tracked_face[2]
        b_y1 = tracked_face[3]

        face_size = 0.5 * abs(b_x1-b_x0) + 0.5 * abs(b_y1-b_y0)
        sampling_value = face_size * 1.0  / subimage_width ###/ net_mins #What is net_mins, the forced scaling factor? why is it relevant?
        sampling_values = [sampling_value]
    elif adaptive_grid_scale:
        min_side = min(im_height, im_width)
        min_box_side = max(20, min_side * smallest_face * 0.825/ net_mins) # * 0.825/0.55) #smallest face patch should be at least 20 pixels!
        min_sampling_value = min_box_side * 1.0 / subimage_width
        sampling_values = []
        sampling_value = min_sampling_value
        
        new_grid_step = (net_maxs/net_mins) / patch_overlap_sampling
        while (subimage_width * sampling_value * net_mins/0.825 < im_width) and (subimage_height * sampling_value * net_mins/0.825 < im_height):
            sampling_values.append(sampling_value)
            sampling_value *= new_grid_step
#        max_box_side = min_side * 0.825 / numpy.sqrt(2)
#        sampling_values.append(max_box_side / regression_width)
        #sampling_values.append(min_side/subimage_width*0.98)
    else:
        min_side = min(im_height, im_width)
        min_box_side = max(20, min_side * smallest_face * 0.825/ net_mins) #smallest_face, at least 20 pixels!
        min_sampling_value = min_box_side * 1.0 / subimage_width
        sampling_values = [min_sampling_value]# default_sampling_values

#    print sampling_values
#    quit()
    benchmark.add_task_from_previous_time("Computed sampling values")

    for sampling_value in sampling_values:        
        benchmark.update_start_time()

        if face_has_been_found and track_single_face:
            patch_width = subimage_width * sampling_value
            patch_height = subimage_height * sampling_value
            
            patch_sepx = net_Dx * 2.0 * patch_width/regression_width
            patch_sepy = net_Dy * 2.0 * patch_height/regression_height

            #posX_values = [tracked_face[0], tracked_face[0]+patch_sepx, tracked_face[0] - patch_sepx,            tracked_face[0], tracked_face[0] ]
            #posY_values = [tracked_face[1],            tracked_face[1],              tracked_face[1], tracked_face[1]+patch_sepy, tracked_face[1]-patch_sepy ]
            posX_values = [tracked_face[0], tracked_face[0]+patch_sepx, tracked_face[0]-patch_sepx]
            posY_values = [tracked_face[1],            tracked_face[1],            tracked_face[1]]
            #posX_values = [tracked_face[0]]
            #posY_values = [tracked_face[1]]
            
        elif adaptive_grid_coords: 
            #Patch width and height in image coordinates
            #This is weird, why using regression_width here? I need logical pixels!!! => subimage_width
            patch_width = subimage_width * sampling_value# regression_width * sampling_value 
            patch_height = subimage_height * sampling_value#  regression_height * sampling_value 
#TODO: add random initialization between 0 and net_Dx * 2.0 * patch_width/regression_width, the same for Y
#These coordinates refer to the scaled image
            if verbose_networks:
                print "net_Dx=", net_Dx, "net_Dy=", net_Dy
            patch_horizontal_separation = net_Dx * 2.0 * patch_width/regression_width
            patch_vertical_separation = net_Dy * 2.0 * patch_height/regression_height

            #posX_values = numpy.arange(rest_horizontal/2, im_width-(patch_width-1), patch_horizontal_separation)
            num_x_patches = numpy.ceil( (1+(im_width-patch_width) / patch_horizontal_separation) * patch_overlap_posx_posy ) 
            posX_values = numpy.linspace(0.0, im_width-patch_width, num_x_patches) #Experimental	
            num_y_patches = numpy.ceil( (1+(im_height-patch_height) / patch_vertical_separation) * patch_overlap_posx_posy ) 
            #posY_values = numpy.arange(rest_vertical/2, im_height-(patch_height-1), patch_vertical_separation)
            posY_values = numpy.linspace(0.0, im_height-patch_height, num_y_patches)
            #A face must be detected by a box with a center distance and scale radio
            #interest points differ from center in these values
            max_Dx_diff = net_Dx * patch_width/regression_width
            max_Dy_diff = net_Dy * patch_height/regression_height
            min_scale_radio = net_mins / 0.825 #1/numpy.sqrt(2.0) #WARNING!!!! 
            max_scale_radio = net_maxs / 0.825 #numpy.sqrt(2.0) #WARNING!!!!

        if verbose_networks:
            print "max_Dx_diff=", max_Dx_diff,"max_Dy_diff=",  max_Dy_diff 
            print "posX_values=", posX_values
            print "posY_values=", posY_values
        
        #actually all resolutions can be processed also at once!
        orig_num_subimages = len(posX_values) * len(posY_values)
        orig_subimage_coordinates = numpy.zeros((orig_num_subimages, 4))
        
        #subimage_width, subimage_height
        for j, posY in enumerate(posY_values):
            for i, posX in enumerate(posX_values):
                orig_subimage_coordinates[j*len(posX_values)+i] = numpy.array([posX, posY, posX+patch_width-1, posY+patch_height-1])
        
        orig_angles = numpy.zeros((orig_num_subimages))

        base_magnitude = patch_width **2 + patch_height**2
        base_side = numpy.sqrt(base_magnitude)

        benchmark.add_task_from_previous_time("Computed grid for current sampling")        
        
        #print "subimage_coordinates", subimage_coordinates
        #base_estimation = orig_subimage_coordinates + 0.0
        #num_images is assumed to be 1 here, this might belong to the TODO
        orig_image_indices = numpy.zeros( 1 * orig_num_subimages, dtype="int")
        for im, image in enumerate(images):
        #    for xx in range(orig_num_subimages):
            orig_image_indices[im * orig_num_subimages:(im+1)*orig_num_subimages] = im 
        
        #Check that this is not memory inefficient
        #subimage_coordinates = subimage_coordinates times num_images
        #image = images[0] #TODO => loop
        
        orig_colors = numpy.random.uniform(0.0, 1.0, size=(orig_num_subimages,3))
        
        curr_num_subimages = orig_num_subimages + 0
        curr_subimage_coordinates = orig_subimage_coordinates + 0
        curr_angles = orig_angles + 0
        curr_invalid_subimages = numpy.zeros(curr_num_subimages, dtype='bool')
        curr_image_indices = orig_image_indices + 0
        curr_orig_index = numpy.arange(curr_num_subimages)

        if face_has_been_found and tracked_face and pygame_display:
            b_x0,b_y0,b_x1,b_y1 = orig_subimage_coordinates[0]
            wx = abs(b_x1 - b_x0)
            wy = abs(b_y1 - b_y0)
            pygame.draw.rect(screen, (150, 255, 150), (b_x0,b_y0,wx,wy), 2)
        
        
        num_plots += 1
        if num_plots > max_num_plots:
            display_plots = False

        if display_plots or show_final_detection:
            im_disp = numpy.asarray(images[0])
            im_disp_rgb = numpy.asarray(images_rgb[0])
        
        if display_plots:
            plots_created = True
            if network_figures_together:
                f0 = plt.figure()
                plt.suptitle("Iterative Face Detection")
                p11 = plt.subplot(3,6,1)
                plt.title("Original")
                p12 = plt.subplot(3,6,2)
                plt.title(network_types[0])
                p13 = plt.subplot(3,6,3)
                plt.title(network_types[1])
                p14 = plt.subplot(3,6,4)
                plt.title(network_types[2])
                p15 = plt.subplot(3,6,5)
                plt.title(network_types[3])
                p16 = plt.subplot(3,6,6)
                plt.title(network_types[4])

                p21 = plt.subplot(3,6,7)
                plt.title(network_types[5])
                p22 = plt.subplot(3,6,8)
                plt.title(network_types[6])
                p23 = plt.subplot(3,6,9)
                plt.title(network_types[7])
                p24 = plt.subplot(3,6,10)
                plt.title(network_types[8])
                p25 = plt.subplot(3,6,11)
                plt.title(network_types[9])
                p26 = plt.subplot(3,6,12)
                plt.title(network_types[10])
                
                p31 = plt.subplot(3,6,13)
                plt.title(network_types[11])
                p32 = plt.subplot(3,6,14)
                plt.title(network_types[12])
                p33 = plt.subplot(3,6,15)
                plt.title(network_types[13])
                p34 = plt.subplot(3,6,16)
                plt.title(network_types[14])
                p35 = plt.subplot(3,6,17)
                plt.title(network_types[15])
                p36 = plt.subplot(3,6,18)
                plt.title(network_types[16])
            else:
                f0 = plt.figure()
                plt.suptitle("Iterative Face Detection 11.")
                p11 = plt.subplot(1,1,1)
                f0 = plt.figure() 
                plt.suptitle("Iterative Face Detection 12"+network_types[0])
                p12 = plt.subplot(1,1,1)
                f0 = plt.figure() 
                plt.suptitle("Iterative Face Detection 13"+network_types[1])
                p13 = plt.subplot(1,1,1)
                f0 = plt.figure() 
                plt.suptitle("Iterative Face Detection 14"+network_types[2])
                p14 = plt.subplot(1,1,1)
                f0 = plt.figure() 
                plt.suptitle("Iterative Face Detection 15"+network_types[3])
                p15 = plt.subplot(1,1,1)
                f0 = plt.figure() 
                plt.suptitle("Iterative Face Detection 16"+network_types[4])
                p16 = plt.subplot(1,1,1)
                f0 = plt.figure() 
                plt.suptitle("Iterative Face Detection 21"+network_types[5])
                p21 = plt.subplot(1,1,1)
                f0 = plt.figure() 
                plt.suptitle("Iterative Face Detection 22"+network_types[6])
                p22 = plt.subplot(1,1,1)
                f0 = plt.figure() 
                plt.suptitle("Iterative Face Detection 23"+network_types[7])
                p23 = plt.subplot(1,1,1)
                f0 = plt.figure() 
                plt.suptitle("Iterative Face Detection 24"+network_types[8])
                p24 = plt.subplot(1,1,1)
                f0 = plt.figure() 
                plt.suptitle("Iterative Face Detection 25"+network_types[9])
                p25 = plt.subplot(1,1,1)
                f0 = plt.figure() 
                plt.suptitle("Iterative Face Detection 26"+network_types[10])
                p26 = plt.subplot(1,1,1)
                f0 = plt.figure() 
                plt.suptitle("Iterative Face Detection 31"+network_types[11])
                p31 = plt.subplot(1,1,1)
                f0 = plt.figure() 
                plt.suptitle("Iterative Face Detection 32"+network_types[12])
                p32 = plt.subplot(1,1,1)
                f0 = plt.figure() 
                plt.suptitle("Iterative Face Detection 33"+network_types[13])
                p33 = plt.subplot(1,1,1)
                f0 = plt.figure() 
                plt.suptitle("Iterative Face Detection 34"+network_types[14])
                p34 = plt.subplot(1,1,1)
                f0 = plt.figure() 
                plt.suptitle("Iterative Face Detection 35"+network_types[15])
                p35 = plt.subplot(1,1,1)
                f0 = plt.figure() 
                plt.suptitle("Iterative Face Detection 36"+network_types[16])
                p36 = plt.subplot(1,1,1)           
    #        print "len(images)",len(images)
    #        quit()
            p11.imshow(im_disp, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
        #color=(r_color[sig], g_color[sig], b_color[sig])
        
            for ii, (x0, y0, x1, y1) in enumerate(orig_subimage_coordinates):
            #    p11.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=colors[ii] )
                p11.plot([x0, x1], [y0, y1], color=orig_colors[ii] )
        
        ###from matplotlib.lines import Line2D
        ##line = mpl.lines.Line2D([10,20,30,40,500], [10,40, 35, 20, 500],
        ##              linewidth=4, color='green', transform=f0.transFigure)
        ##f0.lines.append(line)
        
        if display_plots:
            subplots = [p12, p13, p14, p15, p16, p21, p22, p23, p24, p25, p26, p31, p32, p33, p34, p35, p36, None]
        else:
            subplots = [None,]*20

        benchmark.add_task_from_previous_time("Window creation, and pre computations")        
        
        for num_network in range(num_networks-5):
            benchmark.update_start_time()

            print network_types[num_network],
            network_type = network_types[num_network][0:-1]
            network_serial = int(network_types[num_network][-1])
            interpolation_format = interpolation_formats[network_serial]
            
            skip_image_extraction = 0
            skip_feature_extraction = 0
            if num_network > 0:
                if network_types[num_network-1][0:-1] == "Disc":
                    print "skiping image extraction because previous net is of type Disc!"
                    skip_image_extraction = 1
#                else:
#                    print "NT-1=", network_types[num_network-1][0:-1]
            if networks[num_network] is None:
                skip_image_extraction = 1
                skip_feature_extraction = 1

            if skip_image_extraction == 0:
                #Get arrays
                print "P",
                    
                #subimages = extract_subimages_rotate_ar_parallel(images, curr_image_indices, curr_subimage_coordinates, -1*curr_angles, (subimage_width, subimage_height), interpolation_format)
                subimages = extract_subimages_rotate(images, curr_image_indices, curr_subimage_coordinates, -1*curr_angles, (subimage_width, subimage_height), interpolation_format)
                #subimages_arr = subimages
                if len(subimages) > 0:
                    subimages_arr = images_asarray(subimages)+0.0 ##### Warning!!!
                    contrast_normalize = True and False
                    if contrast_normalize:
                        #print "min and max image array intensities are:", subimages_arr.min(), subimages_arr.max()
                        print "Orig mean=", subimages_arr.mean(), " and std=", subimages_arr.std(axis=1).mean(), " min=", subimages_arr.min(axis=1).mean(), " max=", subimages_arr.max(axis=1).mean()
                        image_array_contrast_normalize_avg_std(subimages_arr, 137.5, 0.40*255)
                        print "After contrast norm: mean=", subimages_arr.mean(), " and std=", subimages_arr.std(axis=1).mean(), " min=", subimages_arr.min(axis=1).mean(), " max=", subimages_arr.max(axis=1).mean()
                        #quit()
                else:
                    subimages_arr = numpy.zeros((0,0))

            else:
                print "reusing existing subimage patches, since network is None or previous net is Disc"
            
            benchmark.add_task_from_previous_time("Extraction of subimages patches")        
             
#            if num_network == num_networks-3: #last network before eye detection networks
#                print "curr_angles=", curr_angles
#                for i in range(len(subimages)):
#                    print "saving patch"
#                    subimages[i].save("patch_im%03d_PAngle%f.jpg"%(i,curr_angles[i]))
            reference = "networks" #"net%d"%num_network
            if len(subimages_arr) > 0:          
                if skip_feature_extraction == 0:
                    t_afterloading = time.time()
                    print "E",
                    benchmark.update_start_time(reference=reference)
                    benchmark.set_default_reference(reference)
                    sl = networks[num_network].execute(subimages_arr, benchmark=benchmark)
                    benchmark.set_default_reference("a")

                    #sl = networks[num_network].execute(subimages_arr, benchmark=benchmark)
                    if verbose_networks:
                        print "Network %d processed all subimages"%num_network
                else:
                    #sl = sl #Nothing to do, tThis has been updated already
                    print "reusing slow features previously computed, since network is None or previous network is Disc" 
                num_boxes[num_network] += sl.shape[0]            
                
                reg_num_signals = classifiers[num_network].input_dim
                
                benchmark.add_task_from_previous_time("Feature extraction")        

                t_class = time.time()
                avg_labels = classifiers[num_network].avg_labels #TODO: IT IS NOT CLEAN TO HAVE THIS EXTRA ARGUMENT, IMPROVE CUICUILCO_RUN, ETC TO REMOVE IT

                if sl.shape[0] > 0:
                    print "C",
                    reg_out = classifiers[num_network].regression(sl[:,0:reg_num_signals], avg_labels) #GaussianRegression(sl[:,0:reg_num_signals])
                else:
                    reg_out = numpy.zeros(0)
                print "R",
                benchmark.add_task_from_previous_time("Regression")        

                #print "reg_out=", reg_out
            
#                if num_network in [0, 4, 8, 12]:
#                    network_type = "discrimination"
#                elif num_network in [1, 5, 9]:
#                    network_type = "posX"
#                elif num_network in [2, 6, 10]:
#                    network_type = "posY"
#                elif num_network in [3, 7, 11]:
#                    network_type = "scale"

                if network_type == "Disc":
                    pass #WARNING!
                elif network_type == "PosX": #POS_X       
                    width = curr_subimage_coordinates[:, 2] - curr_subimage_coordinates[:, 0]
                    reg_out = reg_out * width / regression_width
            #        print "Regression Output scaled:", reg_out
            #        print "Correcting coordinates (X)"
                    curr_subimage_coordinates[:, 0] = curr_subimage_coordinates[:, 0] - reg_out  #X0
                    curr_subimage_coordinates[:, 2] = curr_subimage_coordinates[:, 2] - reg_out  #X1
                elif network_type == "PosY":     #POS_Y
                    height = curr_subimage_coordinates[:, 3] - curr_subimage_coordinates[:, 1]
                    reg_out = reg_out * height / regression_height
            #        print "Regression Output scaled:", reg_out
            #        print "Correcting coordinates (Y)"
                    curr_subimage_coordinates[:, 1] = curr_subimage_coordinates[:, 1] - reg_out  #Y0
                    curr_subimage_coordinates[:, 3] = curr_subimage_coordinates[:, 3] - reg_out  #Y1
                elif network_type == "PAng":     #PAng
                    #curr_angles = curr_angles + reg_out #THIS SIGN IS NOT CLEAR AT ALL!
                    curr_angles = curr_angles + reg_out #0.0 #reg_out ##+ reg_out #THIS SIGN IS NOT CLEAR AT ALL!
                    
#                    height = curr_subimage_coordinates[:, 3] - curr_subimage_coordinates[:, 1]
#                    reg_out = reg_out * height / regression_height
#            #        print "Regression Output scaled:", reg_out
#            #        print "Correcting coordinates (Y)"
#                    curr_subimage_coordinates[:, 1] = curr_subimage_coordinates[:, 1] - reg_out  #Y0
#                    curr_subimage_coordinates[:, 3] = curr_subimage_coordinates[:, 3] - reg_out  #Y1
                elif network_type == "Scale": #SCALE
                    old_width = curr_subimage_coordinates[:, 2] - curr_subimage_coordinates[:, 0]
                    old_height = curr_subimage_coordinates[:, 3] - curr_subimage_coordinates[:, 1]
                    x_center = (curr_subimage_coordinates[:, 2] + curr_subimage_coordinates[:, 0])/2.0
                    y_center = (curr_subimage_coordinates[:, 3] + curr_subimage_coordinates[:, 1])/2.0
            
                    desired_sampling = 0.825 #1 or better: (0.55 + 1.1)/2
            
                    width = old_width / reg_out * desired_sampling
                    height = old_height / reg_out * desired_sampling
            #        print "Regression Output scaled:", reg_out
            #        print "Correcting scale (X)"
                    curr_subimage_coordinates[:, 0] = x_center - width / 2.0
                    curr_subimage_coordinates[:, 2] = x_center + width / 2.0
                    curr_subimage_coordinates[:, 1] = y_center - height / 2.0
                    curr_subimage_coordinates[:, 3] = y_center + height / 2.0
                else:
                    print "Network type unknown!!!: ", network_type
                    quit()
                    pass #regression does not need to modify subimage coordinates
            
                if network_type in ["PosX","PosY","PAng", "Scale"]: 
                    #out of image
                    out_of_borders_images = (curr_subimage_coordinates[:,0]<0) | (curr_subimage_coordinates[:,1]<0) | \
                    (curr_subimage_coordinates[:,2]>=im_width) | (curr_subimage_coordinates[:,3]>=im_height)      

                    #too large or small w.r.t. initial patch
                    subimage_magnitudes = ((curr_subimage_coordinates[:,0:2] - curr_subimage_coordinates[:,2:4])**2).sum(axis=1)
                    subimage_sides = numpy.sqrt(subimage_magnitudes)
                    #sqrt(2)/2*orig_diagonal = 1/sqrt(2)*orig_diagonal < subimage_diagonal < sqrt(2)*orig_diagonal ???
                    too_large_small_images = (subimage_sides/base_side > max_scale_radio*tolerance_scale_deviation) | (subimage_sides/base_side < min_scale_radio/tolerance_scale_deviation)      

                    #too far away horizontally from initial pach
                    subimage_deltas_x = (curr_subimage_coordinates[:,2] + curr_subimage_coordinates[:,0])/2 - (orig_subimage_coordinates[curr_orig_index][:,2] + orig_subimage_coordinates[curr_orig_index][:,0])/2
                    subimage_deltas_y = (curr_subimage_coordinates[:,3] + curr_subimage_coordinates[:,1])/2 - (orig_subimage_coordinates[curr_orig_index][:,3] + orig_subimage_coordinates[curr_orig_index][:,1])/2

                    #too much rotation w.r.t. initial patch            
                    too_rotated_images = (curr_angles > orig_angles[curr_orig_index] + net_Dang*tolerance_angle_deviation) | (curr_angles < orig_angles[curr_orig_index] -net_Dang*tolerance_angle_deviation)
                    x_far_images = numpy.abs(subimage_deltas_x) > (max_Dx_diff * tolerance_posxy_deviation)
                    y_far_images = numpy.abs(subimage_deltas_y) > (max_Dy_diff * tolerance_posxy_deviation)

                    #new_wrong_images = out_of_borders_images | too_large_small_images | x_far_images | y_far_images | too_rotated_images
                    new_wrong_images = too_large_small_images | x_far_images | y_far_images | too_rotated_images
        
                    debug_net_discrimination=False
                    if debug_net_discrimination:
#                        print "subimage_deltas_x is: ", subimage_deltas_x
#                        print "subimage_deltas_y is: ", subimage_deltas_y
                        print "Patch discarded. Wrong x_center is:", (curr_subimage_coordinates[:,2][x_far_images] + curr_subimage_coordinates[:,0][x_far_images])/2
                        print "Patch discarded. Wrong x_center was:", (orig_subimage_coordinates[:,2][curr_orig_index[x_far_images]] + orig_subimage_coordinates[:,0][curr_orig_index[x_far_images]])/2
                        print "Patch discarded. Wrong y_center is:", (curr_subimage_coordinates[:,3][y_far_images] + curr_subimage_coordinates[:,1][y_far_images])/2
                        print "Patch discarded. Wrong y_center was:", (orig_subimage_coordinates[:,3][curr_orig_index[y_far_images]] + orig_subimage_coordinates[:,1][curr_orig_index[y_far_images]])/2
                        print "new_wrong_images %d = out_of_borders_images %d + too_large_small_images %d + x_far_images %d + y_far_images %d" % \
                        (new_wrong_images.sum(), out_of_borders_images.sum(), too_large_small_images.sum(), x_far_images.sum(), y_far_images.sum())
                    else:
                        pass
                else:
                    cut_off_face = cut_offs_face[network_serial]
                    new_wrong_images = reg_out >= cut_off_face
                    #print "new_wrong_images",new_wrong_images
                
                #TODO: Make shure box_side is defined if all boxes are eliminated after the first run
            
                #Update subimage patch information, on temporal variables
                new_num_subimages = curr_num_subimages - new_wrong_images.sum()
                new_subimage_coordinates = curr_subimage_coordinates[new_wrong_images==0]+0.0
                new_angles = curr_angles[new_wrong_images==0]+0.0
                new_invalid_subimages = numpy.zeros(new_num_subimages, dtype='bool')
                new_image_indices = curr_image_indices[new_wrong_images==0] + 0
                new_orig_index = curr_orig_index[new_wrong_images==0] + 0
            

                if verbose_networks or network_type=="Disc":            
                    print "%d / %d valid images"%(new_num_subimages, curr_num_subimages)
                #Overwrite current values
                curr_num_subimages = new_num_subimages
                curr_subimage_coordinates = new_subimage_coordinates +0.0
                curr_angles = new_angles
                curr_invalid_subimages = new_invalid_subimages
                curr_image_indices = new_image_indices 
                curr_orig_index = new_orig_index
                sl = sl[new_wrong_images==0] + 0 #.copy()
                subimages_arr = subimages_arr[new_wrong_images==0,:] + 0

                if network_type == "Disc":
                    curr_confidence = reg_out[new_wrong_images==0] + 0            

                benchmark.add_task_from_previous_time("Adjusted according to regression")
                
                subplot = subplots[num_network]
                if subplot!=None:
                    subplot.imshow(im_disp_rgb, aspect=1.0, interpolation='nearest', origin='upper')
                    #subplot.imshow(im_disp, aspect='auto', interpolation='nearest', origin='upper', cmap=mpl.pyplot.cm.gray)
                    
                    for j, (x0, y0, x1, y1) in enumerate(curr_subimage_coordinates):
                        color = orig_colors[curr_orig_index[j]]
                        if show_confidences and (network_type == "Disc"):
                            color =  (curr_confidence[j], curr_confidence[j], curr_confidence[j])
                            color = (0.25,0.5,1.0)
                        if display_only_diagonal == True:
                            subplot.plot([x0, x1], [y0, y1], color=color )
                        else:
                            #subplot.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=(1.0,1.0,1.0), linewidth=2)
                            subplot.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=color, linewidth=2)                            
                #        if invalid_subimages[j] == False and False:
                #        subplot.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=orig_colors[curr_orig_index[j]] )
                #        if invalid_subimages[j] == False:
                        #subplot.plot([x0, x1], [y0, y1], color=orig_colors[curr_orig_index[j]] )
                        cx = (x0+x1-1)/2.0
                        cy = (y0+y1-1)/2.0
                        mag = 0.4*(x1-x0)
                        pax = cx + mag*numpy.cos(curr_angles[j]*numpy.pi/180+numpy.pi/2)
                        pay = cy - mag*numpy.sin(curr_angles[j]*numpy.pi/180+numpy.pi/2)
                        subplot.plot([cx, pax], [cy, pay], color=color, linewidth=2)

            #    print "subimage_coordinates", subimage_coordinates
                benchmark.add_task_from_previous_time("Created network plot")
        
            ##print "Done loading subimages_train: %d Samples"%len(subimages_train)
            ##flow, layers, benchmark, Network = cache.load_obj_from_cache(network_hash, network_base_dir, "Network", verbose=True)   
            ##print "Done loading network: " + Network.name
        
                display_face = True
                display_aface = True
        
#            TODO: Make sure this works when nothing was found at all inside an image

            #WHICH PLOT IS USED HERE????? The last one used by the network.
            #Display Faces to find:
            if true_coordinates != None:
                sampled_face_coordinates = true_coordinates[image_filenames[im_number]]
                (el_x, el_y, er_x, er_y, n_x, n_y, m_x, m_y, fc_x, fc_y, b_x0, b_y0, b_x1, b_y1) = sampled_face_coordinates
                eye_coords = numpy.array([el_x, el_y, er_x, er_y])
                if display_face and display_plots:
                    #Face Box
                    subplot.plot([b_x0, b_x1, b_x1, b_x0, b_x0], [b_y0, b_y0, b_y1, b_y1, b_y0], "r")
                    #Left eye, right eye and face center
                    #subplot.plot([el_x, er_x, fc_x], [el_y, er_y, fc_y], "ro")
                    subplot.plot([el_x, er_x, fc_x], [el_y, er_y, fc_y], "ro")
                    
            
            #For each face on the image,now there is only one
            #For each remaining image patch
            #Compute FRR, FAR, Error
                box_detected = False
                face_detected = False
                for j in range(len(curr_subimage_coordinates)):
        ##            print "********************"
                    orig_sub_coords = orig_subimage_coordinates[curr_orig_index[j]] #Original patch coordinates
                    (ab_x0, ab_y0, ab_x1, ab_y1) = curr_sub_coords = curr_subimage_coordinates[j] #Current patch coordinates
                    afc_x = (ab_x0 + ab_x1)/2.0 #Here "a" prefix seems to mean "after" the patch was normalized (current positon), "f"=face, "c"=center
                    afc_y = (ab_y0 + ab_y1)/2.0
            
                    bcenter_x_orig = (orig_sub_coords[0]+orig_sub_coords[2])/2.0 #Box center at original position
                    bcenter_y_orig = (orig_sub_coords[1]+orig_sub_coords[3])/2.0 #Here "b" prefix seems to mean box
            
                    bcenter_x = (ab_x0 + ab_x1)/2.0 #Box center at current postion
                    bcenter_y = (ab_y0 + ab_y1)/2.0
                               
                    box_side =  numpy.abs(b_x1-b_x0) #side (width) of the real face_sampled face box ###equals 0.825???
                    abox_side =  numpy.abs(ab_x1-ab_x0) #Side (width) of patch at current "actual" position
                    box_side_orig =numpy.abs(orig_sub_coords[2]-orig_sub_coords[0]) #Side (width) of patch at original position
    
                    #Errors in image pixels
                    bx_error_orig = fc_x - bcenter_x_orig #Face center X error w.r.t. original patch
                    by_error_orig = fc_y - bcenter_y_orig #Face center Y error w.r.t. original patch
                    bx_error = fc_x - bcenter_x #Face center X error w.r.t. actual patch
                    by_error = fc_y - bcenter_y #Face center Y error w.r.t. actual patch
                    #Errors in regression image pixels
                    rel_bx_error = (bx_error / box_side) * regression_width 
                    rel_by_error = (by_error / box_side) * regression_height      
            
                    scale_error = box_side / abox_side - 1.0 
                    #Error with respect to the goal sampling value of 0.825
                    rel_scale_error = scale_error #* 0.825 ####WARNING!!!        
                    # rel_scale_error = 0.825 / box_side * abox_side - 0.825
            
                    (ael_x, ael_y, aer_x, aer_y) = sampled_app_eye_coords = compute_approximate_eye_coordinates(curr_sub_coords, face_sampling=0.825)
                    app_eye_coords = sampled_app_eye_coords 
                    #Error in image pixels
                    rel_eyes_pix_error = (app_eye_coords - eye_coords) / box_side * regression_width
                    #Normalized eye error, goal is a relative error < 0.25
                    rel_eye_error = relative_error_detection(app_eye_coords, eye_coords)        
            
        ##            print "bx_error_orig = %f/%f/%f"%(bx_error_orig, max_Dx_diff, 1.0)   
        ##            print "by_error_orig = %f/%f/%f"%(by_error_orig, max_Dy_diff, 1.0)
        ##    #        print "bx_error = %f/%f/%f"%(bx_error, max_Dx_diff, 1.0)   
        ##    #        print "by_error = %f/%f/%f"%(by_error, max_Dy_diff, 1.0)
        ##            print "rel_eyes_pix_error = ", rel_eyes_pix_error
        ##            
        ##            #relative errors are errors in the original scales of 128 x 128 pixels & true scale = 0.825
        ##            print "rel_bx_error = ", rel_bx_error, "pixels"   
        ##            print "rel_by_error = ", rel_by_error, "pixels"
        ##            print "rel_scale_error =", rel_scale_error, "deviating from 0.825"
        ##            #relative eye error is normalized to the distance between the eyes, should be at most 0.25 for true detection
        ##            print "rel_eye_error =", rel_eye_error        
                    
                    debug_resp_box=False or True
                    
                    #Face is within this box?
                    print "num_network=", num_network, "numpy.abs(rel_bx_error)=", numpy.abs(rel_bx_error), "numpy.abs(rel_by_error)=", numpy.abs(rel_by_error), \
                    "box_side / box_side_orig=", box_side / box_side_orig
                    if numpy.abs(bx_error_orig)<max_Dx_diff and numpy.abs(by_error_orig)<max_Dy_diff and \
                    box_side / box_side_orig > min_scale_radio and box_side / box_side_orig < max_scale_radio:
                        #Bingo, this is responsible of detecting the face
                        if debug_resp_box:
                            print "Responsible box active:",
                            print "box orig_sub_coords=", orig_sub_coords
                            print "box curr_sub_coords=", curr_sub_coords
                        if box_detected == True:
                            print "WTF, face box was already detected!!!"
                        box_detected = True
                        active_boxes[num_network] += 1
                        #Error measures for each image, sampling_value, network_number, box_number
                        #Include rel_bx_error, rel_by_error, rel_scale_error, rel_eye_error, rel_eyes_pix_error,
                        fr_performance[im_number, sampling_value, num_network] = (rel_bx_error, rel_by_error, rel_scale_error, rel_eye_error, rel_eyes_pix_error)
        
                        if debug_resp_box:
                            print "original_eye_coordinates[0][0,4]=", eye_coords
                            print "fc=", fc_x, fc_y
                            print "bx_error=", bx_error
                            print "by_error=", by_error
                            print "app_eye_coords=", app_eye_coords
                            print "rel_eye_error =", rel_eye_error
                        if rel_eye_error < 0.25:
                            face_detected = True
                            print "Face was properly detected"
                            true_positives[num_network] += 1
                        else:
                            print "Face was NOT properly detected (but patch still active), rel_eye_error=", rel_eye_error
                            false_positives[num_network] += 1
                        
                        if display_aface and display_plots:
                            #Face Box
                            if subplot != None:
                                subplot.plot([ab_x0, ab_x1, ab_x1, ab_x0, ab_x0], [ab_y0, ab_y0, ab_y1, ab_y1, ab_y0], "g")
                                #Left eye, right eye and face center
                                subplot.plot([ael_x, aer_x, afc_x], [ael_y, aer_y, afc_y], "go")
                    else:
    #                    if num_network==0:
    #                        print "%f < %f? and %f < %f? and %f < %f/%f=%f < %f ?"%(bx_error_orig, max_Dx_diff, by_error_orig, max_Dy_diff, \
    #                                                                            max_scale_radio, box_side, box_side_orig, box_side /box_side_orig, min_scale_radio)
        ##                pass
        ##                print "box false positive:",
        ##                print orig_sub_coords
        ##                print "box moved to:", curr_sub_coords
        ##                print "face was originally centered in", frgc_original_coordinates[0]
                        false_positives[num_network] += 1
            #      C) If yes count as positive detection, otherwise failed detection and compute FRR     
            #      D) For remaining boxes, all of them are false detections, use them compared to total number of boxes for FAR
                if num_network==0:
                    pass
                    #print "%f < %f/%f=%f < %f?"%(max_scale_radio, box_side, box_side_orig, box_side /box_side_orig, min_scale_radio)
                if face_detected:
                    pass
                    #print "Face was correctly detected at least once"
                elif box_side / box_side_orig > min_scale_radio and box_side / box_side_orig < max_scale_radio:
                    #print "Face was not detected at all"
                    false_negatives[num_network] += 1
                    if not box_detected: 
                        offending_images[num_network].append(im_number)
                else:
                    pass #warning!
#                    print "No face present"
            benchmark.add_task_from_previous_time("Displayed ground truth data (if enabled)")
        #Warning: these are not the true patches that were processed. These are freshly loaded patches
        if save_patches:
            tmp_subimages = extract_subimages_rotate(images, curr_image_indices, curr_subimage_coordinates, -1*curr_angles, (subimage_width, subimage_height) )
            for i in range(len(tmp_subimages)):
                print "saving patch"
                tmp_subimages[i].save("saved_patches/patch_im%+03d_PAngle%f.jpg"%(i,curr_angles[i]))
            benchmark.add_task_from_previous_time("Saving centered face patches")
        
         
        if save_normalized_face_detections:
            print "Saving normalized face detections"
            #Parameters used to emulate face normalization (for pose, not age!)
            normalization_method = "eyes_inferred-mouth_area"
            centering_mode="mid_eyes_inferred-mouth"
            rotation_mode="EyeLineRotation"
            normalized_face_detections_dir = "normalized_face_detections/"
            prefix = "EyeN"
            num_tries = 1        
            allow_random_background = False
            out_size = (256,192)
            convert_format="L"
            integer_rotation_center = True
    
            for i, box_coords in enumerate(curr_subimage_coordinates):
                eyes_coords_orig_app, _, _ = compute_approximate_eye_boxes_coordinates(box_coords, face_sampling=0.825, eye_sampling = 2.3719, rot_angle=curr_angles[i])
            
                float_coords = [eyes_coords_orig_app[0], eyes_coords_orig_app[1], eyes_coords_orig_app[2], eyes_coords_orig_app[3], 0.0, 0.0]
                im2 = face_normalization_tools.normalize_image(None, float_coords, normalization_method=normalization_method, centering_mode=centering_mode, 
                        rotation_mode=rotation_mode, integer_rotation_center=integer_rotation_center, out_size = out_size, convert_format=convert_format, 
                        verbose=False, allow_random_background=allow_random_background, image=images[0])    
                random_number = numpy.random.randint(1000000)
                im2.save(normalized_face_detections_dir + prefix + "%06d.jpg"%(random_number))
            benchmark.add_task_from_previous_time("Saving normalized face detections")
        
        
        

#TODO: POSITION OF EYES SHOULD NOW BE ROTATED ACCORDING TO FACE ANGLE IN PLANE
        eyes_coords_orig = numpy.zeros((len(curr_subimage_coordinates), 4))
        eyesL_box_orig = numpy.zeros((len(curr_subimage_coordinates), 4))
        eyesR_box_orig = numpy.zeros((len(curr_subimage_coordinates), 4))        
        for i, box_coords in enumerate(curr_subimage_coordinates):

            #WARNING: PROBABLY THE SCALE IS INCORRECT! THE METHOD IS NOT ROBUST TO OFFSETS XY even when angles are zero
#            detected_faces.append(compute_approximate_eye_coordinates(eye_coords, face_sampling=0.825))
            eyes_coords_orig[i], eyesL_box_orig[i], eyesR_box_orig[i] = compute_approximate_eye_boxes_coordinates(box_coords, face_sampling=0.825, eye_sampling = 2.3719, rot_angle=curr_angles[i]) 
            if False:
                offset_x = -2.0 #FOR DEBUGING ONLY!!!
                offset_y = -2.0
                offset_a = 0.0 #degrees
                eyes_coords_orig[i][0] += offset_x #eyeL
                eyes_coords_orig[i][2] += offset_x #eyeR
                eyesL_box_orig[i][0] += offset_x
                eyesL_box_orig[i][2] += offset_x
                eyesR_box_orig[i][0] += offset_x
                eyesR_box_orig[i][2] += offset_x
                eyes_coords_orig[i][1] += offset_y #eyeL
                eyes_coords_orig[i][3] += offset_y #eyeR
                eyesL_box_orig[i][1] += offset_y
                eyesL_box_orig[i][3] += offset_y
                eyesR_box_orig[i][1] += offset_y
                eyesR_box_orig[i][3] += offset_y
                curr_angles[i] = 0.0 + offset_a

            display_eye_boxes = True
            if display_eye_boxes and display_plots and subplot != None:
                #left eye box
                bel_x0, bel_y0, bel_x1, bel_y1 =  eyesL_box_orig[i]
                subplot.plot([bel_x0, bel_x1, bel_x1, bel_x0, bel_x0], [bel_y0, bel_y0, bel_y1, bel_y1, bel_y0], "b", linewidth=1.5)
                #right eye box
                ber_x0, ber_y0, ber_x1, ber_y1 =  eyesR_box_orig[i]
                subplot.plot([ber_x0, ber_x1, ber_x1, ber_x0, ber_x0], [ber_y0, ber_y0, ber_y1, ber_y1, ber_y0], "b")
                el_x, el_y, er_x, er_y = eyes_coords_orig[i]
                subplot.plot([el_x, er_x], [el_y, er_y], "or")

        benchmark.add_task_from_previous_time("Approximated eye coordinates from face coordinates and plotted eye boxes (if enabled)")
        #Later the boxes are updated, and this coordinates change. TODO: UNDERSTAND NEXT TWO LINES, NEEDED?
        #eyesL_coords = (eyesL_box_orig[:,0:2]+eyesL_box_orig[:,2:4])/2.0
        #eyesR_coords = (eyesR_box_orig[:,0:2]+eyesR_box_orig[:,2:4])/2.0
        
        ####curr_angles = curr_angles * 0.0 #DEBUGG!!!
        
        eye_xy_too_far_images = numpy.zeros(len(curr_image_indices), dtype=bool) 
        #Left eye only!
        eyesL_box = eyesL_box_orig + 0.0 #FOR DEBUG ONLY!!!
        #print "eyesL_box=",eyesL_box
        eyeL_subimages = extract_subimages_rotate(images, curr_image_indices, eyesL_box, -1*curr_angles, (eye_subimage_width, eye_subimage_height), interpolation_format)

        benchmark.add_task_from_previous_time("Extracted eye patches (L or R)")

        debug = True and False
        if debug:
            for i in range(len(eyeL_subimages)):
                print "saving eyeL patch"
                eyeL_subimages[i].save("saved_patches/eyeL_patch_im%+03d_PAngle%03.3f.jpg"%(i,curr_angles[i]))       

        benchmark.add_task_from_previous_time("Saved eye patches (if enabled)")

        if len(eyeL_subimages) > 0:
            for num_network in [num_networks-5, num_networks-4]:
                subimages_arr = images_asarray(eyeL_subimages)+0.0
                sl = networks[num_network].execute(subimages_arr, benchmark=None)    
                benchmark.add_task_from_previous_time("Extracted eye features (L or R)")

                #print "Network %d processed all subimages"%(num_networks-2)
                reg_num_signals = classifiers[num_network].input_dim
                avg_labels = classifiers[num_network].avg_labels 
                print "avg_labels=",avg_labels
                reg_out = classifiers[num_network].regression(sl[:,0:reg_num_signals], avg_labels)
                print "reg_out_crude:", reg_out
                eye_xy_too_far_images |= numpy.abs(reg_out) >= 9.0 #9.0 

                benchmark.add_task_from_previous_time("Regression eye position (L or R)")

                if network_types[num_network] == "EyeLX": #POS_X     
                    #print "EyeLX"  
                    eyes_box_width = numpy.abs(eyesL_box[:,2]-eyesL_box[:,0])
                    reg_out_x = (reg_out / 2.3719) * eyes_box_width / eye_regression_width 
                    ###reg_out_x = reg_out * eyes_box_width / eye_regression_width / 0.94875
                elif network_types[num_network] == "EyeLY": #POS_Y       
                    #print "EyeLY"  
                    eyes_box_height = numpy.abs(eyesL_box[:,3]-eyesL_box[:,1])
                    reg_out_y = (reg_out / 2.3719) * eyes_box_width / eye_regression_width 
                    ###reg_out_y = reg_out * eyes_box_height / eye_regression_height / 0.94875
                else:
                    print "Unknown network type! (expecting either EyeLX or EyeLY)", network_types[num_network]
                    quit()       

            rot_angle_radian = -1 * curr_angles * numpy.pi/180
            eyesL_dx = reg_out_x * numpy.cos(rot_angle_radian) - reg_out_y * numpy.sin(rot_angle_radian) 
            eyesL_dy = reg_out_y * numpy.cos(rot_angle_radian) + reg_out_x * numpy.sin(rot_angle_radian) 

            eyesL_box[:, 0] = eyesL_box[:, 0] - eyesL_dx  #X0
            eyesL_box[:, 2] = eyesL_box[:, 2] - eyesL_dx #X1                    
            ###print "left eye, X reg_out final=", reg_out
            eyesL_box[:, 1] = eyesL_box[:, 1] - eyesL_dy  #Y0
            eyesL_box[:, 3] = eyesL_box[:, 3] - eyesL_dy  #Y1
            ###print "left eye, Y reg_out final=", reg_out
            benchmark.add_task_from_previous_time("Adjusted eye coordinates accounting for rotation angle (L or R)")

                    
            print "LE found *********************************************************************"
        
        #Right eye only!
        #Swap horizontal coordinates
        eyesRhack_box = eyesR_box_orig + 0.0 #### FOR DEBUG ONLY
        eyesRhack_box[:,0] = eyesR_box_orig[:,2]
        eyesRhack_box[:,2] = eyesR_box_orig[:,0]
        
        #print "eyesRhack_box=",eyesRhack_box
        eyeR_subimages = extract_subimages_rotate(images, curr_image_indices, eyesRhack_box, -1*curr_angles, (eye_subimage_width, eye_subimage_height), interpolation_format)
        benchmark.add_task_from_previous_time("Extracted eye patches (L or R)")

        debug = True and False
        if debug:
            for i in range(len(eyeR_subimages)):
                print "saving eyeR patch"
                eyeR_subimages[i].save("saved_patches/eyeRRpatch_im%+03d_PAngle%03.3f.jpg"%(i,curr_angles[i])) 
        benchmark.add_task_from_previous_time("Saved eye patches (if enabled)")


        if len(eyeR_subimages) > 0:
            for num_network in [num_networks-5, num_networks-4]:
            #eyeR_subimages = extract_subimages_basic(images, curr_image_indices, eyesRhack_box, (eye_subimage_width, eye_subimage_height) )
                subimages_arr = images_asarray(eyeR_subimages)+0.0
                sl = networks[num_network].execute(subimages_arr, benchmark=None)    
                benchmark.add_task_from_previous_time("Extracted eye features (L or R)")

                #print "Network %d processed all subimages"%(num_networks-2)
                reg_num_signals = classifiers[num_network].input_dim
                avg_labels = classifiers[num_network].avg_labels 
                reg_out = classifiers[num_network].regression(sl[:,0:reg_num_signals], avg_labels)
                print "reg_out_crude:", reg_out
                benchmark.add_task_from_previous_time("Regression eye position (L or R)")
                
                eye_xy_too_far_images |= numpy.abs(reg_out) >= 9.0 #9.0 
                if network_types[num_network] == "EyeLX": #POS_X     
                    #print "EyeLX"  
                    eyes_box_width = numpy.abs(eyesRhack_box[:,2]-eyesRhack_box[:,0])
                    reg_out_x = (reg_out / 2.3719) * eyes_box_width / eye_regression_width  
                    ### reg_out_x = reg_out * eyes_box_width / eye_regression_width * 0.94875 #PREVIOUS WORKING VERSION 
                elif network_types[num_network] == "EyeLY": #POS_Y       
                    #print "EyeLY"  
                    eyes_box_height = numpy.abs(eyesRhack_box[:,3]-eyesRhack_box[:,1])
                    reg_out_y = (reg_out / 2.3719) * eyes_box_width / eye_regression_width
                    ### reg_out_y = reg_out * eyes_box_height / eye_regression_height * 0.94875
                    print "right eye, Y reg_out final=", reg_out
                else:
                    print "Unknown network type!", network_types[num_network]
                    quit()
            print "RE found *********************************************************************"

            rot_angle_radian = curr_angles * numpy.pi/180
            eyesR_dx = reg_out_x * numpy.cos(rot_angle_radian) - reg_out_y * numpy.sin(rot_angle_radian) 
            eyesR_dy = reg_out_y * numpy.cos(rot_angle_radian) + reg_out_x * numpy.sin(rot_angle_radian) 
            
            eyesRhack_box[:, 0] = eyesRhack_box[:, 0] + eyesR_dx  #X0
            eyesRhack_box[:, 2] = eyesRhack_box[:, 2] + eyesR_dx  #X1
            print "right eye, X reg_out final=", reg_out
            eyesRhack_box[:, 1] = eyesRhack_box[:, 1] - eyesR_dy  #Y0
            eyesRhack_box[:, 3] = eyesRhack_box[:, 3] - eyesR_dy  #Y1

            benchmark.add_task_from_previous_time("Adjusted eye coordinates accounting for rotation angle (L or R)")

        #Undo horizontal swap of coordinates
        eyesR_box = eyesRhack_box + 0.0
        eyesR_box[:,0] = eyesRhack_box[:,2]
        eyesR_box[:,2] = eyesRhack_box[:,0]

        #Update both eye coordinates (first one was already updated anyway)
        eyesL_coords = (eyesL_box[:,0:2]+eyesL_box[:,2:4])/2.0
        eyesR_coords = (eyesR_box[:,0:2]+eyesR_box[:,2:4])/2.0



        discard_too_far_images = False or True
        if discard_too_far_images:
            print "Number of images discarded due to 'eye_xy_too_far':",  eye_xy_too_far_images.sum()
            eyesL_coords= eyesL_coords[eye_xy_too_far_images==0]
            eyesR_coords= eyesR_coords[eye_xy_too_far_images==0]
            curr_subimage_coordinates = curr_subimage_coordinates[eye_xy_too_far_images==0,:]

        benchmark.update_start_time()
        
        #This actually displays the found eyes
        display_eye_boxes = True            
        if display_eye_boxes and display_plots and subplot!=None:
            for el_x, el_y in eyesL_coords:
                subplot.plot([el_x], [el_y], "bo")
            for er_x, er_y in eyesR_coords:
                subplot.plot([er_x], [er_y], "yo")

#                    if display_aface and display_plots:
#                        #Face Box
#                        if subplot != None:
#                            subplot.plot([ab_x0, ab_x1, ab_x1, ab_x0, ab_x0], [ab_y0, ab_y0, ab_y1, ab_y1, ab_y0], "w")
#                            #Left eye, right eye and face center
#                            subplot.plot([ael_x, aer_x, afc_x], [ael_y, aer_y, afc_y], "wo")
        benchmark.add_task_from_previous_time("Displayed all found and not discarded eyes")
              
        for j, box_coords in enumerate(curr_subimage_coordinates):            
            eyes_coords = eyes_coords_orig[j]
            box_eyes_coords_confidence = numpy.array([box_coords[0], box_coords[1], box_coords[2], box_coords[3], eyesL_coords[j][0], eyesL_coords[j][1], eyesR_coords[j][0], eyesR_coords[j][1], curr_confidence[j]])                       
            detected_faces_eyes_confidences.append(box_eyes_coords_confidence)
            
        #Performance computation
        num_network = num_networks-3
        num_boxes[num_network] += len(curr_subimage_coordinates) 

        #Compute error after eye networks
        if coordinates_filename and len(curr_subimage_coordinates)>0:
            sampled_face_coordinates = database_current_image_coordinates
            #Redundant
            (el_x, el_y, er_x, er_y, n_x, n_y, m_x, m_y, fc_x, fc_y, b_x0, b_y0, b_x1, b_y1) = sampled_face_coordinates
            eye_coords = numpy.array([el_x, el_y, er_x, er_y])

            box_detected = False
            face_detected = False
            for j in range(len(curr_subimage_coordinates)):
        ##            print "********************"
                orig_sub_coords = orig_subimage_coordinates[curr_orig_index[j]]
                (ab_x0, ab_y0, ab_x1, ab_y1) = curr_sub_coords = curr_subimage_coordinates[j]
                afc_x = (ab_x0 + ab_x1)/2.0
                afc_y = (ab_y0 + ab_y1)/2.0
            
                bcenter_x_orig = (orig_sub_coords[0]+orig_sub_coords[2])/2.0
                bcenter_y_orig = (orig_sub_coords[1]+orig_sub_coords[3])/2.0
            
                bcenter_x = (ab_x0 + ab_x1)/2.0
                bcenter_y = (ab_y0 + ab_y1)/2.0
    
                           
                box_side =  numpy.abs(b_x1-b_x0) #side of the real face_sampled face box, equals 0.825
                abox_side =  numpy.abs(ab_x1-ab_x0)
                box_side_orig =numpy.abs(orig_sub_coords[2]-orig_sub_coords[0])
    
                #Errors in image pixels
                bx_error_orig = fc_x - bcenter_x_orig 
                by_error_orig = fc_y - bcenter_y_orig
                bx_error = fc_x - bcenter_x 
                by_error = fc_y - bcenter_y
                #Errors in regression image pixels
                rel_bx_error = (bx_error / box_side) * regression_width
                rel_by_error = (by_error / box_side) * regression_height      
            
                scale_error = box_side / abox_side - 1.0
                #Error with respect to the goal sampling value of 0.825
                rel_scale_error = scale_error #* 0.825 ###WARNING!!!        
                # rel_scale_error = 0.825 / box_side * abox_side - 0.825
                               
                debug_resp_box=False
                #Face is within this box? (if yes, contabilize true error)
                if numpy.abs(bx_error_orig)<max_Dx_diff and numpy.abs(by_error_orig)<max_Dy_diff and \
                box_side / box_side_orig > min_scale_radio and box_side / box_side_orig < max_scale_radio:
                    #Bingo, this is responsible of detecting the face
                    if debug_resp_box:
                        print "Responsible box active:",
                        print "box orig_sub_coords=", orig_sub_coords
                        print "box curr_sub_coords=", curr_sub_coords
                    if box_detected == True:
                        print "WTF, face box was already detected!!!"
                    box_detected = True
                    active_boxes[num_network] += 1
                    #Error measures for each image, sampling_value, network_number, box_number
                    #Include rel_bx_error, rel_by_error, rel_scale_error, rel_eye_error, rel_eyes_pix_error
                    #(ael_x, ael_y, aer_x, aer_y) = sampled_app_eye_coords = compute_approximate_eye_coordinates(curr_sub_coords, face_sampling=0.825)
                    app_eye_coords = numpy.array([eyesL_coords[j][0], eyesL_coords[j][1], eyesR_coords[j][0], eyesR_coords[j][1]])                    
                    #Error in image pixels
                    rel_eyes_pix_error = (app_eye_coords - eye_coords) / box_side * regression_width
                    #Normalized eye error, goal is a relative error < 0.25
                    rel_eye_error = relative_error_detection(app_eye_coords, eye_coords)

                    fr_performance[im_number, sampling_value, num_network] = (rel_bx_error, rel_by_error, rel_scale_error, rel_eye_error, rel_eyes_pix_error)
                            
                    if debug_resp_box:
                        print "database_original_eye_coordinates[0][0,4]=", eye_coords
                        print "fc=", fc_x, fc_y
                        print "bx_error=", bx_error
                        print "by_error=", by_error
                        print "app_eye_coords=", app_eye_coords
                        print "rel_eye_error =", rel_eye_error
                    if rel_eye_error < 0.25:
                        face_detected = True
                        print "Face was properly detected"
                        true_positives[num_network] += 1
                    else:
                        print "Face was NOT properly detected"
                        false_positives[num_network] += 1                    
                else:
    #                    if num_network==0:
    #                        print "%f < %f? and %f < %f? and %f < %f/%f=%f < %f ?"%(bx_error_orig, max_Dx_diff, by_error_orig, max_Dy_diff, \
    #                                                                            max_scale_radio, box_side, box_side_orig, box_side /box_side_orig, min_scale_radio)
        ##                pass
        ##                print "box false positive:",
        ##                print orig_sub_coords
        ##                print "box moved to:", curr_sub_coords
        ##                print "face was originally centered in", frgc_original_coordinates[0]
                        false_positives[num_network] += 1
            #      C) If yes count as positive detection, otherwise failed detection and compute FRR     
            #      D) For remaining boxes, all of them are false detections, use them compared to total number of boxes for FAR
#                if num_network==0:
#                    pass
                    #print "%f < %f/%f=%f < %f?"%(max_scale_radio, box_side, box_side_orig, box_side /box_side_orig, min_scale_radio)
            if face_detected:
                pass
                #print "Face was correctly detected at least once"
            elif box_side / box_side_orig > min_scale_radio and box_side / box_side_orig < max_scale_radio:
                #print "Face was not detected at all"
                false_negatives[num_network] += 1
                if not box_detected: 
                    offending_images[num_network].append(im_number)
            else:
                pass #warning!

            
        benchmark.add_task_from_previous_time("Computed detection rates and precisions (if enabled)")
                   
        
#            TODO: Make sure this works when nothing was found at all inside an image

        display_face = True
        #Display Faces to find:
        if coordinates_filename:
            sampled_face_coordinates = database_current_image_coordinates
#           print "sfc=", sampled_face_coordinates
            (el_x, el_y, er_x, er_y, n_x, n_y, m_x, m_y, fc_x, fc_y, b_x0, b_y0, b_x1, b_y1) = sampled_face_coordinates
            eye_coords = numpy.array([el_x, el_y, er_x, er_y])
            if display_face and display_plots:
                #Face Box
                if subplot != None:
                    subplot.plot([b_x0, b_x1, b_x1, b_x0, b_x0], [b_y0, b_y0, b_y1, b_y1, b_y0], "r")
                    #Left eye, right eye and face center
                    subplot.plot([el_x, er_x, fc_x], [el_y, er_y, fc_y], "ro")

        benchmark.add_task_from_previous_time("Plotted true face box and positions of eyes and face center (if enabled)")
        
    #print "Faces/Eyes before purge:", detected_faces_eyes_confidence
    detected_faces_eyes_confidences_purgued = purgueDetectedFacesEyesConfidence(detected_faces_eyes_confidences)
    benchmark.add_task_from_previous_time("Purgued repeated face detections")

    print "\n%d Faces/Eyes detected after purge:"%len(detected_faces_eyes_confidences_purgued), detected_faces_eyes_confidences_purgued

    if estimate_age or estimate_race or estimate_gender:
        #Parameters used to emulate face normalization
        normalization_method = "eyes_inferred-mouth_areaZ"
        centering_mode="mid_eyes_inferred-mouth"
        rotation_mode="EyeLineRotation"
        allow_random_background = False
        out_size = (256,260)
        age_image_width = out_size[0]
        age_image_height = out_size[1]
        integer_rotation_center = True
        
        #Parameters used to emulate image loading (load_data_from_sSeq)
        age_base_scale = 1.14 
        ####age_scale_offset = 0.00 #0.08 0.04  
        
        age_obj_std_base = 0.16 # ** 0.16
    
        reduction_factor = 160.0/96 # (affects only sampling, subimage size, and translations)
        age_sampling = age_base_scale * reduction_factor
        age_pre_mirroring="none"
        age_contrast_enhance="AgeContrastEnhancement_Avg_Std"
        age_obj_avg_std=0.0
        age_obj_std=age_obj_std_base
        
        age_subimage_first_row = age_image_height/2.0 - age_subimage_height * age_sampling/2.0
        age_subimage_first_column = age_image_width/2.0 - age_subimage_width * age_sampling/2.0
        age_translations_x = 0.0 / reduction_factor
        age_translations_y = -6.0 / reduction_factor

        #1)Extract face patches (usually 96x96, centered according to the eye positions)
        num_faces_found = len(detected_faces_eyes_confidences_purgued)
        age_estimates = numpy.zeros(num_faces_found)
        age_stds = numpy.zeros(num_faces_found)
        race_estimates = 10 * numpy.ones(num_faces_found)
        gender_estimates = 10 * numpy.ones(num_faces_found)

        benchmark.add_task_from_previous_time("Prepared for age, gender, and race estimation")

        for j, box_eyes_coords_confidence in enumerate(detected_faces_eyes_confidences_purgued):            
            #A) Generate normalized image
            #contents: eyeL_x, eyeL_y, eyeR_x, eyeR_y, mouth_x, mouth_y
            benchmark.update_start_time()
            float_coords = [box_eyes_coords_confidence[4], box_eyes_coords_confidence[5], box_eyes_coords_confidence[6], box_eyes_coords_confidence[7], 0.0, 0.0]
            im2 = face_normalization_tools.normalize_image(None, float_coords, normalization_method=normalization_method, centering_mode=centering_mode, 
                    rotation_mode=rotation_mode, integer_rotation_center=integer_rotation_center, out_size = out_size, convert_format="L", 
                    verbose=False, allow_random_background=allow_random_background, image=images[0])            
            benchmark.add_task_from_previous_time("Age/race/gender: computed normalized image")

            ##print im2
            ##print type(im2)
            ##print isinstance(im2, numpy.ndarray)
            ##print isinstance(im2, Image.Image)            
            #B)Extract actual patches
            age_subimages_arr = load_image_data_monoprocessor(image_files=["nofile"], image_array=im2, image_width=age_image_width, image_height=age_image_height, 
                    subimage_width=age_subimage_width, subimage_height=age_subimage_height, pre_mirroring_flags=False, pixelsampling_x = age_sampling, pixelsampling_y = age_sampling, 
                    subimage_first_row=age_subimage_first_row, subimage_first_column=age_subimage_first_column, add_noise = False, convert_format="L", 
                    translations_x=age_translations_x, translations_y=age_translations_y, trans_sampled=True, rotation=0.0, contrast_enhance = age_contrast_enhance, obj_avgs=0.0, 
                    obj_stds=age_obj_std, background_type=None, color_background_filter=None, subimage_reference_point = 0, verbose=False)

            benchmark.add_task_from_previous_time("Age/race/gender: extracted image array")
                
            im_raw = numpy.reshape(age_subimages_arr[0], (96, 96))
            im = scipy.misc.toimage(im_raw, mode="L")
            im.save("ImageForAgeEstimation%03d.jpg"%(number_saved_image_age_estimation))
            number_saved_image_age_estimation += 1 
        #2)Apply the age estimation network
            num_network = num_networks-3
            sl = networks[num_network].execute(age_subimages_arr, benchmark=None)             
            benchmark.add_task_from_previous_time("Age/race/gender: feature extraction")

            if estimate_age:
                reg_num_signals = classifiers[num_network].input_dim
                avg_labels = classifiers[num_network].avg_labels 
                reg_out, std_out = classifiers[num_network].regression(sl[:,0:reg_num_signals], avg_labels, estimate_std=True)
                benchmark.add_task_from_previous_time("Computed age regression")

                age_estimates[j] = reg_out[0]
                age_stds[j] = std_out[0]
                print "age estimation:", reg_out[0], "+/-", std_out[0]
            if estimate_race:
                num_network = num_networks-2
                reg_num_signals = classifiers[num_network].input_dim
                avg_labels = classifiers[num_network].avg_labels
                #reg_out = classifiers[num_network].label(sl[:,0:reg_num_signals])
                reg_out = classifiers[num_network].regression(sl[:,0:reg_num_signals], avg_labels)
                benchmark.add_task_from_previous_time("Computed race classification")

                race_estimates[j] = reg_out[0]
                print "race estimation:", reg_out[0]
            if estimate_gender:
                num_network = num_networks-1
                reg_num_signals = classifiers[num_network].input_dim
                avg_labels = classifiers[num_network].avg_labels
                #reg_out = classifiers[num_network].label(sl[:,0:reg_num_signals])
                reg_out = classifiers[num_network].regression(sl[:,0:reg_num_signals], avg_labels)
                benchmark.add_task_from_previous_time("Computed gender classification")

                gender_estimates[j] = reg_out[0]
                print "gender estimation:", reg_out[0]
        #3)Interpret the results
        gender_confidences = numpy.abs(gender_estimates)
        race_confidences = numpy.abs(race_estimates)/2.0

        gender_estimates = map_real_gender_labels_to_strings(gender_estimates, long_text=True)
        race_estimates = map_real_race_labels_to_strings(race_estimates, long_text=True)
        print "Age estimates:", age_estimates
        print "Age stds: ", age_stds
        print "Race estimates:", race_estimates
        print "Race confidences:", race_confidences
        print "Gender estimates:", gender_estimates
        print "Gender confidences:", gender_confidences
    if track_single_face:
        if len(detected_faces_eyes_confidences_purgued) > 0:
            face_eyes_coords = detected_faces_eyes_confidences_purgued[0]
            tracked_face = (face_eyes_coords[0], face_eyes_coords[1], face_eyes_coords[2], face_eyes_coords[3])
            face_has_been_found = True
        else:
            face_has_been_found = False      

    benchmark.update_start_time()               
    if show_final_detection:
        f0 = plt.figure()
        plt.suptitle("Final face detections")
        ax = f0.add_subplot(111)
        ax.imshow(im_disp_rgb, aspect=1.0, interpolation='nearest', origin='upper')
        for j, face_eyes_coords in enumerate(detected_faces_eyes_confidences_purgued):
            b_x0, b_y0, b_x1, b_y1, el_x, el_y, er_x, er_y, conf = face_eyes_coords
            #color = (conf, conf, conf)
            color = (0.25,0.5,1.0)
            ax.plot([b_x0, b_x1, b_x1, b_x0, b_x0], [b_y0, b_y0, b_y1, b_y1, b_y0], color=color, linewidth=3)
            ax.plot([el_x], [el_y], "bo")
            ax.plot([er_x], [er_y], "yo")
            
            if estimate_age:
                separation_y = (b_y1-b_y0)/20
                color = (0.25,0.5,1.0)
                #ax.text(b_x0, b_y0-separation_y, '%2.1f years +/- %2.1f, \n%s (%.1f %%)\n%s (%.1f %%)'%(age_estimates[j], age_stds[j], race_estimates[j], \
                #        race_confidences[j]*100, gender_estimates[j], gender_confidences[j]*100), verticalalignment='bottom', horizontalalignment='left', color=color, fontsize=12)
                ax.text(b_x0 + separation_y*0.5, b_y0 - separation_y, '%2.0f years\n%s\n%s' % (age_estimates[j], race_estimates[j], gender_estimates[j]),
                        verticalalignment='bottom', horizontalalignment='left', color=color, fontsize=12)
    benchmark.add_task_from_previous_time("Final display: eyes, face box, and age, gender, race labels (if enabled)")

    if pygame_display:
        for j, face_eyes_coords in enumerate(detected_faces_eyes_confidences_purgued):
            b_x0, b_y0, b_x1, b_y1, el_x, el_y, er_x, er_y, conf = map(int, map(round, face_eyes_coords))
            
            #plt.plot([b_x0, b_x1, b_x1, b_x0, b_x0], [b_y0, b_y0, b_y1, b_y1, b_y0], color=color, linewidth=2)
            pygame.draw.rect(screen, (255, 255, 255), (b_x0,b_y0,b_x1-b_x0,b_y1-b_y0), 2)
            #plt.plot([el_x], [el_y], "bo")
            pygame.draw.circle(screen, (0, 0, 255), (el_x,el_y), 5, 0)
            #plt.plot([er_x], [er_y], "yo")
            pygame.draw.circle(screen, (255, 255, 0), (er_x,er_y), 5, 0)
            pygame.draw.circle(screen, (255, 255, 255), (0,0), 1, 0)

            if estimate_age:
                separation_y = (b_y1-b_y0)/20
                pygame_text1 = '%2.1f years +/- %2.1f'%(age_estimates[j], age_stds[j])
                pygame_text2 = '%s'%race_estimates[j]
                pygame_text3 = '%s'%gender_estimates[j]
                text_color = (205,255,255)
                pygame_labels = [myfont.render(pygame_text1, 1, text_color), myfont.render(pygame_text2, 1, text_color), myfont.render(pygame_text3, 1, text_color)]

                label_rect1 = pygame_labels[0].get_rect()
                label_rect2 = pygame_labels[1].get_rect()
                label_rect3 = pygame_labels[2].get_rect()
                
                screen.blit(pygame_labels[0], (b_x0, b_y0-separation_y*0 - label_rect1.height - label_rect2.height - label_rect3.height))
                screen.blit(pygame_labels[1], (b_x0, b_y0-separation_y*0 - label_rect2.height - label_rect3.height))
                screen.blit(pygame_labels[2], (b_x0, b_y0-separation_y*0 - label_rect3.height))
                ####screen.draw.text(pygame_text, color = (255, 50, 50), bottomleft=(b_x0, b_y0-separation_y))            

        pygame.display.update()
    benchmark.add_task_from_previous_time("Final display (pygame): eyes, face box, and (TODO) age, gender, race labels (if enabled)")

                
    if write_results:
        print "writing face/eyes positions to disk. File:", output_filenames[im_number]
        fd = open(output_filenames[im_number], 'a')
        try:
            for face_eyes_coords in detected_faces_eyes_confidences_purgued:
                int_feyes = numpy.round(face_eyes_coords[0:8])
                if right_screen_eye_first: 
                    fd.write("%d, %d, %d, %d, %d, %d, %d, %d"%(int_feyes[0],int_feyes[1],int_feyes[2],int_feyes[3], int_feyes[4],int_feyes[5],int_feyes[6],int_feyes[7]))
                else:
                    fd.write("%d, %d, %d, %d, %d, %d, %d, %d"%(int_feyes[0],int_feyes[1],int_feyes[2],int_feyes[3], int_feyes[6],int_feyes[7],int_feyes[4],int_feyes[5]))                
                if write_age_gender_confidence:
                    fd.write(", -1, unknown, %f"%face_eyes_coords[8])
                fd.write(" \n")
        except:
            print "Error while trying to write to output file %s \n"%output_filenames[im_number]
        fd.close()

    benchmark.add_task_from_previous_time("Results were written to output file (if enabled)")
        
#        for (msg,t) in benchmark:
#            print msg, " ", t
#        
#        print "Total detection time: ", t_latest - t3, "patches processed:", orig_num_subimages

if display_errors:
    print "fr_performance[im_number, sampling_value, network] = (rel_bx_error, rel_by_error, rel_scale_error, rel_eye_error, rel_eyes_pix_error)"
    for (im_number, sampling_value, net_num) in fr_performance.keys():
        if net_num == 3 and im_number >= 0:
            print (im_number, sampling_value, net_num), "=>",
            print fr_performance[(im_number, sampling_value, net_num)]    
    
    for i in range(num_networks):
        print "Offending images after net %d (%s):"%(i, network_types[i]), offending_images[i]          

if coordinates_filename:
    for i in range(num_networks):
        number_of_boxes= false_positives[i] + true_positives[i] #number of boxes AFTER the network, before the network num_boxes
        num_faces = true_positives[i] + false_negatives[i]
        num_nofaces = number_of_boxes - num_faces
        print "After Network %d (%s): %d true_positives %d active_boxes %d initial boxes / %d false_positives, %d false_negatives: FAR=%f, FRR=%f"%(i, network_types[i], true_positives[i], active_boxes[i], \
            num_boxes[i], false_positives[i], false_negatives[i], FAR(false_positives[i], num_nofaces), FRR(false_negatives[i], num_faces))
else:
    if display_errors:
        for i in range(num_networks):
            print "Before Network %d: %d initial boxes "%(i, num_boxes[i])



if display_errors:
    for selected_network in range(num_networks):
#    for sampling_value in sampling_values:
        rel_bx_errors = []
        rel_by_errors = []
        rel_scale_errors = []
        rel_eye_errors = []
        rel_eyes_pix_errors = []
        
        print "************ Errors after network ", selected_network, network_types[selected_network]
        for (im_number, sampling_value, net_num) in fr_performance.keys():
            if net_num == selected_network:
                (rel_bx_error, rel_by_error, rel_scale_error, rel_eye_error, rel_eyes_pix_error) = fr_performance[(im_number, sampling_value, net_num)]
                rel_bx_errors.append(rel_bx_error)
                rel_by_errors.append(rel_by_error)
                rel_scale_errors.append(rel_scale_error)
                rel_eye_errors.append(rel_eye_error)
                rel_eyes_pix_errors.append(rel_eyes_pix_error)
        
        rel_bx_errors = numpy.array(rel_bx_errors)
        rel_by_errors = numpy.array(rel_by_errors)
        rel_scale_errors = numpy.array(rel_scale_errors)
        rel_eye_errors = numpy.array(rel_eye_errors)
        rel_eyes_pix_errors = numpy.array(rel_eyes_pix_errors)
        
        rel_bx_errors_std = rel_bx_errors.std()
        rel_bx_errors_std = rel_bx_errors.std()
        rel_by_errors_std = rel_by_errors.std()
        rel_scale_errors_std = rel_scale_errors.std()
        rel_eye_errors_std = rel_eye_errors.std()
        rel_eyes_pix_errors_std = rel_eyes_pix_errors.std(axis=0)
        
        rel_bx_errors_mean = rel_bx_errors.mean()
        rel_bx_errors_mean = rel_bx_errors.mean()
        rel_by_errors_mean = rel_by_errors.mean()
        rel_scale_errors_mean = rel_scale_errors.mean()
        rel_eye_errors_mean = rel_eye_errors.mean()
        rel_eyes_pix_errors_mean = rel_eyes_pix_errors.mean(axis=0)
        
        rel_bx_errors_rmse = numpy.sqrt((rel_bx_errors**2).mean())
        rel_bx_errors_rmse = numpy.sqrt((rel_bx_errors**2).mean())
        rel_by_errors_rmse = numpy.sqrt((rel_by_errors**2).mean())
        rel_scale_errors_rmse = numpy.sqrt((rel_scale_errors**2).mean())
        rel_eye_errors_rmse = numpy.sqrt((rel_eye_errors**2).mean())
        rel_eyes_pix_errors_rmse = numpy.sqrt((rel_eyes_pix_errors**2).mean(axis=0))        
        
        print "rel_bx_errors_mean =", rel_bx_errors_mean,
        print "rel_by_errors_mean =", rel_by_errors_mean,
        print "rel_scale_errors_mean =", rel_scale_errors_mean,
        print "rel_eye_errors_mean =", rel_eye_errors_mean
        print "rel_eyes_pix_errors_mean =", rel_eyes_pix_errors_mean
        
        print "rel_bx_errors_std =", rel_bx_errors_std,
        print "rel_by_errors_std =", rel_by_errors_std,
        print "rel_scale_errors_std =", rel_scale_errors_std,
        print "rel_eye_errors_std =", rel_eye_errors_std
        print "rel_eyes_pix_errors_std =", rel_eyes_pix_errors_std
        
        print "rel_bx_errors_rmse =", rel_bx_errors_rmse,
        print "rel_by_errors_rmse =", rel_by_errors_rmse,
        print "rel_scale_errors_rmse =", rel_scale_errors_rmse
        print "rel_eye_errors_rmse =", rel_eye_errors_rmse,
        print "rel_eyes_pix_errors_rmse =", rel_eyes_pix_errors_rmse


if plots_created or show_final_detection:
    print "Displaying one or more plots"
    plt.show()

benchmark.display()  

if pygame_display:
    while True:
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                print "e=", e
                quit()
        

print "Program successfully finished"

