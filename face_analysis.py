import string
import numpy
import scipy
from cuicuilco.image_loader import extract_subimages_rotate, images_asarray, load_image_data_monoprocessor
import face_normalization_tools


# Returns the eye coordinates in the same scale as the box, already considering correction face_sampling
# First left eye, then right eye. Notice, left_x > right_x and eye_y < center_y
# Given approximate box coordinates, corresponding to a box with some face_sampling, approximates the
# positions of the eyes according to the normalization criteria.
# face_sampling < 1 means that the face is larger inside the box
def compute_approximate_eye_coordinates(box_coordinates, face_sampling=0.825, leftscreen_on_left=True):
    x0, y0, x1, y1 = box_coordinates
    fc_x = (x0 + x1) / 2.0
    fc_y = (y0 + y1) / 2.0

    if leftscreen_on_left:
        factor = 1
    else:
        factor = -1
    # eye deltas with respect to the face center
    eye_dx = 37.0 / 2.0 * numpy.abs(x1 - x0) / 128 / face_sampling
    eye_dy = 42.0 / 2.0 * numpy.abs(y1 - y0) / 128 / face_sampling
    eye_left_x = fc_x - factor * eye_dx
    eye_right_x = fc_x + factor * eye_dx
    eye_y = fc_y - eye_dy

    return numpy.array([eye_left_x, eye_y, eye_right_x, eye_y])


# In addition to the eye coordinates, it gives two boxes containing the left and right eyes
def compute_approximate_eye_boxes_coordinates(box_coordinates, face_sampling=0.825, eye_sampling=2.3719,
                                              leftscreen_on_left=True, rot_angle=None):
    x0, y0, x1, y1 = box_coordinates
    fc_x = (x0 + x1) / 2.0
    fc_y = (y0 + y1) / 2.0

    if leftscreen_on_left:
        mirroring_factor = 1
    else:
        mirroring_factor = -1

    if rot_angle is None:
        rot_angle = 0

    # eye deltas with respect to the face center in original image coordinates
    # *0.825 ### 37.0/2.0 * numpy.abs(x1-x0) * 0.825 / 128 / face_sampling
    eye_dx = (37.0 / 2.0) * (numpy.abs(x1 - x0) / 64.0) / (2 * 0.825)
    eye_dy = (42.0 / 2.0) * (numpy.abs(y1 - y0) / 64.0) / (2 * 0.825)

    (numpy.abs(x1 - x0) / 64.0) * 64 * (2 * 0.825)

    box_width = (numpy.abs(x1 - x0) / (64.0 * 2 * 0.825)) * (64 * 2.3719 / 2)
    # 64 * numpy.abs(x1-x0) / 128 * eye_sampling / face_sampling
    box_height = box_width + 0.0
    # box_width = 64 * numpy.abs(x1-x0) / 128 * eye_sampling / face_sampling
    # 64 * numpy.abs(x1-x0) / 128 * eye_sampling / face_sampling
    # box_height = 64 * numpy.abs(y1-y0) / 128 * eye_sampling / face_sampling
    rot_angle_radians = rot_angle * numpy.pi / 180

    eye_right_dx_rotated = eye_dx * numpy.cos(rot_angle_radians) - eye_dy * numpy.sin(rot_angle_radians)
    eye_right_dy_rotated = eye_dy * numpy.cos(rot_angle_radians) + eye_dx * numpy.sin(rot_angle_radians)
    eye_left_dx_rotated = (-1 * eye_dx) * numpy.cos(rot_angle_radians) - eye_dy * numpy.sin(rot_angle_radians)
    eye_left_dy_rotated = eye_dy * numpy.cos(rot_angle_radians) + (-1 * eye_dx) * numpy.sin(rot_angle_radians)

    eye_left_x = fc_x + mirroring_factor * eye_left_dx_rotated
    eye_right_x = fc_x + mirroring_factor * eye_right_dx_rotated
    eye_left_y = fc_y - eye_left_dy_rotated
    eye_right_y = fc_y - eye_right_dy_rotated
    box_left_x0 = eye_left_x - box_width / 2.0
    box_left_x1 = eye_left_x + box_width / 2.0
    box_right_x0 = eye_right_x - box_width / 2.0
    box_right_x1 = eye_right_x + box_width / 2.0
    box_left_y0 = eye_left_y - box_height / 2.0
    box_left_y1 = eye_left_y + box_height / 2.0
    box_right_y0 = eye_right_y - box_height / 2.0
    box_right_y1 = eye_right_y + box_height / 2.0

    # [coordinates of both eyes], [left eye box], [right eye box] 
    return numpy.array([eye_left_x, eye_left_y, eye_right_x, eye_right_y]), numpy.array(
        [box_left_x0, box_left_y0, box_left_x1, box_left_y1]), numpy.array(
        [box_right_x0, box_right_y0, box_right_x1, box_right_y1])


# Face midpoint is the average of the point between the eyes and the mouth
def compute_face_midpoint(eye_left_x, eye_left_y, eye_right_x, eye_right_y, mouth_x, mouth_y):
    eye_center_x = (eye_left_x + eye_right_x) / 2.0
    eye_center_y = (eye_left_y + eye_right_y) / 2.0
    midpoint_x = (eye_center_x + mouth_x) / 2.0
    midpoint_y = (eye_center_y + mouth_y) / 2.0
    return midpoint_x, midpoint_y


# Error in the (Euclidean) distance relative to the distance between the eyes
def relative_error_detection(app_eye_coords, eye_coords):
    dist_left = eye_coords[0:2] - app_eye_coords[0:2]  # left eye
    dist_left = numpy.sqrt((dist_left ** 2).sum())
    dist_right = eye_coords[2:4] - app_eye_coords[2:4]  # right eye
    dist_right = numpy.sqrt((dist_right ** 2).sum())
    dist_eyes = eye_coords[0:2] - eye_coords[2:4]
    dist_eyes = numpy.sqrt((dist_eyes ** 2).sum())
    return max(dist_left, dist_right) / dist_eyes


def face_detected(app_eye_coords, eye_coords, factor=0.25):
    rel_error = relative_error_detection(app_eye_coords, eye_coords)
    if rel_error < factor:
        return True
    else:
        return False


def false_acceptance_rate(faces_wrongly_detected, total_nofaces):
    return faces_wrongly_detected * 1.0 / total_nofaces


def false_rejection_rate(faces_wrongly_rejected, total_faces):
    return faces_wrongly_rejected * 1.0 / total_faces


# TODO:USE SCORES TO SEE WHICH DETECTIONS ARE RETAINED AND WHICH ELIMINATED
# Detection confidence: 0.0=most likely a detection, 1.0=unlikely a detection
def purgue_detected_faces_eyes_confidence(detected_faces_eyes_confidences, weight_confidences_by_area=True):
    # detected_faces_eyes might also contain confidence values or other information
    detected_faces_eyes_confidences = numpy.array(detected_faces_eyes_confidences)

    if len(detected_faces_eyes_confidences) > 1:
        detection_confidences = detected_faces_eyes_confidences[:, -1]
        if weight_confidences_by_area:
            # Eye distance is used to compute an "area"
            detection_areas = ((detected_faces_eyes_confidences[:, 6] - detected_faces_eyes_confidences[:, 4]) ** 2 +
                               (detected_faces_eyes_confidences[:, 7] - detected_faces_eyes_confidences[:, 5]) ** 2) ** 0.5
            weighted_confidences = (1.0 - detection_confidences) * detection_areas
            weighted_confidences = weighted_confidences / weighted_confidences.max()
        else:
            weighted_confidences = detection_confidences.copy()
        ordering = numpy.argsort(weighted_confidences)[::-1]
        # print "original confidences =", detected_faces_eyes_confidences[:,-1]
        detected_faces_eyes_confidences = detected_faces_eyes_confidences[ordering, :]

        # print "ordering=", ordering
        # print "ordered confidences =", detected_faces_eyes_confidences[:,-1]
        print "sorted detected_faces_eyes_confidences", detected_faces_eyes_confidences[:, -1]
        print "sorted weighted confidences:", weighted_confidences[ordering]

        unique_faces_eyes_confidences = []
        unique_faces_eyes_confidences.append(detected_faces_eyes_confidences[0])
        for face_eye_coords in detected_faces_eyes_confidences:
            min_d = 10000
            for face_eye_coords2 in unique_faces_eyes_confidences:
                error = relative_error_detection(face_eye_coords[4:8], face_eye_coords2[4:8])
                if error < min_d:
                    min_d = error
            if min_d > 0.25:  # 25: #entries are different enough
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
    #        print "Each line in the batch file should be an input image_filename followed with another line containing
    # the corresponding output_filename"
    #        exit(0)

    image_filenames = []
    output_filenames = []

    for i in range(len(lines) / 2):
        image_filename = lines[2 * i].rstrip()
        output_filename = lines[2 * i + 1].rstrip()
        image_filenames.append(image_filename)
        output_filenames.append(output_filename)
    return image_filenames, output_filenames


def load_true_coordinates(base_dir, true_coordinates_file):
    # true_coordinates_file is a "normalization" file, with the following
    # FILE STRUCTURE. For each image: filename \n le_x le_y re_x re_y m_x m_y
    normalization_file = open(true_coordinates_file, "r")
    count = 0
    working = 1
    max_count = 200000
    image_filenames = []
    coordinates_dir = {}
    while working == 1 and count < max_count:
        filename = normalization_file.readline().rstrip()
        if filename == "":
            working = 0
        else:
            coords_str = normalization_file.readline()
            coords = string.split(coords_str, sep=" ")
            float_coords = map(float, coords)

            if len(float_coords) == 8:
                (LeftEyeCenter_x, LeftEyeCenter_y, RightEyeCenter_x, RightEyeCenter_y, Nose_x,
                 Nose_y, Mouth_x, Mouth_y) = float_coords
            else:  # Should be 6
                LeftEyeCenter_x, LeftEyeCenter_y, RightEyeCenter_x, RightEyeCenter_y, Nose_x, Nose_y = float_coords
                # Approximating mouth position as if face were perfectly vertical.
                Mouth_x = (LeftEyeCenter_x + RightEyeCenter_x) / 2
                Mouth_y = (LeftEyeCenter_y + RightEyeCenter_y) / 2 + (RightEyeCenter_x - LeftEyeCenter_x) * 42.0 / 37.0

            eyes_x_m = (RightEyeCenter_x + LeftEyeCenter_x) / 2.0
            eyes_y_m = (RightEyeCenter_y + LeftEyeCenter_y) / 2.0
            midpoint_eyes_mouth_x = (eyes_x_m + Mouth_x) / 2.0
            midpoint_eyes_mouth_y = (eyes_y_m + Mouth_y) / 2.0
            dist_eyes = numpy.sqrt(
                (LeftEyeCenter_x - RightEyeCenter_x) ** 2 + (LeftEyeCenter_y - RightEyeCenter_y) ** 2)

            height_triangle = numpy.sqrt((eyes_x_m - Mouth_x) ** 2 + (eyes_y_m - Mouth_y) ** 2)

            current_area = dist_eyes * height_triangle / 2.0
            desired_area = (37.0 / 0.825) * (42.0 / 0.825) / 2.0

            # if normalization_method == "mid_eyes_mouth":
            scale_factor = numpy.sqrt(current_area / desired_area)
            # WARNING!!!
            ori_width = 128 * scale_factor  # / 0.825 #regression_width*scale_factor * 0.825
            ori_height = 128 * scale_factor  # / 0.825 #regression_height*scale_factor * 0.825

            box_x0 = midpoint_eyes_mouth_x - ori_width / 2.0
            box_x1 = midpoint_eyes_mouth_x + ori_width / 2.0
            box_y0 = midpoint_eyes_mouth_y - ori_height / 2.0
            box_y1 = midpoint_eyes_mouth_y + ori_height / 2.0

            # 8 coordinates +  6 coordinates
            all_coordinates = (LeftEyeCenter_x, LeftEyeCenter_y, RightEyeCenter_x, RightEyeCenter_y, Nose_x, Nose_y,
                               Mouth_x, Mouth_y, midpoint_eyes_mouth_x, midpoint_eyes_mouth_y, box_x0, box_y0,
                               box_x1, box_y1)
            if (max_count is not None) and (count > max_count):
                break

            if (base_dir is not None) and (base_dir != ""):
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
    # print "XXXX ", subimages_array[0].mean(), subimages_array[0].std(),
    # subimages_array[0].min(), subimages_array[0].max()
    subimages_array -= subimages_array.mean(axis=1).reshape((-1, 1))
    subimages_array /= (subimages_array.std(axis=1).reshape(
        (-1, 1)) / std) + 0.00000001  # std ends up being multiplied, division over zero is avoided
    # print "min =", subimages_array.min(axis=1).mean()
    subimages_array += mean
    # print "after mean addition: min =", subimages_array.min(axis=1).mean()
    numpy.clip(subimages_array, 0.0, 255.0, subimages_array)
    # print "after clip: min =", subimages_array.min(axis=1).mean()
    # print "XXXX ", subimages_array[0].mean(), subimages_array[0].std(),
    # subimages_array[0].min(), subimages_array[0].max()


def map_real_gender_labels_to_strings(gender_label_array, long_text=True):
    strings = []
    # label = -1 => Male, label = 1 => Female
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
            er = "Unrecognized label: " + str(label)
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
            er = "Unrecognized label: " + str(label)
            raise Exception(er)
    return strings


def load_networks_from_pipeline(pipeline_filename, cache_obj, networks_base_dir, classifiers_base_dir, verbose_pipeline=True, verbose_networks=True):    
    pipeline_file = open(pipeline_filename, "rb")
    
    num_networks = int(pipeline_file.readline())
    if verbose_pipeline:
        print "Pipeline contains %d network/classifier pairs" % num_networks
    # num_networks = 1
    
    # The first line describes the face detection networks
    tmp_string = pipeline_file.readline()
    tmp_strings = string.split(tmp_string, " ")
    net_Dx = int(tmp_strings[0])
    net_Dy = int(tmp_strings[1])
    net_Dang = float(tmp_strings[2])
    net_mins = float(tmp_strings[3])
    net_maxs = float(tmp_strings[4])
    
    # Now read data for eye networks
    # This is the scale in which the image patches are generated from the input image (usually 64x64)
    # Pixel functions use this scale
    subimage_width = int(tmp_strings[5])
    subimage_height = int(tmp_strings[6])
    
    # This is the scale in which the labels are given (usually 128x128)
    # Functions related to regression/classification use this scale
    regression_width = int(tmp_strings[7])
    regression_height = int(tmp_strings[8])
    
    # The second line describes the eye detection networks
    tmp_string = pipeline_file.readline()
    tmp_strings = string.split(tmp_string, " ")
    eye_Dx = int(tmp_strings[0])
    eye_Dy = int(tmp_strings[1])
    eye_mins = float(tmp_strings[2])
    eye_maxs = float(tmp_strings[3])
    
    # This is the scale in which the image patches are generated from the input image (usually 32x32 or 64x64)
    # Pixel functions use this scale
    eye_subimage_width = int(tmp_strings[4])
    eye_subimage_height = int(tmp_strings[5])
    
    # This is the scale in which the labels are given (usually 128x128 or 64x64)
    # Functions related to regression/classification use this scale
    eye_regression_width = int(tmp_strings[6])
    eye_regression_height = int(tmp_strings[7])
    
    # regression_width = regression_height = 128 #Regression data assumes subimage has this size
    
    # The third line describes the age estimation network
    tmp_string = pipeline_file.readline()
    tmp_strings = string.split(tmp_string, " ")
    age_Dx = int(tmp_strings[0])
    age_Dy = int(tmp_strings[1])
    age_mins = float(tmp_strings[2])
    age_maxs = float(tmp_strings[3])
    age_subimage_width = int(tmp_strings[4])  # Size of image patches
    age_subimage_height = int(tmp_strings[5])
    age_regression_width = int(tmp_strings[6])  # Original size of image patches, with original scale of labels
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
    
    network_types += ["None"] * (18 - len(network_types))
    if verbose_networks:
        print "network types:", network_types
        print "networks:", network_filenames
        print "classifiers:", classifier_filenames
    
    networks = []
    for network_filename in network_filenames:
        # load_obj_from_cache(self, hash_value=None, base_dir = None, base_filename=None, verbose=True)
        # [flow, layers, benchmark, Network]
        print "loading network or flow:", network_filename, "...",
        if network_filename != "None0":
            all_data = cache_obj.load_obj_from_cache(None, base_dir=networks_base_dir, base_filename=network_filename,
                                                     verbose=True)
    
            # for layer_node in all_data.flow:
            #    if isinstance(layer_node, mdp.hinet.Layer):
            #        for node in layer_node.nodes:
            #            if isinstance(node, mdp.nodes.IEVMLRecNode):
            #                print "deleting unnecessary data"
            #                if "cov_mtx" in node.sfa_node.__dict__:
            #                    del node.sfa_node.cov_mtx
            #                    del node.sfa_node.dcov_mtx
            # cache_obj.update_cache(all_data, None, networks_base_dir, network_filename+"OUT", overwrite=True,
            # use_hash=None, verbose=True)
        else:
            all_data = None
    
        if isinstance(all_data, (list, tuple)):
            print "(Network flow was in a tuple)",
            networks.append(all_data[0])  # Keep only the flows
        else:
            print "(Network flow was not in a tuple)",
            networks.append(all_data)  # It is only a flow
        print "done"
    # quit()
    
    classifiers = []
    for classifier_filename in classifier_filenames:
        # load_obj_from_cache(self, hash_value=None, base_dir = None, base_filename=None, verbose=True)
        classifier = cache_obj.load_obj_from_cache(None, base_dir=classifiers_base_dir, base_filename=classifier_filename,
                                                   verbose=True)
        classifiers.append(classifier)
    
    
    image_information_net = (net_Dx, net_Dy, net_Dang, net_mins, net_maxs, subimage_width, subimage_height, regression_width, regression_height)
    image_information_eye = (eye_Dx, eye_Dy, eye_mins, eye_maxs, eye_subimage_width, eye_subimage_height, eye_regression_width, eye_regression_height)
    image_information_age = (age_Dx, age_Dy, age_mins, age_maxs, age_subimage_width, age_subimage_height, age_regression_width, age_regression_height)
    return image_information_net, image_information_eye, image_information_age, num_networks, network_types, networks, classifiers
            




def load_ground_truth_coordinates(coordinates_filename, image_filenames):
    coordinates_file = open(coordinates_filename, "r")

    database_original_points = {}
    working = True
    while working:
        filename = coordinates_file.readline().rstrip()
        if filename == "":
            working = False
        else:
            coords_str = coordinates_file.readline()
            coords = string.split(coords_str, sep=" ")
            float_coords = map(float, coords)
            # Here person based coordinate system for the eyes
            RightEyeCenter_x, RightEyeCenter_y, LeftEyeCenter_x, LeftEyeCenter_y, Mouth_x, Mouth_y = float_coords

            n_x = 0
            n_y = 0  # we dont know much about the nose...

            eyes_x_m = (RightEyeCenter_x + LeftEyeCenter_x) / 2.0
            eyes_y_m = (RightEyeCenter_y + LeftEyeCenter_y) / 2.0

            midpoint_eyes_mouth_x = (eyes_x_m + Mouth_x) / 2.0
            midpoint_eyes_mouth_y = (eyes_y_m + Mouth_y) / 2.0

            dist_eyes = numpy.sqrt(
                (LeftEyeCenter_x - RightEyeCenter_x) ** 2 + (LeftEyeCenter_y - RightEyeCenter_y) ** 2)

            # Triangle formed by the eyes and the mouth.
            height_triangle = numpy.sqrt((eyes_x_m - Mouth_x) ** 2 + (eyes_y_m - Mouth_y) ** 2)

            # Assumes eye line is perpendicular to the line from eyes_m to mouth
            current_area = dist_eyes * height_triangle / 2.0
            desired_area = 37.0 * 42.0 / 2.0

            # if normalization_method == "mid_eyes_mouth":
            scale_factor = numpy.sqrt(current_area / desired_area)
            # Warning, is it subimage or regression???
            # regression is fine: subimage is used only for the physical sampling of the box, but its logical size
            # is given by regression
            #        ori_width = subimage_width*scale_factor
            #        ori_height = subimage_height*scale_factor
            ori_width = regression_width * scale_factor * 0.825
            ori_height = regression_height * scale_factor * 0.825

            # WARNING, using subpixel coordinates!
            #        box_x0 = int(midpoint_eyes_mouth_x-ori_width/2)
            #        box_x1 = int(midpoint_eyes_mouth_x+ori_width/2)
            #        box_y0 = int(midpoint_eyes_mouth_y-ori_height/2)
            #        box_y1 = int(midpoint_eyes_mouth_y+ori_height/2)
            box_x0 = midpoint_eyes_mouth_x - ori_width / 2
            box_x1 = midpoint_eyes_mouth_x + ori_width / 2
            box_y0 = midpoint_eyes_mouth_y - ori_height / 2
            box_y1 = midpoint_eyes_mouth_y + ori_height / 2
            #############

            coordinates = (LeftEyeCenter_x, LeftEyeCenter_y, RightEyeCenter_x,
                           RightEyeCenter_y, n_x, n_y, Mouth_x, Mouth_y)
            more_coordinates = (midpoint_eyes_mouth_x, midpoint_eyes_mouth_y, box_x0, box_y0, box_x1, box_y1)
            all_coordinates = list(coordinates) + list(more_coordinates)  # 8 coordinates +  6 coordinates

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
    
    return database_image_coordinates


def compute_sampling_values(im_width, im_height, subimage_width, subimage_height, smallest_face, net_mins, net_maxs, patch_overlap_sampling, adaptive_grid_scale, track_single_face, face_has_been_found, tracked_face):
    if face_has_been_found and track_single_face:
        b_x0 = tracked_face[0]
        b_y0 = tracked_face[1]
        b_x1 = tracked_face[2]
        b_y1 = tracked_face[3]

        face_size = 0.5 * abs(b_x1 - b_x0) + 0.5 * abs(b_y1 - b_y0)
        sampling_value = face_size * 1.0 / subimage_width  # # / net_mins #What is net_mins, the forced scaling factor?
        # why is it relevant?
        sampling_values = [sampling_value]
    elif adaptive_grid_scale:
        min_side = min(im_height, im_width)
        min_box_side = max(20, min_side * smallest_face * 0.825 / net_mins)  # * 0.825/0.55)
        # smallest face patch should be at least 20 pixels!
        min_sampling_value = min_box_side * 1.0 / subimage_width
        sampling_values = []
        sampling_value = min_sampling_value

        new_grid_step = (net_maxs / net_mins) / patch_overlap_sampling
        while (subimage_width * sampling_value * net_mins / 0.825 < im_width) and (
                        subimage_height * sampling_value * net_mins / 0.825 < im_height):
            sampling_values.append(sampling_value)
            sampling_value *= new_grid_step
        #        max_box_side = min_side * 0.825 / numpy.sqrt(2)
        #        sampling_values.append(max_box_side / regression_width)
        # sampling_values.append(min_side/subimage_width*0.98)
    else:
        min_side = min(im_height, im_width)
        min_box_side = max(20, min_side * smallest_face * 0.825 / net_mins)  # smallest_face, at least 20 pixels!
        min_sampling_value = min_box_side * 1.0 / subimage_width
        sampling_values = [min_sampling_value]  # default_sampling_values
    return sampling_values


def compute_posX_posY_values(im_width, im_height, subimage_width, subimage_height, regression_width, regression_height, sampling_value, net_Dx, net_Dy, patch_overlap_posx_posy, face_has_been_found, track_single_face, adaptive_grid_coords, verbose=False):
    if face_has_been_found and track_single_face:
        patch_width = subimage_width * sampling_value
        patch_height = subimage_height * sampling_value

        patch_sepx = net_Dx * 2.0 * patch_width / regression_width
        patch_sepy = net_Dy * 2.0 * patch_height / regression_height

        # posX_values = [tracked_face[0], tracked_face[0]+patch_sepx, tracked_face[0] - patch_sepx,
        #  tracked_face[0], tracked_face[0] ]
        # posY_values = [tracked_face[1], tracked_face[1], tracked_face[1], tracked_face[1]+patch_sepy,
        # tracked_face[1]-patch_sepy ]
        posX_values = [tracked_face[0], tracked_face[0] + patch_sepx, tracked_face[0] - patch_sepx]
        posY_values = [tracked_face[1], tracked_face[1], tracked_face[1]]
        # posX_values = [tracked_face[0]]
        # posY_values = [tracked_face[1]]

    elif adaptive_grid_coords:
        # Patch width and height in image coordinates
        # This is weird, why using regression_width here? I need logical pixels!!! => subimage_width
        patch_width = subimage_width * sampling_value  # regression_width * sampling_value
        patch_height = subimage_height * sampling_value  # regression_height * sampling_value
        # TODO: add random initialization between 0 and net_Dx * 2.0 * patch_width/regression_width, the same for Y
        # These coordinates refer to the scaled image
        if verbose:
            print "net_Dx=", net_Dx, "net_Dy=", net_Dy            
        patch_horizontal_separation = net_Dx * 2.0 * patch_width / regression_width
        patch_vertical_separation = net_Dy * 2.0 * patch_height / regression_height

        # posX_values = numpy.arange(rest_horizontal/2, im_width-(patch_width-1), patch_horizontal_separation)
        num_x_patches = numpy.ceil(
                (1 + (im_width - patch_width) / patch_horizontal_separation) * patch_overlap_posx_posy)
        posX_values = numpy.linspace(0.0, im_width - patch_width, num_x_patches)  # Experimental
        num_y_patches = numpy.ceil(
                (1 + (im_height - patch_height) / patch_vertical_separation) * patch_overlap_posx_posy)
        # posY_values = numpy.arange(rest_vertical/2, im_height-(patch_height-1), patch_vertical_separation)
        posY_values = numpy.linspace(0.0, im_height - patch_height, num_y_patches)
        # A face must be detected by a box with a center distance and scale radio
        # interest points differ from center in these values
    
    max_Dx_diff = net_Dx * patch_width / regression_width
    max_Dy_diff = net_Dy * patch_height / regression_height

    if verbose:
        print "max_Dx_diff=", max_Dx_diff, "max_Dy_diff=", max_Dy_diff
        print "posX_values=", posX_values
        print "posY_values=", posY_values
    return posX_values, posY_values, patch_width, patch_height, max_Dx_diff, max_Dy_diff


#TODO: This function may be optimized considerably
def compute_subimage_coordinates_from_posX_posY_values(posX_values, posY_values, patch_width, patch_height):
    orig_num_subimages = len(posX_values) * len(posY_values)
    orig_subimage_coordinates = numpy.zeros((orig_num_subimages, 4))

    for j, posY in enumerate(posY_values):
        for i, posX in enumerate(posX_values):
            orig_subimage_coordinates[j * len(posX_values) + i] = numpy.array(
                [posX, posY, posX + patch_width - 1, posY + patch_height - 1])
    return orig_num_subimages, orig_subimage_coordinates



def create_network_plots(plt, network_figures_together, network_types):
    if network_figures_together:
        f0 = plt.figure()
        plt.suptitle("Iterative Face Detection")
        p11 = plt.subplot(3, 6, 1)
        plt.title("Original")
        p12 = plt.subplot(3, 6, 2)
        plt.title(network_types[0])
        p13 = plt.subplot(3, 6, 3)
        plt.title(network_types[1])
        p14 = plt.subplot(3, 6, 4)
        plt.title(network_types[2])
        p15 = plt.subplot(3, 6, 5)
        plt.title(network_types[3])
        p16 = plt.subplot(3, 6, 6)
        plt.title(network_types[4])

        p21 = plt.subplot(3, 6, 7)
        plt.title(network_types[5])
        p22 = plt.subplot(3, 6, 8)
        plt.title(network_types[6])
        p23 = plt.subplot(3, 6, 9)
        plt.title(network_types[7])
        p24 = plt.subplot(3, 6, 10)
        plt.title(network_types[8])
        p25 = plt.subplot(3, 6, 11)
        plt.title(network_types[9])
        p26 = plt.subplot(3, 6, 12)
        plt.title(network_types[10])

        p31 = plt.subplot(3, 6, 13)
        plt.title(network_types[11])
        p32 = plt.subplot(3, 6, 14)
        plt.title(network_types[12])
        p33 = plt.subplot(3, 6, 15)
        plt.title(network_types[13])
        p34 = plt.subplot(3, 6, 16)
        plt.title(network_types[14])
        p35 = plt.subplot(3, 6, 17)
        plt.title(network_types[15])
        p36 = plt.subplot(3, 6, 18)
        plt.title(network_types[16])
    else:
        f0 = plt.figure()
        plt.suptitle("Iterative Face Detection 11.")
        p11 = plt.subplot(1, 1, 1)
        f0 = plt.figure()
        plt.suptitle("Iterative Face Detection 12" + network_types[0])
        p12 = plt.subplot(1, 1, 1)
        f0 = plt.figure()
        plt.suptitle("Iterative Face Detection 13" + network_types[1])
        p13 = plt.subplot(1, 1, 1)
        f0 = plt.figure()
        plt.suptitle("Iterative Face Detection 14" + network_types[2])
        p14 = plt.subplot(1, 1, 1)
        f0 = plt.figure()
        plt.suptitle("Iterative Face Detection 15" + network_types[3])
        p15 = plt.subplot(1, 1, 1)
        f0 = plt.figure()
        plt.suptitle("Iterative Face Detection 16" + network_types[4])
        p16 = plt.subplot(1, 1, 1)
        f0 = plt.figure()
        plt.suptitle("Iterative Face Detection 21" + network_types[5])
        p21 = plt.subplot(1, 1, 1)
        f0 = plt.figure()
        plt.suptitle("Iterative Face Detection 22" + network_types[6])
        p22 = plt.subplot(1, 1, 1)
        f0 = plt.figure()
        plt.suptitle("Iterative Face Detection 23" + network_types[7])
        p23 = plt.subplot(1, 1, 1)
        f0 = plt.figure()
        plt.suptitle("Iterative Face Detection 24" + network_types[8])
        p24 = plt.subplot(1, 1, 1)
        f0 = plt.figure()
        plt.suptitle("Iterative Face Detection 25" + network_types[9])
        p25 = plt.subplot(1, 1, 1)
        f0 = plt.figure()
        plt.suptitle("Iterative Face Detection 26" + network_types[10])
        p26 = plt.subplot(1, 1, 1)
        f0 = plt.figure()
        plt.suptitle("Iterative Face Detection 31" + network_types[11])
        p31 = plt.subplot(1, 1, 1)
        f0 = plt.figure()
        plt.suptitle("Iterative Face Detection 32" + network_types[12])
        p32 = plt.subplot(1, 1, 1)
        f0 = plt.figure()
        plt.suptitle("Iterative Face Detection 33" + network_types[13])
        p33 = plt.subplot(1, 1, 1)
        f0 = plt.figure()
        plt.suptitle("Iterative Face Detection 34" + network_types[14])
        p34 = plt.subplot(1, 1, 1)
        f0 = plt.figure()
        plt.suptitle("Iterative Face Detection 35" + network_types[15])
        p35 = plt.subplot(1, 1, 1)
        f0 = plt.figure()
        plt.suptitle("Iterative Face Detection 36" + network_types[16])
        p36 = plt.subplot(1, 1, 1)
        #        print "len(images)",len(images)
        #        quit()
    return p11, p12, p13, p14, p15, p16, p21, p22, p23, p24, p25, p26, p31, p32, p33, p34, p35, p36


def load_network_subimages(images, curr_image_indices, curr_subimage_coordinates, curr_angles, subimage_width, subimage_height, interpolation_format, contrast_normalize):
    # Get arrays
    print "P",

    # subimages = extract_subimages_rotate_ar_parallel(images, curr_image_indices,
    # curr_subimage_coordinates, -1*curr_angles, (subimage_width, subimage_height), interpolation_format)
    subimages = extract_subimages_rotate(images, curr_image_indices, curr_subimage_coordinates,
                                         -1 * curr_angles, (subimage_width, subimage_height),
                                         interpolation_format)
    # subimages_arr = subimages
    if len(subimages) > 0:
        subimages_arr = images_asarray(subimages)   
        # contrast_normalize = True and False 
        if contrast_normalize:
            # print "min and max image array intensities are:", subimages_arr.min(), subimages_arr.max()
            print "Orig mean=", subimages_arr.mean(), " and std=", subimages_arr.std(
                axis=1).mean(), " min=", subimages_arr.min(axis=1).mean(), " max=", subimages_arr.max(
                axis=1).mean()
            image_array_contrast_normalize_avg_std(subimages_arr, 137.5, 0.40 * 255)
            print "After contrast norm: mean=", subimages_arr.mean(), " and std=", subimages_arr.std(
                axis=1).mean(), " min=", subimages_arr.min(axis=1).mean(), " max=", subimages_arr.max(
                axis=1).mean()
            # quit()
    else:
            subimages_arr = numpy.zeros((0,0))
    return subimages_arr


def update_current_subimage_coordinates(network_type, curr_subimage_coordinates, curr_angles, reg_out, regression_width, regression_height, desired_sampling):
    if network_type == "Disc":
        pass  # WARNING!
    elif network_type == "PosX":  # POS_X
        width = curr_subimage_coordinates[:, 2] - curr_subimage_coordinates[:, 0]
        reg_out = reg_out * width / regression_width
        #        print "Regression Output scaled:", reg_out
        #        print "Correcting coordinates (X)"
        curr_subimage_coordinates[:, 0] = curr_subimage_coordinates[:, 0] - reg_out  # X0
        curr_subimage_coordinates[:, 2] = curr_subimage_coordinates[:, 2] - reg_out  # X1
    elif network_type == "PosY":  # POS_Y
        height = curr_subimage_coordinates[:, 3] - curr_subimage_coordinates[:, 1]
        reg_out = reg_out * height / regression_height
        #        print "Regression Output scaled:", reg_out
        #        print "Correcting coordinates (Y)"
        curr_subimage_coordinates[:, 1] = curr_subimage_coordinates[:, 1] - reg_out  # Y0
        curr_subimage_coordinates[:, 3] = curr_subimage_coordinates[:, 3] - reg_out  # Y1
    elif network_type == "PAng":  # PAng
        curr_angles = curr_angles + reg_out  # 0.0 #reg_out ##+ reg_out #THIS SIGN IS NOT CLEAR AT ALL!
    elif network_type == "Scale":  # SCALE
        old_width = curr_subimage_coordinates[:, 2] - curr_subimage_coordinates[:, 0]
        old_height = curr_subimage_coordinates[:, 3] - curr_subimage_coordinates[:, 1]
        x_center = (curr_subimage_coordinates[:, 2] + curr_subimage_coordinates[:, 0]) / 2.0
        y_center = (curr_subimage_coordinates[:, 3] + curr_subimage_coordinates[:, 1]) / 2.0

        width = old_width / reg_out * desired_sampling
        height = old_height / reg_out * desired_sampling
        #        print "Regression Output scaled:", reg_out
        #        print "Correcting scale (X)"
        curr_subimage_coordinates[:, 0] = x_center - width / 2.0
        curr_subimage_coordinates[:, 2] = x_center + width / 2.0
        curr_subimage_coordinates[:, 1] = y_center - height / 2.0
        curr_subimage_coordinates[:, 3] = y_center + height / 2.0
    else:
        er = "Network type unknown!!!: ", network_type
        raise Exception(er)
        
    return curr_subimage_coordinates, curr_angles
    
def identify_patches_to_discard(network_type, curr_subimage_coordinates, curr_angles, curr_disc, base_side, 
                                     im_width, im_height, curr_orig_index, orig_subimage_coordinates, orig_angles,  
                                     max_Dx_diff, max_Dy_diff, tolerance_posxy_deviation, max_scale_radio, min_scale_radio, tolerance_scale_deviation, net_Dang, tolerance_angle_deviation, cut_off_face):   
    if network_type == "PosX":
        # out of image
        out_of_x_borders_images = (curr_subimage_coordinates[:, 0] < 0) |  (curr_subimage_coordinates[:, 2] >= im_width) 

        # too far away horizontally from initial patch
        subimage_deltas_x = (curr_subimage_coordinates[:, 2] + curr_subimage_coordinates[:, 0]) / 2 - \
                            (orig_subimage_coordinates[curr_orig_index][:, 2] +
                             orig_subimage_coordinates[curr_orig_index][:, 0]) / 2

        x_far_images = numpy.abs(subimage_deltas_x) > (max_Dx_diff * tolerance_posxy_deviation)
        new_wrong_images = x_far_images
    elif network_type == "PosY":
        # out of image
        out_of_borders_images = (curr_subimage_coordinates[:, 1] < 0) | (curr_subimage_coordinates[:, 3] >= im_height)

        # too far away vertically from initial patch
        subimage_deltas_y = (curr_subimage_coordinates[:, 3] + curr_subimage_coordinates[:, 1]) / 2 - \
                            (orig_subimage_coordinates[curr_orig_index][:, 3] +
                             orig_subimage_coordinates[curr_orig_index][:, 1]) / 2

        y_far_images = numpy.abs(subimage_deltas_y) > (max_Dy_diff * tolerance_posxy_deviation)
        new_wrong_images = y_far_images
    elif network_type == "PAng":
        too_rotated_images = (curr_angles > orig_angles[
            curr_orig_index] + net_Dang * tolerance_angle_deviation) | (curr_angles < orig_angles[
            curr_orig_index] - net_Dang * tolerance_angle_deviation)
        new_wrong_images = too_rotated_images
    elif network_type == "Scale":
        # too large or small w.r.t. initial patch
        subimage_magnitudes = ((curr_subimage_coordinates[:, 0:2] -
                                curr_subimage_coordinates[:, 2:4]) ** 2).sum(axis=1)
        subimage_sides = numpy.sqrt(subimage_magnitudes)
        # sqrt(2)/2*orig_diagonal = 1/sqrt(2)*orig_diagonal < subimage_diagonal < sqrt(2)*orig_diagonal ???
        too_large_small_images = (subimage_sides / base_side > max_scale_radio *
                                  tolerance_scale_deviation) | (subimage_sides / base_side <
                                                                min_scale_radio / tolerance_scale_deviation)
        new_wrong_images = too_large_small_images
    elif network_type == "Disc":
        new_wrong_images = curr_disc >= cut_off_face
    else:
        er = "Unknown network type:"+str(network_type)
        raise Exception(er)
    return new_wrong_images

def identify_patches_to_discard_slow(network_type, curr_subimage_coordinates, curr_angles, curr_disc, base_side, 
                                     im_width, im_height, curr_orig_index, orig_subimage_coordinates, orig_angles,  
                                     max_Dx_diff, max_Dy_diff, tolerance_posxy_deviation, max_scale_radio, min_scale_radio, tolerance_scale_deviation, net_Dang, tolerance_angle_deviation, cut_off_face):   
    if network_type in ["PosX", "PosY", "PAng", "Scale"]:
        # out of image
        out_of_borders_images = (curr_subimage_coordinates[:, 0] < 0) | \
                                (curr_subimage_coordinates[:, 1] < 0) | \
                                (curr_subimage_coordinates[:, 2] >= im_width) | \
                                (curr_subimage_coordinates[:, 3] >= im_height)

        # too large or small w.r.t. initial patch
        subimage_magnitudes = ((curr_subimage_coordinates[:, 0:2] -
                                curr_subimage_coordinates[:, 2:4]) ** 2).sum(axis=1)
        subimage_sides = numpy.sqrt(subimage_magnitudes)
        # sqrt(2)/2*orig_diagonal = 1/sqrt(2)*orig_diagonal < subimage_diagonal < sqrt(2)*orig_diagonal ???
        too_large_small_images = (subimage_sides / base_side > max_scale_radio *
                                  tolerance_scale_deviation) | (subimage_sides / base_side <
                                                                min_scale_radio / tolerance_scale_deviation)

        # too far away horizontally from initial patch
        subimage_deltas_x = (curr_subimage_coordinates[:, 2] + curr_subimage_coordinates[:, 0]) / 2 - \
                            (orig_subimage_coordinates[curr_orig_index][:, 2] +
                             orig_subimage_coordinates[curr_orig_index][:, 0]) / 2
        subimage_deltas_y = (curr_subimage_coordinates[:, 3] + curr_subimage_coordinates[:, 1]) / 2 - \
                            (orig_subimage_coordinates[curr_orig_index][:, 3] +
                             orig_subimage_coordinates[curr_orig_index][:, 1]) / 2

        # too much rotation w.r.t. initial patch
        too_rotated_images = (curr_angles > orig_angles[
            curr_orig_index] + net_Dang * tolerance_angle_deviation) | (curr_angles < orig_angles[
            curr_orig_index] - net_Dang * tolerance_angle_deviation)
        x_far_images = numpy.abs(subimage_deltas_x) > (max_Dx_diff * tolerance_posxy_deviation)
        y_far_images = numpy.abs(subimage_deltas_y) > (max_Dy_diff * tolerance_posxy_deviation)

        # new_wrong_images = out_of_borders_images | too_large_small_images | x_far_images |
        # y_far_images | too_rotated_images
        new_wrong_images = too_large_small_images | x_far_images | y_far_images | too_rotated_images

        debug_net_discrimination = False
        if debug_net_discrimination:
            #                        print "subimage_deltas_x is: ", subimage_deltas_x
            #                        print "subimage_deltas_y is: ", subimage_deltas_y
            print "Patch discarded. Wrong x_center is:", (curr_subimage_coordinates[:, 2][x_far_images] +
                                                          curr_subimage_coordinates[:, 0][x_far_images]) / 2
            print "Patch discarded. Wrong x_center was:", (orig_subimage_coordinates[:, 2][
                                                               curr_orig_index[x_far_images]] +
                                                           orig_subimage_coordinates[:, 0][
                                                               curr_orig_index[x_far_images]]) / 2
            print "Patch discarded. Wrong y_center is:", (curr_subimage_coordinates[:, 3][y_far_images] +
                                                          curr_subimage_coordinates[:, 1][y_far_images]) / 2
            print "Patch discarded. Wrong y_center was:", (orig_subimage_coordinates[:, 3][
                                                               curr_orig_index[y_far_images]] +
                                                           orig_subimage_coordinates[:, 1][
                                                               curr_orig_index[y_far_images]]) / 2
            print ("new_wrong_images %d = out_of_borders_images %d + too_large_small_images %d " +
                   "x_far_images %d + y_far_images %d") % (new_wrong_images.sum(),
                                                         out_of_borders_images.sum(),
                                                         too_large_small_images.sum(), x_far_images.sum(),
                                                         y_far_images.sum())
        else:
            pass
    else:
        new_wrong_images = curr_disc >= cut_off_face
    return new_wrong_images


def plot_current_subimage_coordinates_angles_confidences(subplot, network_type, show_confidences, display_only_diagonal, im_disp_rgb, orig_colors, curr_orig_index, curr_subimage_coordinates, curr_angles, curr_confidence):
    if subplot is not None:
        subplot.imshow(im_disp_rgb, aspect=1.0, interpolation='nearest', origin='upper')
        # subplot.imshow(im_disp, aspect='auto', interpolation='nearest', origin='upper',
        #     cmap=mpl.pyplot.cm.gray)

        for j, (x0, y0, x1, y1) in enumerate(curr_subimage_coordinates):
            color = orig_colors[curr_orig_index[j]]
            if show_confidences and (network_type == "Disc"):
                color = (curr_confidence[j], curr_confidence[j], curr_confidence[j])
                color = (0.25, 0.5, 1.0)
            if display_only_diagonal:
                subplot.plot([x0, x1], [y0, y1], color=color)
            else:
                # subplot.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=(1.0,1.0,1.0), linewidth=2)
                subplot.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=color, linewidth=2)
                #        if invalid_subimages[j] == False and False:
                #        subplot.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
                #           color=orig_colors[curr_orig_index[j]] )
                #        if invalid_subimages[j] == False:
            # subplot.plot([x0, x1], [y0, y1], color=orig_colors[curr_orig_index[j]] )
            cx = (x0 + x1 - 1) / 2.0
            cy = (y0 + y1 - 1) / 2.0
            mag = 0.4 * (x1 - x0)
            pax = cx + mag * numpy.cos(curr_angles[j] * numpy.pi / 180 + numpy.pi / 2)
            pay = cy - mag * numpy.sin(curr_angles[j] * numpy.pi / 180 + numpy.pi / 2)
            subplot.plot([cx, pax], [cy, pay], color=color, linewidth=2)

def save_pose_normalized_face_detections(image, curr_subimage_coordinates, curr_angles):
    print "Saving normalized face detections"
    # Parameters used to emulate face normalization (for pose, not age!)
    normalization_method = "eyes_inferred-mouth_area"
    centering_mode = "mid_eyes_inferred-mouth"
    rotation_mode = "EyeLineRotation"
    normalized_face_detections_dir = "normalized_face_detections/"
    prefix = "EyeN"
    num_tries = 1
    allow_random_background = False
    out_size = (256, 192)
    convert_format = "L"
    integer_rotation_center = True

    for i, box_coords in enumerate(curr_subimage_coordinates):
        eyes_coords_orig_app, _, _ = compute_approximate_eye_boxes_coordinates(box_coords, face_sampling=0.825,
                                                                               eye_sampling=2.3719,
                                                                               rot_angle=curr_angles[i])
        float_coords = [eyes_coords_orig_app[0], eyes_coords_orig_app[1], eyes_coords_orig_app[2],
                        eyes_coords_orig_app[3], 0.0, 0.0]
        im2 = face_normalization_tools.normalize_image(None, float_coords,
                                                       normalization_method=normalization_method,
                                                       centering_mode=centering_mode,
                                                       rotation_mode=rotation_mode,
                                                       integer_rotation_center=integer_rotation_center,
                                                       out_size=out_size, convert_format=convert_format,
                                                       verbose=False,
                                                       allow_random_background=allow_random_background,
                                                       image=image)
        random_number = numpy.random.randint(1000000)
        im2.save(normalized_face_detections_dir + prefix + "%06d.jpg" % random_number)
        
        
        
        
def find_Left_eyes(images, curr_image_indices, curr_angles, eyesL_box_orig, eye_subimage_width, eye_subimage_height, benchmark, num_networks, network_types, networks, classifiers, eye_regression_width, eye_regression_height, interpolation_format, contrast_enhance="AgeContrastEnhancement_Avg_Std", obj_avg=0.11, obj_std=0.15, tolerance_xy_eye = 9.0 ):
    return find_Left_Right_eyes(images, curr_image_indices, curr_angles, eyesL_box_orig, eye_subimage_width, eye_subimage_height, benchmark, num_networks, network_types, networks, classifiers, eye_regression_width, eye_regression_height, interpolation_format, contrast_enhance="AgeContrastEnhancement_Avg_Std", obj_avg=0.11, obj_std=0.15, tolerance_xy_eye = 9.0, left_eye=1)


def find_Right_eyes(images, curr_image_indices, curr_angles, eyesR_box_orig, eye_subimage_width, eye_subimage_height, benchmark, num_networks, network_types, networks, classifiers, eye_regression_width, eye_regression_height, interpolation_format, contrast_enhance="AgeContrastEnhancement_Avg_Std", obj_avg=0.11, obj_std=0.15, tolerance_xy_eye = 9.0 ):
    # Swap horizontal coordinates
    eyesRhack_box = eyesR_box_orig.copy()  # FOR DEBUG ONLY
    eyesRhack_box[:, 0] = eyesR_box_orig[:, 2]
    eyesRhack_box[:, 2] = eyesR_box_orig[:, 0]
    eyesRhack_box, eyeR_xy_too_far_images = find_Left_Right_eyes(images, curr_image_indices, curr_angles, eyesR_box_orig, eye_subimage_width, eye_subimage_height, benchmark, num_networks, network_types, networks, classifiers, eye_regression_width, eye_regression_height, interpolation_format, contrast_enhance="AgeContrastEnhancement_Avg_Std", obj_avg=0.11, obj_std=0.15, tolerance_xy_eye = 9.0, left_eye=1)
    # Undo horizontal swap of coordinates
    eyesR_box = eyesRhack_box.copy()
    eyesR_box[:, 0] = eyesRhack_box[:, 2]
    eyesR_box[:, 2] = eyesRhack_box[:, 0]

    return eyesR_box, eyeR_xy_too_far_images


def find_Left_Right_eyes(images, curr_image_indices, curr_angles, eyesL_box_orig, eye_subimage_width, eye_subimage_height, benchmark, num_networks, network_types, networks, classifiers, eye_regression_width, eye_regression_height, interpolation_format, contrast_enhance="AgeContrastEnhancement_Avg_Std", obj_avg=0.11, obj_std=0.15, tolerance_xy_eye = 9.0, left_eye=1):
    eye_xy_too_far_images = numpy.zeros(len(curr_image_indices), dtype=bool)
    # Left eye only!
    eyesL_box = eyesL_box_orig.copy() # FOR DEBUG ONLY!!!
    # print "eyesL_box=",eyesL_box
    # TODO: Use new normalization method used for eye localization
    eyeL_subimages = extract_subimages_rotate(images, curr_image_indices, eyesL_box, -1 * curr_angles,
                                              (eye_subimage_width, eye_subimage_height), interpolation_format,
                                              contrast_enhance="AgeContrastEnhancement_Avg_Std",
                                              obj_avg=0.11, obj_std=0.15)

    benchmark.add_task_from_previous_time("Extracted eye patches (L or R)")

    debug = True and False
    if debug:
        for i in range(len(eyeL_subimages)):
            if left_eye:
                print "saving eyeL patch"
                eyeL_subimages[i].save("saved_patches/eyeL_patch_im%+03d_PAngle%03.3f.jpg" % (i, curr_angles[i]))
            else:
                print "saving eyeR patch"
                eyeL_subimages[i].save("saved_patches/eyeR_patch_im%+03d_PAngle%03.3f.jpg" % (i, curr_angles[i]))
    
    benchmark.add_task_from_previous_time("Saved eye patches (if enabled)")

    if len(eyeL_subimages) > 0:
        for num_network in [num_networks - 5, num_networks - 4]:
            subimages_arr = images_asarray(eyeL_subimages)
            sl = networks[num_network].execute(subimages_arr, benchmark=None)
            benchmark.add_task_from_previous_time("Extracted eye features (L or R)")

            # print "Network %d processed all subimages"%(num_networks-2)
            reg_num_signals = classifiers[num_network].input_dim
            avg_labels = classifiers[num_network].avg_labels
            print "avg_labels=", avg_labels
            reg_out = classifiers[num_network].regression(sl[:, 0:reg_num_signals], avg_labels)
            print "reg_out_crude:", reg_out
            eye_xy_too_far_images |= numpy.abs(reg_out) >= tolerance_xy_eye

            benchmark.add_task_from_previous_time("Regression eye position (L or R)")

            if network_types[num_network] == "EyeLX":  # POS_X
                # print "EyeLX"
                eyes_box_width = numpy.abs(eyesL_box[:, 2] - eyesL_box[:, 0])
                reg_out_x = (reg_out / 2.3719) * eyes_box_width / eye_regression_width
                # reg_out_x = reg_out * eyes_box_width / eye_regression_width / 0.94875
            elif network_types[num_network] == "EyeLY":  # POS_Y
                # print "EyeLY"
                eyes_box_height = numpy.abs(eyesL_box[:, 3] - eyesL_box[:, 1])
                reg_out_y = (reg_out / 2.3719) * eyes_box_height / eye_regression_height
                # reg_out_y = reg_out * eyes_box_height / eye_regression_height / 0.94875
            else:
                print "Unknown network type! (expecting either EyeLX or EyeLY)", network_types[num_network]
                quit()

        if left_eye:
            factor = 1
        else:
            factor = -1

        rot_angle_radian = -1 * factor * curr_angles * numpy.pi / 180
        eyesL_dx = reg_out_x * numpy.cos(rot_angle_radian) - reg_out_y * numpy.sin(rot_angle_radian)
        eyesL_dy = reg_out_y * numpy.cos(rot_angle_radian) + reg_out_x * numpy.sin(rot_angle_radian)

        eyesL_box[:, 0] = eyesL_box[:, 0] - factor * eyesL_dx  # X0
        eyesL_box[:, 2] = eyesL_box[:, 2] - factor * eyesL_dx  # X1
        # # # print "left eye, X reg_out final=", reg_out
        eyesL_box[:, 1] = eyesL_box[:, 1] - eyesL_dy  # Y0
        eyesL_box[:, 3] = eyesL_box[:, 3] - eyesL_dy  # Y1
        # # # print "left eye, Y reg_out final=", reg_out
        benchmark.add_task_from_previous_time("Adjusted eye coordinates accounting for rotation angle (L or R)")

        print "LE found *********************************************************************"
    return eyesL_box, eye_xy_too_far_images


# #         # print "eyesRhack_box=",eyesRhack_box
# #         eyeR_subimages = extract_subimages_rotate(images, curr_image_indices, eyesRhack_box, -1 * curr_angles,
# #                                                   (eye_subimage_width, eye_subimage_height), interpolation_format,
# #                                                   contrast_enhance="AgeContrastEnhancement_Avg_Std",
# #                                                   obj_avg=0.11, obj_std=0.15)
# #         benchmark.add_task_from_previous_time("Extracted eye patches (L or R)")
# # 
# #         debug = True and False
# #         if debug:
# #             for i in range(len(eyeR_subimages)):
# #                 print "saving eyeR patch"
# #                 eyeR_subimages[i].save("saved_patches/eyeRRpatch_im%+03d_PAngle%03.3f.jpg" % (i, curr_angles[i]))
# #         benchmark.add_task_from_previous_time("Saved eye patches (if enabled)")
# # 
# #         if len(eyeR_subimages) > 0:
# #             for num_network in [num_networks - 5, num_networks - 4]:
# #                 # eyeR_subimages = extract_subimages_basic(images, curr_image_indices, eyesRhack_box,
# #                 # (eye_subimage_width, eye_subimage_height) )
# #                 subimages_arr = images_asarray(eyeR_subimages) + 0.0
# #                 sl = networks[num_network].execute(subimages_arr, benchmark=None)
# #                 benchmark.add_task_from_previous_time("Extracted eye features (L or R)")
# # 
# #                 # print "Network %d processed all subimages"%(num_networks-2)
# #                 reg_num_signals = classifiers[num_network].input_dim
# #                 avg_labels = classifiers[num_network].avg_labels
# #                 reg_out = classifiers[num_network].regression(sl[:, 0:reg_num_signals], avg_labels)
# #                 print "reg_out_crude:", reg_out
# #                 benchmark.add_task_from_previous_time("Regression eye position (L or R)")
# # 
# #                 eye_xy_too_far_images |= numpy.abs(reg_out) >= 9.0  # 9.0
# #                 if network_types[num_network] == "EyeLX":  # POS_X
# #                     # print "EyeLX"
# #                     eyes_box_width = numpy.abs(eyesRhack_box[:, 2] - eyesRhack_box[:, 0])
# #                     reg_out_x = (reg_out / 2.3719) * eyes_box_width / eye_regression_width
# #                     # reg_out_x = reg_out * eyes_box_width / eye_regression_width * 0.94875 #PREVIOUS WORKING VERSION
# #                 elif network_types[num_network] == "EyeLY":  # POS_Y
# #                     # print "EyeLY"
# #                     eyes_box_height = numpy.abs(eyesRhack_box[:, 3] - eyesRhack_box[:, 1])
# #                     reg_out_y = (reg_out / 2.3719) * eyes_box_width / eye_regression_width
# #                     # reg_out_y = reg_out * eyes_box_height / eye_regression_height * 0.94875
# #                     print "right eye, Y reg_out final=", reg_out
# #                 else:
# #                     print "Unknown network type!", network_types[num_network]
# #                     quit()
# #             print "RE found *********************************************************************"
# # 
# #             rot_angle_radian = curr_angles * numpy.pi / 180
# #             eyesR_dx = reg_out_x * numpy.cos(rot_angle_radian) - reg_out_y * numpy.sin(rot_angle_radian)
# #             eyesR_dy = reg_out_y * numpy.cos(rot_angle_radian) + reg_out_x * numpy.sin(rot_angle_radian)
# # 
# #             eyesRhack_box[:, 0] = eyesRhack_box[:, 0] + eyesR_dx  # X0
# #             eyesRhack_box[:, 2] = eyesRhack_box[:, 2] + eyesR_dx  # X1
# #             print "right eye, X reg_out final=", reg_out
# #             eyesRhack_box[:, 1] = eyesRhack_box[:, 1] - eyesR_dy  # Y0
# #             eyesRhack_box[:, 3] = eyesRhack_box[:, 3] - eyesR_dy  # Y1
# # 
# #             benchmark.add_task_from_previous_time("Adjusted eye coordinates accounting for rotation angle (L or R)")

def estimate_age_race_gender(image, detected_faces_eyes_confidences_purgued, benchmark, number_saved_image_age_estimation, age_subimage_width, age_subimage_height, num_networks, networks, classifiers, estimate_age=True, estimate_race=True, estimate_gender=True, verbose=True):
    # Parameters used to emulate face normalization
    normalization_method = "eyes_inferred-mouth_areaZ"
    centering_mode = "mid_eyes_inferred-mouth"
    rotation_mode = "EyeLineRotation"
    allow_random_background = False
    out_size = (256, 260)
    age_image_width = out_size[0]
    age_image_height = out_size[1]
    integer_rotation_center = True

    # Parameters used to emulate image loading (load_data_from_sSeq)
    age_base_scale = 1.14
    # # # # age_scale_offset = 0.00 # 0.08 0.04

    age_obj_std_base = 0.16  # ** 0.16

    reduction_factor = 160.0 / 96  # (affects only sampling, subimage size, and translations)
    age_sampling = age_base_scale * reduction_factor
    age_pre_mirroring = "none"
    age_contrast_enhance = "AgeContrastEnhancement_Avg_Std"
    age_obj_avg_std = 0.0
    age_obj_std = age_obj_std_base

    age_subimage_first_row = age_image_height / 2.0 - age_subimage_height * age_sampling / 2.0
    age_subimage_first_column = age_image_width / 2.0 - age_subimage_width * age_sampling / 2.0
    age_translations_x = 0.0 / reduction_factor
    age_translations_y = -6.0 / reduction_factor

    # 1)Extract face patches (usually 96x96, centered according to the eye positions)
    num_faces_found = len(detected_faces_eyes_confidences_purgued)
    age_estimates = numpy.zeros(num_faces_found)
    age_stds = numpy.zeros(num_faces_found)
    race_estimates = 10 * numpy.ones(num_faces_found)
    gender_estimates = 10 * numpy.ones(num_faces_found)

    benchmark.add_task_from_previous_time("Prepared for age, gender, and race estimation")

    for j, box_eyes_coords_confidence in enumerate(detected_faces_eyes_confidences_purgued):
        # A) Generate normalized image
        # contents: eyeL_x, eyeL_y, eyeR_x, eyeR_y, mouth_x, mouth_y
        benchmark.update_start_time()
        float_coords = [box_eyes_coords_confidence[4], box_eyes_coords_confidence[5], box_eyes_coords_confidence[6],
                        box_eyes_coords_confidence[7], 0.0, 0.0]
        im2 = face_normalization_tools.normalize_image(None, float_coords,
                                                       normalization_method=normalization_method,
                                                       centering_mode=centering_mode,
                                                       rotation_mode=rotation_mode,
                                                       integer_rotation_center=integer_rotation_center,
                                                       out_size=out_size, convert_format="L",
                                                       verbose=False,
                                                       allow_random_background=allow_random_background,
                                                       image=image)
        benchmark.add_task_from_previous_time("Age/race/gender: computed normalized image")

        # # print im2
        # # print type(im2)
        # # print isinstance(im2, numpy.ndarray)
        # # print isinstance(im2, Image.Image)
        # B)Extract actual patches
        # TODO: why am I using the monoprocessor version???
        age_subimages_arr = load_image_data_monoprocessor(image_files=["nofile"], image_array=im2,
                                                          image_width=age_image_width,
                                                          image_height=age_image_height,
                                                          subimage_width=age_subimage_width,
                                                          subimage_height=age_subimage_height,
                                                          pre_mirroring_flags=False, pixelsampling_x=age_sampling,
                                                          pixelsampling_y=age_sampling,
                                                          subimage_first_row=age_subimage_first_row,
                                                          subimage_first_column=age_subimage_first_column,
                                                          add_noise=False, convert_format="L",
                                                          translations_x=age_translations_x,
                                                          translations_y=age_translations_y, trans_sampled=True,
                                                          rotations=0.0, rotate_before_translation=True, contrast_enhance=age_contrast_enhance,
                                                          obj_avgs=0.0,
                                                          obj_stds=age_obj_std, background_type=None,
                                                          color_background_filter=None, subimage_reference_point=0,
                                                          verbose=False)

        benchmark.add_task_from_previous_time("Age/race/gender: extracted image array")

        im_raw = numpy.reshape(age_subimages_arr[0], (96, 96))
        im = scipy.misc.toimage(im_raw, mode="L")
        im.save("ImageForAgeEstimation%03d.jpg" % number_saved_image_age_estimation)
        number_saved_image_age_estimation += 1
        # 2)Apply the age estimation network
        num_network = num_networks - 3
        sl = networks[num_network].execute(age_subimages_arr, benchmark=None)
        benchmark.add_task_from_previous_time("Age/race/gender: feature extraction")

        if estimate_age:
            reg_num_signals = classifiers[num_network].input_dim
            avg_labels = classifiers[num_network].avg_labels
            reg_out, std_out = classifiers[num_network].regression(sl[:, 0:reg_num_signals], avg_labels,
                                                                   estimate_std=True)
            benchmark.add_task_from_previous_time("Computed age regression")

            age_estimates[j] = reg_out[0]
            age_stds[j] = std_out[0]
            if verbose:
                print "age estimation:", reg_out[0], "+/-", std_out[0]
        if estimate_race:
            num_network = num_networks - 2
            reg_num_signals = classifiers[num_network].input_dim
            avg_labels = classifiers[num_network].avg_labels
            # reg_out = classifiers[num_network].label(sl[:,0:reg_num_signals])
            reg_out = classifiers[num_network].regression(sl[:, 0:reg_num_signals], avg_labels)
            benchmark.add_task_from_previous_time("Computed race classification")

            race_estimates[j] = reg_out[0]
            if verbose:
                print "race estimation:", reg_out[0]
        if estimate_gender:
            num_network = num_networks - 1
            reg_num_signals = classifiers[num_network].input_dim
            avg_labels = classifiers[num_network].avg_labels
            # reg_out = classifiers[num_network].label(sl[:,0:reg_num_signals])
            reg_out = classifiers[num_network].regression(sl[:, 0:reg_num_signals], avg_labels)
            benchmark.add_task_from_previous_time("Computed gender classification")

            gender_estimates[j] = reg_out[0]
            if verbose:
                print "gender estimation:", reg_out[0]
    # 3)Interpret the results
    gender_confidences = numpy.abs(gender_estimates)
    race_confidences = numpy.abs(race_estimates) / 2.0

    gender_estimates = map_real_gender_labels_to_strings(gender_estimates, long_text=True)
    race_estimates = map_real_race_labels_to_strings(race_estimates, long_text=True)
    if verbose:
        print "Age estimates:", age_estimates
        print "Age stds: ", age_stds
        print "Race estimates:", race_estimates
        print "Race confidences:", race_confidences
        print "Gender estimates:", gender_estimates
        print "Gender confidences:", gender_confidences
    return number_saved_image_age_estimation, age_estimates, age_stds, race_estimates, gender_estimates
