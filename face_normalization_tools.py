#Functions to generate normalized datasets
#Rotations enabled on 22.05.2012
#Library version first adapted on 2.9.2010
#Alberto Escalante. alberto.escalante@ini.rub.de

import string
import sys
#from lxml import etree
import PIL
from PIL import Image
from PIL import ImageOps
import numpy
import scipy
import math

sys.path.append("/home/escalafl/workspace4/cuicuilco_MDP3.2/src")
from cuicuilco import image_loader

def compute_approximate_mouth_coordinates(eye_coordinates):
    """ Given the coordinates of the eyes of a face, the function approximates the position of the mouth.
    It assumes that the face is mostly frontal (except for the screen plane rotation), and has a similar shape as computer generated ones.
    Standard face shape: dist_eyes=37.0, height_triangle=42.0 (eyes to mouth).
    Triangle's area is kept constant at 37.0*42.0/2
    eye_coordinates -- 4 tuple with values eye_left_x, eye_left_y, eye_right_x, eye_right_y
    output -- numpy.array([mouth_x, mouth_y])
    """
    eye_left_x, eye_left_y, eye_right_x, eye_right_y = eye_coordinates
    eye_dx = eye_right_x-eye_left_x
    eye_dy = eye_right_y-eye_left_y
    dist_eyes = numpy.sqrt((eye_dx)**2 + (eye_dy)**2)
    midpoint_eyes_x = (eye_right_x+eye_left_x)/2.0
    midpoint_eyes_y = (eye_right_y+eye_left_y)/2.0

    #Standard shape: dist_eyes=37.0, height_triangle=42.0
    mouth_x = midpoint_eyes_x - (42.0 / 37.0) * eye_dy 
    mouth_y = midpoint_eyes_y + (42.0 / 37.0) * eye_dx
    
    return numpy.array([mouth_x, mouth_y])

#WARNING, this function is NOT Final!!!!!
#TODO:make this function safe wrt. the transform operation real coords=(x0,y0,x1+1,y1+1)
#BUG:Seems to shrink/expand image horizontally!!!
def im_transform_randombackground(im, out_size, transf, data, filter):
    if transf != Image.EXTENT:
        print "transformation not supported in im_transform_randombackground:", transf
        return None
#    x0, y0, x1, y1 = map(int, data)
    x0, y0, x1, y1 = data
    im_width, im_height = im.size
    if x0>=0 and y0>=0 and x1 <= im_width and y1 <= im_height: #WWW changed im_width-1 to im_width, the same with im_height
        return im.transform(out_size, Image.EXTENT, data, filter)

    patch_width = out_size[0]
    patch_height = out_size[1]    
    noise = numpy.random.randint(256, size=(patch_height, patch_width))

    out_im = scipy.misc.toimage(noise, mode="L")

    xp0, yp0, xp1, yp1 = x0, y0, x1, y1 #The {xy}p{01} contain the true origin location of the pixels to be mapped
    Xp0, Yp0, Xp1, Yp1 = 0, 0, patch_width, patch_height #The {XY}p{01} contain the true destination location of the pixels to be mapped
    
    if x0 < 0:
        xp0=0
        Xp0=(xp0-x0) * patch_width/(x1-x0)
    if y0 < 0:
        yp0=0
        Yp0=(yp0-y0) * patch_height/(y1-y0)
    if x1 > im_width:
        xp1= im_width
        Xp1=(xp1-x0) * patch_width/(x1-x0)
    if y1 > im_height:
        yp1= im_height
        Yp1=(yp1-y0) * patch_height/(y1-y0)

    #Warning, small error introduced here, image size  int(Xp1+0.5)-int(Xp0+0.5) not equal to int(Xp1-Xp0+0.5)
    out_size2 = (int(Xp1+0.5) - int(Xp0+0.5), int(Yp1+0.5) - int(Yp0+0.5))
    #Warning, are integer coordinates really needed? another error introduced here
    Xp0, Yp0, Xp1, Yp1 = map(int, [Xp0+0.5, Yp0+0.5, Xp1+0.5, Yp1+0.5])
#   is this test correct? i guess the variables are inverted!
#   it was: if xp0 < 0 or yp0 < 0 or xp1 > im_width-1 or yp1 > im_height-1:
#   Modified version:
    if xp1 <= 0 or yp1 <= 0 or xp0 >= im_width or yp0 >= im_height:
        print "transformation warning: patch fully out of image"
        quit()
        return out_im
    
    #Get Image patch from original image
    data2 = (xp0, yp0, xp1, yp1)
    print out_size2, data2

    if out_size2[0] > 0 and out_size2[1] > 0:     
        im_tmp = im.transform(out_size2, transf, data2, filter)    
        #Copy sampled image patch into noise image
        #Warning! paste cannot be made with subpixel accuracy, thus small error introduced here
        out_im.paste(im_tmp, (Xp0, Yp0, Xp1, Yp1))

    return out_im

#Coordinates: [eyeL, eyeR, mouth] (no nose!)
def normalize_image(filename, coordinates, normalization_method = "eyes_mouth_area", centering_mode = "mid_eyes_mouth", rotation_mode = "noRotation", integer_rotation_center=True ,out_size = (256,192), convert_format="L", verbose=False, allow_random_background=True, image=None):
    """ Opens an image "filename", normalizes its size depending on the "normalization_method", and centers it to the point specified by "centering_mode" 
    coordinates -- coordinates of the face given by the tuple (LeftEyeCenter_x, LeftEyeCenter_y, RightEyeCenter_x, RightEyeCenter_y, Mouth_x, Mouth_y) 
    normalization_method -- either "eyes_mouth_area", "eyes_inferred-mouth_areaZ", "eyes_inferred-mouth_areaZ-Test"
    
    """

    LeftEyeCenter_x, LeftEyeCenter_y, RightEyeCenter_x, RightEyeCenter_y, Mouth_x, Mouth_y = coordinates
    
    if image is None:
        print "filename=", filename, 
        try:
            im = Image.open(filename)
            im = im.convert(convert_format)
        except:
            print "failed opening image", filename
            return None
    else:
        print "using provided image"
        im = image

    #Read coordinate data

    #Load and convert source image

    #Compute important coordinates and measures
    eyes_x_m = (RightEyeCenter_x + LeftEyeCenter_x) / 2.0
    eyes_y_m = (RightEyeCenter_y + LeftEyeCenter_y) / 2.0

    midpoint_eyes_mouth_x = (eyes_x_m + Mouth_x) / 2.0
    midpoint_eyes_mouth_y = (eyes_y_m + Mouth_y) / 2.0

    dist_eyes = numpy.sqrt((LeftEyeCenter_x - RightEyeCenter_x)**2 + (LeftEyeCenter_y - RightEyeCenter_y)**2) 
        
    #Triangle formed by the eyes and the mouth.
    height_triangle = numpy.sqrt((eyes_x_m - Mouth_x)**2 + (eyes_y_m - Mouth_y)**2) 

    #angle of eye_line with respect to horizontal axis, measured counter-clock wise
    #print RightEyeCenter_y - LeftEyeCenter_y, RightEyeCenter_x - LeftEyeCenter_x
    eye_line_angle = math.atan2(RightEyeCenter_y - LeftEyeCenter_y, RightEyeCenter_x - LeftEyeCenter_x) * 180 / math.pi
      
    if LeftEyeCenter_x > RightEyeCenter_x:
        print "Warning: the eyes are ordered incorrectly!!! in ", filename
        exit()

    #Assumes eye line is perpendicular to the line from eyes_m to mouth
    current_area = dist_eyes * height_triangle / 2.0
#    desired_area = 37.0 * 42.0 / 2.0 
    desired_area = 37.0 * 42.0 / 2.0 * (37.5/37.0)**2

    inferred_mouth_x, inferred_mouth_y = compute_approximate_mouth_coordinates((LeftEyeCenter_x, LeftEyeCenter_y, RightEyeCenter_x, RightEyeCenter_y))
    print "inferred_mouth_x, inferred_mouth_y=",  inferred_mouth_x, inferred_mouth_y 
    
    height_triangle_with_inferred_mouth = numpy.sqrt((eyes_x_m - inferred_mouth_x)**2 + (eyes_y_m - inferred_mouth_y)**2)
    current_area_with_inferred_mouth = dist_eyes * height_triangle_with_inferred_mouth / 2.0
    midpoint_eyes_inferred_mouth_x = (eyes_x_m + inferred_mouth_x) / 2.0
    midpoint_eyes_inferred_mouth_y = (eyes_y_m + inferred_mouth_y) / 2.0
    
    print "Normalization:"+normalization_method, "Centering_mode="+centering_mode, "rotation_mode="+rotation_mode

    #Compute scale: ori_width, ori_height
    if normalization_method == "eyes_mouth_area":        
        scale_factor =  numpy.sqrt(current_area / desired_area )
        ori_width = out_size[0]*scale_factor 
        ori_height = out_size[1]*scale_factor
    elif normalization_method == "eyes_inferred-mouth_area": 
        scale_factor =  numpy.sqrt(current_area_with_inferred_mouth / desired_area )
        ori_width = out_size[0]*scale_factor 
        ori_height = out_size[1]*scale_factor
    elif normalization_method == "eyes_inferred-mouth_areaZ": 
        scale_factor =  numpy.sqrt(current_area_with_inferred_mouth / desired_area )
        ori_width = out_size[0]*scale_factor/2 
        ori_height = out_size[1]*scale_factor/2
    elif normalization_method == "eyes_inferred-mouth_areaZ-Test": 
        desired_area_test = 8.0 * (8.0 * 42.0 / 37) / 2.0
        scale_factor =  numpy.sqrt(current_area_with_inferred_mouth / desired_area_test )
        print "current_area_with_inferred_mouth=", current_area_with_inferred_mouth
        print "desired_area_test=", desired_area_test
        ori_width = out_size[0]*scale_factor
        ori_height = out_size[1]*scale_factor
    else:
        er = "Error in normalization: Unknown Method:" + str(normalization_method)
        raise Exception(er)

    if centering_mode=="mid_eyes_mouth":
        rotation_center_x = midpoint_eyes_mouth_x
        rotation_center_y = midpoint_eyes_mouth_y
    elif centering_mode == "mid_eyes_inferred-mouth":
        rotation_center_x = midpoint_eyes_inferred_mouth_x
        rotation_center_y = midpoint_eyes_inferred_mouth_y
    elif centering_mode == "eyeL":
        rotation_center_x = LeftEyeCenter_x
        rotation_center_y = LeftEyeCenter_y
    elif centering_mode == "eyeR":
        rotation_center_x = RightEyeCenter_x
        rotation_center_y = RightEyeCenter_y
    elif centering_mode == "noFace":
        angle = numpy.random.uniform(0, 2*numpy.pi)
        rotation_center_x = midpoint_eyes_mouth_x + 0.75*ori_width * numpy.cos(angle)
        rotation_center_y = midpoint_eyes_mouth_y + 0.75*ori_height * numpy.sin(angle)
        ori_width = ori_width / 2.0 #zoom-in to further avoid faces
        ori_height = ori_height / 2.0
    else:
        er = "Error in centering_mode: Unknown Method:"+str(centering_mode)
        raise Exception(er)        


    if rotation_mode == "noRotation":
        rotation_angle = 0.0
    else:
        rotation_angle = eye_line_angle

    #print "rotation angle=", rotation_angle
    
    #Rotate (even if it is only 0 degrees). Needed for consistency.
    #Find nearest pixel to rotate
    if integer_rotation_center:
        rotation_center_x_int = int(rotation_center_x+0.5)
        rotation_center_y_int = int(rotation_center_y+0.5)  
    else:
        rotation_center_x_int = rotation_center_x
        rotation_center_y_int = rotation_center_y  
        
    rotation_window_width = 2 * max(im.size[0]-1-rotation_center_x_int+0.5, rotation_center_x_int+0.5) #Result is always integer and odd
    rotation_window_height = 2 * max(im.size[1]-1-rotation_center_y_int+0.5, rotation_center_y_int+0.5) # 

    Delta_x = rotation_center_x - rotation_center_x_int
    Delta_y = rotation_center_y - rotation_center_y_int

    #print "Rotation angle=", rotation_angle, "degrees"
    rotation_angle_rad =  -rotation_angle * numpy.pi/180.0 #ROTATE IN OPOSITE DIRECTION, SEE AXIS
    Delta_x_rotated = Delta_x  * numpy.cos(rotation_angle_rad) - Delta_y * numpy.sin(rotation_angle_rad)
    Delta_y_rotated = Delta_y  * numpy.cos(rotation_angle_rad) + Delta_x * numpy.sin(rotation_angle_rad)
    
    rotation_crop_x0 = rotation_center_x_int-(rotation_window_width-1)/2.0 # integer crop values
    rotation_crop_y0 = rotation_center_y_int-(rotation_window_height-1)/2.0 #
    rotation_crop_x1 = rotation_center_x_int+(rotation_window_width-1)/2.0+1 #rotation_angle
    rotation_crop_y1 = rotation_center_y_int+(rotation_window_height-1)/2.0+1 #              

    crop_size = (int(rotation_window_width+0.5), int(rotation_window_height+0.5)) #integer values anyway
    crop_coordinates = (rotation_crop_x0, rotation_crop_y0, rotation_crop_x1, rotation_crop_y1)
    #print "crop_size=", crop_size
    #print "crop_coordinates=", crop_coordinates
    #print "rotation_center_x_int, rotation_center_y_int=", rotation_center_x_int, rotation_center_y_int
    #print "rotation_center_x, rotation_center_y=", rotation_center_x, rotation_center_y
    #print "Delta_x_rotated, Delta_y_rotated=", Delta_x_rotated, Delta_y_rotated
    
    im_crop_first = im.transform(crop_size, Image.EXTENT, crop_coordinates) #, Image.BICUBIC)
                    
    #print delta_ang
    if rotation_angle != 0 or True:        
        im_rotated_shifted = image_loader.rotate_improved(im_crop_first, rotation_angle, Image.BICUBIC)
#    else:
#        im_rotated_shifted = im_crop_first
#        print "not rotating"

    #FIX ROTATION!!! (Translate one pixel left-up)
#    im_rotated = im_rotated_shifted.transform(crop_size, Image.EXTENT, (1,1, crop_size[0]+1, crop_size[1]+1))
    im_rotated = im_rotated_shifted
  
#    print "crop width mod 2=%d, crop height mod 2=%d, im_rotated.size[0] mod 2=%d, im_rotated.size[1] mod 2=%d"%(\
#            crop_size[0] & 1, crop_size[1] & 1, im_rotated.size[0] & 1, im_rotated.size[1] & 1)
    #im_out=im_rotated
    #im_out.save("im_out.jpg", "JPEG", quality=95)
    #im_outT = im_out.transpose(Image.FLIP_LEFT_RIGHT)
    #im_outT.save("im_outT.jpg", "JPEG", quality=95)
    
    #####Main execution path verified and working (no rotation)    
    #### Now extract goal image from rotated image
    center_x_geometric = (crop_size[0]-1)/2.0
    center_y_geometric = (crop_size[1]-1)/2.0
    #print "After rotation: geometric center_x, rotation_center_y=", center_x_geometric, rotation_center_y_geometric
    
    new_center_x = center_x_geometric + Delta_x_rotated #Real value!
    new_center_y = center_y_geometric + Delta_y_rotated #Real value!
    #print "After rotation: center_x, center_y=", center_x, center_y
    #print "center_x, center_y=", center_x, center_y
    #print "ori_width, ori_height=", ori_width, ori_height
    
    #WWW removed int
    x0 = (new_center_x-(ori_width-1)/2.0) #WWW added -1
    x1 = (new_center_x+(ori_width-1)/2.0)+1 #WWW added -1 and +1
    y0 = (new_center_y-(ori_height-1)/2.0) #WWW added -1
    y1 = (new_center_y+(ori_height-1)/2.0)+1 #WWW added -1 and +1
    
    #print "transform_coords = x0, x1, y0, y1=", x0, x1, y0, y1
    transform_coords = (x0,y0,x1,y1)
    
    if allow_random_background: ####WWWWW
        im_out = im_transform_randombackground(im_rotated, out_size, Image.EXTENT, transform_coords, Image.BILINEAR)
        print "Aborted!"
        quit()
    else:
        if False: #x0<0 or y0<0 or x1>im_rotated.size[0] or y1>im_rotated.size[1]:
            print "Normalization Failed: Not enough background to cut"
            im_out = None
        else:
            im_out = im_rotated.transform(out_size, Image.EXTENT, transform_coords, Image.BICUBIC)

    if centering_mode == "eyeR": #do a final mirroring step in this case only
        im_out = ImageOps.mirror(im_out)
        
    return im_out

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "Incorrect number of parameters. Usage: %s coordinate_file output_pattern mode"%sys.argv[0]
        print "coordinate_file -- File with the face coordinates, for instance: CAS_PEAL_coordinates.txt"
        print "output_pattern -- pattern for the filename used to store the normalized images, e.g. 'image%05d.jpg' or 'None' to preserve original filename"
        print "mode -- either 'mid_eyes_mouth' or 'mid_eyes_inferred-mouth' or 'background' or 'mid_eyes_inferred-mouthZ_horiz' or 'mid_eyes_inferred-mouthZ2_horiz' or 'mid_eyes_inferred-mouthZ2_horiz-Test'"
#        print "centering_mode -- either 'mid_eyes_mouth' or 'mid_eyes_inferred-mouth' or 'noFace' " 
#        print "rotation_mode -- either 'noRotation' or 'EyeLineRotation' "
        quit()
        
    #CREATE NORMALIZATION FILE
    normalization_filename = sys.argv[1] #CAS_PEAL_coordinates.txt
    output_pattern = sys.argv[2] # image%05d.jpg
    mode = sys.argv[3]
    
#    centering_mode = sys.argv[4]
#    rotation_mode = sys.argv[5]
#
#    if normalization_mode not in ["eyes_mouth_area", "eyes_inferred-mouth_area", "eyes_inferred-mouth_areaZ"]:
#        er = "normalization mode unsupported:"+normalization_mode
#        raise Exception(er)
#
#    if rotation_mode not in ["noRotation", "EyeLineRotation"]):
#        er = "Rotation mode unsupported:"+rotation_mode
#        raise Exception(er)
#
#    if centering_mode not in ["mid_eyes_mouth", "mid_eyes_inferred-mouth", "noFace"]:
#         er = "Centering mode unsupported:"+centering_mode
#        raise Exception(er)
         
    out_size = (256,192)
    convert_format="L"
    integer_rotation_center = True
    if mode == "mid_eyes_mouth_horiz": #Default for face detection (true mouth coordinates)
        normalization_method = "eyes_mouth_area"
        centering_mode="mid_eyes_mouth"
        rotation_mode="EyeLineRotation"
        out_dir = "normalized_h/"
        prefix = ""
        num_tries = 1        
        allow_random_background=True and False
    elif mode == "mid_eyes_inferred-mouth_horiz": #Default for face detection (no mouth coordinates)
        normalization_method = "eyes_inferred-mouth_area"
        centering_mode="mid_eyes_inferred-mouth"
        rotation_mode="EyeLineRotation"
        out_dir = "normalizedEyes_h/"
        prefix = "EyeN"
        num_tries = 1        
        allow_random_background=True and False #This introduces an additional transformation that might decrease performance, instead have black background.
    elif mode == "mid_eyes_inferred-mouthZ_horiz":
        normalization_method = "eyes_inferred-mouth_areaZ"
        centering_mode="mid_eyes_inferred-mouth"
        rotation_mode="EyeLineRotation"
        out_dir = "normalizedEyesZ_h/"
        prefix = "EyeNZ"
        num_tries = 1        
        allow_random_background=True and False
    elif mode == "mid_eyes_inferred-mouthZ4_horiz": #mode used for age estimation
        normalization_method = "eyes_inferred-mouth_areaZ"
        centering_mode="mid_eyes_inferred-mouth"
        rotation_mode="EyeLineRotation"
        out_dir = "normalizedEyesZ4_h/" # "normalizedEyesZ2_h_RGB/"
        prefix = "EyeNZ4"
        num_tries = 1        
        allow_random_background=True and False
        out_size = (256,260) #width, height
#        convert_format = "RGB"
    elif mode == "mid_eyes_inferred-mouthZ4_horiz-Test": #mode used for debugging age estimation
        normalization_method = "eyes_inferred-mouth_areaZ-Test"
        centering_mode="mid_eyes_inferred-mouth"
        rotation_mode="EyeLineRotation"
        out_dir = "normalizedEyesZ4_h-Test/" # "normalizedEyesZ2_h_RGB/"
        prefix = "EyeNZ4"
        num_tries = 1        
        allow_random_background=True and False
        out_size = (17,20)
    elif mode == "background":
        normalization_method = "eyes_mouth_area"
        centering_mode="noFace"
        rotation_mode="noRotation"
        out_dir = "noFace/"
        prefix = ""
        num_tries = 10
        allow_random_background=False
    elif mode == "leftEye": #Method used for estimation of eye position
        normalization_method = "eyes_inferred-mouth_areaZ"
        centering_mode="eyeL"
        rotation_mode="EyeLineRotation"
        out_dir = "normalized_EyeL/"
        prefix = ""
        num_tries = 1        
        allow_random_background=True and False
    elif mode == "rightEye": #Method used for estimation of eye position, the right eye is mirrored horizontally
        normalization_method = "eyes_inferred-mouth_areaZ"
        centering_mode="eyeR"
        rotation_mode="EyeLineRotation"
        out_dir = "normalized_EyeR/"
        prefix = ""
        num_tries = 1        
        allow_random_background=True and False
    else:
        print "Aborting. Unknown normalization/centering_mode mode:", mode
        quit()
        
    normalization_file = open(normalization_filename, "r")
    count = 0
    working = 1
    max_count = 200000
    while working==1 and count < max_count:
        filename = normalization_file.readline().rstrip()
        if filename == "":
            working = 0
        else: 
            coords_str = normalization_file.readline()
            coords = string.split(coords_str, sep=" ")
            float_coords = map(float, coords)
            dist_eyes = numpy.sqrt((float_coords[2]-float_coords[0]) ** 2 + (float_coords[3]-float_coords[1])**2)
            
            if dist_eyes < 5: #15: #20
                print "image ", filename, "has a too small face: dist_eyes = %f pixels"%dist_eyes
            else:
                for repetition in range(num_tries):    
                    im2 = normalize_image(filename, float_coords, normalization_method=normalization_method, centering_mode=centering_mode, rotation_mode=rotation_mode, integer_rotation_center=integer_rotation_center, out_size = out_size, convert_format=convert_format, verbose=False, allow_random_background=allow_random_background)
                    if im2 == None:
                        print "image ", filename, "was not properly normalized"
                    else:
                        if output_pattern == "None":
                            filename_components = string.split(filename, sep="/")
                            filename_short = prefix+filename_components[-1]
#                            print "filename_short is:", filename_short
                            im2.save(out_dir+filename_short, "JPEG", quality=90)                       
                        else:
#                            skip
                            im2.save(out_dir+output_pattern%count, "JPEG", quality=90)
                        count += 1
    normalization_file.close() 

#Example python facenormalization.py Caltech_coordinates.txt image%05d.jpg mid_eyes_mouth
#Output files are in the directory normalized for this mode, otherwise in directory no Face
