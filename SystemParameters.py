import mdp
import more_nodes
import patch_mdp


TOP_LEFT_CORNER = 0

#from  eban_SFA_libs import *

#Attention: Parameters here are the only ones returned as significant parameters (for cacheing)
#the function __values__() returns the elements that define the hash of the object
class ParamsNetwork (object):
    def __init__(self):
        self.name = "test hierarchical network"
        self.L0 = None
        self.L1 = None
        self.L2 = None
        self.L3 = None
        self.L4 = None
        self.L5 = None
        self.L6 = None
        self.L7 = None
        self.L8 = None
        self.L9 = None
        self.L10 = None
    def __values__(self):
        return (self.L0, self.L1, self.L2, self.L3, self.L4, self.L5)
        
class ParamsSystem (object):
    def __init__(self):
        self.name = "test system"
        self.network = None
        self.iTraining = None
        self.sTraining = None
        self.iSeenid = None
        self.sSeenid = None
        self.iNewid = None
        self.sNewid = None
        self.analysis =  None
        self.block_size = None
        self.train_mode = None
        #Warning: just guessing common values here
        self.enable_reduced_image_sizes = True
        self.reduction_factor = 2.0
        self.hack_image_size = 64
        self.enable_hack_image_size = True
        
    def __values__(self):
        return (self.network, self.iTraining, self.sTraining, self.iSeenid, self.sSeenid, self.iNewid, self.sNewid)
        
class ParamsInput(object):
    def __init__(self):
        self.name = "test input"
        self.data_base_dir = None
        self.ids = None
        self.ages = [999]
        self.MIN_GENDER = -3
        self.MAX_GENDER =  3
        self.GENDER_STEP = 0.1
        self.genders = [0]
        self.racetweens = [999]
        self.expressions =  [0]
        self.morphs = [0]
        self.poses =  [0]
        self.lightings =  [0]
        self.slow_signal =  0
        self.step =  1
        self.offset = 0
        self.correct_labels = None
        self.correct_classes = None
#    def __values__(self):
#        return (self.network, self.iTraining, self.sTraining, self.iSeenid, self.sSeenid, self.iNewid, self.sNewid)


class ParamsDataLoading(object):
    def __init__(self):
        self.name = "test input data"
        self.input_files = []
        self.num_images = 0
        self.image_width = 256
        self.image_height = 192
        self.subimage_width = 135
        self.subimage_height = 135 
        self.pixelsampling_x = 1
        self.pixelsampling_y =  1
        self.subimage_pixelsampling = 2
        self.subimage_first_row =  0
        self.subimage_first_column = 0
        self.subimage_reference_point = TOP_LEFT_CORNER 
        self.add_noise_L0 = True
        self.convert_format = "L"
        self.background_type = None
        self.translation = 0
        self.translations_x = None
        self.translations_y = None
        self.trans_sampled = True
       
#SFALayer: PInvSwitchboard, pca_node, ord_node, gen_exp, red_node, clip_node, sfa_node
class ParamsSFALayer(object):
    def __init__(self):
        self.name = "SFA Layer"
        self.x_field_channels=3
        self.y_field_channels=3
        self.x_field_spacing=3
        self.y_field_spacing=3
        self.nx_value = None
        self.ny_value = None
        
        self.in_channel_dim = 1
        self.pca_node_class = None
        self.pca_out_dim = 0.99999
        self.pca_args = {"block_size": 1}

        self.ord_node_class = None
        self.ord_args = {}

        self.exp_funcs = None
        self.inv_use_hint = True
        self.inv_max_steady_factor=0.35
        self.inv_delta_factor=0.6
        self.inv_min_delta=0.0001

        self.red_node_class = None
        self.red_out_dim = 0.99999
        self.red_args = {"block_size": 1, "cutoff": 4}

        self.clip_func = None
        self.clip_inv_func = None
#        self.clip_func = lambda x: clipping_sigma(x, 4)
#        self.clip_inv_func = lambda x: inv_clipping_sigma(x, 4)

        self.sfa_node_class = None
        self.sfa_out_dim = 15
        self.sfa_args = {"block_size": 1, "cutoff": 4}
        self.cloneLayer = True
        self.node_list = None
        self.layer_number = None
        
class ExecSFALayer(object):
    def __init__(self):
        self.name = "SFA Layer"
        self.params = None

#SFASuperNode: pca_node, ord_node, gen_exp, red_node, clip_node, sfa_node       
class ParamsSFASuperNode(object):
    def __init__(self):
        self.name = "SFA Supernode"
        self.in_channel_dim=1
        self.pca_node_class = mdp.nodes.WhiteningNode
        self.pca_out_dim = 0.99999
        self.pca_args = {"block_size": 1}
        self.ord_node_class = None
        self.ord_args = {}
        self.exp_funcs = None
        self.inv_use_hint = True
        self.inv_max_steady_factor=0.35
        self.inv_delta_factor=0.6
        self.inv_min_delta=0.0001
        self.red_node_class = mdp.nodes.WhiteningNode
        self.red_out_dim = 0.99999
        self.red_args = {"block_size": 1, "cutoff": 4}
        self.clip_func = None
        self.clip_inv_func = None
#        self.clip_func = lambda x: clipping_sigma(x, 100.5)
#        self.clip_inv_func = lambda x: inv_clipping_sigma(x, 100.5)
        self.sfa_node_class = mdp.nodes.SFANode
        self.sfa_out_dim = 15
        self.sfa_args = {"block_size": 1, "cutoff": 4}
#        self.block_size = block_size
#class ParamsSFALayer(object):
#    def __init__(self):
#        self.name = "SFA Layer"
#        self.x_field_channels=3
#        self.y_field_channels=3
#        self.x_field_spacing=3
#        self.y_field_spacing=3
#        self.in_channel_dim=1
#        self.pca_node = mdp.nodes.WhiteningNode
#        self.pca_out_dim = 0.99999
#        self.pca_args = {"block_size": 1}
#        self.exp_funcs = None
#        self.red_node = mdp.nodes.WhiteningNode
#        self.red_out_dim = 0.99999
#        self.red_args = {"block_size": 1, "cutoff": 4}
#        self.sfa_node = mdp.nodes.SFANode
#        self.sfa_out_dim = 15
        self.node_list = None

class ExecSFASuperNode(object):
    def __init__(self):
        self.name = "SFA Layer"
        self.params = None
        
#CODE PENDING!!!!!!!!!!
class ExperimentResult(object):
    def __init__(self):
        self.name = "Simulation Results" 
        self.network_name = None
        self.layers_name = None
        
        self.reg_num_signals = None

        self.iTrain = None
        self.sTrain = None
        self.typical_delta_train = None
        self.typical_eta_train = None
        self.brute_delta_train  = None
        self.brute_eta_train = None
        self.class_rate_train = None
        self.mse_train = None
        self.msegauss_train= None

        self.iSeenid = None
        self.sSeenid = None
        self.typical_delta_seenid = None
        self.typical_eta_seenid = None
        self.brute_delta_seenid  = None
        self.brute_eta_seenid = None
        self.class_rate_seenid = None
        self.mse_seenid = None
        self.msegauss_seenid= None


        self.iNewid = None
        self.sNewid = None
        self.typical_delta_newid = None
        self.typical_eta_newid = None
        self.brute_delta_newid  = None
        self.brute_eta_newid = None
        self.class_rate_newid = None
        self.mse_newid = None
        self.msegauss_newid= None

#class_rate_train = correct_classif_rate(correct_classes_training, classes_training)
#mse_train = distance_squared_Euclidean(correct_labels_training, labels_training)/len(labels_training)
#msegauss_train = distance_squared_Euclidean(correct_labels_training, regression_training)/len(labels_training)
#
#class_rate_seenid = correct_classif_rate(correct_classes_seenid, classes_seenid)
#mse_seenid = distance_squared_Euclidean(correct_labels_seenid, labels_seenid)/len(labels_seenid)
#msegauss_seenid = distance_squared_Euclidean(correct_labels_seenid, regression_seenid)/len(labels_seenid)
#
#class_rate_newid = correct_classif_rate(correct_classes_newid, classes_newid)
#mse_newid = distance_squared_Euclidean(correct_labels_newid, labels_newid)/len(labels_newid)
#msegauss_newid = distance_squared_Euclidean(correct_labels_newid, regression_newid)/len(labels_newid)



class NetworkOutputs(object):       
    def __init__(self):
        self.num_samples = 0
        self.sl = []
        self.correct_classes = []
        self.correct_labels = []
        self.classes = []
        self.labels = []
        self.block_size = []
        self.eta_values = []
        self.delta_values = []
        self.class_rate = 0
        self.gauss_class_rate = 0
        self.reg_mse = 0
        self.gauss_reg_mse = 0
        




def test_object_contents(object):
    dict = object.__dict__
    list_none_elements = []
    for w in dict.keys():
        if dict[w] == None:
            list_none_elements.append(str(w))
    if len(list_none_elements) > 0:
        print "Warning!!! object %s contains 'None' fields: "%(str(object)), list_none_elements



    