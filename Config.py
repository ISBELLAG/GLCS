
import os
import torch
import time
import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

use_cuda = torch.cuda.is_available()
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)

cosineLR = True # whether use cosineLR or not
n_channels = 3
n_labels = 1
epochs = 2000
img_size = 224
print_frequency = 1
save_frequency = 5000
vis_frequency = 10
early_stopping_patience = 50
lr  = 0.00004
Dlr = 0.0004
pretrain = False
task_name = 'Crag50' # GlaS MoNuSeg

learning_rate = 1e-3
dlearning_rate= 1e-4
batch_size = 2

model_type = 'UNet'
# model_name = 'UCTransNet'
model_name = 'UNet'
discriminator_show = './show' + task_name + '/Discriminator_show'
image_show= './datasets/' + task_name + '/image_show'
# discriminator_show = './datasets/' + task_name + '/Discriminator_show'
sup_dataset = './datasets/'+ task_name+ '/Train_Folder/sup'
unsup_dataset = './datasets/'+ task_name+ '/Train_Folder/unsup'

val_dataset = './datasets/'+ task_name+ '/Val_Folder/'
test_dataset = './datasets/'+ task_name+ '/Test_Folder/'

# test_dataset = './datasets/'+ task_name
# test_dataset = sup_dataset = './datasets/'+ task_name+ '/Test_Folder/'
session_name       = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path          = task_name +'/'+ model_name +'/' + session_name + '/'
db_save_path       = task_name +'/'+ model_name +'/' + session_name + '/'
model_path         = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path        = save_path + session_name + ".log"
visualize_path     = save_path + 'visualize_val/'




# used in testing phase, copy the session name in training phase
test_session = "Test_session_05.28_14h57"

