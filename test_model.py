
import torch.optim
from DataSet_loader import ValGenerator, ImageToImage2D
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")
import Config as config
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from nets.UNet import UNet
from utils import *
import cv2


def show_image_with_dice(predict_save, labs, save_path):
    tmp_lbl = (labs).astype(np.float32)
    tmp_3dunet = (predict_save).astype(np.float32)
    dice_pred = 2 * np.sum(tmp_lbl * tmp_3dunet) / (np.sum(tmp_lbl) + np.sum(tmp_3dunet) + 1e-5)
    # dice_show = "%.3f" % (dice_pred)
    iou_pred = jaccard_score(tmp_lbl.reshape(-1),tmp_3dunet.reshape(-1))
    # fig, ax = plt.subplots()
    # plt.gca().add_patch(patches.Rectangle(xy=(4, 4),width=120,height=20,color="white",linewidth=1))
    if config.task_name is "Crag":
        predict_save = cv2.pyrUp(predict_save,(224,224))
        predict_save = cv2.resize(predict_save,(224,224))
        # kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32) #定义一个核
        # predict_save = cv2.filter2D(predict_save, -1, kernel=kernel)
        cv2.imwrite(save_path,predict_save * 255)
    else:
        cv2.imwrite(save_path,predict_save * 255)
    # plt.imshow(predict_save * 255,cmap='gray')
    # plt.text(x=10, y=24, s="Dice:" + str(dice_show), fontsize=5)
    # plt.axis("off")
    # remove the white borders
    # height, width = predict_save.shape
    # fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
    # plt.gca().xaxis.set_major_locator(plt.NullLocator())
    # plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
    # plt.margins(0, 0)
    # plt.savefig(save_path, dpi=2000)
    # plt.close()
    return dice_pred, iou_pred

def vis_and_save_heatmap(model, input_img, img_RGB, labs, vis_save_path, dice_pred, dice_ens):
    model.eval()

    output = model(input_img.cuda())
    pred_class = torch.where(output>0.5,torch.ones_like(output),torch.zeros_like(output))
    predict_save = pred_class[0].cpu().data.numpy()
    predict_save = np.reshape(predict_save, (config.img_size, config.img_size))
    dice_pred_tmp, iou_tmp = show_image_with_dice(predict_save, labs, save_path=vis_save_path+'_predict'+model_type+'.jpg')
    return dice_pred_tmp, iou_tmp



if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    test_session = config.test_session
    if config.task_name is "Gl25":
        test_num = 33
        model_type = config.model_name
        # model_path = "./Glas/" + model_type + "/" + test_session + "/models/model-" + model_type + "-192" + ".pth"
        model_path = "./Gl25/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth"

    elif config.task_name is "Crag25":
        test_num = 44
        model_type = config.model_name
        model_path = "./Crag25/" + model_type + "/" + test_session + "/models/best_model-" + model_type + ".pth"
        # model_path = "./Crag/" + model_type + "/" + test_session + "/models/model-" + model_type +"-216" + ".pth"

    elif config.task_name is "T20":
        test_num = 7
        model_type = config.model_name
        model_path = "./T20/" + model_type + "/" + test_session + "/models/model-" + model_type + "-65.pth.tar"

        # model_path = "./MoNuSeg/" + model_type + "/" + test_session + "/models/best_model-" + model_type + "13.pth.tar"

    model_type = 'UNet'
    save_path  = config.task_name +'/'+ model_type +'/' + test_session + '/'
    vis_path = "./" + config.task_name + '_visualize_test/'
    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    checkpoint = torch.load(model_path, map_location='cuda')


    if model_type == 'UNet':
        model = UNet(3,1)

    elif model_type == 'UCTransNet_pretrain':
        config_vit = config.get_CTranS_config()
        model = UCTransNet(config_vit,n_channels=config.n_channels,n_classes=config.n_labels)

    else: raise TypeError('Please enter a valid name for the model type')

    model = model.cuda()
    if torch.cuda.device_count() > 1:
        print ("Let's use {0} GPUs!".format(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=[0,1])
    model.load_state_dict(checkpoint['state_dict'])
    print('Model loaded !')
    tf_test = ValGenerator(output_size=[config.img_size, config.img_size])
    test_dataset = ImageToImage2D(config.test_dataset, tf_test,image_size=config.img_size)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    dice_pred = 0.0
    iou_pred = 0.0
    dice_ens = 0.0

    with tqdm(total=test_num, desc='Test visualize', unit='img', ncols=70, leave=True) as pbar:
        for i, (sampled_batch, names) in enumerate(test_loader, 1):
            test_data, test_label = sampled_batch['image'], sampled_batch['label']
            arr=test_data.numpy()
            arr = arr.astype(np.float32())
            lab=test_label.data.numpy()
            img_lab = np.reshape(lab, (lab.shape[1], lab.shape[2])) * 255
            fig, ax = plt.subplots()
            plt.imshow(img_lab, cmap='gray')
            plt.axis("off")
            height, width = config.img_size, config.img_size
            fig.set_size_inches(width / 100.0 / 3.0, height / 100.0 / 3.0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            plt.subplots_adjust(top=1, bottom=0, left=0, right=1, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(vis_path+str(i)+"_lab.png", dpi=2000)
            plt.close()
            input_img = torch.from_numpy(arr)
            dice_pred_t,iou_pred_t = vis_and_save_heatmap(model, input_img, None, lab,
                                                          vis_path+str(i),
                                               dice_pred=dice_pred, dice_ens=dice_ens)
            dice_pred+=dice_pred_t
            iou_pred+=iou_pred_t
            torch.cuda.empty_cache()
            pbar.update()
    print ("dice_pred",dice_pred/test_num)
    print ("iou_pred",iou_pred/test_num)





