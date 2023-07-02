import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from args import parser
import os
import PIL.Image as Image
import imageio
from pathlib import Path
from tqdm import tqdm

connectivity_dict = {"H36M":[[0, 2, 1], [1, 2, 1], [2, 3, 1], [3, 4, 1], [4, 12, 1],#left arm
                             [5, 12, 0], [5, 6, 0], [6, 7, 0], [7, 8, 0], [7, 9, 0],#right arm
                              [10, 11, 0], [11, 12, 0], [12, 13, 0],#torso
                              [14, 13, 0],  [14, 15, 0], [15, 16, 0], [16, 17, 0],#left leg
                               [18, 13, 1],[18, 19, 1],[19, 20, 1], [20, 21, 1]]#right leg
                    ,"CMU":[[0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 4, 1], [4, 5, 1], [5, 15, 1],#left arm
                             [6, 7, 0], [7, 8, 0], [8, 9, 0], [9, 10, 0], [10, 11, 0], [11, 15, 0],#right arm
                              [12, 13, 0], [13, 14, 0], [14, 15, 0], [15, 16, 0],#torso
                              [16, 17, 1],  [17, 18, 1], [18, 19, 1], [19, 20, 1],#left leg
                               [16, 21, 0],[21, 22, 0],[22, 23, 0], [23, 24, 0]]#right leg
                    ,"3DPW":[[11, 5, 0], [5, 6, 0], [6, 7, 0], [7, 8, 0], [8, 9, 0],  # left arm
                                [14, 19, 0], [19, 20, 0], [20, 21, 0], [21, 22, 0], # left leg
                                [10, 11, 0], [11, 12, 0], [12, 13, 0], [13, 14, 0], # torso
                                [0, 1, 1], [1, 2, 1], [2, 3, 1], [3, 4, 1], [4, 11, 1], # right arm
                                [14, 15, 1], [15, 16, 1], [16, 17, 1], [17, 18, 1], # right leg
                                ]}

parser.add_argument("--activity", type=str, default="walking", help="The activity you want to draw")
opt = parser.parse_args()

def draw3Dpose(pred_3d, gt_3d, ax, dataset):

    for i in connectivity_dict[dataset]:
        x, y, z = [np.array([pred_3d[i[0], j], pred_3d[i[1], j]]) for j in range(3)]
        linestyle='-'
        lcolor="#3498db"#blue
        rcolor="#e74c3c"#orange
        ax.plot(x, y, z, lw=2, linestyle=linestyle, c=lcolor if i[2] else rcolor)

        x, y, z = [np.array([gt_3d[i[0], j], gt_3d[i[1], j]]) for j in range(3)]
        linestyle='dotted'
        lcolor="indigo"
        rcolor="darkred"
        ax.plot(x, y, z, lw=2, linestyle=linestyle, c=lcolor if i[2] else rcolor)
    
    if dataset=="H36M" or dataset=="CMU":
        RADIUS = 800  # space around the subject
    elif dataset=="3DPW":
        RADIUS = 0.8
    
    xroot, yroot, zroot = gt_3d[13, 0], gt_3d[13, 1], gt_3d[13, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")


# 图片背景透明化
def transPNG(img):
    # img = img.convert("RGBA")
    datas = img.getdata()
    newData = list()
    for item in datas:
        if item[0] > 230 and item[1] > 230 and item[2] > 230:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    return img


# 图片背景由黑变白
def transPNGbw(img):
    # img = img.convert("RGBA")
    datas = img.getdata()
    newData = list()
    for item in datas:
        if item[0] < 25 and item[1] < 25 and item[2] < 25:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)
    img.putdata(newData)
    return img


def image_compose(IMAGES_PATH, IMAGE_SIZE, IMAGE_COLUMN, IMAGE_ROW, IMAGE_SAVE_PATH,
                  LEFT_DIST, RIGHT_DIST, UPPER_DIST, LOWER_DIST, OVERLOP):
    """图像拼接函数"""
    if os.path.exists(IMAGE_SAVE_PATH):
        os.remove(IMAGE_SAVE_PATH)
    aft_image = Image.new('RGB', (IMAGE_COLUMN * (IMAGE_SIZE[0]-LEFT_DIST-RIGHT_DIST) - OVERLOP*(IMAGE_COLUMN-1),
                                  IMAGE_ROW * (IMAGE_SIZE[1]-UPPER_DIST-LOWER_DIST)))  # 创建一个新图
    file_list_gt = os.listdir(IMAGES_PATH)
    file_list_gt.sort(key=lambda x: int(x.split('_')[-1][:-4]))  # .png所以是[:-4]
    image_names_gt = [name for name in file_list_gt]

    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, IMAGE_ROW+1):
        for x in range(1, IMAGE_COLUMN+1):
            from_image_gt = Image.open(IMAGES_PATH + image_names_gt[IMAGE_COLUMN * (y - 1) + x - 1]).resize(
                (IMAGE_SIZE[0], IMAGE_SIZE[1]), Image.Resampling.LANCZOS).crop(
                (LEFT_DIST, UPPER_DIST, IMAGE_SIZE[0]-RIGHT_DIST, IMAGE_SIZE[1]-LOWER_DIST))
            from_image_gt = transPNG(from_image_gt)
            r, g, b, a = from_image_gt.split()
            aft_image.paste(from_image_gt, ((x - 1) * (IMAGE_SIZE[0] - LEFT_DIST - RIGHT_DIST - OVERLOP),
                                         (y - 1) * (IMAGE_SIZE[1] - UPPER_DIST - LOWER_DIST)), mask=a)
            aft_image = transPNGbw(aft_image)

    return aft_image.save(IMAGE_SAVE_PATH)  # 保存新图

def imgs2gif(imgPaths, saveName, duration=None, loop=0, fps=None):
    """
    Generate animated poses as GIF
    :param imgPaths: Path to images
    :param saveName: The name of the GIF
    :param duration: The duration time of each frame
    :param fps: The frame rate(you may choose either duration or fps to decide the frame rate of the GIF)
    :param loop: The loops the GIF plays, 0 means endless looping
    :return:
    """
    if fps:
        duration = 1 / fps
    images = [imageio.v2.imread(str(img_path)) for img_path in imgPaths]
    imageio.mimsave(saveName, images, "gif", duration=duration, loop=loop)

if __name__ == '__main__':
    assert opt.dataset in ["H36M", "CMU", "3DPW"]
    if opt.dataset == "3DPW":
        specific_3d_skeleton_truth = np.load('./seq/truth_seq_'+opt.dataset+'.npy')# (batch_size, frame_num, joint_num, 3)
        specific_3d_skeleton_clean = np.load('./seq/clean_seq_'+opt.dataset+'.npy')  
        specific_3d_skeleton_pert = np.load('./seq/pert_seq_'+opt.dataset+'.npy')
    else:
        specific_3d_skeleton_truth = np.load('./seq/truth_seq_'+opt.dataset+'_'+opt.activity+'.npy')# (batch_size, frame_num, joint_num, 3)
        specific_3d_skeleton_clean = np.load('./seq/clean_seq_'+opt.dataset+'_'+opt.activity+'.npy')  
        specific_3d_skeleton_pert = np.load('./seq/pert_seq_'+opt.dataset+'_'+opt.activity+'.npy')
    img_dir = os.path.join("./{}".format("img_"+opt.dataset))
    batch_size, frame_num, joint_num, _ = specific_3d_skeleton_truth.shape
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()

    #Only draw 8 or less sequence
    if batch_size>=8:
        seq_num = 8
    else:
        seq_num = batch_size

    for i in tqdm(range(seq_num), desc='Draw images for each pose'):
        for j in range(frame_num):
            #Print each poses
            #Process the ground truth
            specific_3d_skeleton_truth[i][j] = specific_3d_skeleton_truth[i][j][:, [0, 2, 1]]
            if opt.dataset == "H36M":
                ax.view_init(elev=30, azim=60)
            elif opt.dataset == "CMU":
                ax.view_init(elev=15, azim=45)
            elif opt.dataset == "3DPW":
                ax.view_init(elev=15, azim=180)
            plt.axis('off')
            #For clean results and truth
            fig_path = os.path.join(img_dir, 'clean_batch_'+str(i))
            ax.lines.clear()  
            specific_3d_skeleton_clean[i][j] = specific_3d_skeleton_clean[i][j][:, [0, 2, 1]]
            draw3Dpose(specific_3d_skeleton_clean[i][j], specific_3d_skeleton_truth[i][j], ax, opt.dataset)
              
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            plt.savefig(os.path.join(fig_path, 'clean_'+str(j)+'.png'))

            #For perturbed results and truth
            fig_path = os.path.join(img_dir, 'pert_batch_'+str(i))
            # print('Saving to: '+fig_path)
            ax.lines.clear()  
            specific_3d_skeleton_pert[i][j] = specific_3d_skeleton_pert[i][j][:, [0, 2, 1]]
            draw3Dpose(specific_3d_skeleton_pert[i][j], specific_3d_skeleton_truth[i][j], ax, opt.dataset) 
            if not os.path.exists(fig_path):
                os.makedirs(fig_path)
            plt.savefig(os.path.join(fig_path, 'pert_'+str(j)+'.png'))
            
    for i in tqdm(range(seq_num),desc='Draw images & GIFs for each sequence'):
        fig_path_clean = os.path.join(img_dir, 'clean_batch_' + str(i)+'/')# the address where the poses' images are
        fig_path_pert = os.path.join(img_dir, 'pert_batch_' + str(i)+'/')
        size = [640, 480]  # the size of each image
        
        row = frame_num 
        column = 1  
        save_path = os.path.join(img_dir,'All_poses')  
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_path_clean = os.path.join(img_dir,'All_poses','clean_'+ str(i)+'.png')
        save_path_pert = os.path.join(img_dir,'All_poses','pert_'+ str(i)+'.png')
        left_dist = 180  
        right_dist = 180  
        upper_dist = 120  
        lower_dist = 20  
        overlap = 50  
        image_compose(fig_path_clean, size, row, column, save_path_clean, left_dist, right_dist, upper_dist, lower_dist, overlap)
        image_compose(fig_path_pert, size, row, column, save_path_pert, left_dist, right_dist, upper_dist, lower_dist, overlap)

        #Generate gif
        if opt.dataset == "H36M" or opt.dataset == "CMU":
            frame_rate = 25
        elif opt.dataset == "3DPW":
            frame_rate = 30
        p_lis=[]
        for j in range(frame_num):
            p_lis.append(os.path.join(img_dir, 'clean_batch_' + str(i)+'/clean_'+str(j)+'.png'))

        imgs2gif(p_lis, os.path.join(img_dir, 'clean_batch_' + str(i)+'.gif'), None, 0, frame_rate)
        # print('GIFs of the clean sequences is saved')

        p_lis=[]
        for j in range(frame_num):
            p_lis.append(os.path.join(img_dir, 'pert_batch_' + str(i)+'/pert_'+str(j)+'.png'))

        imgs2gif(p_lis, os.path.join(img_dir, 'pert_batch_' + str(i)+'.gif'), None, 0, frame_rate)
        # print('GIFs of the perturbed sequences is saved')

