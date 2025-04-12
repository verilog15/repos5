import random
import warnings
import cv2
import kornia
import torch.nn
import torch.nn.functional as F
import torch.optim
from nvidia import cudnn
from sklearn.metrics import roc_auc_score
from utils import *
from network.discriminator import Discriminator
from network.distortions.deepfakes.FaceShifter.face_modules.model import Backbone
from network.distortions.deepfakes.FaceShifter.face_modules.mtcnn import MTCNN
from network.distortions.deepfakes.FaceShifter.network.AEI_Net import AEI_Net
from FaceSecurity.BBW.network.distortions.deepfakes.simswap.obfuscate import SimSwap
from network.distortions.deepfakes.starganV2master.StarGANV2 import StarGANv2
from utils import basedatasets
from utils.img_utils import *
from lpips import lpips
from torchvision.transforms import transforms
from network.Vector import vector_var
from network.model import *
from network.Unet_common import *
from PIL import Image
import torchvision.transforms as T
from torchmetrics.classification import BinaryAccuracy
from torchvision import datasets
from torchmetrics import ROC, F1Score, Recall



def load(name):
    state_dicts = torch.load(name)
    print(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    net.load_state_dict(network_state_dict)
    discriminator.load_state_dict(state_dicts['discriminator'])
    template_init.load_state_dict(state_dicts['template_init'])

    try:
        optim.load_state_dict(state_dicts['opt'])
        optim_d.load_state_dict(state_dicts['optim_d'])
        optim_template.load_state_dict(state_dicts['optim_template_init'])

    except:
        print('Cannot load optimizer for some reason or other')


def gauss_noise(shape):
    noise = torch.zeros(shape).to(c.device)
    for i in range(noise.shape[0]):
        noise[i] = torch.randn(noise[i].shape).to(c.device)
    return noise


def image_resize(steg_img, down_scale=0.5):
    image = steg_img
    down = F.interpolate(image, size=(int(down_scale * image.shape[2]), int(down_scale * image.shape[3])),
                         mode='nearest')
    up = F.interpolate(down, size=(image.shape[2], image.shape[3]), mode='nearest')
    return up


# cv2 真实jpeg压缩
def image_jpeg(img, quality):
    B, _, _, _ = img.shape
    img = img.mul(255).add_(0.5).clamp_(0, 255)
    quality = torch.tensor(quality).repeat(B, 1).to(device)

    # Init list to store compressed images
    image_rgb_jpeg = []
    # Code each image
    for index in range(B):
        # Make encoding parameter
        encode_parameters = (int(cv2.IMWRITE_JPEG_QUALITY), int(quality[index].item()))
        # Encode image note CV2 is using [B, G, R]
        _, encoding = cv2.imencode(".jpeg", img[index].flip(0).permute(1, 2, 0).cpu().numpy(), encode_parameters)
        image_rgb_jpeg.append(torch.from_numpy(cv2.imdecode(encoding, 1)).permute(2, 0, 1).flip(0))

    # Stack images
    image_rgb_jpeg = torch.stack(image_rgb_jpeg, dim=0).float() / 255

    return image_rgb_jpeg.to(device)


def to_rgb(image):
    rgb_image = Image.new("RGB", image.size)
    rgb_image.paste(image)
    return rgb_image


def read_img(src):
    image = Image.open(src)
    image = to_rgb(image)
    return transform_val(image)


def image_blur(steg_img, blur_sigma):
    blur_transform = kornia.augmentation.RandomGaussianBlur(kernel_size=(3, 3), sigma=(blur_sigma, blur_sigma), p=1)
    return blur_transform(steg_img)


def image_gaussnoise(image, std=0.1):
    transform_gauss = kornia.augmentation.RandomGaussianNoise(mean=0, std=std, p=1)
    return transform_gauss(image)


def image_identity(image, param=None):
    return image


# classes = 0->test,1->val
def get_target(input_dataset, classes=0, size=8):
    index = [i for i, t in enumerate(input_dataset.targets) if t == classes]
    rand_inds = np.random.choice(index, size=size, replace=False)
    imgs = [input_dataset[i][0] for i in rand_inds]
    return torch.stack(imgs).to(c.device)


# classes = 0->test,1->val 固定target
def get_image_by_class_and_index(target_dataset, classes, idx):
    # 找到所有属于指定类别的图片索引
    indices = [i for i, target in enumerate(target_dataset.targets) if target == classes]
    # 使用模运算获取有效的索引
    valid_idx = idx % len(indices)
    target_index = indices[valid_idx]  # 防止 valid_idx 超出当前类别的图片数量

    # 提取并返回对应的图片
    image, label = target_dataset[target_index]
    return torch.tensor(image).unsqueeze(0).to(device)


def random_black(img):
    mask = torch.ones_like(img)
    w = random.randint(50, 70)
    h = random.randint(50, 70)
    x = random.randint(0, img.size()[2] - w)
    y = random.randint(0, img.size()[3] - h)
    # 将随机区域置为0
    mask[:, :, y:y + h, x:x + w] = 0
    return img * mask


def all_distortion(img, step):
    # 定义所有变换方法和参数 ,共27中变换方法
    transformations = [
        (image_blur, [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]),
        (image_gaussnoise, [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]),
        (image_resize, [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]),
        (image_jpeg, [50, 75, 95]),
        (image_identity, [None])
    ]

    # 生成一个扁平化的变换方法和参数列表
    flat_transformations = [(func, param) for func, params in transformations for param in params]

    # 根据图片的下标选择变换方法和参数
    total_transforms = len(flat_transformations)
    index = step % total_transforms
    func, param = flat_transformations[index]

    # 执行选择的变换
    transformed_img = func(img, param)

    return transformed_img


def nonLinearTrans(input_image):
    input_image = input_image + 0.5
    input_image = torch.pow(input_image, 2.0)
    return input_image.clamp(0, 1)


def simswap(source, idx):
    # classes = 0->test,1->val
    target = get_image_by_class_and_index(target_dataset, classes=1, idx=idx)

    res = torch.empty(source.size())
    for i in range(source.size()[0]):
        res[i, :, :, :] = swap_model.forward(source[i], target[i])
    return res.to(c.device)


def face_shifter(xt, idx):
    xs = get_image_by_class_and_index(target_dataset, classes=1, idx=idx)  # target->xs
    xt = face_shifter_transform(xt)
    xs = face_shifter_transform(xs)
    _, _, w, h = xt.size()
    with torch.no_grad():
        embeds = arcface(F.interpolate(xs, (112, 112), mode='bilinear', align_corners=True))
        yt, _ = G(xt, embeds)
        return (yt + 1) / 2


def starGANV2_swap(src, idx):
    ref_id = random.randint(0, 1)
    ref = get_image_by_class_and_index(startGAN_target_dataset, classes=ref_id, idx=idx)
    return StarGANV2_model.forward(src, ref, ref_id)

def seed_torch(seed=25):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)


def decoded_message_error_rate(message, decoded_message):
    length = message.shape[0]
    message = message.gt(0)
    decoded_message = decoded_message.gt(0)
    error_rate = float(sum(message != decoded_message)) / length
    return error_rate


def decoded_message_error_rate_batch(messages, decoded_messages):
    error_rate = []
    batch_size = len(messages)
    for i in range(batch_size):
        error_rate.append(decoded_message_error_rate(messages[i], decoded_messages[i]))
    return error_rate


def test(trans_fun, param, swap_fun, use_one_distortion):
    with torch.no_grad():
        fprAll = []
        tprAll = []
        aucAll = []
        psnr_c = []
        psnr_s = []
        psnr_s_f = []
        ssim_c = []
        ssim_s = []
        lpips_c = []
        lpips_s = []

        net.eval()
        discriminator.eval()
        template_init.eval()

        acc.reset()
        F1.reset()
        recall.reset()

        saved_iterations = np.random.choice(np.arange(1, len(basedatasets.testloader) + 1), size=3, replace=False)
        saved_all = None

        for i, data in enumerate(basedatasets.testloader):
            cover = data.to(c.device)
            secret = template.repeat([cover.size()[0], 1, 1, 1])

            cover_input = dwt(cover)
            secret_input = dwt(secret)

            input_img = torch.cat((cover_input, secret_input), 1)

            #################
            #    forward:   #
            #################
            output = net(input_img)
            output_steg = output.narrow(1, 0, 4 * c.channels_in)

            steg = iwt(output_steg)

            if use_one_distortion:
                steg_nm = trans_fun(steg, param)
            else:
                steg_nm = all_distortion(steg, step=i)

            steg_nm_input_val = dwt(steg_nm)

            output_z = output.narrow(1, 4 * c.channels_in, output.shape[1] - 4 * c.channels_in)
            output_z = gauss_noise(output_z.shape)

            try:
                steg_mp_val = swap_fun(steg, i)
            except Exception as e:
                print(e)
                continue

            if use_one_distortion:
                steg_mp_val = trans_fun(steg_mp_val, param)
            else:
                steg_mp_val = all_distortion(steg_mp_val, step=i)

            steg_mp_input_val = dwt(steg_mp_val).to(c.device)

            #################
            #   backward:   #
            #################
            input_rev = torch.cat((steg_nm_input_val, output_z), 1)
            input_rev_mp_val = torch.cat((steg_mp_input_val, output_z), 1)

            output_image = net(input_rev, rev=True)
            output_image_mp = net(input_rev_mp_val, rev=True)

            secret_rev_val = output_image.narrow(1, 4 * c.channels_in, output_image.shape[1] - 4 * c.channels_in)
            secret_rev_val = iwt(secret_rev_val)

            secret_rev_mp_val = output_image_mp.narrow(1, 4 * c.channels_in, output_image_mp.shape[1] - 4 * c.channels_in)
            secret_rev_mp_val = iwt(secret_rev_mp_val)

            #################
            # discriminator #
            #################

            diff_nm_secret_val = secret - secret_rev_val
            diff_mp_secret_val = secret - secret_rev_mp_val

            # 真和假数据同时输入到预测模型中
            all_inputs = torch.cat((diff_nm_secret_val, diff_mp_secret_val), dim=0)
            target_labels = torch.cat((valid_label_val, fake_label_val), dim=0)

            preds = discriminator(all_inputs).reshape(-1)

            fpr, tpr, _ = roc(preds, target_labels)

            fprAll.append(fpr.cpu())
            tprAll.append(tpr.cpu())

            acc.update(preds, target_labels)
            F1.update(preds, target_labels)
            recall.update(preds, target_labels)

            aucAll.append(roc_auc_score(target_labels.cpu().numpy(), preds.cpu().numpy()))

            psnr_temp_c = - kornia.losses.psnr_loss(cover.detach(), steg.detach(), 2).cpu().numpy()
            psnr_c.append(psnr_temp_c)

            psnr_temp_s = - kornia.losses.psnr_loss(secret.detach(), secret_rev_val.detach(), 2).cpu().numpy()
            psnr_s.append(psnr_temp_s)

            psnr_temp_s_f = - kornia.losses.psnr_loss(secret.detach(), secret_rev_mp_val.detach(), 2).cpu().numpy()
            psnr_s_f.append(psnr_temp_s_f)

            ssim_temp = 1 - 2 * kornia.losses.ssim_loss(cover.detach(), steg, window_size=11, reduction="mean")
            ssim_c.append(ssim_temp.cpu().numpy())

            ssim_s_temp = 1 - 2 * kornia.losses.ssim_loss(secret.detach(), secret_rev_val.detach(), window_size=11,
                                                          reduction="mean")
            ssim_s.append(ssim_s_temp.cpu().numpy())

            lpips_temp = LPIPSLoss(cover.detach().to(c.device), steg.detach().to(c.device))
            lpips_c.append(lpips_temp.cpu().numpy())

            lpips_s_temp = LPIPSLoss(secret.detach().to(c.device), secret_rev_val.detach().to(c.device))
            lpips_s.append(lpips_s_temp.cpu().numpy())

            if i in saved_iterations:
                if saved_all is None:
                    saved_all = get_random_images(cover, secret, steg, steg_mp_val, diff_nm_secret_val,
                                                  diff_mp_secret_val)
                else:
                    saved_all = concatenate_images(saved_all, cover, secret, steg, steg_mp_val, diff_nm_secret_val,
                                                   diff_mp_secret_val)

        save_images(saved_all, './img', str(swap_fun))

        print(
            "acc={:.5f},fpr={:.4f},tpr={:.4f},F1={:.4f},recall={:.4f},auc={:.10f},psnr_c={:.4f},psnr_s={:.4f},psnr_s_f={:.4f},ssim_c={:.4f},lpips_c={:.4f},ssim_s={:.4f},lpips_s={:.4f}".format(
                acc.compute(),
                np.mean(fprAll),
                np.mean(tprAll),
                F1.compute(),
                recall.compute(),
                np.mean(aucAll),
                np.mean(psnr_c),
                np.mean(psnr_s),
                np.mean(psnr_s_f),
                np.mean(ssim_c),
                np.mean(lpips_c),
                np.mean(ssim_s),
                np.mean(lpips_s),
            )
        )


if __name__ == '__main__':

    seed_torch(25)
    warnings.filterwarnings("ignore")

    # init models
    net = Model().to(c.device)
    init_model(net)
    net = torch.nn.DataParallel(net, device_ids=c.device_ids)
    params_trainable = (list(filter(lambda p: p.requires_grad, net.parameters())))
    optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optim, c.weight_step, gamma=c.gamma)

    # simswap
    swap_model = SimSwap().to(c.device)
    data_transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量
    ])

    # starGANV2
    startGAN_target_dataset = datasets.ImageFolder(root='/home/cw/ysc/proFace/FaceSecurity/BBW/network/distortions/deepfakes/starganV2master/data/celeba_hq/val/', transform=data_transform)
    StarGANV2_model = StarGANv2().to(c.device)

    # faceshifter model init
    detector = MTCNN()
    G = AEI_Net(c_id=512).eval()
    G.load_state_dict(torch.load('/home/cw/ysc/proFace/FaceSecurity/BBW/network/distortions/deepfakes/FaceShifter/saved_models/G_latest.pth', map_location=device))
    G = G.to(device)
    arcface = Backbone(50, 0.6, 'ir_se').to(device).eval()
    arcface.load_state_dict(
        torch.load('/home/cw/ysc/proFace/FaceSecurity/BBW/network/distortions/deepfakes/FaceShifter/face_modules/model_ir_se50.pth', map_location=device),
        strict=False)
    face_shifter_transform = transforms.Compose([  # 传入faceshifter之前将数据从[0,1]->[-1,1]
        transforms.Normalize(mean=0.5, std=0.5)
    ])

    discriminator = Discriminator(input_shape=(c.channels_in, c.cropsize, c.cropsize)).to(device)
    optim_d = torch.optim.Adam(discriminator.parameters(), lr=c.discriminator_lr, weight_decay=c.weight_decay)

    # init template vector
    template_init = vector_var(size=c.cropsize).to(c.device)
    optim_template = torch.optim.Adam(template_init.parameters(), lr=c.template_lr, weight_decay=c.weight_decay)
    template = template_init()

    # init face swap target datasets
    target_dir = c.TARGET_PATH
    target_dataset = datasets.ImageFolder(root=target_dir, transform=data_transform)

    LPIPSLoss = lpips.LPIPS(net='vgg').to(c.device)

    dwt = DWT()
    iwt = IWT()

    threshold = 0.50

    acc = BinaryAccuracy(threshold=threshold).to(c.device)
    roc = ROC(task="binary", thresholds=[threshold]).to(c.device)
    F1 = F1Score(task="binary", threshold=threshold).to(c.device)
    recall = Recall(task="binary", threshold=threshold).to(c.device)

    # test
    valid_label_val = torch.ones(c.batchsize_val, dtype=torch.long, requires_grad=False).to(c.device)  # 真实标签，都是1
    fake_label_val = torch.zeros(c.batchsize_val, dtype=torch.long, requires_grad=False).to(c.device)  # 假标签，都是0

    transform_val = T.Compose([
        T.ToTensor(),
    ])

    use_one_distortion = True

    deepfake_func = [("simswap", simswap), ("face_shifter", face_shifter), ("starGANV2_swap", starGANV2_swap)]

    # 定义所有变换方法和参数
    transformations = [
        (image_jpeg, [55, 65, 75, 85, 95]),
        (image_blur, [2.00, 1.80, 1.60, 1.40, 1.20]),
        (image_gaussnoise, [0.05, 0.04, 0.03, 0.02, 0.01]),
        (image_resize, [0.50, 0.60, 0.70, 0.80, 0.90]),
    ]
    flat_transformations = [(func, param) for func, params in transformations for param in params]

    load(c.MODEL_PATH_FRE + c.suffix_fre)

    for j in range(len(deepfake_func)):
        for i in range(len(flat_transformations)):
            func, param = flat_transformations[i]
            print("****************************************")
            print(str(func).split(" ")[1] + " " + deepfake_func[j][0] + " " + str(param))
            test(trans_fun=func, param=param, swap_fun=deepfake_func[j][1], use_one_distortion=use_one_distortion)
