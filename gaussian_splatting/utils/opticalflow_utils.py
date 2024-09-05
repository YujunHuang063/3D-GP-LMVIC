import pdb
import sys
import math
import torch
from torch import nn
import torch.nn.functional as F

from utils.graphics_utils import torch_warp_simple

try:
    from .correlation import correlation # the custom cost volume layer
except:
    sys.path.insert(0, './correlation'); import correlation # you should consider upgrading python

def bilinearupsacling(inputfeature):
    inputheight = inputfeature.size(2)
    inputwidth = inputfeature.size(3)
    outfeature = F.interpolate(
        inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear', align_corners=False)

    return outfeature

def bilineardownsacling(inputfeature):
    inputheight = inputfeature.size(2)
    inputwidth = inputfeature.size(3)
    outfeature = F.interpolate(
        inputfeature, (inputheight // 2, inputwidth // 2), mode='bilinear', align_corners=False)
    return outfeature

class MEBasic(nn.Module):
    def __init__(self, complexity_level=0):
        super().__init__()
        self.relu = nn.ReLU()
        self.by_pass = False
        if complexity_level < 0:
            self.by_pass = True
        elif complexity_level == 0:
            self.conv1 = nn.Conv2d(8, 32, 7, 1, padding=3)
            self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)
            self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)
            self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
            self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)
        elif complexity_level == 3:
            self.conv1 = nn.Conv2d(8, 32, 5, 1, padding=2)
            self.conv2 = nn.Conv2d(32, 64, 5, 1, padding=2)
            self.conv3 = nn.Conv2d(64, 32, 5, 1, padding=2)
            self.conv4 = nn.Conv2d(32, 16, 5, 1, padding=2)
            self.conv5 = nn.Conv2d(16, 2, 5, 1, padding=2)

    def forward(self, x):
        if self.by_pass:
            return x[:, -2:, :, :]

        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x


class ME_Spynet(nn.Module):
    def __init__(self):
        super().__init__()
        self.me_8x = MEBasic(0)
        self.me_4x = MEBasic(0)
        self.me_2x = MEBasic(3)
        self.me_1x = MEBasic(3)

    def forward(self, im1, im2):
        if im1.shape != im2.shape:
            raise ValueError("im1 and im2 must have the same shape.")
        batchsize, C, H, W = im1.shape
        pad_h = (8 - (H % 8)) % 8
        pad_w = (8 - (W % 8)) % 8
        im1 = F.pad(im1, (0, pad_w, 0, pad_h), mode='replicate')
        im2 = F.pad(im2, (0, pad_w, 0, pad_h), mode='replicate')

        im1_1x = im1
        im1_2x = F.avg_pool2d(im1_1x, kernel_size=2, stride=2)
        im1_4x = F.avg_pool2d(im1_2x, kernel_size=2, stride=2)
        im1_8x = F.avg_pool2d(im1_4x, kernel_size=2, stride=2)
        im2_1x = im2
        im2_2x = F.avg_pool2d(im2_1x, kernel_size=2, stride=2)
        im2_4x = F.avg_pool2d(im2_2x, kernel_size=2, stride=2)
        im2_8x = F.avg_pool2d(im2_4x, kernel_size=2, stride=2)

        shape_fine = im1_8x.size()
        zero_shape = [batchsize, 2, shape_fine[2], shape_fine[3]]
        flow_8x = torch.zeros(zero_shape, dtype=im1.dtype, device=im1.device)
        flow_8x = self.me_8x(torch.cat((im1_8x, im2_8x, flow_8x), dim=1))

        flow_4x = bilinearupsacling(flow_8x) * 2.0
        flow_4x = flow_4x + self.me_4x(torch.cat((im1_4x,
                                                  torch_warp_simple(im2_4x, flow_4x.permute(0, 2, 3, 1)),
                                                  flow_4x),
                                                 dim=1))

        flow_2x = bilinearupsacling(flow_4x) * 2.0
        flow_2x = flow_2x + self.me_2x(torch.cat((im1_2x,
                                                  torch_warp_simple(im2_2x, flow_2x.permute(0, 2, 3, 1)),
                                                  flow_2x),
                                                 dim=1))

        flow_1x = bilinearupsacling(flow_2x) * 2.0
        flow_1x = flow_1x + self.me_1x(torch.cat((im1_1x,
                                                  torch_warp_simple(im2_1x, flow_1x.permute(0, 2, 3, 1)),
                                                  flow_1x),
                                                 dim=1))
        flow_1x = flow_1x[:, :, :H, :W]
        return flow_1x
    
Backward_tensorGrid = {}

def Backward(tensorInput, tensorFlow):
	if str(tensorFlow.size()) not in Backward_tensorGrid:
		tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
		tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))

		Backward_tensorGrid[str(tensorFlow.size())] = torch.cat([ tensorHorizontal, tensorVertical ], 1).cuda()

	tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

	return torch.nn.functional.grid_sample(input=tensorInput, grid=(Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow).permute(0, 2, 3, 1), mode='bilinear'
										, padding_mode='border')

class SpyNet(torch.nn.Module):
    def __init__(self):
        super(SpyNet, self).__init__()

        class Preprocess(torch.nn.Module):
            def __init__(self):
                super(Preprocess, self).__init__()

            def forward(self, tensorInput):
                tensorRed = (tensorInput[:, 0:1, :, :] - 0.485) / 0.229
                tensorGreen = (tensorInput[:, 1:2, :, :] - 0.456) / 0.224
                tensorBlue = (tensorInput[:, 2:3, :, :] - 0.406) / 0.225
                return torch.cat([ tensorRed, tensorGreen, tensorBlue ], 1)

        class Basic(torch.nn.Module):
            def __init__(self, intLevel):
                super(Basic, self).__init__()

                self.moduleBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                )

            def forward(self, tensorInput):
                return self.moduleBasic(tensorInput)

        self.modulePreprocess = Preprocess()

        self.moduleBasic = torch.nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])

    def forward(self, tensorFirst, tensorSecond):
        if tensorFirst.shape != tensorSecond.shape:
            raise ValueError("tensorFirst and tensorSecond must have the same shape.")
        batchsize, C, H, W = tensorFirst.shape
        pad_h = (32 - (H % 32)) % 32
        pad_w = (32 - (W % 32)) % 32
        tensorFirst = F.pad(tensorFirst, (0, pad_w, 0, pad_h), mode='replicate')
        tensorSecond = F.pad(tensorSecond, (0, pad_w, 0, pad_h), mode='replicate')
        tensorFlow = []
        tensorFirst = [ self.modulePreprocess(tensorFirst) ]
        tensorSecond = [ self.modulePreprocess(tensorSecond) ]

        for intLevel in range(5):
            if tensorFirst[0].size(2) > 32 or tensorFirst[0].size(3) > 32:
                tensorFirst.insert(0, torch.nn.functional.avg_pool2d(input=tensorFirst[0], kernel_size=2, stride=2, count_include_pad=False))
                tensorSecond.insert(0, torch.nn.functional.avg_pool2d(input=tensorSecond[0], kernel_size=2, stride=2, count_include_pad=False))

        tensorFlow = tensorFirst[0].new_zeros([ tensorFirst[0].size(0), 2, int(math.floor(tensorFirst[0].size(2) / 2.0)), int(math.floor(tensorFirst[0].size(3) / 2.0)) ])

        for intLevel in range(len(tensorFirst)):
            tensorUpsampled = torch.nn.functional.interpolate(input=tensorFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0

            if tensorUpsampled.size(2) != tensorFirst[intLevel].size(2): tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[ 0, 0, 0, 1 ], mode='replicate')
            if tensorUpsampled.size(3) != tensorFirst[intLevel].size(3): tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')

            tensorFlow = self.moduleBasic[intLevel](torch.cat([ tensorFirst[intLevel], Backward(tensorInput=tensorSecond[intLevel], tensorFlow=tensorUpsampled), tensorUpsampled ], 1)) + tensorUpsampled

        tensorFlow = tensorFlow[:, :, :H, :W]
        return tensorFlow
    
backwarp_tenGrid = {}
backwarp_tenPartial = {}

def backwarp(tenInput, tenFlow):
    if str(tenFlow.shape) not in backwarp_tenGrid:
        tenHor = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, -1).repeat(1, 1, tenFlow.shape[2], 1)
        tenVer = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, -1, 1).repeat(1, 1, 1, tenFlow.shape[3])

        backwarp_tenGrid[str(tenFlow.shape)] = torch.cat([ tenHor, tenVer ], 1).cuda()

    if str(tenFlow.shape) not in backwarp_tenPartial:
        backwarp_tenPartial[str(tenFlow.shape)] = tenFlow.new_ones([ tenFlow.shape[0], 1, tenFlow.shape[2], tenFlow.shape[3] ])

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] * (2.0 / (tenInput.shape[3] - 1.0)), tenFlow[:, 1:2, :, :] * (2.0 / (tenInput.shape[2] - 1.0)) ], 1)
    tenInput = torch.cat([ tenInput, backwarp_tenPartial[str(tenFlow.shape)] ], 1)
    tenOutput = torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.shape)] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='zeros', align_corners=True)
    tenMask = tenOutput[:, -1:, :, :]; tenMask[tenMask > 0.999] = 1.0; tenMask[tenMask < 1.0] = 0.0
    return tenOutput[:, :-1, :, :] * tenMask

class PWC_Net(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class Extractor(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=128, out_channels=196, kernel_size=3, stride=2, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=196, out_channels=196, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

            def forward(self, tenInput):
                tenOne = self.netOne(tenInput)
                tenTwo = self.netTwo(tenOne)
                tenThr = self.netThr(tenTwo)
                tenFou = self.netFou(tenThr)
                tenFiv = self.netFiv(tenFou)
                tenSix = self.netSix(tenFiv)

                return [ tenOne, tenTwo, tenThr, tenFou, tenFiv, tenSix ]

        class Decoder(torch.nn.Module):
            def __init__(self, intLevel):
                super().__init__()

                intPrevious = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 1]
                intCurrent = [ None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None ][intLevel + 0]

                if intLevel < 6: self.netUpflow = torch.nn.ConvTranspose2d(in_channels=2, out_channels=2, kernel_size=4, stride=2, padding=1)
                if intLevel < 6: self.netUpfeat = torch.nn.ConvTranspose2d(in_channels=intPrevious + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=4, stride=2, padding=1)
                if intLevel < 6: self.fltBackwarp = [ None, None, None, 5.0, 2.5, 1.25, 0.625, None ][intLevel + 1]

                self.netOne = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netTwo = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128, out_channels=128, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netThr = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128, out_channels=96, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFou = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96, out_channels=64, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netFiv = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64, out_channels=32, kernel_size=3, stride=1, padding=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1)
                )

                self.netSix = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=intCurrent + 128 + 128 + 96 + 64 + 32, out_channels=2, kernel_size=3, stride=1, padding=1)
                )

            def forward(self, tenOne, tenTwo, objPrevious):
                tenFlow = None
                tenFeat = None

                if objPrevious is None:
                    tenFlow = None
                    tenFeat = None

                    tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=tenTwo), negative_slope=0.1, inplace=False)

                    tenFeat = torch.cat([ tenVolume ], 1)

                elif objPrevious is not None:
                    tenFlow = self.netUpflow(objPrevious['tenFlow'])
                    tenFeat = self.netUpfeat(objPrevious['tenFeat'])

                    tenVolume = torch.nn.functional.leaky_relu(input=correlation.FunctionCorrelation(tenOne=tenOne, tenTwo=backwarp(tenInput=tenTwo, tenFlow=tenFlow * self.fltBackwarp)), negative_slope=0.1, inplace=False)

                    tenFeat = torch.cat([ tenVolume, tenOne, tenFlow, tenFeat ], 1)

                # end

                tenFeat = torch.cat([ self.netOne(tenFeat), tenFeat ], 1)
                tenFeat = torch.cat([ self.netTwo(tenFeat), tenFeat ], 1)
                tenFeat = torch.cat([ self.netThr(tenFeat), tenFeat ], 1)
                tenFeat = torch.cat([ self.netFou(tenFeat), tenFeat ], 1)
                tenFeat = torch.cat([ self.netFiv(tenFeat), tenFeat ], 1)

                tenFlow = self.netSix(tenFeat)

                return {
                    'tenFlow': tenFlow,
                    'tenFeat': tenFeat
                }

        class Refiner(torch.nn.Module):
            def __init__(self):
                super().__init__()

                self.netMain = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=81 + 32 + 2 + 2 + 128 + 128 + 96 + 64 + 32, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=128, out_channels=96, kernel_size=3, stride=1, padding=8, dilation=8),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=96, out_channels=64, kernel_size=3, stride=1, padding=16, dilation=16),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
                    torch.nn.LeakyReLU(inplace=False, negative_slope=0.1),
                    torch.nn.Conv2d(in_channels=32, out_channels=2, kernel_size=3, stride=1, padding=1, dilation=1)
                )

            def forward(self, tenInput):
                return self.netMain(tenInput)

        self.netExtractor = Extractor()

        self.netTwo = Decoder(2)
        self.netThr = Decoder(3)
        self.netFou = Decoder(4)
        self.netFiv = Decoder(5)
        self.netSix = Decoder(6)

        self.netRefiner = Refiner()

        # self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.hub.load_state_dict_from_url(url='http://content.sniklaus.com/github/pytorch-pwc/network-' + args_strModel + '.pytorch', file_name='pwc-' + args_strModel).items() })

    def forward(self, tenOne, tenTwo):
        if tenOne.shape != tenTwo.shape:
            raise ValueError("tenOne and tenTwo must have the same shape.")
        batchsize, C, H, W = tenOne.shape
        pad_h = (64 - (H % 64)) % 64
        pad_w = (64 - (W % 64)) % 64
        tenOne = F.pad(tenOne, (0, pad_w, 0, pad_h), mode='replicate')
        tenTwo = F.pad(tenTwo, (0, pad_w, 0, pad_h), mode='replicate')
        tenOne = self.netExtractor(tenOne)
        tenTwo = self.netExtractor(tenTwo)

        objEstimate = self.netSix(tenOne[-1], tenTwo[-1], None)
        objEstimate = self.netFiv(tenOne[-2], tenTwo[-2], objEstimate)
        objEstimate = self.netFou(tenOne[-3], tenTwo[-3], objEstimate)
        objEstimate = self.netThr(tenOne[-4], tenTwo[-4], objEstimate)
        objEstimate = self.netTwo(tenOne[-5], tenTwo[-5], objEstimate)

        tenFlow = (objEstimate['tenFlow'] + self.netRefiner(objEstimate['tenFeat'])) * 20.0
        tenFlow = torch.nn.functional.interpolate(input=tenFlow, size=(H + pad_h, W + pad_w), mode='bilinear', align_corners=False)
        tenFlow = tenFlow[:, :, :H, :W]

        return tenFlow

TRAIN_SIZE = [432, 960]

def compute_grid_indices(image_shape, patch_size=TRAIN_SIZE, min_overlap=20):
    if min_overlap >= patch_size[0] or min_overlap >= patch_size[1]:
        raise ValueError("!!")
    hs = list(range(0, image_shape[0], patch_size[0] - min_overlap))
    ws = list(range(0, image_shape[1], patch_size[1] - min_overlap))
    # Make sure the final patch is flush with the image boundary
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    return [(h, w) for h in hs for w in ws]

def compute_weight(hws, image_shape, patch_size=TRAIN_SIZE, sigma=1.0, wtype='gaussian'):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5
    h, w = h - c_h, w - c_w
    weights_hw = (h ** 2 + w ** 2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h:h+patch_size[0], w:w+patch_size[1]] = weights_hw
    weights = weights.cuda()
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(weights[:, idx:idx+1, h:h+patch_size[0], w:w+patch_size[1]])

    return patch_weights

TRAIN_SIZE = [432, 960]
hws_dic = {}
weights_dic = {}
def flowformerpp_forward(model, image1, image2, sigma=0.05):
    IMAGE_SIZE = (image1.shape[2], image1.shape[3])
    if str(image1.shape) not in hws_dic:
        hws_dic[str(image1.shape)] = compute_grid_indices(IMAGE_SIZE)
        weights_dic[str(image1.shape)] = compute_weight(hws_dic[str(image1.shape)], IMAGE_SIZE, TRAIN_SIZE, sigma)
    hws = hws_dic[str(image1.shape)]
    weights = weights_dic[str(image1.shape)]
    flows = 0
    flow_count = 0

    for idx, (h, w) in enumerate(hws):
        image1_tile = image1[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
        image2_tile = image2[:, :, h:h+TRAIN_SIZE[0], w:w+TRAIN_SIZE[1]]
        flow_pre, flow_low = model(image1_tile, image2_tile)

        padding = (w, IMAGE_SIZE[1]-w-TRAIN_SIZE[1], h, IMAGE_SIZE[0]-h-TRAIN_SIZE[0], 0, 0)
        flows += F.pad(flow_pre * weights[idx], padding)
        flow_count += F.pad(weights[idx], padding)

    flow_pre = flows / flow_count
    return flow_pre