import argparse
import os
import cv2
import numpy as  np
import torch

from models.src import dp
from models import pipelines


def tti(tensor):
    img = tensor[0].permute(1, 2, 0)
    img = img.detach().cpu().numpy()
    return img


def load_data(samples_root, source_sample, target_sample, device='cuda:0'):
    source_uv_path = os.path.join(samples_root, 'source_uv', source_sample + '.npy')
    source_img_path = os.path.join(samples_root, 'source_img', source_sample + '.jpg')
    target_uv_path = os.path.join(samples_root, 'target_uv', target_sample + '.npy')

    print(source_img_path)

    source_uv = np.load(source_uv_path)
    source_img = cv2.imread(str(source_img_path))[..., [2, 1, 0]] / 255
    target_uv = np.load(target_uv_path)

    [source_uv, source_img, target_uv] = [torch.FloatTensor(x).permute(2, 0, 1).unsqueeze(0).to(device) for x in
                                          [source_uv, source_img, target_uv]]

    return dict(source_uv=source_uv,
                source_img=source_img,
                target_uv=target_uv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--checkpoint_path', type=str, default='data/checkpoint')
    parser.add_argument('--samples_root', type=str, default='data/samples')
    parser.add_argument('--out_dir', type=str, default='data/results')
    parser.add_argument('--source_sample', type=str)
    parser.add_argument('--target_sample', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = args.device

    inpainter_file = os.path.join(args.checkpoint_path, 'inpainter.pth')
    refiner_file = os.path.join(args.checkpoint_path, 'refiner.pth')

    inpainter = dp.GatedHourglass(32, 5, 2).to(device)
    refiner = dp.ResHourglassDeformableSkip(8, 10, 3, ngf=256).to(device)

    inpainter.load_state_dict(torch.load(inpainter_file))
    refiner.load_state_dict(torch.load(refiner_file))

    pipeline = pipelines.DeformablePipe(inpainter, refiner).eval()

    data_dict = load_data(args.samples_root, args.source_sample, args.target_sample, device=device)

    output_dict = pipeline(data_dict)
    pred_img = (tti(output_dict['refined']) * 255).astype(np.uint8)

    pair_str = args.source_sample + '_to_' + args.target_sample
    out_path = os.path.join(args.out_dir, pair_str+'.png')
    os.makedirs(args.out_dir, exist_ok=True)

    cv2.imwrite(out_path, pred_img[..., ::-1])


