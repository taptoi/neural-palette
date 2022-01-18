import torch
from torchvision import transforms
import numpy as np
from PIL import Image
from scipy.interpolate import interp1d
from skimage.color import rgb2lab, lab2rgb
from model import PaletteNet
from pre_process import center_crop
from pre_process import get_image_palette
import click

def normalize_to_network_input(lab):
    """
    Normalize from LAB color space to [-1.,1.] range network input
        The input is expected in HxWxC format
    """
    range_l = [0., 100.]
    range_a = [-86.185, 98.254]
    range_b = [-107.863, 94.482]
    range_target = [-1., 1.]
    
    clip_l = np.clip(lab[..., 0], range_l[0], range_l[1])
    clip_a = np.clip(lab[..., 1], range_a[0], range_a[1])
    clip_b = np.clip(lab[..., 2], range_b[0], range_b[1])

    map_l = interp1d(range_l, range_target)
    map_a = interp1d(range_a, range_target)
    map_b = interp1d(range_b, range_target)

    # map value range of l's:
    lab[..., 0] = map_l(clip_l)
    # map value range of a's:
    lab[..., 1] = map_a(clip_a)
    # map value range of b's:
    lab[..., 2] = map_b(clip_b)
    
    return lab

def denormalize_to_lab(t):
    """
    Denormalize a numpy array from [-1.,1.] range to 
    valid lab color space range
    """
    range_l = [0., 100.]
    range_a = [-86.185, 98.254]
    range_b = [-107.863, 94.482]
    map_l = interp1d([-1., 1.], range_l)
    map_a = interp1d([-1., 1.], range_a)
    map_b = interp1d([-1., 1.], range_b)
    # map value range of l's:
    t[:, 0, :, :] = map_l(t[:, 0, :, :])
    # map value range of a's:
    t[:, 1, :, :] = map_a(t[:, 1, :, :])
    # map value range of b's:
    t[:, 2, :, :] = map_b(t[:, 2, :, :])
    return t


def match_lab_lightness(outputs, targets):
    """
    Replace the L layer of LAB image in outputs 
    by the ones in targets
    """
    outputs = outputs.permute(0, 2, 3, 1)
    targets = targets.permute(0, 2, 3, 1)
    outputs[..., 0] = targets[..., 0]
    return outputs.permute(0, 3, 1, 2)

def load_checkpoint(model, load_path, device, optimizer=None):
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    return model, optimizer, epoch

def evaluate(input, model):
    """
    Inference entry point corresponding to predict_fn of aws sagemaker
    :param tuple of PIL image (1 x H x W x C), Palette (1 x 1 x n_p x C)) input 
    where n_p is the number of palette colors
    :param PaletteNet model: The model with pretrained weights
    :returns numpy nd array with rgb image data
    """
    # setup device:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # unpack the input tuple:
    image, palette = input
    # setup transform:
    tform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # Ensure dimensions supported by model:
    w, h = image.size
    w_target = int(w / 64) * 64
    h_target = int(h / 64) * 64
    image = center_crop(image, w_target, h_target)
    # PIL to numpy - convert to LAB:
    img = np.array(image, dtype=np.float64) / 255.
    img_lab = rgb2lab(img)
    palette_lab = rgb2lab(palette / 255.)
    img_lab = tform(normalize_to_network_input(img_lab))
    palette_lab = tform(normalize_to_network_input(palette_lab))
    # reshape to batch size 1:
    img_lab = img_lab[np.newaxis, :, :, :]
    palette_lab = palette_lab[np.newaxis, :, :, :]

    input_img = torch.tensor(img_lab, dtype=torch.float).to(device)
    input_palette = torch.tensor(palette_lab, dtype=torch.float).to(device)

    # evaluate
    model.eval()
    with torch.no_grad():
        output = model(input_img, input_palette)
    # match lightness
    ll = input_img[:, 0, :, :].reshape(-1, 1, input_img.shape[2], input_img.shape[3])
    output = torch.hstack((ll, output))
    # transfer and denormalize output
    output = output.clone().detach().cpu().numpy()
    output = denormalize_to_lab(output)
    output = lab2rgb(np.transpose(output[0], (1, 2, 0))) * 255.
    output_pil = Image.fromarray(output.astype('uint8'), 'RGB')
    return output_pil


@click.command()
@click.option('--checkpoint_path', default='./checkpoints/good-darkness-93-run_1697036_checkpoint_epoch_11.pt', help='Path to network checkpoint')
@click.option('--img_in_path', default='eval_in.png', help='Path to input image')
@click.option('--img_pal_path', default='eval_pal.png', help='Path to image to extract palette from')
@click.option('--img_out_path', default='eval_out.png', help='Path to output image')
def test_evaluate(checkpoint_path,
                    img_in_path,
                    img_pal_path,
                    img_out_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Palette colors:
    n_c = 6
    # Load model:
    model = PaletteNet().to(device)
    load_checkpoint(model, checkpoint_path, device)
    print(f"Loaded checkpoint {checkpoint_path}")
    # Load input image:
    image = Image.open(img_in_path)
    # Load input palette image:
    palette_image = Image.open(img_pal_path)
    # Extract target palette by k-means:
    _, target_palette = get_image_palette(palette_image, n_c)
    # Add axis to add batch dimension:
    target_palette = target_palette[np.newaxis, :, :]
    # Perform the prediction:
    output = evaluate((image, target_palette), model)
    # Save final output:
    output.save(img_out_path)

if __name__ == '__main__':
    test_evaluate()
    
