import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from scipy.interpolate import interp1d
from skimage.color import rgb2lab, lab2rgb
import os.path
import os
from model import PaletteNet
from pre_process import get_image_palette
import wandb
import click

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]

class NeuralPaletteDataset(Dataset):
    def __init__(self, data_folder, transform):
        super().__init__()
        self.data_folder = data_folder
        self.transform = transform

    def __len__(self):
        img_folder = self.data_folder + "/source"
        return len(get_immediate_subdirectories(img_folder))

    # The custom data loader loads image-palette pairs
    # and their color augmented counterparts in compressed format
    # (they are generated from raw image data by pre_process.py)
    def __getitem__(self, idx):
        sources_root = self.data_folder + "/source"
        targets_root = self.data_folder + "/target"
        source_folders = get_immediate_subdirectories(sources_root)
        source_folder = os.path.join(sources_root, source_folders[idx])
        folder_base = os.path.basename(source_folder)
        target_folder = os.path.join(targets_root, folder_base)
        angles = range(40, 360, 40)
        random_selector = np.random.randint(0, len(angles) - 1)
        random_angle_prefix = str(angles[random_selector]).rjust(3, '0') + "_"
        
        source_img_path = os.path.join(source_folder, "img.npz")
        target_img_path = os.path.join(target_folder, random_angle_prefix + "img.npz")
        source_pal_path = os.path.join(source_folder, "pal.npz")
        target_pal_path = os.path.join(target_folder, random_angle_prefix + "pal.npz")
        
        try:
            source_img = np.load(source_img_path)['arr_0']
            source_pal = np.load(source_pal_path)['arr_0']
            source_pal = source_pal.reshape(1, source_pal.shape[0], source_pal.shape[1])
            target_img = np.load(target_img_path)['arr_0']
            target_pal = np.load(target_pal_path)['arr_0']
            target_pal = target_pal.reshape(1, target_pal.shape[0], target_pal.shape[1])
        except:
            print("Error loading training data from source path: ", source_img_path)
            

        if(self.transform):
            source_img = self.transform(normalize_to_network_input(source_img))
            source_pal = self.transform(normalize_to_network_input(source_pal))
            target_img = self.transform(normalize_to_network_input(target_img))
            target_pal = self.transform(normalize_to_network_input(target_pal))


        return source_img, source_pal, target_img, target_pal

        
def batch_map(batched, fun):
    """
    Trivial non-parallel function application
    on a batched nd-array
    """
    N = batched.shape[0]
    ret = np.zeros_like(batched)
    for i in range(N):
        ret[i] = fun(batched[i])
    return ret
    

def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)

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


def to_wandb_image_grid(tensor, down_smp_factor, num_images):
    """Convert image tensor to Weights & Biases image list"""
    res = (tensor).clone().detach().cpu().numpy()
    res = denormalize_to_lab(res)
    res = batch_map(np.transpose(res, (0, 2, 3, 1)), lab2rgb)
    res = res[:num_images, ::down_smp_factor, ::down_smp_factor, :]
    return [wandb.Image(img) for img in res]

def load_checkpoint(model, load_path, optimizer=None):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    return model, optimizer, epoch

def match_lab_lightness(outputs, targets):
    """
    Replace the L layer of LAB image in outputs 
    by the ones in targets
    """
    outputs = outputs.permute(0, 2, 3, 1)
    targets = targets.permute(0, 2, 3, 1)
    outputs[..., 0] = targets[..., 0]
    return outputs.permute(0, 3, 1, 2)

@click.command()
@click.option('--num_epochs', default=200, help='Number of epochs to train')
@click.option('--report_wandb', default=False, help='Use Weights & Biases for reporting')
@click.option('--wandb_entity_name', default="", help='Weights & Biases entity name')
@click.option('--restore_from_checkpoint', default=False, help='Restore training from checkpoint')
@click.option('--checkpoint_path', default="", help='Path to network checkpoint')
@click.option('--checkpoint_save_every', default="5", help='Save checkpoint N epochs')
def train(num_epochs, 
        report_wandb,
        wandb_entity_name,
        restore_from_checkpoint, 
        checkpoint_path,
        checkpoint_save_every):
    """
    The main training function
    """
    runid = np.random.randint(9999999)
    workers = 4
    # Batch size during training
    batch_size = 4
    # Number of channels in training images
    c_im = 3
    # Palette colors
    n_c = 6
    # Number of channels in training samples
    c = c_im + n_c * c_im

    # Learning rate
    learning_rate = 0.00005

    # normalization and denormalization
    #normalize = transforms.Normalize(MEAN.tolist(), STD.tolist())
    tform = transforms.Compose([
        transforms.ToTensor(),
        #normalize,
    ])

    dataset = NeuralPaletteDataset("./dataset", tform)
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                            shuffle=True, num_workers=workers)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = init_model(c, device)
    model = PaletteNet().to(device)

    loss = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.9995), weight_decay=0.01)

    if restore_from_checkpoint:
        load_checkpoint(model, checkpoint_path, optimizer)
        print(f"Loaded checkpoint {checkpoint_path}")

    iters = 0
    if report_wandb:
        wandb.init(project="neural-palette", entity=wandb_entity_name)
        wandb.config = {
                "algorithm": "neural-palette",
                "learning_rate_0": learning_rate,
                "epochs": num_epochs,
                "batch_size": batch_size,
                }

    print("Starting Training Loop...")
    for epoch in range(num_epochs):
        loss_acc = 0.
        loss_count = 0
        for batch_idx, data in enumerate(dataloader, 0):
            im_in, palette_in, im_aug, palette_aug = data
            print(f"Training epoch {epoch + 1}, iteration {iters}...")
            model.zero_grad()
            # switch places of input and augmented by probability:
            if np.random.rand() < 0.5:
                temp = im_in
                im_in = im_aug
                im_aug = temp
                palette_aug = palette_in

            input = torch.tensor(im_in, dtype=torch.float).to(device)
            im_aug = torch.tensor(im_aug, dtype=torch.float).to(device)
            palette_aug = torch.tensor(palette_aug, dtype=torch.float).to(device)
            output = model(input, palette_aug)
            # Concatenate the LAB lightness from input
            ll = input[:, 0, :, :].reshape(-1, 1, input.shape[2], input.shape[3])
            output = torch.hstack((ll, output))
            # Calculate loss on input batch:
            err = loss(output, im_aug)
            # Calculate gradients
            err.backward()
            optimizer.step()
            iters += 1
            loss_acc += err.item()
            loss_count += 1
            if batch_idx == 0 and report_wandb:
                result_sample = to_wandb_image_grid(output, down_smp_factor=4, num_images=4)
                input_sample = to_wandb_image_grid(input, down_smp_factor=4, num_images=4)
                im_aug_sample = to_wandb_image_grid(im_aug, down_smp_factor=4, num_images=4)
                wandb.log({"Target": im_aug_sample,
                            "Result": result_sample,
                            "Input": input_sample})

        loss_acc /= loss_count
        print(f"Loss for epoch {epoch}: {loss_acc}")
        if report_wandb:
            wandb.log({"Loss": loss_acc,
                        "Epoch": epoch
                        })
        save_path = f"run_{runid}_checkpoint_epoch_{epoch+1}.pt"
        if(epoch % checkpoint_save_every == 0):
            save_checkpoint(model, optimizer, save_path, epoch)


    

if __name__ == '__main__':
    train()
    
