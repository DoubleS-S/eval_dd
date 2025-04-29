import numpy as np
import torch
import torch.nn as nn
import argparse
import copy
import time
from utils import (get_dataset, get_network, evaluate_synset, get_time, DiffAugment, ParamDiffAug)

def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--generated_data', type=str, default='gen.npz', help='path to the generated data file')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=50, help='image(s) per class')
    parser.add_argument('--num_eval', type=int, default=3, help='number of evaluations')
    parser.add_argument('--data_path', type=str, default='./data', help='dataset path')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training')
    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--dsa', type=str, default='True', help='whether to use DSA augmentation during evaluation')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--device', type=str, default='cuda', help='device for training')
    args = parser.parse_args()
    
    args.dsa = True if args.dsa == 'True' else False
    args.dsa_param = ParamDiffAug()
    
    # Load the gen.npz file containing all generated images
    print(f"Loading generated images from 'gen.npz'...")
    data = np.load(args.generated_data)
    image_syn_all = data['x']  # All synthetic images
    label_syn_all = data['y']  # All synthetic labels
    
    # Get dataset information
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset, args.data_path)
    
    print(f"Total generated images: {len(image_syn_all)}, targeting {args.ipc} images per class")
    
    # Simplified approach: Directly find indices for each class and sample from them
    selected_images = []
    selected_labels = []
    
    for c in range(num_classes):
        # Find indices for current class
        class_indices = np.where(label_syn_all == c)[0]
        print(f"Class {c}: {len(class_indices)} images available")
        
        # Check if we have enough images for this class
        if len(class_indices) <= args.ipc:
            # Take all images for this class
            selected_indices = class_indices
            print(f"Warning: Only {len(class_indices)} images available for class {c}, using all of them")
        else:
            # Randomly select ipc images
            selected_indices = np.random.choice(class_indices, args.ipc, replace=False)
            print(f"Randomly selected {args.ipc} images for class {c}")
        
        # Add the selected images and their labels
        selected_images.append(image_syn_all[selected_indices])
        selected_labels.append(label_syn_all[selected_indices])
    
    # Combine all classes
    image_syn = np.concatenate(selected_images, axis=0)
    label_syn = np.concatenate(selected_labels, axis=0)
    
    print(f"Final dataset for evaluation: {image_syn.shape[0]} images total, {args.ipc} per class")
    
    # Convert to torch tensors
    image_syn = torch.from_numpy(image_syn).float()
    label_syn = torch.from_numpy(label_syn).long()
    
    # Evaluate the selected subset
    model_eval = args.model
    accs = []
    for it_eval in range(args.num_eval):
        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model
        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn), copy.deepcopy(label_syn)
        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval, testloader, args)
        accs.append(acc_test)
        
    print('Evaluate %d random %s, mean = %.1f std = %.1f\n-------------------------'%(len(accs), model_eval, np.mean(accs), np.std(accs)))

if __name__ == '__main__':
    main()