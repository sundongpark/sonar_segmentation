import torch
from sonar_loader import *
from model import *
from eval import *
import tqdm
import random
import csv
import os
from loss import FocalLoss, CBFocalLoss

import warnings
warnings.filterwarnings(action='ignore')
'''
random_seed = 42
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(random_seed)
random.seed(random_seed)
'''

pre_path = './checkpoints/best_model.pth'
save_csv = True
csv_path = os.path.expanduser('~/GoogleDrive/SONAR_Semantic_Segmentation/results/')

def train_function(data, model, optimizer, loss_function, scheduler, device):
    model.train()
    epoch_loss = 0

    for index, sample_batch in enumerate(tqdm.tqdm(data)):
        imgs = sample_batch['image']
        gt_mask = sample_batch['mask']

        imgs = imgs.to(device)
        true_masks = gt_mask.to(device)

        outputs = model(imgs)

        # prediction vis
        probs = torch.softmax(outputs, dim=1)
        masks_pred = torch.argmax(probs, dim=1)

        loss = loss_function(outputs, true_masks)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch finished ! Loss: {epoch_loss / index:.4f}, lr:{scheduler.get_last_lr()}')

def validation_epoch(model, val_loader, num_class, device, epoch, model_name):
    class_iou, mean_iou = eval_net_loader(model, val_loader, num_class, device, epoch)
    print('Class IoU:', ' '.join(f'{x:.4f}' for x in class_iou), f'  |  Mean IoU: {mean_iou:.4f}')
    if save_csv and epoch == 'test':
        with open(f'{csv_path}{model_name}_{dataset[:-1]}_results_{iter}.csv', 'w', newline='') as f:
            w = csv.writer(f, delimiter='\n')
            w.writerow(class_iou)
            w.writerow([mean_iou])
    return mean_iou

def main(mode='', gpu_id=0, num_epoch=31, train_batch_size=2, test_batch_size=1, classes=[], pretrained=False, save_path='', model_name = 'unet', loss_fn = torch.nn.CrossEntropyLoss(), dataset = ''):
    lr = 0.001
    save_term = 5

    dir_checkpoint = './checkpoints/'

    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')

    data_path = './data/segmentation/' + dataset

    dataset_train = sonarDataset(data_path + 'train', classes)
    dataset_val = sonarDataset(data_path + 'val', classes)
    dataset_test = sonarDataset(data_path + 'test', classes)

    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=0
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=test_batch_size, shuffle=True, num_workers=0
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=test_batch_size, shuffle=False, num_workers=0
    )

    if model_name == 'resnet18':
        model = ResNetUNet(in_channels=1, n_classes=len(classes), encoder=models.resnet18).to(device).train() 
    elif model_name == 'resnet34':
        model = ResNetUNet(in_channels=1, n_classes=len(classes), encoder=models.resnet34).to(device).train() 
    elif model_name == 'resnet50':
        model = DeepResUnet(in_channels=1, n_classes=len(classes), encoder=models.resnet50).to(device).train()
    elif model_name == 'resnet101':
        model = DeepResUnet(in_channels=1, n_classes=len(classes), encoder=models.resnet101).to(device).train()
    elif model_name == 'resnet152':
        model = DeepResUnet(in_channels=1, n_classes=len(classes), encoder=models.resnet152).to(device).train()
    elif model_name == 'vgg16':
        model = VGGUnet(in_channels=1, n_classes=len(classes), encoder=models.vgg16).to(device).train()
    elif model_name == 'vgg19':
        model = VGGUnet(in_channels=1, n_classes=len(classes), encoder=models.vgg19).to(device).train()
    elif model_name == 'unet':
        model = UNet(in_channels=1, n_classes=len(classes)).to(device).train()

    if 'train' in mode:
        if pretrained:
            model.load_state_dict(torch.load(pre_path))
            print('Model loaded from {}'.format(pre_path))

        print('\nStarting training:\n'
              f'Epochs: {num_epoch}\n'
              f'Batch size: {train_batch_size}\n'
              f'Learning rate: {lr}\n'
              f'Training size: {len(data_loader.dataset)}\n')


        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
        loss_function = loss_fn   # torch.nn.CrossEntropyLoss()

        max_score = 0
        max_score_epoch = 0
        for epoch in range(1, num_epoch+1):
            print('*** Starting epoch {}/{}. ***'.format(epoch, num_epoch))

            train_function(data_loader, model, optimizer, loss_function, lr_scheduler, device)
            lr_scheduler.step()

            mean_iou = validation_epoch(model, data_loader_val, len(classes), device, epoch, model_name)

            state_dict = model.state_dict()
            if device == "cuda":
                state_dict = model.module.state_dict()
            if epoch % save_term == 0:
                state_dict = model.state_dict()
                if device == "cuda":
                    state_dict = model.module.state_dict()
                torch.save(state_dict, dir_checkpoint + f'{epoch}.pth')
                print('Checkpoint epoch: {} saved !'.format(epoch))
            if max_score < mean_iou:
                max_score = mean_iou
                max_score_epoch = epoch
                print('Best Model saved!')
                torch.save(state_dict, dir_checkpoint + 'best_model.pth')
            '''
            if epoch >= max_score_epoch + 3:
                break
            '''
            print('****************************')
        print('*** Test ***')
        model.load_state_dict(torch.load(pre_path))
        validation_epoch(model, data_loader_test, len(classes), device, 'test', model_name)

if __name__ =="__main__":

    CLASSES = ['background', 'bottle', 'can', 'chain',
               'drink-carton', 'hook', 'propeller', 'shampoo-bottle',
               'standing-bottle', 'tire', 'valve', 'wall']
    device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device: {str(device)}\n')
    
    beta = 0.999999
    gamma = 0.6

    for iter in range(10):
        for model_name in ['unet','vgg16','vgg19','resnet18','resnet34','resnet50','resnet101','resnet152']:
            batch_size = 4
            if model_name == 'resnet18':
                batch_size = 16
            elif model_name == 'resnet34':
                batch_size = 16 
            elif model_name == 'resnet50':
                batch_size = 8
            elif model_name == 'resnet101':
                batch_size = 4
            elif model_name == 'resnet152':
                batch_size = 4
            elif model_name == 'vgg16':
                batch_size = 4
            elif model_name == 'vgg19':
                batch_size = 4
            elif model_name == 'unet':
                batch_size = 4

            for dataset in ['single02/', 'single05/','single10/', 'single11/','single12/','multi02/','multi05/','multi10/','multi11/','multi12/']:
                print(f'model: {model_name}')
                print(f'dataset: {dataset}')
                print(f'iter: {iter}')
                main(mode='train', gpu_id=0, num_epoch=30,
                    train_batch_size=batch_size, test_batch_size=1, classes=CLASSES,
                    pretrained=False, save_path='', loss_fn=CBFocalLoss(beta, gamma), model_name=model_name, dataset=dataset)
