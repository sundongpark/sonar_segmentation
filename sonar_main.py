import torch
from sonar_loader import *
from model import *
from eval import *
import tqdm
import random
from loss import FocalLoss
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

def validation_epoch(model, val_loader, num_class, device, epoch):
    class_iou, mean_iou = eval_net_loader(model, val_loader, num_class, device, epoch)
    print('Class IoU:', ' '.join(f'{x:.4f}' for x in class_iou), f'  |  Mean IoU: {mean_iou:.4f}')

    return mean_iou

def main(mode='', gpu_id=0, num_epoch=31, train_batch_size=2, test_batch_size=1, classes=[], pretrained=False, save_path=''):
    lr = 0.001
    save_term = 5

    dir_checkpoint = './checkpoints/'

    device = torch.device(f'cuda:{gpu_id}') if torch.cuda.is_available() else torch.device('cpu')

    data_path = './data/segmentation/'

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

    # model = DeepResUnet(in_channels=1, n_classes=len(classes), encoder=models.resnet152).to(device).train()
    # model = UNet(in_channels=1, n_classes=len(classes)).to(device).train()
    model = ResNetUNet(in_channels=1, n_classes=len(classes), encoder=models.resnet18).to(device).train() 

    if 'train' in mode:
        if pretrained:
            model.load_state_dict(torch.load(pre_path))
            print('Model loaded from {}'.format(pre_path))

        print('Starting training:\n'
              f'Epochs: {num_epoch}\n'
              f'Batch size: {train_batch_size}\n'
              f'Learning rate: {lr}\n'
              f'Training size: {len(data_loader.dataset)}\n'
              f'Device: {str(device)}\n')

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=1)
        loss_function = torch.nn.CrossEntropyLoss() # FocalLoss(0.25)   # torch.nn.CrossEntropyLoss()

        max_score = 0
        max_score_epoch = 0
        for epoch in range(1, num_epoch+1):
            print('*** Starting epoch {}/{}. ***'.format(epoch, num_epoch))

            train_function(data_loader, model, optimizer, loss_function, lr_scheduler, device)
            lr_scheduler.step()

            mean_iou = validation_epoch(model, data_loader_val, len(classes), device, epoch)

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
            if epoch >= max_score_epoch + 3:
                break

            print('****************************')
        print('*** Test ***')
        model.load_state_dict(torch.load(pre_path))
        validation_epoch(model, data_loader_test, len(classes), device, epoch='test')

if __name__ =="__main__":

    CLASSES = ['background', 'bottle', 'can', 'chain',
               'drink-carton', 'hook', 'propeller', 'shampoo-bottle',
               'standing-bottle', 'tire', 'valve', 'wall']


    main(mode='train', gpu_id=0, num_epoch=30,
         train_batch_size=16, test_batch_size=1, classes=CLASSES,
         pretrained=False, save_path='')