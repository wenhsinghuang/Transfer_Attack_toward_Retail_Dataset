import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.models import resnet50
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms
import cv2
import timm
import copy
from PIL import Image
from tqdm import tqdm


"""## Dataset Class"""

class GroceryDataset(Dataset):
    def __init__(self, annotations_files, img_dir, transform=None, target_transform=None):
        
        self.img_paths = []
        self.img_labels = []
        for annotation_file in annotations_files:
            with open(annotation_file) as f:
                for line in f.readlines():
                    self.img_paths.append(line.split()[0])
                    self.img_labels.append(int(line.split()[1]))
                    
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.img_dir + self.img_paths[idx]), cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        label = np.array(self.img_labels[idx])

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            label = self.target_transform(label)
        # print("After normalization:", image.min().item(), image.max().item())  # Add this line
        return image, label

"""## Model"""

def train_model(model, model_name, criterion, optimizer, scheduler, train_dataloader, val_dataloader, num_epochs=25, start_epoch=0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(start_epoch, num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = train_dataloader
                dataset_size = len(train_dataloader.dataset)
            else:
                model.eval()
                dataloader = val_dataloader
                dataset_size = len(val_dataloader.dataset)

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_size
            epoch_acc = running_corrects.double() / dataset_size

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': criterion,
                }, DIR_PATH + f'{model_name}/{model_name}_checkpoint.pth')

        print()

    print(f'Best val Acc: {best_acc:.4f}')

    # Load the best model weights
    model.load_state_dict(best_model_wts)
    return model

"""## Restore Model"""

def load_checkpoint_and_resume(model, checkpoint_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Loading the model's state_dict from the checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])

    # Loading the optimizer's state_dict from the checkpoint (if available)
    optimizer = None
    if 'optimizer_state_dict' in checkpoint:
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Loading the scheduler's state_dict from the checkpoint (if available)
    scheduler = None
    if 'scheduler_state_dict' in checkpoint:
        scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Loading the last completed epoch from the checkpoint (if available)
    last_epoch = checkpoint.get('epoch', -1)

    return model, optimizer, scheduler, last_epoch

"""# Load Dataset"""

DIR_PATH = "./"

# transform
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

batch_size = 256

# The Freiburg Groceries Dataset
img_dir = DIR_PATH+'images/'
train_annotations_files = ['splits/train0.txt','splits/train1.txt','splits/train2.txt','splits/train3.txt','splits/train4.txt']
train_annotations_files = [DIR_PATH+x for x in train_annotations_files]

# load dataset
train_grocery_dataset = GroceryDataset(train_annotations_files, img_dir, data_transforms['train'])

# load dataloader
train_loader = DataLoader(train_grocery_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

"""# Load Pre-Trained Model"""

# Get the number of unique classes in your dataset
num_classes = len(set(train_grocery_dataset.img_labels))

# ViT model

# ResNet18 model
resnet_model = resnet18(pretrained=True)
resnet_model.fc = nn.Linear(resnet_model.fc.in_features, num_classes)

# Modify the ViT model
vit_model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
vit_model.head = nn.Linear(vit_model.head.in_features, num_classes)

# ResNet50 model
resnet50_model = resnet50(pretrained=True)
resnet50_model.fc = nn.Linear(resnet50_model.fc.in_features, num_classes)


criterion = nn.CrossEntropyLoss()

# Set up the optimizer and learning rate scheduler for each model
vit_optimizer = optim.SGD(vit_model.parameters(), lr=0.001, momentum=0.9)
vit_scheduler = lr_scheduler.StepLR(vit_optimizer, step_size=7, gamma=0.1)

resnet_optimizer = optim.SGD(resnet_model.parameters(), lr=0.001, momentum=0.9)
resnet_scheduler = lr_scheduler.StepLR(resnet_optimizer, step_size=7, gamma=0.1)

"""## Load Trained ResNet18 model"""

# restore training from best ckpt
checkpoint_path = DIR_PATH + "ckpts/resnet18_checkpoint.pth"

resnet_model, resnet_optimizer, resnet_scheduler, last_epoch = load_checkpoint_and_resume(
    resnet_model, checkpoint_path)

"""## Load Trained ResNet50 model"""

# restore training from best ckpt



checkpoint_path = DIR_PATH + "ckpts/resnet50_checkpoint.pth"

resnet50_model, resnet50_optimizer, resnet50_scheduler, last_epoch = load_checkpoint_and_resume(
    resnet50_model, checkpoint_path)

"""## Load Trained ViT model"""

# restore vit_model from best ckpt
checkpoint_path = DIR_PATH + "ckpts/vit_checkpoint.pth"

vit_model, vit_optimizer, vit_scheduler, last_epoch = load_checkpoint_and_resume(
    vit_model, checkpoint_path)

"""# Evaluation"""

def evaluate_model(model, dataloader):
    model.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    correct = 0
    total = 0
    
    count = 0
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='evaluate'):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

"""## Clean Eval"""

# The Freiburg Groceries Dataset
img_dir = DIR_PATH+'images/'
val_annotations_files = ['splits/test0.txt','splits/test1.txt','splits/test2.txt','splits/test3.txt','splits/test4.txt']
val_annotations_files = [DIR_PATH+x for x in val_annotations_files]

# load dataset
val_grocery_dataset = GroceryDataset(val_annotations_files, img_dir, data_transforms['val'])

# # load dataloader
# val_loader = DataLoader(val_grocery_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# test_accuracy = evaluate_model(resnet_model, val_loader)
# print()
# print(f"ResNet eval accuracy: {test_accuracy:.4f}")

# test_accuracy = evaluate_model(vit_model, val_loader)
# print()
# print(f"ViT eval accuracy: {test_accuracy:.4f}")

# test_accuracy = evaluate_model(resnet50_model, val_loader)
# print()
# print(f"resnet50 eval accuracy: {test_accuracy:.4f}")

"""## AdversarialLoader & Noise Functions"""
import os
import numpy as np
from PIL import Image
import foolbox as fb

def apply_pgd_attack(model, inputs, labels, attack_type='pgd', epsilon=0.03, nb_iter=10, device=torch.device('cpu')):
    original_mode = model.training  # Store the original mode of the model
    model.eval()  # Set the model to evaluation mode temporarily
    model.to(device)
    
    fmodel = fb.PyTorchModel(model, bounds=(-2.2, 2.8))
    
    if attack_type == 'pgd':
        attack = fb.attacks.LinfPGD(steps=nb_iter, abs_stepsize=epsilon / 4)
    elif attack_type == 'l2_pgd':
        attack = fb.attacks.L2ProjectedGradientDescentAttack(rel_stepsize=0.025, abs_stepsize=None, steps=nb_iter, random_start=True)
    elif attack_type == 'fgsm':
        attack = fb.attacks.FGSM()
    elif attack_type == 'inf_pgd':
        attack = fb.attacks.LinfPGD()
    elif attack_type == 'l2_cw':
        attack = fb.attacks.L2CarliniWagnerAttack()
    else:
        raise ValueError(f"Unsupported attack type: {attack_type}")
    
    with torch.enable_grad():  # Enable gradients for the attack
        adversarial_inputs, success, _ = attack(fmodel, inputs.to(device), labels.to(device), epsilons=epsilon)
    
    model.train(mode=original_mode)  # Revert the model to its original mode
    return adversarial_inputs

from torch.utils.data import DataLoader

class AdversarialLoader(DataLoader):
    def __init__(self, dataset, model, attack_type='pgd', *args, **kwargs):
        super().__init__(dataset, *args, **kwargs)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.attack_type = attack_type

    def __iter__(self):
        for inputs, labels in super().__iter__():
            print(self.model, inputs, labels)
            adversarial_inputs = apply_pgd_attack(self.model, inputs, labels, attack_type=self.attack_type, device=self.device)
            yield adversarial_inputs, labels

"""## Save Adversarial Images"""

# Create the AdversarialLoader
attack_type='inf_pgd'
attack_model_name = 'ResNet18'

if attack_model_name == 'ResNet18':
    attacked_model = resnet_model
elif attack_model_name == 'ResNet50':
    attacked_model = resnet50_model
elif attack_model_name == 'VIT ':
    attacked_model = vit_model
save_images_folder= f"adversarial_images_{attack_model_name}_{attack_type}/"
val_adversarial_loader = AdversarialLoader(val_grocery_dataset, attacked_model, attack_type=attack_type, batch_size=64, shuffle=False, num_workers=4)



# Create a directory to store the adversarial images
adversarial_images_dir = os.path.join(DIR_PATH, save_images_folder)
# print(adversarial_images_dir)
if not os.path.exists(adversarial_images_dir):
    os.makedirs(adversarial_images_dir)

# Extract mean and std for denormalization from data_transforms
normalize_transform = data_transforms['val'].transforms[-1]
mean = np.array(normalize_transform.mean)
std = np.array(normalize_transform.std)

# Save the adversarial images and their labels to the directory
counter = 0
with open(adversarial_images_dir+"labels.txt", "w") as labels_file:
    for adversarial_inputs, labels in tqdm(val_adversarial_loader, desc='adversarial'):
        for img, label in zip(adversarial_inputs, labels):
            img_np = img.cpu().numpy()
            img_denorm = (img_np * std[:, None, None] + mean[:, None, None]).transpose((1, 2, 0))
            # img_denorm = np.clip(img_denorm, 0, 1)  # Clip values to the range [0, 1]
            img_denorm = (img_denorm * 255).astype("uint8")

            img_pil = Image.fromarray(img_denorm)
            img_pil.save(os.path.join(adversarial_images_dir, f"adversarial_image_{counter}.png"))

            # Save the label to the text file
            labels_file.write(f"adversarial_image_{counter}.png {label.item()}\n")

            counter += 1

"""## Adversarial evaluation"""

adv_files = [DIR_PATH+ f'{save_images_folder}/labels.txt']
adv_img_dir = DIR_PATH+ f'{save_images_folder}/'
val_adversarial_dataset = GroceryDataset(adv_files, adv_img_dir, data_transforms['val'])
val_adversarial_loader = DataLoader(val_adversarial_dataset, batch_size=batch_size, shuffle=False, num_workers=8)

# Evaluate the model on the adversarial examples
resnet_adversarial_accuracy = evaluate_model(resnet_model, val_adversarial_loader)
print(f'ResNet18 model accuracy on adversarial examples: {resnet_adversarial_accuracy * 100:.2f}%')

# Evaluate the model on the adversarial examples
resnet50_adversarial_accuracy = evaluate_model(resnet50_model, val_adversarial_loader)
print(f'ResNet50 model accuracy on adversarial examples: {resnet50_adversarial_accuracy * 100:.2f}%')

vit_adversarial_accuracy = evaluate_model(vit_model, val_adversarial_loader)
print(f'ViT model accuracy on adversarial examples: {vit_adversarial_accuracy * 100:.2f}%')
