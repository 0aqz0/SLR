import torch
from sklearn.metrics import accuracy_score

def test(model, criterion, dataloader, device, epoch, logger, writer):
    model.eval()
    losses = []
    all_label = []
    all_pred = []

    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            # get the inputs and labels
            inputs, labels = data['data'].to(device), data['label'].to(device)
            # forward
            outputs = model(inputs)
            if isinstance(outputs, list):
                outputs = outputs[0]
            # compute the loss
            loss = criterion(outputs, labels.squeeze())
            losses.append(loss.item())
            # collect labels & prediction
            prediction = torch.max(outputs, 1)[1]
            all_label.extend(labels.squeeze())
            all_pred.extend(prediction)
    # Compute the average loss & accuracy
    test_loss = sum(losses)/len(losses)
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    test_acc = accuracy_score(all_label.squeeze().cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    # Log
    writer.add_scalars('Loss', {'test': test_loss}, epoch+1)
    writer.add_scalars('Accuracy', {'test': test_acc}, epoch+1)
    logger.info("Average Test Loss: {:.6f} | Acc: {:.2f}%".format(test_loss, test_acc*100))

if __name__ == '__main__':
    import os
    import argparse
    from torch.utils.data import DataLoader
    import torchvision.transforms as transforms
    from dataset import CSL_Isolated
    from models.Conv3D import resnet18, resnet34, resnet50, r2plus1d_18
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='/home/haodong/Data/CSL_Isolated/color_video_125000',
        type=str, help='Data path for testing')
    parser.add_argument('--label_path', default='/home/haodong/Data/CSL_Isolated/dictionary.txt',
        type=str, help='Label path for testing')
    parser.add_argument('--model', default='3dresnet18',
        type=str, help='Choose a model for testing')
    parser.add_argument('--model_path', default='3dresnet18.pth',
        type=str, help='Model state dict path')
    parser.add_argument('--num_classes', default=500,
        type=int, help='Number of classes for testing')
    parser.add_argument('--batch_size', default=32,
        type=int, help='Batch size for testing')
    parser.add_argument('--sample_size', default=128,
        type=int, help='Sample size for testing')
    parser.add_argument('--sample_duration', default=16,
        type=int, help='Sample duration for testing')
    parser.add_argument('--no_cuda', action='store_true',
        help='If true, dont use cuda')
    parser.add_argument('--cuda_devices', default='2',
        type=str, help='Cuda visible devices')
    args = parser.parse_args()

    # Path setting
    data_path = args.data_path
    label_path = args.label_path
    model_path = args.model_path
    # Use specific gpus
    os.environ["CUDA_VISIBLE_DEVICES"]=args.cuda_devices
    # Device setting
    if torch.cuda.is_available() and not args.no_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Hyperparams
    num_classes = args.num_classes
    batch_size = args.batch_size
    sample_size = args.sample_size
    sample_duration = args.sample_duration

    # Start testing
    # Load data
    transform = transforms.Compose([transforms.Resize([sample_size, sample_size]),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.5], std=[0.5])])
    test_set = CSL_Isolated(data_path=data_path, label_path=label_path, frames=sample_duration,
        num_classes=num_classes, train=False, transform=transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # Create model
    if args.model == '3dresnet18':
        model = resnet18(pretrained=True, progress=True, sample_size=sample_size,
            sample_duration=sample_duration, num_classes=num_classes).to(device)
    elif args.model == '3dresnet34':
        model = resnet34(pretrained=True, progress=True, sample_size=sample_size,
            sample_duration=sample_duration, num_classes=num_classes).to(device)
    elif args.model == '3dresnet50':
        model = resnet50(pretrained=True, progress=True, sample_size=sample_size,
            sample_duration=sample_duration, num_classes=num_classes).to(device)
    elif args.model == 'r2plus1d':
        model = r2plus1d_18(pretrained=True, num_classes=num_classes).to(device)
    # Run the model parallelly
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    # Load model
    model.load_state_dict(torch.load(model_path))

    # Test the model
    model.eval()
    all_label = []
    all_pred = []

    with torch.no_grad():
        for batch_idx, data in enumerate(test_loader):
            # get the inputs and labels
            inputs, labels = data['data'].to(device), data['label'].to(device)
            # forward
            outputs = model(inputs)
            # collect labels & prediction
            prediction = torch.max(outputs, 1)[1]
            all_label.extend(labels.squeeze())
            all_pred.extend(prediction)
    # Compute the average loss & accuracy
    all_label = torch.stack(all_label, dim=0)
    all_pred = torch.stack(all_pred, dim=0)
    test_acc = accuracy_score(all_label.squeeze().cpu().data.squeeze().numpy(), all_pred.cpu().data.squeeze().numpy())
    print("Test Acc: {:.2f}%".format(test_acc*100))
