    # This code was created using jelmerwolterink's tutorial on github
    # and has since been modified for my own use
    # Title: ISMRMTutorial
    # Author: jelmerwolterink
    # Availability: https://github.com/jelmerwolterink/ISMRMTutorial

if __name__ == '__main__':
    import torch
    import monai
    import numpy as np
    import matplotlib.pyplot as plt
    import glob
    import tqdm
    from tqdm import tqdm

    datapath = r'C:\Users\jonat\Documents\stanfordProjectData\Tutorial_Training_Intellij'


    def visualize_data(pt_dict, batch=False):
        image = pt_dict["image"].squeeze()
        label = pt_dict["label"].squeeze()
        if batch:
            image = image.permute((1, 2, 0))
            label = label.permute((1, 2, 0))
        plt.figure(figsize=(20,20))
        for z in range(image.shape[2]):
            plt.subplot(int(np.ceil(np.sqrt(image.shape[2]))), int(np.ceil(np.sqrt(image.shape[2]))), 1 + z)
            plt.imshow(image[:, :, z], cmap='gray')
            plt.axis('off')
            plt.imshow(np.ma.masked_where(label[:, :, z]!=2, label[:, :, z]==2), alpha=0.6, cmap='Blues', clim=(0, 1))
            plt.imshow(np.ma.masked_where(label[:, :, z]!=3, label[:, :, z]==3), alpha=0.6, cmap='Greens', clim=(0, 1))
            plt.imshow(np.ma.masked_where(label[:, :, z]!=1, label[:, :, z]==1), alpha=0.6, cmap='Reds', clim=(0, 1))
            plt.title('Slice {}'.format(z + 1))
        plt.show()


    file_dict = []
    for ptid in range(1, 101):
        gt_filenames = glob.glob(r'{}/patient{}/*_gt.nii.gz'.format(datapath, str(ptid).zfill(3)))
        file_dict.append({'image': gt_filenames[0].replace('_gt', ''), 'label': gt_filenames[0]})
        file_dict.append({'image': gt_filenames[1].replace('_gt', ''), 'label': gt_filenames[1]})

    transform = monai.transforms.Compose([
        monai.transforms.LoadImageD(("image", "label")),
        monai.transforms.AddChannelD(("image", "label")),
        monai.transforms.ScaleIntensityRangePercentilesd(keys=("image"), lower=5, upper=95, b_min=0, b_max=1, clip=True),
        monai.transforms.RandSpatialCropD(keys=("image", "label"), roi_size=(128, 128, 1), random_center=True, random_size=False),
        monai.transforms.SqueezeDimd(keys=("image", "label"), dim=-1),
        monai.transforms.ToTensorD(("image", "label"))
    ])

    dataset = monai.data.Dataset(data = file_dict, transform = transform)
    dataloader = monai.data.DataLoader(dataset, batch_size=16, shuffle=False)
    print('The dataset contains {} images'.format(len(dataset)))

    val_ids = set(range(1, 101, 5))
    train_ids = set(range(1, 101)) - val_ids

    file_dict_train = []
    for ptid in train_ids:
        gt_filenames = glob.glob(r'{}/patient{}/*_gt.nii.gz'.format(datapath, str(ptid).zfill(3)))
        file_dict_train.append({'image': gt_filenames[0].replace('_gt', ''), 'label': gt_filenames[0]})
        file_dict_train.append({'image': gt_filenames[1].replace('_gt', ''), 'label': gt_filenames[1]})

    file_dict_val = []
    for ptid in val_ids:
        gt_filenames = glob.glob(r'{}/patient{}/*_gt.nii.gz'.format(datapath, str(ptid).zfill(3)))
        file_dict_val.append({'image': gt_filenames[0].replace('_gt', ''), 'label': gt_filenames[0]})
        file_dict_val.append({'image': gt_filenames[1].replace('_gt', ''), 'label': gt_filenames[1]})

    # This transform should be altered to add data augmentation
    transform_train = monai.transforms.Compose([
        monai.transforms.LoadImageD(("image", "label")),
        monai.transforms.AddChannelD(("image", "label")),
        monai.transforms.ScaleIntensityRangePercentilesd(keys=("image"), lower=5, upper=95, b_min=0, b_max=1, clip=True),
        monai.transforms.RandSpatialCropD(keys=("image", "label"), roi_size=(128, 128, 1), random_center=True, random_size=False),
        monai.transforms.SqueezeDimd(keys=("image", "label"), dim=-1),
        monai.transforms.ToTensorD(("image", "label")),
    ])

    transform_val = monai.transforms.Compose([
        monai.transforms.LoadImageD(("image", "label")),
        monai.transforms.AddChannelD(("image", "label")),
        monai.transforms.ScaleIntensityRangePercentilesd(keys=("image"), lower=5, upper=95, b_min=0, b_max=1, clip=True),
        monai.transforms.ToTensorD(("image", "label")),
    ])

    dataset_train = monai.data.CacheDataset(data = file_dict_train, transform = transform_train, progress=False)
    dataset_val = monai.data.Dataset(data = file_dict_val, transform = transform_val)

    dataloader_train = monai.data.DataLoader(dataset_train, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    dataloader_val = monai.data.DataLoader(dataset_val, batch_size=1, shuffle=False)

    print('The training set contains {} MRI scans.'.format(len(dataset_train)))
    print('The test set contains {} MRI scans.'.format(len(dataset_val)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    print(device)
    model = monai.networks.nets.UNet(
        dimensions=2,
        in_channels=1,
        out_channels=4,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)

    loss_function = monai.losses.DiceLoss(softmax=True, to_onehot_y=True, batch=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    training_losses = list()

    for epoch in tqdm(range(20)):
        model.train()
        epoch_loss = 0
        step = 0
        print("epoch")
        print(dataloader_train)
        for batch_data in dataloader_train:
            print("batch_data")
            step += 1
            optimizer.zero_grad()
            outputs = model(batch_data["image"].to(device))
            loss = loss_function(outputs, batch_data["label"].to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        training_losses.append(epoch_loss/step)

    # Store the network parameters
    torch.save(model.state_dict(), r'trainedUNet.pt')

    # model.load_state_dict(torch.load(r'trainedUNet.pt'))

    # plt.figure()
    # plt.plot(np.asarray(training_losses))
    # plt.xlabel('Epoch')
    # plt.ylabel('Dice loss')
    # plt.show()
    # plt.draw()

    #TODO: Uncomment

    model.eval()
    postprocess = monai.transforms.Compose([
        monai.transforms.AsDiscrete(argmax=True, to_onehot=True, n_classes=4, threshold_values=False),
        monai.transforms.KeepLargestConnectedComponent(applied_labels=(1, 2, 3), independent=False, connectivity=None)
    ])

    for val_batch in dataloader_val:
        outputs_val = monai.inferers.sliding_window_inference(val_batch["image"].squeeze(1).permute(3, 0, 1, 2).to(device), (128, 128), 32, model, overlap = 0.8)
        outputs_val = outputs_val.permute(1, 2, 3, 0)
        print(outputs_val.shape)
        outputs_val = postprocess(outputs_val)
        result = {"image": val_batch["image"].squeeze(),
                  "label": torch.argmax(outputs_val, dim=0).squeeze().cpu()}
        #     print("The outputs val shape is: {}".format(outputs_val.shape))
        #     print("The outputs val_batch is: {}".format(val_batch["image"].squeeze().shape))
        #     print("The outputs outputs_val argmax is: {}".format(torch.argmax(outputs_val, dim=0).shape))
        visualize_data(result)
        visualize_data(val_batch)

        dice_metric = monai.metrics.DiceMetric()
        #     print('The output val size is: {} .'.format(outputs_val.cpu().shape))
        #     print("The val batch size is: {}".format(val_batch["label"].shape))
        #     print(monai.networks.utils.one_hot(val_batch["label"].squeeze().unsqueeze(0).unsqueeze(0), 4).shape)
        dsc = dice_metric(outputs_val.cpu().unsqueeze(0), monai.networks.utils.one_hot(val_batch["label"].squeeze().unsqueeze(0).unsqueeze(0), 4))
        hd_metric = monai.metrics.HausdorffDistanceMetric()
        hd = hd_metric(outputs_val.cpu().unsqueeze(0), monai.networks.utils.one_hot(val_batch["label"].squeeze().unsqueeze(0).unsqueeze(0), 4))
        #     print(dsc.mean())
        print('Average DSC {:.2f}, average Hausdorff distance {:.2f} mm'.format(dsc.mean(), hd.mean()))
