import os
import time
import torch
from apex import amp
from args import parse
from torch.utils.data.dataloader import DataLoader

from torch.utils.tensorboard import SummaryWriter

import get_data
from encoder_net import ResNet_BYOL, MLPHead
from byol import BYOL
from utils import distribute_over_GPUs, create_model_training_folder, save_model, logfile


def train(args, model, optimizer, path, train_loader):
        niter = 0

        for epoch in range(args.num_epochs):
            print("Start epoch [{} / {}]". format(epoch + 1, args.num_epochs))
            start_time = time.time()

            for step, (x_1, x_2, target) in enumerate(train_loader):

                x_1 = x_1.to(args.device)
                x_2 = x_2.to(args.device)

                loss = model(x_1, x_2)

                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                niter += 1
                print_loss = loss.item()
            epoch_time = time.time() - start_time
            print("Epoch [{}/{}] : Loss {:.4f}  Time {:.2f}".format(epoch + 1, args.num_epochs, print_loss, epoch_time))
            logfile(path, epoch + 1, args.num_epochs, print_loss)

        if (epoch + 1) % 10 == 0:
            save_model(os.path.join(path, 'model-{}.pth').format(epoch + 1))



def main():
    args = parse()
    batch_size = args.batch_size
    num_workers = args.num_workers
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Training with: {device}")

    writer = SummaryWriter()
    create_model_training_folder(writer, files_to_same=["args.py", "main_byol.py"])
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')

    train_dataset, _, _ = get_data.get_dataset(args.dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=num_workers, drop_last=False,
                              shuffle=True)

    online = ResNet_BYOL(args.network, args.mlp_size, args.pro_size).cuda()
    predictor = MLPHead(online.projetion.net[-1].out_features,
                          args.mlp_size, args.pro_size).cuda()
    target = ResNet_BYOL(args.network, args.mlp_size, args.pro_size)

    optimizer = torch.optim.SGD(list(online.parameters()) + list(predictor.parameters()),
                                lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    online, optimizer = amp.initialize(online, optimizer, opt_level='O1')
    online = distribute_over_GPUs(args.device, online)
    predictor = distribute_over_GPUs(args.device, predictor)
    target = distribute_over_GPUs(args.device, target)

    model = BYOL( online, target, predictor, args)
    train(args, model, optimizer, model_checkpoints_folder, train_loader)


if __name__ == '__main__':
    main()
