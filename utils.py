import os
import torch
import torch.nn as nn
from shutil import copyfile


def distribute_over_GPUs(device, model, num_GPU=None):
    ## distribute over GPUs
    if num_GPU is None:
        model = nn.DataParallel(model)
        num_GPU = torch.cuda.device_count()
    else:
        assert (
            num_GPU <= torch.cuda.device_count()
        ), "You cant use more GPUs than you have."
        model = nn.DataParallel(model, device_ids=list(range(num_GPU)))

    model = model.to(device)
    print("Let's use", num_GPU, "GPUs!")

    return model


def create_model_training_folder(writer, files_to_same):
    model_checkpoints_folder = os.path.join(writer.log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            copyfile(file, os.path.join(model_checkpoints_folder, os.path.basename(file)))


def logfile(path, epoch, num_epochs, loss, rtnext=True):
    output = "Epoch:[{}/{}]\t \t Loss: \t \t {:.4f}".format(epoch + 1, num_epochs, loss)
    filename = os.path.join(path, 'log_file.txt')
    fo = open(filename, "a")
    if rtnext:
        fo.write(output+'\n')
    else:
        fo.write(output)
    fo.close()


def save_model(self, PATH):

        torch.save({
            'online_network_state_dict': self.online_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, PATH)