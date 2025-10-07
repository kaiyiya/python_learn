
from dataset import *
from model.net import *
from model.training import *
import configs.config_loader as cfg_loader



if __name__ == '__main__':

    args = cfg_loader.get_config()


    dataset = MyDataset('data/train/img', 'data/train/gt')
    dataloader = DataLoader(dataset=dataset, batch_size=4, shuffle=True)

    model = UNet(in_channels=1, out_channels=1)

    trainer = Trainer(dataloader, model, args)
    trainer.train_model()

