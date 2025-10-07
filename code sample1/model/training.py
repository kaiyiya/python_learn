from __future__ import division
import torch



class Trainer(object):
    def __init__(self,train_loader,model, opt):
        self.args = opt
        self.train_loader = train_loader
        self.model = model
        self.criterion = torch.nn.functional.binary_cross_entropy
        self.optimizer = torch.optim.Adam(lr=0.003, params=model.parameters())
        self.epochs = 200
        self.model.cuda()



    def train_model(self):
        for epoch in range(self.epochs):
            losses = []
            for i, (img, mask) in enumerate(self.train_loader):
                img, mask = img.cuda(), mask.float().cuda()

                self.optimizer.zero_grad()
                output = self.model(img)
                loss = self.criterion(output, mask)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                if i % 10 == 0:
                    print(f'setp{i}---train_loss, ', loss.item())

            print(f'epoch{epoch}-----------loss:{sum(losses) / len(losses)}')
            if epoch % 10 == 0:
                torch.save(self.model, f'unet-{epoch}.pth.tar')

        torch.save(self.model, f'Unet-epochs{self.epochs}.pth')


