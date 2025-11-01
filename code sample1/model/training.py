from __future__ import division
import torch



class Trainer(object):
    def __init__(self, train_loader, model, opt, device):
        self.args = opt
        self.train_loader = train_loader
        self.model = model
        self.device = device
        self.criterion = torch.nn.functional.binary_cross_entropy
        self.optimizer = torch.optim.Adam(lr=0.001, params=model.parameters())
        self.epochs = 200
        self.model.to(device)



    def train_model(self):
        print(f'开始训练，共 {self.epochs} 个epoch')
        for epoch in range(self.epochs):
            losses = []
            for i, (img, mask) in enumerate(self.train_loader):
                img, mask = img.to(self.device), mask.float().to(self.device)

                self.optimizer.zero_grad()
                output = self.model(img)
                print(f"Image range: [{img.min():.3f}, {img.max():.3f}]")
                print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
                print(f"Mask range: [{mask.min():.3f}, {mask.max():.3f}]")
                loss = self.criterion(output, mask)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                print(f'epoch {epoch}, step {i} --- train_loss: {loss.item():.6f}')

            avg_loss = sum(losses) / len(losses)
            print(f'epoch {epoch} ----------- avg_loss: {avg_loss:.6f}')

            if epoch % 10 == 0:
                torch.save(self.model.state_dict(), f'unet-{epoch}.pth')
                print(f'模型已保存: unet-{epoch}.pth')

        torch.save(self.model.state_dict(), f'Unet-epochs{self.epochs}.pth')
        print(f'训练完成，最终模型已保存: Unet-epochs{self.epochs}.pth')


