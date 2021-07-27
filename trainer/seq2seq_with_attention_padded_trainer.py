import torch
from trainer.base_trainer import BaseTrainer
#from model.accuracy import binary_accuracy


class Trainer(BaseTrainer):
    def __init__(self, resume: bool, model, loss_function, optimizer, scheduler, train_dataloader, test_dataloader, epochs, save_checkpoint_interval,
                 test_interval, output_dir, checkpoints_dir, find_max):
        super(Trainer, self).__init__(resume, model, loss_function, optimizer, scheduler, epochs, save_checkpoint_interval, test_interval, output_dir, checkpoints_dir, find_max)
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader


    def _train_epoch(self, epoch):
        total_loss = 0.0
        len_dataset = len(self.train_dataloader)
        step = 0
        for batch in self.train_dataloader:
            source, src_len = batch.src
            target = batch.trg

            self.optimizer.zero_grad()

            output = self.model(source, src_len, target)

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            target = target[1:].view(-1)

            loss = self.loss_function(output, target)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()

            #if step % self.global_step == 0:
            if step % 1000 == 0:
                print("Epoch {} | Global Step {} | Step: {}/{} | Train Loss {:.2f}".format(epoch, self.global_step, step, len_dataset, loss.item() ))

            step += 1
            self.global_step += 1

        print("Train Loss {:.2f}".format(total_loss / len_dataset))
        self.writer.add_scalar(f"Loss/Train", total_loss / len_dataset, epoch)


    @torch.no_grad()
    def _test_epoch(self, epoch):
        total_loss = 0.0
        len_dataset = len(self.test_dataloader)

        step = 0
        for batch in self.test_dataloader:
            source, src_len = batch.src
            target = batch.trg

            output = self.model(source, src_len, target, 0) #turn off teacher forcing

            '''
            print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(labels),
                                                                   100. * correct / len(labels),))
            '''
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            target = target[1:].view(-1)

            loss = self.loss_function(output, target)
            total_loss += loss.item()

            #if step % self.global_step == 0:
            if step % 1000 == 0:
                print("Epoch {} | Global Step {} | Step: {}/{} | Test Loss {:.2f} ".format(epoch, self.global_step, step, len_dataset, loss.item()))

            step += 1
            self.global_step += 1

        print("Test Loss {:.2f} ".format(total_loss / len_dataset) )
        self.writer.add_scalar(f"Loss/Test", total_loss / len_dataset, epoch)

        return total_acc / len_dataset
