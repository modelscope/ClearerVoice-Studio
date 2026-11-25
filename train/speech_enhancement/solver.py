import time, os
import torch
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from losses.loss import loss_frcrn_se_16k, loss_mossformer2_se_48k, loss_mossformergan_se_16k
from utils.misc import power_compress, power_uncompress, stft, istft, EPS

import warnings
warnings.filterwarnings("ignore")
class Solver(object):
    def __init__(self, args, model, optimizer, discriminator, optimizer_disc, train_data, validation_data, test_data):
        self.train_data = train_data
        self.validation_data = validation_data
        self.test_data = test_data
        self.args = args
        self.device = self.args.device

        self.print = False
        if (self.args.distributed and self.args.local_rank ==0) or not self.args.distributed:
            self.print = True
            self.writer = SummaryWriter('%s/tensorboard/' % args.checkpoint_dir)

        self.model = model
        self.optimizer=optimizer
        self.discriminator = discriminator
        self.optimizer_disc = optimizer_disc
        if self.args.distributed:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[self.args.local_rank],find_unused_parameters=True)
            if self.discriminator is not None:
                self.discriminator = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.discriminator)
                self.discriminator = DDP(self.discriminator, device_ids=[self.args.local_rank],find_unused_parameters=True)
        self._init()
 
        if self.args.network == 'FRCRN_SE_16K':
            self._run_one_epoch = self._run_one_epoch_frcrn_se_16k
        elif self.args.network == 'MossFormer2_SE_48K':
            self._run_one_epoch = self._run_one_epoch_mossformer2_se_48k
        elif args.network=='MossFormerGAN_SE_16K':
            self._run_one_epoch = self._run_one_epoch_mossformergan_se_16k
        else:
            print(f'_run_one_epoch is not implemented for {self.args.network}!')


    def _init(self):
        self.halving = False
        self.epoch = 0
        self.step = 0
        self.best_val_loss = float("inf")
        self.val_no_impv = 0

        if self.args.train_from_last_checkpoint:
            flag = self._load_model()
            if flag == 0:
                ckpt_name = os.path.join(self.args.checkpoint_dir, 'last_checkpoint')
                if not os.path.isfile(ckpt_name):
                    print('[!] Last checkpoints are not found. Start new training ...')
                else:
                    with open(ckpt_name, 'r') as f:
                        model_name = f.readline().strip()
                    checkpoint_path = os.path.join(self.args.checkpoint_dir, model_name)
                    self._load_pretrained_model(checkpoint_path)
        elif self.args.init_checkpoint_path != 'None':
            self._load_pretrained_model(self.args.init_checkpoint_path)
        else:
            if self.print: print('Start new training')

        self.model.to(self.device)
        if self.discriminator is not None:
            self.discriminator.to(self.device)

    def _load_model(self, mode='last_checkpoint', use_cuda=True, strict=True):
        ckpt_name = os.path.join(self.args.checkpoint_dir, mode)
        if not os.path.isfile(ckpt_name):
            mode = 'last_best_checkpoint'
            ckpt_name = os.path.join(self.args.checkpoint_dir, mode)
            if not os.path.isfile(ckpt_name):
                print('[!] Last checkpoints are not found. Start new training ...')
                self.epoch = 0
                self.step = 0
        else:
            print(f'Loading checkpoint: {ckpt_name}')
            with open(ckpt_name, 'r') as f:
                model_name = f.readline().strip()
                disc_name = f.readline().strip()
            checkpoint_path = os.path.join(self.args.checkpoint_dir, model_name)
            checkpoint = self.load_checkpoint(checkpoint_path)
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'], strict=strict)
            else:
                print('Checkpoint {ckpt_name} has no model key, will try _load_pretrained_model')
                return 0
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except Exception as e:
                print(f"self.optimizer is not loaded: {e}")
                if torch.__version__ >= '1.7.1' and self.args.network == 'MossFormer2_SE_48K':
                    print(f'your torch version is too high for {self.args.network}, please use torch {torch.__version__} or lower')
                if torch.__version__ >= '2.0.1' and self.args.network == 'MossFormerGAN_SE_16K':
                    print(f'your torch version is too low for {self.args.network}, please use torch {torch.__version__} or higher')

            self.epoch = checkpoint['epoch']
            self.step = checkpoint['step']
            if self.discriminator is not None and disc_name is not None:
                #load discriminator
                discriminator_path = os.path.join(self.args.checkpoint_dir, disc_name)
                checkpoint = self.load_checkpoint(discriminator_path)
                self.discriminator.load_state_dict(checkpoint['discriminator'], strict=strict)
                self.optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
            print('=> Reloaded previous model and optimizer. Continue training ...')

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path, map_location=lambda storage, loc: storage)
        return checkpoint
        
    def _load_pretrained_model(self, checkpoint_path, load_optimizer=False, load_training_stat=False):
        if os.path.isfile(checkpoint_path):
            print(f'Loading checkpoint: {checkpoint_path}\n')
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            # load model weights
            if 'model' in checkpoint:
                pretrained_model = checkpoint['model']
            else:
                pretrained_model = checkpoint
            state = self.model.state_dict()
            for key in state.keys():
                if key in pretrained_model and state[key].shape == pretrained_model[key].shape:
                     state[key] = pretrained_model[key]
                elif key.replace('module.', '') in pretrained_model and state[key].shape == pretrained_model[key.replace('module.', '')].shape:     
                     state[key] = pretrained_model[key.replace('module.', '')]
                elif 'module.'+key in pretrained_model and state[key].shape == pretrained_model['module.'+key].shape:
                     state[key] = pretrained_model['module.'+key]
                elif self.print: print(f'{key} not loaded')
            self.model.load_state_dict(state)
            for g in self.optimizer.param_groups:
                g['lr'] = self.args.finetune_learning_rate 
            if self.discriminator is not None:
                discriminator_path = checkpoint_path.replace('.pt', '.disc.pt')
                checkpoint = self.load_checkpoint(discriminator_path)
                self.discriminator.load_state_dict(checkpoint['discriminator'], strict=False)
                self.optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
            print(f'==> Done model init from {checkpoint_path}. Start finetune training ...')
        else:
            print(f'{checkpoint_path} is not found. Start new training ...')
            self.epoch = 0
            self.step = 0

        # load optimizer only
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            for g in self.optimizer.param_groups:
                g['lr'] = self.args.learning_rate

        # load the training states
        if load_training_stat:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epoch=checkpoint['epoch']
            self.step = checkpoint['step']
            if self.print: print("Resume training from epoch: {}".format(self.epoch))
        
    def save_checkpoint(self, mode='last_checkpoint'):
        checkpoint_path = os.path.join(
            self.args.checkpoint_dir, 'model.ckpt-{}-{}.pt'.format(self.epoch, self.step))
        torch.save({'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': self.epoch,
                    'step': self.step}, checkpoint_path)

        if self.discriminator is not None:
            checkpoint_path = os.path.join(self.args.checkpoint_dir, 'discriminator.ckpt-{}-{}.pt'.format(self.epoch, self.step))
            torch.save({'discriminator': self.discriminator.state_dict(),
                'optimizer_disc': self.optimizer_disc.state_dict(),
                'epoch': self.epoch,
                'step': self.step}, checkpoint_path)

        with open(os.path.join(self.args.checkpoint_dir, mode), 'w') as f:
            f.write('model.ckpt-{}-{}.pt\n'.format(self.epoch, self.step))
            if self.discriminator is not None:
                f.write('discriminator.ckpt-{}-{}.pt\n'.format(self.epoch, self.step))
        print("=> Save checkpoint:", checkpoint_path)

    def train(self):
        start_epoch = self.epoch
        for epoch in range(start_epoch, self.args.max_epoch+1):
            if self.args.distributed: self.args.train_sampler.set_epoch(epoch)
            # Train
            self.model.train()
            start = time.time()            
            tr_loss = self._run_one_epoch(data_loader = self.train_data)
            if self.args.distributed: tr_loss = self._reduce_tensor(tr_loss.to(self.device))
            if self.print: print(f'Train Summary | End of Epoch {epoch} | Time {time.time() - start:2.3f}s | Train Loss {tr_loss:2.4f}')

            # Validation
            self.model.eval()
            start = time.time()
            with torch.no_grad():
                val_loss = self._run_one_epoch(data_loader = self.validation_data, state='val')
                if self.args.distributed: val_loss = self._reduce_tensor(val_loss.to(self.device))
            if self.print: print(f'Valid Summary | End of Epoch {epoch} | Time {time.time() - start:2.3f}s | Valid Loss {val_loss:2.4f}')

            if self.args.tt_list is not None:
                # Test
                self.model.eval()
                start = time.time()
                with torch.no_grad():
                    test_loss = self._run_one_epoch(data_loader = self.test_data, state='test')
                    if self.args.distributed: test_loss = self._reduce_tensor(test_loss.to(self.device))
                if self.print: print(f'Test Summary | End of Epoch {epoch} | Time {time.time() - start:2.3f}s | Test Loss {test_loss:2.4f}')

            # Check whether to early stop and to reduce learning rate
            find_best_model = False
            if val_loss >= self.best_val_loss:
                self.val_no_impv += 1
                if self.val_no_impv == 5:
                    self.halving = True
                elif self.val_no_impv >= 10:
                    if self.print: print("No imporvement for 10 epochs, early stopping.")
                    break
            else:
                self.val_no_impv = 0
                self.best_val_loss = val_loss
                find_best_model=True

            # Halfing the learning rate
            if self.halving:
                self.halving = False
                self._load_model(mode='last_best_checkpoint')
                if self.print: print('reload from last best checkpoint')

                optim_state = self.optimizer.state_dict()
                optim_state['param_groups'][0]['lr'] *= 0.5
                self.optimizer.load_state_dict(optim_state)
                if self.print: print('Learning rate adjusted to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))
                

            if self.print:
                # Tensorboard logging
                self.writer.add_scalar('Train_loss', tr_loss, epoch)
                self.writer.add_scalar('Validation_loss', val_loss, epoch)
                if self.args.tt_list is not None:
                    self.writer.add_scalar('Test_loss', test_loss, epoch)

            # Save model
            self.save_checkpoint()
            if find_best_model:
                self.save_checkpoint(mode='last_best_checkpoint')
                print("Found new best model, dict saved")
            self.epoch = self.epoch + 1

    def _run_one_epoch_mossformergan_se_16k(self, data_loader, state='train'):
        num_batch = len(data_loader)
        gen_loss_print = 0.0
        disc_loss_print = 0.0
        total_loss = 0.0
        self.accu_count = 0
        self.optimizer.zero_grad()
        self.optimizer_disc.zero_grad()
        stime = time.time()
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            c = torch.sqrt(inputs.size(-1) / (torch.sum((inputs ** 2.0), dim=-1)+EPS))
            inputs = torch.transpose(inputs, 0, 1) #, torch.transpose(labels, 0, 1)
            inputs = torch.transpose(inputs * c, 0, 1) #, torch.transpose(labels * c, 0, 1)
            inputs_spec = stft(inputs, self.args, center=True)
            inputs_spec = inputs_spec.to(torch.float32)
            inputs_spec = power_compress(inputs_spec).permute(0, 1, 3, 2)

            Out_List = self.model(inputs_spec)

            loss_gen, loss_disc = loss_mossformergan_se_16k(self.args, inputs, labels, Out_List, c, self.discriminator, self.device)

            if state=='train':
                if self.args.accu_grad:
                    if loss_gen is not None:
                        self.accu_count += 1
                        loss_gen_scaled = loss_gen/(self.args.effec_batch_size / self.args.batch_size)
                        loss_gen_scaled.backward()            
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                        loss_disc_scaled = loss_disc/(self.args.effec_batch_size / self.args.batch_size)
                        loss_disc_scaled.backward()
                        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.args.clip_grad_norm)
                        if self.accu_count == (self.args.effec_batch_size / self.args.batch_size):
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            self.optimizer_disc.step()
                            self.optimizer_disc.zero_grad()
                            self.accu_count = 0
                else:
                    if loss_gen is not None:
                        loss_gen.backward()
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.args.clip_grad_norm)
                        self.optimizer_disc.step()
                        self.optimizer_disc.zero_grad()
                self.step += 1
                ##cal losses for printing
                if loss_gen is not None:
                    gen_loss_print += loss_gen.data.cpu()
                    disc_loss_print += loss_disc.data.cpu()
                if (i + 1) % self.args.print_freq == 0:
                    eplashed = time.time() - stime
                    speed_avg = eplashed / (i+1)
                    gen_loss_print_avg = gen_loss_print / self.args.print_freq
                    disc_loss_print_avg = disc_loss_print / self.args.print_freq
                    print('Train Epoch: {}/{} Step: {}/{} | {:2.3f}s/batch | lr {:1.4e} |'
                      '| Gen_Loss {:2.4f}'
                      '| Disc_Loss {:2.4f}'
                      .format(self.epoch, self.args.max_epoch,
                          i+1, num_batch, speed_avg, self.optimizer.param_groups[0]["lr"],
                          gen_loss_print_avg,
                          disc_loss_print_avg,
                        ))
                    gen_loss_print = 0.0
                    disc_loss_print = 0.0
                if (i + 1) % self.args.checkpoint_save_freq == 0:
                    self.save_checkpoint()
            
            if loss_gen is not None:
                total_loss += loss_gen.data.cpu()

        return total_loss / (i+1)


    def _run_one_epoch_frcrn_se_16k(self, data_loader, state='train'):
        num_batch = len(data_loader)
        sisnr_print = 0.0
        mix_loss_print = 0.0
        mask_loss_print = 0.0

        total_loss = 0
        self.accu_count = 0
        self.optimizer.zero_grad()
        stime = time.time()
        for i, (inputs, labels) in enumerate(data_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            Out_List = self.model(inputs)
            loss, CMask_MSE_Loss, SiSNR_loss = loss_frcrn_se_16k(self.args, inputs, labels, Out_List, self.device)

            if torch.isnan(loss):
                print('loss is nan, skip this batch')
                continue

            if state=='train':
                if self.args.accu_grad:
                    self.accu_count += 1
                    loss_scaled = loss/(self.args.effec_batch_size / self.args.batch_size)
                    loss_scaled.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    if self.accu_count == (self.args.effec_batch_size / self.args.batch_size):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.accu_count = 0
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.step += 1
                ##cal losses for printing
                mix_loss_print += loss.data.cpu()
                mask_loss_print += CMask_MSE_Loss.data.cpu()
                sisnr_print += SiSNR_loss.data.cpu()
                if (i + 1) % self.args.print_freq == 0:
                    eplashed = time.time() - stime
                    speed_avg = eplashed / (i+1)
                    mix_loss_print_avg = mix_loss_print / self.args.print_freq
                    mask_loss_print_avg = mask_loss_print / self.args.print_freq
                    sisnr_print_avg = sisnr_print / self.args.print_freq
                    print('Train Epoch: {}/{} Step: {}/{} | {:2.3f}s/batch | lr {:1.4e} |'
                      '| Total_Loss {:2.4f}'
                      '| CMask_Loss {:2.4f}'
                      '| SiSNR_Loss {:2.4f}'
                      .format(self.epoch, self.args.max_epoch,
                          i+1, num_batch, speed_avg, self.optimizer.param_groups[0]["lr"],
                          mix_loss_print_avg,
                          mask_loss_print_avg,
                          sisnr_print_avg
                        ))
                    sisnr_print = 0.0
                    mix_loss_print = 0.0
                    mask_loss_print = 0.0
                if (i + 1) % self.args.checkpoint_save_freq == 0:
                    self.save_checkpoint()
            total_loss += loss.data.cpu()
        return total_loss / (i+1)
    
    def _run_one_epoch_mossformer2_se_48k(self, data_loader, state='train'):
        num_batch = len(data_loader)
        mask_loss_print = 0.0

        total_loss = 0
        self.accu_count = 0
        self.optimizer.zero_grad()
        stime = time.time()
        for i, (inputs, labels, fbanks) in enumerate(data_loader):
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            fbanks = fbanks.to(self.device)
            #seq_lens = seq_lens.to(self.device)

            Out_List = self.model(fbanks)
            loss = loss_mossformer2_se_48k(self.args, inputs, labels, Out_List, self.device)
            if torch.isnan(loss):
                print('loss is nan, skip this batch')
                continue
 
            if state=='train':
                if self.args.accu_grad:
                    self.accu_count += 1
                    loss_scaled = loss/(self.args.effec_batch_size / self.args.batch_size)
                    loss_scaled.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    if self.accu_count == (self.args.effec_batch_size / self.args.batch_size):
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self.accu_count = 0
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                self.step += 1
                ##cal losses for printing
                mask_loss_print += loss.data.cpu()
                if (i + 1) % self.args.print_freq == 0:
                    eplashed = time.time() - stime
                    speed_avg = eplashed / (i+1)
                    mask_loss_print_avg = mask_loss_print / self.args.print_freq
                    print('Train Epoch: {}/{} Step: {}/{} | {:2.3f}s/batch | lr {:1.4e} |'
                      '| Mask_Loss {:2.4f}'
                      .format(self.epoch, self.args.max_epoch,
                          i+1, num_batch, speed_avg, self.optimizer.param_groups[0]["lr"],
                          mask_loss_print_avg,
                        ))
                    mask_loss_print = 0.0
                if (i + 1) % self.args.checkpoint_save_freq == 0:
                    self.save_checkpoint()
            
            total_loss += loss.data.cpu()
        return total_loss / (i+1)
            
    def _reduce_tensor(self, tensor):
        if not self.args.distributed: return tensor
        rt = tensor.clone()
        dist.all_reduce(rt, op=dist.ReduceOp.SUM)
        rt /= self.args.world_size
        return rt
