import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from tensorboardX import SummaryWriter
from torchvision.utils import save_image
import os
import torch
import torch.nn as nn
import random
import numpy as np
import shutil

cmd_train="  "
assert cmd_train!=None, "the command is null"



np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
# torch.backends.cudnn.benchmark = True


def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
def train(opt, data_loader, model, writer):
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    total_steps = opt.epoch_count * dataset_size

    for epoch in range(opt.epoch_count, opt.niter+opt.niter_decay+1):
            #model.netD_depth.apply(init_weights)
            epoch_start_time = time.time()
            epoch_iter = 0


            for i, data in enumerate(dataset):

                iter_start_time = time.time()
                total_steps += opt.batchSize
                epoch_iter += opt.batchSize

                model.set_input(data)
                # psnr_epoch, ssim_epoch = model.test()
                if epoch <=90:
                    model.optimize_parameters_stage()
                elif epoch>90 and epoch <=110:
                    model.optimize_parameters_stage2()
                elif epoch>110:
                    model.optimize_parameters_stage3()
                errors = model.get_current_errors()


                # display on visdom
                if total_steps % opt.display_freq == 0:
                    writer.add_images('syn_haze_img', (model.syn_haze_img[0:1] + 1) / 2.0, total_steps,
                                      dataformats='NCHW')

                    writer.add_images('syn_dehazing_img', (model.syn_dehazing_img[0:1] + 1) / 2.0, total_steps,
                                      dataformats='NCHW')

                    writer.add_images('clear_img', (model.clear_img[0:1] + 1) / 2.0, total_steps, dataformats='NCHW')

                    writer.add_images('real_haze_img', (model.real_haze_img[0:1] + 1) / 2.0, total_steps,
                                      dataformats='NCHW')
                    writer.add_images('r_clahe_img', (model.real_clahe_img[0:1] + 1) / 2.0, total_steps,
                                      dataformats='NCHW')
                    writer.add_images('real_dehazing_img', (model.real_dehazing_img[0:1] + 1) / 2.0, total_steps,
                                      dataformats='NCHW')
                    writer.add_images('real_syn_dehazing_img ', (model.real_syn_dehazing_img [0:1] + 1) / 2.0, total_steps,dataformats='NCHW')
                    writer.add_images('rs_recon_haze_img ', (model.rs_recon_img[0:1] + 1) / 2.0,total_steps, dataformats='NCHW')
                    writer.add_scalar('loss_Syn_Dehazing', model.loss_Syn_Dehazing.item(), total_steps)
                    writer.add_scalar('loss_Syn_vgg', model.loss_Syn_vgg.item(), total_steps)

                    writer.add_scalar('loss_R_syn_Dehazing_TV', model.loss_R_syn_Dehazing_TV.item(), total_steps)
                    #writer.add_scalar('loss_R_syn_Dehazing_DC', model.loss_R_syn_Dehazing_DC.item(), total_steps)

                    writer.add_scalar('loss_D_real_un', model.loss_D_real_un.item(), total_steps)
                    writer.add_scalar('loss_D_fake_un', model.loss_D_fake_un.item(), total_steps)
                    writer.add_scalar('loss_G_un', model.loss_G_un.item(), total_steps)
                    writer.add_scalar('loss_recon', model.loss_recon.item(), total_steps)
                    writer.add_scalar('loss_vgg_A', model.loss_vgg_A.item(), total_steps)

                    writer.add_scalar('loss_real_vgg', model.loss_real_vgg.item(), total_steps)
                    writer.add_scalar('loss_real_Dehazing', model.loss_real_Dehazing.item(), total_steps)
                    writer.add_scalar('loss_R_Dehazing_TV', model.loss_R_Dehazing_TV.item(), total_steps)
                    writer.add_scalar('loss_real_Dehazing_feature', model.loss_real_Dehazing_feature.item(), total_steps)

                    writer.add_scalar('loss_G', model.loss_G.item(), total_steps)

                if (epoch+1) % opt.save_epoch_freq == 0:

                    save_image((model.syn_haze_img[0:1] + 1) / 2,
                               os.path.join(visual_dir, str(epoch) + "_syn_haze.png"),
                               nrow=1)
                    save_image((model.syn_dehazing_img[0:1] + 1) / 2,
                               os.path.join(visual_dir, str(epoch) + "_syn_dehazing_img.png"), nrow=1)
                    save_image((model.clear_img[0:1] + 1) / 2, os.path.join(visual_dir, str(epoch) + "_clear_img.png"),
                               nrow=1)

                    save_image((model.real_haze_img[0:1] + 1) / 2,
                               os.path.join(visual_dir, str(epoch) + "_real_haze_img.png"), nrow=1)
                    save_image((model.real_clahe_img[0:1] + 1) / 2,
                               os.path.join(visual_dir, str(epoch) + "_real_clahe_img.png"), nrow=1)
                    save_image((model.real_dehazing_img[0:1] + 1) / 2,
                               os.path.join(visual_dir, str(epoch) + "_real_dehazing_img.png"), nrow=1)

                    save_image((model.real_syn_dehazing_img[0:1] + 1) / 2.0,
                               os.path.join(visual_dir, str(epoch) + "_real_syn_dehazing_img.png"), nrow=1)
                    save_image((model.rs_recon_img[0:1] + 1) / 2.0,
                               os.path.join(visual_dir, str(epoch) + "_real_recon_haze_img.png"), nrow=1)

            if epoch % opt.save_epoch_freq == 0:
                print('saving the model at the end of epoch %d, iters %d' %
                      (epoch, total_steps))

                model.save_networks(epoch)
            if epoch>20:
                opt.save_epoch_freq=1


            model.update_learning_rate(epoch)


            print('End of epoch %d / %d \t Time Taken: %d sec\t learning rate %.7f ' %
                  (epoch, opt.niter + opt.niter_decay,time.time() - epoch_start_time,model.optimizers[0].param_groups[0]['lr']))
            with open(log_dir,"a") as training_log:
                training_log.write('End of epoch %d / %d \t Time Taken: %d sec\t learning rate %.7f ' %
                  (epoch, opt.niter + opt.niter_decay,time.time() - epoch_start_time,model.optimizers[0].param_groups[0]['lr']))
                training_log.write('\n')

    # for epoch in range(opt.niter, opt.niter + opt.niter_decay + 1):
    #         #model.netD_depth.apply(init_weights)
    #         epoch_start_time = time.time()
    #         epoch_iter = 0
    #
    #
    #         for i, data in enumerate(dataset):
    #
    #             iter_start_time = time.time()
    #             total_steps += opt.batchSize
    #             epoch_iter += opt.batchSize
    #
    #             model.set_input(data)
    #
    #             model.optimize_parameters_stage2(i)
    #
    #
    #
    #             if total_steps % opt.display_freq == 0:
    #                 writer.add_images('syn_haze_img', (model.syn_haze_img[0:1] + 1) / 2.0, total_steps,
    #                                   dataformats='NCHW')
    #
    #                 writer.add_images('syn_dehazing_img', (model.syn_dehazing_img[0:1] + 1) / 2.0, total_steps,
    #                                   dataformats='NCHW')
    #
    #                 writer.add_images('clear_img', (model.clear_img[0:1] + 1) / 2.0, total_steps, dataformats='NCHW')
    #
    #                 writer.add_images('real_haze_img', (model.real_haze_img[0:1] + 1) / 2.0, total_steps,
    #                                   dataformats='NCHW')
    #                 writer.add_images('r_clahe_img', (model.real_clahe_img[0:1] + 1) / 2.0, total_steps,
    #                                   dataformats='NCHW')
    #                 writer.add_images('real_syn_dehazing_img', (model.real_syn_dehazing_img[0:1] + 1) / 2.0,total_steps,
    #                                   dataformats='NCHW')
    #
    #                 writer.add_images('r_dehazing_img', (model.real_dehazing_img[0:1] + 1) / 2.0, total_steps,
    #                                   dataformats='NCHW')
    #                 # writer.add_images('real_recon_haze_img ', (model.real_recon_img[0:1] + 1) / 2.0, total_steps,
    #                 #                   dataformats='NCHW')
    #
    #                 writer.add_scalar('loss_Syn_Dehazing', model.loss_Syn_Dehazing.item(), total_steps)
    #                 writer.add_scalar('loss_R_Dehazing_TV', model.loss_R_Dehazing_TV.item(), total_steps)
    #                 writer.add_scalar('loss_D_real_un', model.loss_D_real_un.item(), total_steps)
    #                 writer.add_scalar('loss_D_fake_un', model.loss_D_fake_un.item(), total_steps)
    #                 writer.add_scalar('loss_G_un', model.loss_G_un.item(), total_steps)
    #                 writer.add_scalar('R_Dehazing_DC', model.loss_R_Dehazing_DC.item(), total_steps)
    #
    #                 writer.add_scalar('loss_vgg_A', model.loss_vgg_A.item(), total_steps)
    #                 writer.add_scalar('loss_recon', model.loss_recon.item(), total_steps)
    #
    #
    #
    #         if epoch % opt.save_epoch_freq == 0:
    #             print('saving the model at the end of epoch %d, iters %d' %
    #                   (epoch, total_steps))
    #             model.save_networks(epoch)
    #             save_image((model.syn_haze_img[0:1] + 1) / 2,
    #                        os.path.join(visual_dir, str(epoch) + "_syn_haze.png"),
    #                        nrow=1)
    #
    #             save_image((model.syn_dehazing_img[0:1] + 1) / 2,
    #                        os.path.join(visual_dir, str(epoch) + "_syn_dehazing_img.png"), nrow=1)
    #             save_image((model.clear_img[0:1] + 1) / 2, os.path.join(visual_dir, str(epoch) + "_clear_img.png"),
    #                        nrow=1)
    #             save_image((model.real_haze_img[0:1] + 1) / 2,
    #                        os.path.join(visual_dir, str(epoch) + "_real_haze_img.png"), nrow=1)
    #             save_image((model.real_clahe_img[0:1] + 1) / 2,
    #                        os.path.join(visual_dir, str(epoch) + "_r_clahe_img.png"), nrow=1)
    #             save_image((model.real_dehazing_img[0:1] + 1) / 2,
    #                        os.path.join(visual_dir, str(epoch) + "_r_dehazing_img.png"), nrow=1)
    #             save_image((model.real_syn_dehazing_img[0:1]+ 1)/ 2.0,
    #                        os.path.join(visual_dir, str(epoch) + "real_syn_dehazing_img.png"), nrow=1)
    #             # save_image((model.real_recon_img[0:1] + 1) / 2.0,
    #             #            os.path.join(visual_dir, str(epoch) + "_real_recon_haze_img.png"), nrow=1)
    #
    #         if epoch>opt.niter:
    #             opt.save_epoch_freq=1
    #         model.update_learning_rate(epoch)
    #
    #
    #         print('End of epoch %d / %d \t Time Taken: %d sec\t learning rate %d ' %
    #               (epoch, opt.niter + opt.niter_decay,time.time() - epoch_start_time,model.opt.lr))
    #         with open(log_dir,"a") as training_log:
    #             training_log.write('End of epoch %d / %d \t Time Taken: %d sec\t learning rate %d ' %
    #               (epoch, opt.niter + opt.niter_decay,time.time() - epoch_start_time,opt.lr))
    #             training_log.write('\n')
    #         if epoch > opt.niter:
    #             model.update_learning_rate()
opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
model = create_model(opt)
# visualizer = Visualizer(opt)
log_dir=os.path.join(opt.checkpoints_dir,opt.name,'opt.txt')
# dst_dir=os.path.join(opt.checkpoints_dir,opt.name,'dehaze2stagescoor_model.py')
# src_dir ='models/dehaze2stagescoor_model.py'
# shutil.copy(src_dir,dst_dir)
visual_dir = os.path.join(opt.checkpoints_dir,opt.name,"visual")
if not os.path.exists(visual_dir):
    os.mkdir(visual_dir)
writer= SummaryWriter(visual_dir)
with open(log_dir,"a") as record_command:
    record_command.write(cmd_train)
    record_command.write('\n')
train(opt, data_loader, model, writer)
writer.close()
