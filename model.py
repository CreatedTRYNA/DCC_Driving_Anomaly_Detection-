import torch
from torch import nn
from models import shufflenet, shufflenetv2, resnet, mobilenet, mobilenetv2, resnetcds
from utils import _construct_depth_model
from models import resnet2D
from models.vision_transformer import QuickGELU, TransformerFusedBlock, TransformerConcatFusedBlock


def gen_transformer_layer(num_layers: int = 6,
                          feature_dim: int = 512,
                          out_feature_dim: int = 512,
                          num_heads: int = 8,
                          mlp_factor: float = 4.0,
                          act: nn.Module = QuickGELU,
                          encoder_decoder : str = "decoder"):
    if encoder_decoder == "decoder":
        block = TransformerFusedBlock(feature_dim=feature_dim,
                                      out_feature_dim=feature_dim,
                                      num_layers=num_layers,
                                      num_heads=num_heads,
                                      mlp_factor=mlp_factor,
                                      act=act)
    else:
        feature_dim = 1024
        block = TransformerConcatFusedBlock(feature_dim=feature_dim,
                                            out_feature_dim=out_feature_dim,
                                            num_layers=num_layers,
                                            num_heads=num_heads,
                                            mlp_factor=mlp_factor,
                                            act=act)
    return block


def gen_model(model_name: str = 'resnet18'):
    if model_name == 'resnet18':
        model = resnet2D.resnet18()
        model_weight_path = "premodels/resnet18.pth"  # 权重路径
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(model_weight_path), strict=False)  # 载入模型权重
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = torch.nn.Identity()  # 重新确定全连接层
    else:
        model = resnet2D.resnet50()
        model_weight_path = "premodels/resnet50.pt"  # 权重路径
        missing_keys, unexpected_keys = model.load_state_dict(torch.load(model_weight_path), strict=False)  # 载入模型权重
        model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = torch.nn.Linear(512 * 4, 512)  # 重新确定全连接层

    return model


def generate_model(args):
    assert args.model_type in ['resnet', 'shufflenet', 'shufflenetv2', 'mobilenet', 'mobilenetv2']
    if args.pre_train_model == False or args.mode == 'test':
        print('Without Pre-trained model')
        if args.model_type == 'resnet':
            assert args.model_depth in [18, 50, 101]
            if args.model_depth == 18:

                model = resnet.resnet18(
                    output_dim=args.feature_dim,
                    sample_size=args.sample_size,
                    sample_duration=args.sample_duration,
                    shortcut_type=args.shortcut_type,
                    tracking=args.tracking,
                    pre_train=args.pre_train_model,
                    use_cbam=args.cbam
                )

            elif args.model_depth == 50:
                model = resnet.resnet50(
                    output_dim=args.feature_dim,
                    sample_size=args.sample_size,
                    sample_duration=args.sample_duration,
                    shortcut_type=args.shortcut_type,
                    tracking=args.tracking,
                    pre_train=args.pre_train_model,
                    use_cbam=args.cbam
                )

            elif args.model_depth == 101:
                model = resnet.resnet101(
                    output_dim=args.feature_dim,
                    sample_size=args.sample_size,
                    sample_duration=args.sample_duration,
                    shortcut_type=args.shortcut_type,
                    tracking=args.tracking,
                    pre_train=args.pre_train_model,
                    use_cbam=args.cbam
                )

        elif args.model_type == 'shufflenet':
            model = shufflenet.get_model(
                groups=args.groups,
                width_mult=args.width_mult,
                output_dim=args.feature_dim,
                pre_train=args.pre_train_model
            )
        elif args.model_type == 'shufflenetv2':
            model = shufflenetv2.get_model(
                output_dim=args.feature_dim,
                sample_size=args.sample_size,
                width_mult=args.width_mult,
                pre_train=args.pre_train_model
            )
        elif args.model_type == 'mobilenet':
            model = mobilenet.get_model(
                sample_size=args.sample_size,
                width_mult=args.width_mult,
                pre_train=args.pre_train_model
            )
        elif args.model_type == 'mobilenetv2':
            model = mobilenetv2.get_model(
                sample_size=args.sample_size,
                width_mult=args.width_mult,
                pre_train=args.pre_train_model
            )

        # model = nn.DataParallel(model, device_ids=None)
    else:
        if args.model_type == 'resnet':
            # pre_model_path = './premodels/resnet-18-kinetics.pth'
            pre_model_path = './premodels/r3d18_K_200ep.pth'
            ###default pre-trained model is trained on kinetics dataset which has 600 classes
            if args.model_depth == 18:
                model = resnet.resnet18(
                    output_dim=args.feature_dim,
                    sample_size=args.sample_size,
                    sample_duration=args.sample_duration,
                    shortcut_type='A',
                    tracking=args.tracking,
                    pre_train=args.pre_train_model,
                    use_ssmctb=False
                )


            elif args.model_depth == 50:
                pre_model_path = './premodels/r3d50_K_200ep.pth'
                model = resnet.resnet50(
                    output_dim=args.feature_dim,
                    sample_size=args.sample_size,
                    sample_duration=args.sample_duration,
                    shortcut_type='B',
                    tracking=args.tracking,
                    pre_train=args.pre_train_model,
                    use_ssmctb=args.use_ssmctb
                )

            elif args.model_depth == 101:
                model = resnet.resnet101(
                    output_dim=args.feature_dim,
                    sample_size=args.sample_size,
                    sample_duration=args.sample_duration,
                    shortcut_type='B',
                    tracking=args.tracking,
                    pre_train=args.pre_train_model,
                    use_ssmctb=args.use_ssmctb
                )

        elif args.model_type == 'shufflenet':
            pre_model_path = './premodels/kinetics_shufflenet_' + str(args.width_mult) + 'x_G3_RGB_16_best.pth'
            model = shufflenet.get_model(
                groups=args.groups,
                width_mult=args.width_mult,
                output_dim=args.feature_dim,
                pre_train=args.pre_train_model

            )

        elif args.model_type == 'shufflenetv2':
            pre_model_path = './premodels/kinetics_shufflenetv2_' + str(args.width_mult) + 'x_RGB_16_best.pth'
            model = shufflenetv2.get_model(
                output_dim=args.feature_dim,
                sample_size=args.sample_size,
                width_mult=args.width_mult,
                pre_train=args.pre_train_model
            )
        elif args.model_type == 'mobilenet':
            pre_model_path = './premodels/kinetics_mobilenet_' + str(args.width_mult) + 'x_RGB_16_best.pth'
            model = mobilenet.get_model(
                sample_size=args.sample_size,
                width_mult=args.width_mult,
                pre_train=args.pre_train_model
            )
        elif args.model_type == 'mobilenetv2':
            pre_model_path = './premodels/kinetics_mobilenetv2_' + str(args.width_mult) + 'x_RGB_16_best.pth'
            model = mobilenetv2.get_model(
                sample_size=args.sample_size,
                width_mult=args.width_mult,
                pre_train=args.pre_train_model
            )

        # model = nn.DataParallel(model, device_ids=None)  # in order to load pre-trained model
        model_dict = model.state_dict()
        pretrained_dict = torch.load(pre_model_path)['state_dict']
        # print(len(pretrained_dict.keys()))
        # print({k for k, v in pretrained_dict.items() if k not in model_dict})
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model = _construct_depth_model(model)
    if args.use_cuda:
        model = model.cuda()
    return model


class DynamicFusingModel(nn.Module):
    def __init__(self, views):
        super().__init__()
