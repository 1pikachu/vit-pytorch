import argparse
import os
import sys
import time

import torch
try:
    from context_func import context_func
except ModuleNotFoundError as e:
    print("!!!pls check how to add context_func.py from launch_benchmark.sh")
    sys.exit(0)


def create_model_input(args):
    # 256 input
    input = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
    # model
    if args.arch == "Token-to-Token-ViT":
        from vit_pytorch.t2t import T2TViT
        # 224 input
        input = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
        model = T2TViT(
            dim = 512,
            image_size = 224,
            depth = 5,
            heads = 8,
            mlp_dim = 512,
            num_classes = 1000,
            t2t_layers = ((7, 4), (3, 2), (3, 2)) # tuples of the kernel size and stride of each consecutive layers of the initial token to token module
        )
    elif args.arch == "LeViT":
        from vit_pytorch.levit import LeViT
        # 224 input
        input = torch.randn(args.batch_size, 3, args.image_size, args.image_size)
        model = LeViT(
            image_size = 224,
            num_classes = 1000,
            stages = 3,             # number of stages
            dim = (256, 384, 512),  # dimensions at each stage
            depth = 4,              # transformer of depth 4 at each stage
            heads = (4, 6, 8),      # heads at each stage
            mlp_mult = 2,
            dropout = 0.1
        )
    elif args.arch == "DeepViT":
        from vit_pytorch.deepvit import DeepViT
        model = DeepViT(
            image_size = 256,
            patch_size = 32,
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
    elif args.arch == "CaiT":
        from vit_pytorch.cait import CaiT
        model = CaiT(
            image_size = 256,
            patch_size = 32,
            num_classes = 1000,
            dim = 1024,
            depth = 12,             # depth of transformer for patch to patch attention only
            cls_depth = 2,          # depth of cross attention of CLS tokens to patch
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1,
            layer_dropout = 0.05    # randomly dropout 5% of the layers
        )
    else:
        from vit_pytorch import ViT
        model = ViT(
            image_size = 256,
            patch_size = 32,
            num_classes = 1000,
            dim = 1024,
            depth = 6,
            heads = 16,
            mlp_dim = 2048,
            dropout = 0.1,
            emb_dropout = 0.1
        )
    return model.eval().to(args.device), input.to(args.device)

def test(args):

    model, input = create_model_input(args)

    # NHWC
    if args.channels_last or args.device == "cuda":
        try:
            model = model.to(memory_format=torch.channels_last)
            print("---- Use NHWC model.")
            input = input.to(memory_format=torch.channels_last)
            print("---- Use NHWC input.")
        except:
            pass
    # JIT
    if args.jit:
        with torch.no_grad():
            try:
                model = torch.jit.trace(model, input, check_trace=False)
                print("---- JIT trace enable.")
            except (RuntimeError, TypeError) as e:
                print("---- JIT trace disable.")
                print("failed to use PyTorch jit mode due to: ", e)
    if args.nv_fuser:
        fuser_mode = "fuser2"
    else:
        fuser_mode = "none"
    print("---- fuser mode:", fuser_mode)

    # compute
    total_sample = 0
    total_time = 0.0

    profile_len = args.num_iter // 2
    with torch.no_grad():
        for i in range(args.num_iter):
            input = torch.randn(args.batch_size, 3, 224, 224)
            if args.channels_last or args.device == "cuda":
                try:
                    input = input.to(memory_format=torch.channels_last)
                    print("---- Use NHWC intput.")
                except:
                    pass
            tic = time.time()
            with context_func(args.profile if i == profile_len else False, args.device, fuser_mode) as prof:
                input = input.to(args.device)
                preds = model(input)
                torch.cuda.synchronize()
            toc = time.time()
            # caculate time
            print("Iteration: {}, inference time: {} sec.".format(i, toc - tic), flush=True)
            if i >= args.num_warmup:
                total_time += (toc - tic)
                total_sample += args.batch_size

    latency = total_time / total_sample * 1000
    throughput = total_sample / total_time
    print("\nLatency: {} ms\nThroughput: {} images/s\n".format(latency, throughput))

    # assert preds.shape == (1, 1000), 'correct logits outputted'


if __name__ == '__main__':
    #
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='ViT',
                        help='model architecture (default: ViT)')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    parser.add_argument('--precision', default="float32", type=str, help='precision')
    parser.add_argument('--channels_last', default=1, type=int, help='Use NHWC or not')
    parser.add_argument('--jit', action='store_true', default=False, help='enable JIT')
    parser.add_argument('--nv_fuser', action='store_true', default=False)
    parser.add_argument('--cuda', action='store_true', default=False, help='Use CUDA')
    parser.add_argument('--profile', action='store_true', default=False, help='collect timeline')
    parser.add_argument('--num_iter', default=200, type=int, help='test iterations')
    parser.add_argument('--num_warmup', default=20, type=int, help='test warmup')
    parser.add_argument('--image_size', default=256, type=int)
    parser.add_argument('--device', default='cuda', choices=['xpu', 'cuda', 'cpu'], type=str)
    args = parser.parse_args()

    if args.device == "xpu":
        import intel_extension_for_pytorch

    if args.arch == "LeViT" or args.arch == "Token-to-Token-ViT":
        args.image_size = 224
    else:
        args.image_size = 256
    print(args)

    # start test
    if args.precision == "bfloat16":
        amp_enable = True
        amp_dtype = torch.bfloat16
    elif args.precision == "float16":
        amp_enable = True
        amp_dtype = torch.float16
    else:
        amp_enable = False
        amp_dtype = torch.float32
    print("----amp enable: {}, amp dtype: {}".format(amp_enable, amp_dtype))

    with torch.autocast(device_type=args.device, enabled=amp_enable, dtype=amp_dtype):
        test(args)

