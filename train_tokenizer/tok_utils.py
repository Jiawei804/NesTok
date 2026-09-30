import torchvision.transforms as transforms  
import torch 
from PIL import Image
import numpy as np
import torch.distributed as dist
import functools
import pickle
import logging
import torchvision

def load_img_list(image_list_path):
    img_dict = []
    with open(image_list_path, 'r') as f:
        for line in f:
            img_dict.append(line[:-1])
    return img_dict

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, size, root, image_list_path): 
        self.root = root
        self.image_list_path = image_list_path
        self.crop_size = size
        self.resize_size = size 
        
        self.img_list = load_img_list(self.image_list_path) 
        self.transform = transforms.Compose([
            transforms.Resize(self.resize_size),
            transforms.RandomCrop(self.crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0., 0., 0.), std=(1, 1, 1))
        ]) 

    def __getitem__(self, index):  
        path = self.root + '/' + self.img_list[index] 
        img = Image.open(path).convert('RGB')
        img_trans = self.transform(img) 
        return img_trans

    def __len__(self):
        return len(self.img_list) 

class LambdaLinearScheduler:
    def __init__(
        self, warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval=0
    ):
        assert (
            len(warm_up_steps)
            == len(f_min)
            == len(f_max)
            == len(f_start)
            == len(cycle_lengths)
        )
        self.lr_warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.cycle_lengths = cycle_lengths
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))
        self.last_f = 0.0
        self.verbosity_interval = verbosity_interval

    def find_in_interval(self, n):
        interval = 0
        for cl in self.cum_cycles[1:]:
            if n <= cl:
                return interval
            interval += 1

    def schedule(self, n, **kwargs):
        cycle = self.find_in_interval(n)
        n = n - self.cum_cycles[cycle]
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(
                    f"current step: {n}, recent lr-multiplier: {self.last_f}, "
                    f"current cycle {cycle}"
                )

        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[
                cycle
            ] * n + self.f_start[cycle]
            self.last_f = f
            return f
        else:
            f = self.f_min[cycle] + (self.f_max[cycle] - self.f_min[cycle]) * (
                self.cycle_lengths[cycle] - n
            ) / (self.cycle_lengths[cycle])
            self.last_f = f
            return f

    def __call__(self, n, **kwargs):
        return self.schedule(n, **kwargs)
    
class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0
    
    def update(self, val, n=1.0):
        self.val = val
        self.sum += val * n
        self.cnt += n
        if self.cnt == 0:
            self.avg = 1
        else:
            self.avg = self.sum / self.cnt

def save_image(tensor, path, nrow=8, normalize=True, value_range=(-1, 1)):
    tensor = tensor.clamp(min(value_range), max(value_range))
    torchvision.utils.save_image(
        tensor,
        path,
        nrow=nrow,
        normalize=normalize,
        value_range=value_range
    )

#------------------------ Basic operations ------------------------#
@torch.no_grad()
def load_state_dict(module, state_dict, drop_prefix=''):
    # find incompatible key-vals
    src, dst = state_dict, module.state_dict()
    if drop_prefix:
        src = type(src)([
            (k[len(drop_prefix):] if k.startswith(drop_prefix) else k, v)
            for k, v in src.items()
        ])
    missing = [k for k in dst if k not in src]
    unexpected = [k for k in src if k not in dst]
    unmatched = [k for k in src.keys() & dst.keys() if src[k].shape != dst[k].shape]

    # keep only compatible key-vals
    incompatible = set(unexpected + unmatched)
    src = type(src)([(k, v) for k, v in src.items() if k not in incompatible])
    module.load_state_dict(src, strict=False)

    # report incompatible key-vals
    if len(missing) != 0:
        print('  Missing: ' + ', '.join(missing), flush=True)
    if len(unexpected) != 0:
        print('  Unexpected: ' + ', '.join(unexpected), flush=True)
    if len(unmatched) != 0:
        print('  Shape unmatched: ' + ', '.join(unmatched), flush=True)
    return {'missing': missing, 'unexpected': unexpected, 'unmatched': unmatched}

def is_dist_initialized():
    return dist.is_available() and dist.is_initialized()

def get_world_size(group=None):
    return dist.get_world_size(group) if is_dist_initialized() else 1

def get_rank(group=None):
    return dist.get_rank(group) if is_dist_initialized() else 0

def all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, **kwargs):
    if get_world_size(group) > 1:
        return dist.all_reduce(tensor, op, group, **kwargs)
    
def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ['gloo', 'nccl']
    device = torch.device('cpu' if backend == 'gloo' else 'cuda')

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        logger = logging.getLogger(__name__)
        logger.warning(
            'Rank {} trying to all-gather {:.2f} GB of data on device'
            '{}'.format(get_rank(), len(buffer) / (1024 ** 3), device))
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor

def _pad_to_largest_tensor(tensor, group):
    world_size = dist.get_world_size(group=group)
    assert world_size >= 1, \
        'gather/all_gather must be called from ranks within' \
        'the give group!'
    local_size = torch.tensor(
        [tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [torch.zeros(
        [1], dtype=torch.int64, device=tensor.device)
        for _ in range(world_size)]

    # gather tensors and compute the maximum size
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # pad tensors to the same size
    if local_size != max_size:
        padding = torch.zeros(
            (max_size - local_size, ),
            dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    return size_list, tensor

@functools.lru_cache()
def get_global_gloo_group():
    backend = dist.get_backend()
    assert backend in ['gloo', 'nccl']
    if backend == 'nccl':
        return dist.new_group(backend='gloo')
    else:
        return dist.group.WORLD
    
def generalized_all_gather(data, group=None):
    if get_world_size(group) == 1:
        return [data]
    if group is None:
        group = get_global_gloo_group()
    
    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving tensors from all ranks
    tensor_list = [torch.empty(
        (max_size, ), dtype=torch.uint8, device=tensor.device)
        for _ in size_list]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))
    return data_list

def all_gather(tensor, uniform_size=True, group=None, **kwargs):
    world_size = get_world_size(group)
    if world_size == 1:
        return [tensor]
    assert tensor.is_contiguous(), \
        'ops.all_gather requires the tensor to be contiguous()'
    
    if uniform_size:
        tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor, group, **kwargs)
        return tensor_list
    else:
        # collect tensor shapes across GPUs
        shape = tuple(tensor.shape)
        shape_list = generalized_all_gather(shape, group)

        # flatten the tensor
        tensor = tensor.reshape(-1)
        size = int(np.prod(shape))
        size_list = [int(np.prod(u)) for u in shape_list]
        max_size = max(size_list)

        # pad to maximum size
        if size != max_size:
            padding = tensor.new_zeros(max_size - size)
            tensor = torch.cat([tensor, padding], dim=0)
        
        # all_gather
        tensor_list = [torch.empty_like(tensor) for _ in range(world_size)]
        dist.all_gather(tensor_list, tensor, group, **kwargs)

        # reshape tensors
        tensor_list = [t[:n].view(s) for t, n, s in zip(
            tensor_list, size_list, shape_list)]
        return tensor_list