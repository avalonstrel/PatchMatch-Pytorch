import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import time
import sys

SHAPE = [64,64]
PSIZE = 2

def get_spatial_feat(feat, pos, psize):
    """
    Take the feat at on position
    Params:
        feat(torch.Tensor) :B*C*H*W
        pos(torch.Tensor): B*2
    Return: Batch feature(torch.Tensor): B*C*psize*psize
    """
    return torch.stack([feat[i, :, pos[i, 0]:pos[i, 0]+psize,
        pos[i, 1]:pos[i, 1]+psize] for i in range(feat.size(0))])

def get_patch_feats(feat1, feat2, pos1, pos2, psize=2):
    """
    Given the features and the position, return the 4 feat for compute energy function
    """
    pos_feat1 = get_spatial_feat(feat1, pos1, psize)
    pos_feat2 = get_spatial_feat(feat2, pos2, psize)

    return (pos_feat1, pos_feat2)

def f_mapping(f, pos):
    """
    Implement f on position, return the position mapped by f
    """
    return torch.stack([f[i, pos[i, 0], pos[i, 1]]  for i in range(pos.size(0))])

def dist_f_mapping(dist_f, pos):
    dist = torch.zeros(pos.size(0))
    for i in range(pos.size(0)):
        dist[i] = dist_f[i, pos[i, 0], pos[i, 1]]
    return dist

def distance(feat1, feat2):
    """
    A distance function return shape B tensor
    """
    batch_size = feat1.size(0)
    dist = F.pairwise_distance((feat1).reshape(batch_size, -1), (feat2).reshape(batch_size, -1))
    return dist

def feat_patch_distance(feat1, feat2, pos, pos_f,  psize):
    """
    Get the patch distance between two patch
    """
    pos_feat1, pos_feat2 = get_patch_feats(feat1, feat2,  pos, pos_f, psize)
    best_dist = distance(pos_feat1, pos_feat2)
    return best_dist

def get_pos_dist(pos, f, feat1, feat2, psize=2):
    """
    Return the mapped position and dist corresponding to the pos and f
    """
    pos_f = f_mapping(f, pos)
    dist = feat_patch_distance(feat1, feat2, pos, pos_f, psize)
    return pos_f, dist

def improve_guess(pos, pos_new_f, best_pos_f, best_dist, feat1, feat2, psize=2):
    """
    Return the best b position and corresponding distance
    Params:
        pos(B*2):position for update
        pos_f(B*2): new_position in b for update
        best_pos_f(B*2): best pos now
        best_dist(B): best distance now
        feat*(B*C*H*W): features
    """

    new_dist = feat_patch_distance(feat1, feat2, pos, pos_new_f, psize)

    for i in range(pos.size(0)):
        if best_dist[i] > new_dist[i]:
            best_pos_f[i] = pos_new_f[i]
            best_dist[i] = new_dist[i]
    return best_pos_f, best_dist

def update_f_dist_f(pos, f, dist_f, best_pos_f, best_dist):
    """
    Use the best position and distance to update the mapping function and distance function
    """
    best_f = f.clone()
    best_dist_f = dist_f.clone()
    for i in range(pos.size(0)):
        best_f[i,pos[i,0], pos[i,1]] = best_pos_f[i]
        best_dist_f[i,pos[i,0], pos[i,1]] = best_dist[i]
    return best_f, best_dist_f

def propagation(pos, change, f, dist_f, feat1, feat2, ae_shape, be_shape, psize=2):
    """
    Batch Propagation in patch match.
    Params:
        pos(torch.Tensor:B*2): batch of position
        change(torch.Tensor:2): direction for propagation
        f(torch.Tensor:B*H*W*2): a \phi_a->b function represented by a tensor relative position
        dist_f(torch.Tensor:B*H*W): a \phi_a->b function represented by a tensor min dist
        feat*(torch.Tensor:B*C*H*W): batch features
    Return best_f(torch.Tensor:B*H*W*2) best_dist_f(torch.Tensor:B*H*W)
    """
    best_pos_f = f_mapping(f, pos)
    best_dist = f_mapping(dist_f, pos)

    # make the change variable
    up_change, left_change  = torch.zeros_like(torch.tensor(change)), torch.zeros_like(torch.tensor(change))
    up_change[0], left_change[1] = change
    # Batch pos adding up_change new B*2

    pos_new = pos - left_change
    if pos_new[0][1] < ae_shape[1] and pos_new[0][1] >= 0:
        pos_new_f = f_mapping(f, pos_new)
        pos_new_f = pos_new_f + left_change
        if pos_new_f[0][1] < be_shape[1] and pos_new_f[0][1] >= 0:
            best_pos_f, best_dist = improve_guess(pos, pos_new_f, best_pos_f, best_dist, feat1, feat2, psize)

    pos_new = pos - up_change
    if pos_new[0][0] < ae_shape[0] and pos_new[0][0] >=0 :
        pos_new_f = f_mapping(f, pos_new)
        pos_new_f = pos_new_f + up_change
        if pos_new_f[0][0] < be_shape[0] and pos_new_f[0][0] >= 0:
            best_pos_f, best_dist = improve_guess(pos, pos_new_f, best_pos_f, best_dist, feat1, feat2, psize)

    return best_pos_f, best_dist

def random_search(pos, f, dist_f, best_pos_f, best_dist, feat1, feat2, \
                    be_shape, alpha=0.5, psize=2):
    """
    Batch Random Search For patch Match
    """
    r = torch.tensor(be_shape).type(torch.FloatTensor)

    while r[0] > 1 and r[1] > 1:
        for j in range(1):
            pos_random_f = torch.zeros_like(best_pos_f)
            for i in range(pos_random_f.size(0)):
                xmin, xmax = max(best_pos_f[i][1].type(torch.FloatTensor)-r[1], 0.), min(best_pos_f[i][1].type(torch.FloatTensor)+r[1]+1, be_shape[1]-1+.0)
                ymin, ymax = max(best_pos_f[i][0].type(torch.FloatTensor)-r[0], 0.), min(best_pos_f[i][0].type(torch.FloatTensor)+r[0]+1, be_shape[0]-1+.0)
                pos_random_f[i] = (torch.tensor([ymin, xmin]) + torch.rand(2)*torch.tensor([ymax-ymin, xmax-xmin])).type(torch.LongTensor)
            best_pos_f, best_dist = improve_guess(pos, pos_random_f, best_pos_f, best_dist, feat1, feat2)

        r = alpha*r

    return best_pos_f, best_dist

def initialize_direction(i, ae_shape):
    if (i) % 2 == 1:
        change = [-1,-1]
        start = [ae_shape[0]-1, ae_shape[1]-1]
        end = [-1, -1]
    else:
        change = [1, 1]
        start = [0, 0]
        end = [ae_shape[0], ae_shape[1]]
    return change, start, end

def get_effective_shape(feat, psize):
    return (feat.size(2)-psize+1, feat.size(3)-psize+1)

def deep_patch_match(feat1, feat2, psize=2, iteration=5, alpha=0.5):
    """
    A deep patch match method based on two pairs data. Formulated in Deep Image Analogy
    Original version only use img1 and img2
    Params: img1(torch.Tensor):  shape B*C*H*W
    """
    ae_shape = get_effective_shape(feat1, psize)
    be_shape = get_effective_shape(feat2, psize)

    # initialization
    f = torch.zeros(feat1.size(0), feat1.size(2), feat1.size(3), 2).type(torch.LongTensor)
    dist_f = torch.zeros(feat1.size(0), feat1.size(2), feat1.size(3))
    for x in range(ae_shape[0]):
        for y in range(ae_shape[1]):
            f[:,x,y] = (torch.rand(2)*torch.tensor(ae_shape).type(torch.FloatTensor)).view(1,2).repeat(feat1.size(0), 1).type(torch.LongTensor)
            pos = torch.tensor([x,y]).view(1,2).repeat(feat1.size(0), 1)
            pos_f = f_mapping(f, pos)
            dist_f[:, x, y] = feat_patch_distance(feat1, feat2, pos, pos_f, psize)

    for i in range(iteration):
        print("Iteration {}: Running".format(i+1))
        change, start, end = initialize_direction(i, ae_shape)
        print('start:{}, end:{}, change:{}'.format(start, end, change))
        end_time = time.time()
        for x in range(int(start[0]), int(end[0]), int(change[0])):
            for y in range(int(start[1]), int(end[1]), int(change[1])):
                pos = torch.tensor([x,y]).view(1,2).repeat(feat1.size(0), 1)

                best_pos_f, best_dist = propagation(pos, change, f, dist_f, feat1, feat2, ae_shape, be_shape, psize)

                best_pos_f, best_dist = random_search(pos, f, dist_f, best_pos_f, best_dist, feat1, feat2, be_shape, psize=psize)


                f, dist_f = update_f_dist_f(pos, f, dist_f, best_pos_f, best_dist)
        print("Iteration {}: Finishing Time : {}".format(i+1, time.time()-end_time))
    return f

def reconstruct_avg(feat2, f, psize=2):
    """
    Reconstruct another batch feat1 from batch feat2 by f
    Params:
        feat2(torch.Tensor:shape (B*C*H*W)): feature 2
        f(torch.Tensor:shape (B*H*W*2)): f : 1->2
    """

    feat1 = torch.zeros_like(feat2)

    for x in range(feat2.size(2)):
        for y in range(feat2.size(3)):
            pos = torch.zeros(feat2.size(0), 2).type(torch.LongTensor)
            pos = pos + torch.tensor([x,y]).type(torch.LongTensor)
            pos_f = f_mapping(f, pos)
            batch_feat = get_spatial_feat(feat2, pos_f, psize)
            b,c,hp,wp = batch_feat.size()
            feat1[:,:,x,y] = batch_feat.view(b,c,hp*wp).mean(dim=2)

    return feat1[:, :, :feat1.size(2)-psize, :feat1.size(3)-psize]

def reshape_test(img):
    return img.view(1, *img.size())

def img_padding(img, psize):
    """
    Input B*C*H*W
    """
    new_img = torch.zeros(img.size(0), img.size(1), img.size(2)+psize, img.size(3)+psize)
    new_img[:,:,:img.size(2), :img.size(3)] = img
    return new_img
def save_img(img, name):
    img = img.transpose(0,1).transpose(1,2)*255
    img = Image.fromarray(img.numpy().astype(np.uint8))
    img.save(name)

def main():
    transforms_fun = transforms.Compose([transforms.Resize(SHAPE),transforms.ToTensor()])
    img1 = transforms_fun(Image.open(sys.argv[1]))
    img2 = transforms_fun(Image.open(sys.argv[2]))
    img1, img2 = reshape_test(img1),reshape_test(img2)
    img1_pad = img_padding(img1, PSIZE)
    img2_pad = img_padding(img2, PSIZE)
    f = deep_patch_match(img1, img2, psize=PSIZE, iteration=5, alpha=0.5)

    re_img1 = reconstruct_avg(img2, f, psize=PSIZE)

    save_img(re_img1[0], sys.argv[3])

if __name__ == '__main__':
    main()
