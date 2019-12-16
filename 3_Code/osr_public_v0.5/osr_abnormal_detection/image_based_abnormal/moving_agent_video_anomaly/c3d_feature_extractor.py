import torch
from torch.autograd import Variable
import numpy as np

from dataset import Video
from spatial_transforms import (Compose, Normalize, Scale, CenterCrop, ToTensor)
from temporal_transforms import LoopPadding

def extract_features(video_dir, video_name, class_names, model, opt, annotation_digit = 5):
    assert opt.mode in ['score', 'feature']

    spatial_transform = Compose([Scale(opt.sample_size),
                                 CenterCrop(opt.sample_size),
                                 ToTensor(),
                                 Normalize(opt.mean, [1, 1, 1])])
    temporal_transform = LoopPadding(opt.sample_duration)
    data = Video(video_dir, spatial_transform=spatial_transform,
                 temporal_transform=temporal_transform,
                 sample_duration=opt.sample_duration)
    data_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size,
                                              shuffle=False, num_workers=opt.n_threads, pin_memory=True)

    # print('reading file from: ', video_dir, 'file name: ', video_name)

    video_outputs = []
    video_segments = []
    model.eval()
    for i, (inputs, segments) in enumerate(data_loader):
        # inputs = Variable(inputs, volatile=True)
        inputs = inputs.cuda()
        outputs = model(inputs)
        # outputs_cpu = outputs.cpu().data.numpy()
        # video_outputs += outputs_cpu
        # video_outputs += outputs.cpu().data
        # np.vstack([video_outputs, outputs_cpu])
        video_outputs.append(outputs.cpu().data)
        # video_outputs.cat(video_outputs, outputs.cpu().data)
        video_segments.append(segments)

    video_outputs = torch.cat(video_outputs)
    video_segments = torch.cat(video_segments)
    results = {
        'video': video_name,
        'clips': []
    }

    _, max_indices = video_outputs.max(dim=1)
    for i in range(video_outputs.size(0)):
        clip_results = {
            'segment': video_segments[i].tolist(),
        }

        if opt.mode == 'score':
            clip_results['label'] = class_names[max_indices[i]]
            clip_results['scores'] = video_outputs[i].tolist()
        elif opt.mode == 'feature':
            clip_results['features'] = video_outputs[i].tolist()
            clip_results['ground_truth_annotaion'] = annotation_digit

        results['clips'].append(clip_results)

    total_feature_vectors = len(results["clips"])
    np_data = np.array([], dtype=np.float64).reshape(0, 2048)
    for features_in_one_video in range(total_feature_vectors):
        # for i in result[1]["clips"]:
        # print (i["scores"])
        one_feature_vector = results["clips"][features_in_one_video]["features"]
        a = np.asarray(one_feature_vector)
        # print(a)
        np_data = np.vstack([np_data, a])


    return np_data
