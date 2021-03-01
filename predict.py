import argparse
import torch
import json
from helper_functions import process_image, load_model_from_checkpoint


def predict(image_path, model, topk=3):
    image = process_image(image_path)
    image = image.unsqueeze(0)

    with torch.no_grad():
        model.eval()
        logps = model.forward(image)
        ps = torch.exp(logps)
        top_p, top_class = ps.topk(topk, dim=1)

    return top_p, top_class


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=str, help='image to classify')
    parser.add_argument('checkpoint', type=str, help='path to checkpoint file')
    parser.add_argument('--top_k', type=int, default='3', help='return top k most likely classes')
    parser.add_argument('--gpu', action='store_true', help='predict on gpu')

    with open('cat_to_name.json', 'r') as f:
        flower_to_name = json.load(f)

    args = parser.parse_args()

    m = load_model_from_checkpoint(args.checkpoint, args.gpu)
    top_p, top_class = predict(args.image, m, args.top_k)

    ground_truth = flower_to_name[args.image.split('/')[-2]]

    top_list = top_class.tolist()[0]
    top_prob = top_p.tolist()[0]
    idx_to_class = {value: key for key, value in m.class_to_idx.items()}
    flower_names = [flower_to_name[cl] for cl in [idx_to_class[x] for x in top_list]]

    print(f"Predicting flower name from image '{args.image}'. Ground truth '{ground_truth}'")
    print("{:>10s} {:<50s}".format('Probability', 'Flower'))
    for name, prob in zip(flower_names, top_prob):
        print("{:>10.2f}% {:<50s}".format(prob * 100, name))
