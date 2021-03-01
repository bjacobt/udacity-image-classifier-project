import argparse
from my_model import MyModel
from helper_functions import get_data_loader, get_testing_transformations, get_training_transformations

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='directory containing data')
    parser.add_argument('--save_dir', type=str, default='.', help='__model checkpoint directory')
    parser.add_argument('--arch', type=str, default='resnet50', choices=['resnet50', 'densenet121'],
                        help='__model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.003)
    parser.add_argument('--hidden_units', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--gpu', action='store_true', help='train on gpu')

    args = parser.parse_args()

    model = MyModel(arch=args.arch, gpu=args.gpu, hidden_layers=args.hidden_units)

    train_dir = f'{args.data_dir}/train'
    test_dir = f'{args.data_dir}/test'
    valid_dir = f'{args.data_dir}/valid'
    train_data, train_loader = get_data_loader(train_dir, get_training_transformations(), shuffle=True)
    valid_data, valid_loader = get_data_loader(test_dir, get_testing_transformations())
    
    model.train(epochs=args.epochs, lr=args.learning_rate, trainloader=train_loader, validloader=valid_loader)
    model.save(args.save_dir, train_data, args.epochs, args.learning_rate)
