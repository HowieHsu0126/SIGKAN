import argparse
import logging
import os
import shutil
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from loader import ChunkSampler, InfluenceDataSet
from layers import AttentionMechanism
from model import BatchGAT, BatchGCN, SIGKAN_Att, SIGKAN_Norm
from sklearn.metrics import (precision_recall_curve,
                             precision_recall_fscore_support, roc_auc_score)
from tensorboard_logger import tensorboard_logger
from torch.utils.data import DataLoader
from loss import compute_total_loss, psi
# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def parse_args():
    """Parse command line arguments to configure model training.

    Returns:
        argparse.Namespace: Parsed arguments from the command line.
    """
    parser = argparse.ArgumentParser(description="Let's train a model.")
    parser.add_argument('--tensorboard-log', type=str, default='',
                        help="Name of this run for TensorBoard logging.")
    parser.add_argument('--model', type=str, default='gcn',
                        choices=['gcn', 'pscn', 'sigkan_norm', 'sigkan_att'],
                        help="Model type to use ('gcn', 'pscn', 'sigkan_norm', 'sigkan_att').")
    parser.add_argument('--delta', type=float, default=1e-5,
                        help='Bounded confidence interval if using SIGKAN.')
    parser.add_argument('--no-cuda', action='store_true',
                        help='Disable CUDA training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--weight-decay', type=float,
                        default=5e-4, help='Weight decay (L2 regularization).')
    parser.add_argument('--dropout', type=float,
                        default=0.2, help='Dropout rate.')
    parser.add_argument('--hidden-units', type=str, default="16,8",
                        help="Comma-separated hidden units per layer.")
    parser.add_argument('--heads', type=str, default="1,1,1",
                        help="Comma-separated heads per layer.")
    parser.add_argument('--batch', type=int, default=2048, help="Batch size.")
    parser.add_argument('--dim', type=int, default=64,
                        help="Embedding dimension.")
    parser.add_argument('--check-point', type=int,
                        default=10, help="Checkpoint interval.")
    parser.add_argument('--instance-normalization',
                        action='store_true', help="Enable instance normalization.")
    parser.add_argument('--shuffle', action='store_true',
                        help="Shuffle dataset.")
    parser.add_argument('--file-dir', type=str,
                        required=True, help="Input file directory.")
    parser.add_argument('--train-ratio', type=float,
                        default=50, help="Training set ratio.")
    parser.add_argument('--valid-ratio', type=float,
                        default=25, help="Validation set ratio.")
    parser.add_argument('--class-weight-balanced',
                        action='store_true', help="Balance class weights.")
    parser.add_argument('--use-vertex-feature', action='store_true',
                        help="Use vertex structural features.")
    # ODE and adversarial loss weights
    parser.add_argument('--alpha', type=float, default=1.0, help='Weight for ODE loss.')
    parser.add_argument('--beta', type=float, default=1.0, help='Weight for adversarial loss.')
    parser.add_argument('--epsilon', type=float, default=0.01, help='Perturbation budget for adversarial training.')

    return parser.parse_args()


def setup_environment(args):
    """Set up random seeds, CUDA configuration, and TensorBoard logging environment.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
    """
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    tensorboard_log_dir = f'tensorboard/{args.model}_{args.tensorboard_log}'
    shutil.rmtree(tensorboard_log_dir, ignore_errors=True)
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    tensorboard_logger.configure(tensorboard_log_dir)
    logger.info('TensorBoard logging to %s', tensorboard_log_dir)


def get_data_loaders(args, dataset):
    """Prepare DataLoader instances for training, validation, and testing.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        dataset (Dataset): The dataset from which to create data loaders.

    Returns:
        tuple: Train, validation, and test data loaders.
    """
    N = len(dataset)
    train_end = int(N * args.train_ratio / 100)
    valid_end = int(N * (args.train_ratio + args.valid_ratio) / 100)

    train_loader = DataLoader(
        dataset, batch_size=args.batch, sampler=ChunkSampler(train_end, 0))
    valid_loader = DataLoader(dataset, batch_size=args.batch, sampler=ChunkSampler(
        valid_end - train_end, train_end))
    test_loader = DataLoader(
        dataset, batch_size=args.batch, sampler=ChunkSampler(N - valid_end, valid_end))

    return train_loader, valid_loader, test_loader


def create_model(args, dataset, n_classes, feature_dim):
    """Instantiate the appropriate model based on provided arguments."""
    n_units = [feature_dim] + \
        [int(x) for x in args.hidden_units.split(",")] + [n_classes]

    if args.model == "gcn":
        return BatchGCN(
            pretrained_emb=dataset.get_embedding(),
            vertex_feature=dataset.get_vertex_features(),
            use_vertex_feature=args.use_vertex_feature,
            n_units=n_units,
            dropout=args.dropout,
            instance_normalization=args.instance_normalization
        )

    elif args.model == "sigkan_norm":
        return SIGKAN_Norm(
            n_units=n_units,
            delta=args.delta,  # Bounded confidence interval
            dropout=args.dropout
        )

    elif args.model == "sigkan_att":
        return SIGKAN_Att(
            n_units=n_units,
            delta=args.delta,
            dropout=args.dropout,
            attention=AttentionMechanism(feature_dim),
        )

    else:
        raise NotImplementedError(f"Model type '{args.model}' not supported.")


def train(epoch, train_loader, valid_loader, test_loader, model, alpha, beta, delta, epsilon, psi):
    """Train the model for one epoch."""
    model.train()
    total_loss, total_samples = 0, 0

    for batch in train_loader:
        adj, features, labels, vertices = [x.cuda() if args.cuda else x for x in batch]
        
        # Compute the total loss
        loss = compute_total_loss(model, features, vertices, adj, labels, alpha, beta, delta, epsilon, psi)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += adj.size(0) * loss.item()
        total_samples += adj.size(0)

    logger.info(f"Epoch {epoch}, Train Loss: {total_loss / total_samples:.4f}")
    tensorboard_logger.log_value('train_loss', total_loss / total_samples, epoch)




def evaluate(epoch, loader, model, thr=None, return_best_thr=False, log_desc='valid_'):
    """Evaluate the model."""
    model.eval()
    total_loss, total_samples = 0, 0
    y_true, y_pred, y_score = [], [], []

    for batch in loader:
        adj, features, labels, vertices = [
            x.cuda() if args.cuda else x for x in batch]
        output = model(features, vertices, adj)[:, -1, :]

        loss_batch = F.nll_loss(output, labels, class_weight)
        total_loss += adj.size(0) * loss_batch.item()
        total_samples += adj.size(0)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(output.max(1)[1].cpu().tolist())
        y_score.extend(output[:, 1].cpu().tolist())

    if thr is not None:
        y_pred = np.where(np.array(y_score) > thr, 1, 0)

    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_score)
    logger.info(
        f"{log_desc}loss: {total_loss / total_samples:.4f} AUC: {auc:.4f} Prec: {prec:.4f} Rec: {rec:.4f} F1: {f1:.4f}")

    tensorboard_logger.log_value(
        f'{log_desc}loss', total_loss / total_samples, epoch)
    tensorboard_logger.log_value(f'{log_desc}auc', auc, epoch)
    tensorboard_logger.log_value(f'{log_desc}prec', prec, epoch)
    tensorboard_logger.log_value(f'{log_desc}rec', rec, epoch)
    tensorboard_logger.log_value(f'{log_desc}f1', f1, epoch)

    if return_best_thr:
        precs, recs, thrs = precision_recall_curve(y_true, y_score)
        f1s = 2 * precs * recs / (precs + recs)
        best_thr = thrs[np.nanargmax(f1s)]
        logger.info(f"Best threshold = {best_thr:.4f}, F1 = {np.max(f1s):.4f}")
        return best_thr
    return None


if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Set up environment
    setup_environment(args)

    # Load dataset
    dataset = InfluenceDataSet(args.file_dir, args.dim, args.seed, args.shuffle, args.model)

    # Determine the number of output classes and input feature dimensions
    n_classes = 2  # Assuming binary classification
    feature_dim = dataset.get_feature_dimension()

    # Set class weights for balanced class weighting if needed
    class_weight = dataset.get_class_weight() if args.class_weight_balanced else torch.ones(n_classes)
    class_weight = class_weight.cuda() if args.cuda else class_weight

    # Create data loaders for training, validation, and testing
    train_loader, valid_loader, test_loader = get_data_loaders(args, dataset)

    # Create the appropriate model based on the args
    model = create_model(args, dataset, n_classes, feature_dim)
    if args.cuda:
        model.cuda()

    # Set optimizer
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Training loop
    logger.info("Starting training process...")
    start_time = time.time()

    for epoch in range(args.epochs):
        train(epoch, train_loader, valid_loader, test_loader, model, args.alpha, args.beta, args.delta, args.epsilon, psi)

    logger.info(f"Training completed in {time.time() - start_time:.4f}s")

    # Retrieve the best threshold
    best_thr = evaluate(args.epochs, valid_loader, model, return_best_thr=True, log_desc='valid_')

    # Final evaluation on the test set
    logger.info("Testing the model with the best threshold...")
    evaluate(args.epochs, test_loader, model, thr=best_thr, log_desc='test_')

