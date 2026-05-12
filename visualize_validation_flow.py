#!/usr/bin/env python3
"""Export a step-by-step visualization of one ImageNet VID validation sample.

The script mirrors the validation path used by ``train.py`` for a single
reference/search pair and writes the intermediate tensors as image files. It is
intended for debugging and explaining the SiamFC data flow rather than for bulk
evaluation.
"""
import argparse
import logging
from os import makedirs
from os.path import abspath, dirname, isfile, join



def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Visualize every major intermediate artifact for one validation sample.")
    parser.add_argument('-d', '--data_dir', required=True,
                        help="Full path to the ImageNet VID dataset root.")
    parser.add_argument('-e', '--exp_name', default='default',
                        help="Experiment folder under training/experiments containing parameters.json.")
    parser.add_argument('-o', '--output_dir', default='validation_flow_viz',
                        help="Directory where the visualization images will be written.")
    parser.add_argument('-i', '--sample_idx', default=0, type=int,
                        help="Validation sample index to visualize.")
    parser.add_argument('-r', '--restore_file', default=None,
                        help="Checkpoint name inside the experiment folder, without .pth.tar.")
    parser.add_argument('-c', '--checkpoint', default=None,
                        help="Optional full path to a .pth.tar checkpoint. Overrides --restore_file.")
    parser.add_argument('-f', '--imutils_flag', default='safe', type=str,
                        choices=['fast', 'safe'],
                        help="Image utility backend. 'safe' avoids requiring jpeg4py/libjpeg-turbo.")
    parser.add_argument('--feature_channels', default=16, type=int,
                        help="Number of embedding channels to include in feature-grid images.")
    return parser.parse_args()


def build_model(params):
    if params.model == 'BaselineEmbeddingNet':
        embedding_net = mdl.BaselineEmbeddingNet()
    elif params.model == 'VGG11EmbeddingNet_5c':
        embedding_net = mdl.VGG11EmbeddingNet_5c()
    elif params.model == 'VGG16EmbeddingNet_8c':
        embedding_net = mdl.VGG16EmbeddingNet_8c()
    else:
        raise ValueError("Unknown model '{}' in parameters.json".format(params.model))
    return mdl.SiameseNet(embedding_net, upscale=params.upscale,
                          corr_map_size=params.final_sz, stride=4)


def load_optional_checkpoint(model, args, exp_dir):
    checkpoint_path = args.checkpoint
    if checkpoint_path is None and args.restore_file is not None:
        checkpoint_path = join(exp_dir, args.restore_file + '.pth.tar')
    if checkpoint_path is None:
        logging.info("No checkpoint provided; visualizing randomly initialized weights.")
        return None
    train_utils.load_checkpoint(checkpoint_path, model)
    logging.info("Loaded checkpoint: %s", checkpoint_path)
    return checkpoint_path


def tensor_image_to_numpy(tensor):
    img = tensor.detach().cpu().numpy()
    img = np.transpose(img, (1, 2, 0))
    return np.clip(img, 0.0, 1.0)


def normalize_map(array):
    array = np.asarray(array, dtype=np.float32)
    min_value = float(np.min(array))
    max_value = float(np.max(array))
    if max_value == min_value:
        return np.zeros_like(array)
    return (array - min_value) / (max_value - min_value)


def save_rgb(path, image, title=None):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.clip(image, 0.0, 1.0))
    if title:
        ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_bbox(path, image, annotation, title):
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(np.uint8(image))
    if annotation['xmin'] is not None:
        width = annotation['xmax'] - annotation['xmin']
        height = annotation['ymax'] - annotation['ymin']
        rect = patches.Rectangle((annotation['xmin'], annotation['ymin']), width, height,
                                 linewidth=2, edgecolor='lime', fill=False)
        ax.add_patch(rect)
        ax.scatter([annotation['xmin'] + width / 2],
                   [annotation['ymin'] + height / 2], c='red', s=32)
    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_heatmap(path, heatmap, title, cmap='inferno', marker=None):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(heatmap, cmap=cmap, interpolation='nearest')
    if marker is not None:
        ax.scatter([marker[1]], [marker[0]], c='cyan', marker='+', s=140, linewidths=2)
    ax.set_title(title)
    ax.axis('off')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_feature_grid(path, feature_tensor, title, max_channels=16):
    feature = feature_tensor.detach().cpu().numpy()[0]
    channels = min(max_channels, feature.shape[0])
    cols = int(np.ceil(np.sqrt(channels)))
    rows = int(np.ceil(channels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    axes = np.atleast_1d(axes).ravel()
    for idx, ax in enumerate(axes):
        ax.axis('off')
        if idx < channels:
            ax.imshow(normalize_map(feature[idx]), cmap='viridis', interpolation='nearest')
            ax.set_title('ch {}'.format(idx), fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_overlay(path, image, heatmap, title, alpha=0.45, marker=None):
    heatmap = normalize_map(heatmap)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(np.clip(image, 0.0, 1.0))
    ax.imshow(heatmap, cmap='inferno', alpha=alpha, interpolation='bilinear',
              extent=(0, image.shape[1], image.shape[0], 0))
    if marker is not None:
        ax.scatter([marker[1]], [marker[0]], c='cyan', marker='+', s=160, linewidths=2)
    ax.set_title(title)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save_summary(path, ref_img, search_img, label_pos, score_sigmoid, pred_idx, loss_value, auc, center_error):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes[0, 0].imshow(ref_img)
    axes[0, 0].set_title('Reference crop')
    axes[0, 1].imshow(search_img)
    axes[0, 1].set_title('Search crop')
    axes[1, 0].imshow(label_pos, cmap='gray', interpolation='nearest')
    axes[1, 0].set_title('Positive label')
    axes[1, 1].imshow(score_sigmoid, cmap='inferno', interpolation='nearest')
    axes[1, 1].scatter([pred_idx[1]], [pred_idx[0]], c='cyan', marker='+', s=140, linewidths=2)
    axes[1, 1].set_title('Sigmoid score map')
    for ax in axes.ravel():
        ax.axis('off')
    fig.suptitle('loss={:.4f} | AUC={:.4f} | center_error={:.2f}px'.format(
        loss_value, auc, center_error))
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(args):
    global np, torch, plt, patches, mdl, train_utils, ImageNetVID_val
    global create_BCELogit_loss_label, losses, met, imutils, device

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np
    import torch

    import training.models as mdl
    from training import train_utils
    from training.datasets import ImageNetVID_val
    from training.labels import create_BCELogit_loss_label
    import training.losses as losses
    import training.metrics as met
    import utils.image_utils as imutils

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    root_dir = dirname(abspath(__file__))
    exp_dir = join(root_dir, 'training', 'experiments', args.exp_name)
    json_path = join(exp_dir, 'parameters.json')
    if not isfile(json_path):
        raise FileNotFoundError("No json configuration file found at {}".format(json_path))

    output_dir = abspath(args.output_dir)
    makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    params = train_utils.Params(json_path)
    model = build_model(params).to(device)
    checkpoint_path = load_optional_checkpoint(model, args, exp_dir)
    model.eval()

    img_read_fcn = imutils.get_decode_jpeg_fcn(flag=args.imutils_flag)
    img_resize_fcn = imutils.get_resize_fcn(flag=args.imutils_flag)
    metadata_val_file = join(exp_dir, 'metadata.val')
    metadata_file = metadata_val_file if isfile(metadata_val_file) else None
    save_metadata = None if metadata_file else join(output_dir, 'metadata.val')

    val_set = ImageNetVID_val(args.data_dir,
                              label_fcn=create_BCELogit_loss_label,
                              pos_thr=params.pos_thr,
                              neg_thr=params.neg_thr,
                              upscale_factor=model.upscale_factor,
                              cxt_margin=params.context_margin,
                              reference_size=params.reference_sz,
                              search_size=params.search_sz,
                              final_size=params.final_sz,
                              img_read_fcn=img_read_fcn,
                              resize_fcn=img_resize_fcn,
                              metadata_file=metadata_file,
                              save_metadata=save_metadata,
                              max_frame_sep=params.max_frame_sep)

    sample = val_set[args.sample_idx]
    seq_idx = sample['seq_idx']
    ref_idx = sample['ref_idx']
    srch_idx = sample['srch_idx']
    ref_path = val_set.frames[seq_idx][ref_idx]
    srch_path = val_set.frames[seq_idx][srch_idx]
    ref_annotation = val_set.annotations[seq_idx][ref_idx]
    srch_annotation = val_set.annotations[seq_idx][srch_idx]

    raw_ref = img_read_fcn(ref_path)
    raw_srch = img_read_fcn(srch_path)
    ref_img = tensor_image_to_numpy(sample['ref_frame'])
    srch_img = tensor_image_to_numpy(sample['srch_frame'])
    label = sample['label']
    label_pos = label[:, :, 0]
    label_mask = label[:, :, 1]

    ref_batch = sample['ref_frame'].unsqueeze(0).to(device)
    srch_batch = sample['srch_frame'].unsqueeze(0).to(device)
    labels_batch = torch.from_numpy(label).unsqueeze(0).to(device)

    with torch.no_grad():
        embed_ref = model.get_embedding(ref_batch)
        embed_srch = model.get_embedding(srch_batch)
        score = model.match_corr(embed_ref, embed_srch)
        loss = losses.BCELogit_Loss(score, labels_batch)
        score_np = score.detach().cpu().numpy()
        label_np = np.expand_dims(label, axis=0)
        auc = met.AUC(score_np, label_np)
        center_error = met.center_error(score_np, label_np,
                                        upscale_factor=model.upscale_factor)
        score_2d = score_np[0, 0]
        score_sigmoid = torch.sigmoid(score).detach().cpu().numpy()[0, 0]

    pred_flat = int(np.argmax(score_2d))
    pred_idx = np.unravel_index(pred_flat, score_2d.shape)

    if score_sigmoid.shape != srch_img.shape[:2]:
        score_for_overlay = torch.nn.functional.interpolate(
            torch.from_numpy(score_sigmoid).view(1, 1, *score_sigmoid.shape),
            size=srch_img.shape[:2], mode='bilinear', align_corners=False)
        score_for_overlay = score_for_overlay.numpy()[0, 0]
        scale_y = srch_img.shape[0] / score_sigmoid.shape[0]
        scale_x = srch_img.shape[1] / score_sigmoid.shape[1]
        overlay_marker = (pred_idx[0] * scale_y, pred_idx[1] * scale_x)
    else:
        score_for_overlay = score_sigmoid
        overlay_marker = pred_idx

    save_bbox(join(output_dir, '00_raw_reference_with_bbox.png'), raw_ref,
              ref_annotation, 'Raw reference frame with bbox')
    save_bbox(join(output_dir, '01_raw_search_with_bbox.png'), raw_srch,
              srch_annotation, 'Raw search frame with bbox')
    save_rgb(join(output_dir, '02_reference_crop_127.png'), ref_img,
             'Reference crop fed to the embedding branch')
    save_rgb(join(output_dir, '03_search_crop_255.png'), srch_img,
             'Search crop fed to the embedding branch')
    save_heatmap(join(output_dir, '04_label_positive.png'), label_pos,
                 'Positive label channel', cmap='gray')
    save_heatmap(join(output_dir, '05_label_loss_mask.png'), label_mask,
                 'Loss mask channel', cmap='gray')
    save_feature_grid(join(output_dir, '06_reference_embedding_grid.png'), embed_ref,
                      'Reference embedding feature maps', args.feature_channels)
    save_feature_grid(join(output_dir, '07_search_embedding_grid.png'), embed_srch,
                      'Search embedding feature maps', args.feature_channels)
    save_heatmap(join(output_dir, '08_score_map_logits.png'), score_2d,
                 'Correlation logits / score map', marker=pred_idx)
    save_heatmap(join(output_dir, '09_score_map_sigmoid.png'), score_sigmoid,
                 'Sigmoid-normalized score map', marker=pred_idx)
    save_overlay(join(output_dir, '10_score_map_overlay_on_search.png'), srch_img,
                 score_for_overlay, 'Score map overlaid on search crop', marker=overlay_marker)
    save_summary(join(output_dir, '11_validation_flow_summary.png'), ref_img, srch_img,
                 label_pos, score_sigmoid, pred_idx, float(loss.item()), auc, center_error)

    manifest_path = join(output_dir, 'manifest.txt')
    with open(manifest_path, 'w') as manifest:
        manifest.write('Validation flow visualization\n')
        manifest.write('experiment: {}\n'.format(args.exp_name))
        manifest.write('sample_idx: {}\n'.format(args.sample_idx))
        manifest.write('sequence: {}\n'.format(val_set.get_seq_name(seq_idx)))
        manifest.write('reference_frame: {}\n'.format(ref_path))
        manifest.write('search_frame: {}\n'.format(srch_path))
        manifest.write('checkpoint: {}\n'.format(checkpoint_path or 'random initialization'))
        manifest.write('loss: {:.6f}\n'.format(float(loss.item())))
        manifest.write('AUC: {:.6f}\n'.format(float(auc)))
        manifest.write('center_error_px: {:.6f}\n'.format(float(center_error)))
        manifest.write('predicted_score_map_yx: {},{}\n'.format(pred_idx[0], pred_idx[1]))

    logging.info("Wrote validation flow visualization to %s", output_dir)
    logging.info("Summary manifest: %s", manifest_path)


if __name__ == '__main__':
    main(parse_arguments())
