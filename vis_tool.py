import numpy as np
import torchvision
from matplotlib import pyplot as plot

#refer to CATEGORY_TO_LABEL_DICT
VOC_BBOX_LABEL_NAMES = ('background',
                        'airplane', 'bike',     'bird',     'boat',     'bottle',
                        'bus',      'car',      'cat',      'chair',    'cow',
                        'table',    'dog',      'horse',    'moto',     'person',
                        'plant',    'sheep',    'sofa',     'train',    'tv')

def vis_image(img, ax=None):
    if ax is None:
        fig = plot.figure()
        ax = fig.add_subplot(1, 1, 1)
    # CHW -> HWC
    #img    = img.transpose((1, 2, 0))
    img     = (img * 0.225 + 0.45).clip(min=0, max=1) * 255 #inverse_normalize
    grid    = torchvision.utils.make_grid(img)
    img     = grid.cpu().numpy().transpose(1, 2, 0)
    ax.imshow(img.astype(np.uint8))

    return ax

def vis_bbox(img, bbox, label=None, score=None, ax=None):
    label_names = list(VOC_BBOX_LABEL_NAMES) + ['bg']
    # add for index `-1`
    if label is not None and not len(bbox) == len(label):
        raise ValueError('The length of label must be same as that of bbox')

    if score is not None and not len(bbox) == len(score):
        raise ValueError('The length of score must be same as that of bbox')

    # Returns newly instantiated matplotlib.axes.Axes object if ax is None
    ax = vis_image(img, ax=ax)

    # If there is no bounding box to display, visualize the image and exit.
    if len(bbox) == 0:
        return ax

    bbox = bbox.detach().cpu()
    for i, bb in enumerate(bbox):
        xy      = (bb[0], bb[1])
        height  = bb[3] - bb[1]
        width   = bb[2] - bb[0]
        ax.add_patch(plot.Rectangle(xy, width, height, fill=False, edgecolor='red', linewidth=2))

        caption = list()
        if label is not None and label_names is not None:
            lb = label[i]
            if not (-1 <= lb < len(label_names)):  # modfy here to add backgroud
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])

        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[0],
                    bb[1],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
    return ax


def fig2data(fig):
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)

    return buf.reshape(h, w, 4)

def visdom_bbox(*args, **kwargs):
    ax  = vis_bbox(*args, **kwargs)
    fig = ax.get_figure()
    img_data = fig2data(fig).astype(np.int32)
    img_data = img_data[:, :, :3].transpose((2, 0, 1)) / 255.0 # HWC->CHW
    plot.close()

    return img_data

