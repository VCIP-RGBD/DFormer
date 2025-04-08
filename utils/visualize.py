import numpy as np
import cv2
import scipy.io as sio


def set_img_color(colors, background, img, pred, gt, show255=False):
    for i in range(0, len(colors)):
        if i != background:
            img[np.where(pred == i)] = colors[i]
    if show255:
        img[np.where(gt == background)] = 255
    return img


def show_prediction(colors, background, img, pred, gt):
    im = np.array(img, np.uint8)
    set_img_color(colors, background, im, pred, gt)
    final = np.array(im)
    return final


def show_img(colors, background, img, clean, gt, *pds):
    im1 = np.array(img, np.uint8)
    # set_img_color(colors, background, im1, clean, gt)
    final = np.array(im1)
    # the pivot black bar
    pivot = np.zeros((im1.shape[0], 15, 3), dtype=np.uint8)
    for pd in pds:
        im = np.array(img, np.uint8)
        # pd[np.where(gt == 255)] = 255
        set_img_color(colors, background, im, pd, gt)
        final = np.column_stack((final, pivot))
        final = np.column_stack((final, im))

    im = np.array(img, np.uint8)
    set_img_color(colors, background, im, gt, True)
    final = np.column_stack((final, pivot))
    final = np.column_stack((final, im))
    return final


def get_colors(class_num):
    colors = []
    for i in range(class_num):
        colors.append((np.random.random((1, 3)) * 255).tolist()[0])

    return colors


def get_ade_colors():
    colors = sio.loadmat("./color150.mat")["colors"]
    colors = colors[
        :,
        ::-1,
    ]
    colors = np.array(colors).astype(int).tolist()
    colors.insert(0, [0, 0, 0])

    return colors


def print_iou(iou, freq_IoU, mean_pixel_acc, pixel_acc, class_names=None, show_no_back=False, no_print=False):
    n = iou.size
    lines = []
    for i in range(n):
        if class_names is None:
            cls = "Class %d:" % (i + 1)
        else:
            cls = "%d %s" % (i + 1, class_names[i])
        lines.append("%-8s\t%.3f%%" % (cls, iou[i] * 100))
    mean_IoU = np.nanmean(iou)
    mean_IoU_no_back = np.nanmean(iou[1:])
    # print(len(iou))
    if show_no_back:
        lines.append(
            "----------     %-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%"
            % (
                "mean_IoU",
                mean_IoU * 100,
                "mean_IoU_no_back",
                mean_IoU_no_back * 100,
                "freq_IoU",
                freq_IoU * 100,
                "mean_pixel_acc",
                mean_pixel_acc * 100,
                "pixel_acc",
                pixel_acc * 100,
            )
        )
    else:
        lines.append(
            "----------     %-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%"
            % (
                "mean_IoU",
                mean_IoU * 100,
                "freq_IoU",
                freq_IoU * 100,
                "mean_pixel_acc",
                mean_pixel_acc * 100,
                "pixel_acc",
                pixel_acc * 100,
            )
        )
    print(
        "----------     %-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%\t%-8s\t%.3f%%"
        % (
            "mean_IoU",
            mean_IoU * 100,
            "mean_IU_no_back",
            mean_IoU_no_back * 100,
            "freq_IoU",
            freq_IoU * 100,
            "mean_pixel_acc",
            mean_pixel_acc * 100,
            "pixel_acc",
            pixel_acc * 100,
        )
    )
    line = "\n".join(lines)
    # if not no_print:
    #     print(line)
    # else:
    print(line[-1])
    return str(mean_IoU * 100)
