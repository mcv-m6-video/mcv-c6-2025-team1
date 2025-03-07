import numpy as np
import png
def read_png_file(flow_file):  # from https://github.com/liruoteng/OpticalFlowToolkit/tree/master
    """
    Read from KITTI .png file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    flow_object = png.Reader(filename=flow_file)
    flow_direct = flow_object.asDirect()
    flow_data = list(flow_direct[2])
    (w, h) = flow_direct[3]['size']
    print("Reading %d x %d flow file in .png format" % (h, w))
    flow = np.zeros((h, w, 3), dtype=np.float64)
    for i in range(len(flow_data)):
        flow[i, :, 0] = flow_data[i][0::3]
        flow[i, :, 1] = flow_data[i][1::3]
        flow[i, :, 2] = flow_data[i][2::3]

    invalid_idx = (flow[:, :, 2] == 0)
    flow[:, :, 0:2] = (flow[:, :, 0:2] - 2 ** 15) / 64.0
    flow[invalid_idx, 0] = 0
    flow[invalid_idx, 1] = 0
    return flow

def compute_flow_metrics(flow, gt): #Team 1 2024 :)
    # Binary mask to discard non-occluded areas
    # non_occluded_areas = gt[:,:,2] != 0

    # Only for the first 2 channels
    square_error_matrix = (flow[:, :, 0:2] - gt[:, :, 0:2]) ** 2
    square_error_matrix_valid = square_error_matrix * np.stack(
        (gt[:, :, 2], gt[:, :, 2]), axis=2
    )
    # 
    # square_error_matrix_valid = square_error_matrix[non_occluded_areas]

    # non_occluded_pixels = np.shape(square_error_matrix_valid)[0]
    non_occluded_pixels = np.sum(gt[:, :, 2] != 0)

    # Compute MSEN
    pixel_error_matrix = np.sqrt(
        np.sum(square_error_matrix_valid, axis=2)
    )  # Pixel error for both u and v
    msen = (1 / non_occluded_pixels) * np.sum(
        pixel_error_matrix
    )  # Average error for all non-occluded pixels

    # Compute PEPN
    erroneous_pixels = np.sum(pixel_error_matrix > 3)
    pepn = erroneous_pixels / non_occluded_pixels

    return msen, pepn, pixel_error_matrix

def calculate_msen(gt_flow, pred_flow):
    mask = gt_flow[:, :, 2] == 1 # mask of the valid points
    error_u = gt_flow[:, :, 0] - pred_flow[:, :, 0]
    error_v = gt_flow[:, :, 1] - pred_flow[:, :, 1]

    sqrt_error = np.sqrt(error_u ** 2 + error_v ** 2)
    sqrt_error_masked = sqrt_error[mask]
    msen = np.mean(sqrt_error_masked)

    return msen


def calculate_pepn(gt_flow, pred_flow, th=3):
    mask = gt_flow[:, :, 2] == 1 # mask of the valid points
    error_u = gt_flow[:, :, 0] - pred_flow[:, :, 0]
    error_v = gt_flow[:, :, 1] - pred_flow[:, :, 1]

    sqrt_error = np.sqrt(error_u ** 2 + error_v ** 2)
    sqrt_error_masked = sqrt_error[mask]

    return np.sum(sqrt_error_masked > th) / len(sqrt_error_masked)