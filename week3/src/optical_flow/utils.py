import numpy as np
import png


def read_png_file(flow_file: str) -> np.ndarray:
    """Read from KITTI .png file.
    
    Taken from: https://github.com/liruoteng/OpticalFlowToolkit/blob/master/lib/flowlib.py#L559 
    
    Args:
        flow_file (str): Name of the flow file

    Returns:
        np.ndarray: Optical flow data in matrix
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


# Team 6 2024 
def calculate_msen(gt_flow: np.ndarray, pred_flow: np.ndarray) -> float:
    """Calculates the MSEN metric given a GT and predicted flows.

    Args:
        gt_flow (np.ndarray): GT optical flow
        pred_flow (np.ndarray): Predicted optical flow

    Returns:
        float: Result of the MSEN metric.
    """
    mask = gt_flow[:, :, 2] == 1 # mask of the valid points
    error_u = gt_flow[:, :, 0] - pred_flow[:, :, 0]
    error_v = gt_flow[:, :, 1] - pred_flow[:, :, 1]

    sqrt_error = np.sqrt(error_u ** 2 + error_v ** 2)
    sqrt_error_masked = sqrt_error[mask]
    msen = np.mean(sqrt_error_masked)

    return msen


def calculate_pepn(gt_flow: np.ndarray, pred_flow: np.ndarray, th: int=3) -> float:
    """Calculates the PEPN metric given a GT and predicted flow under a threshold.

    Args:
        gt_flow (np.ndarray): Ground Truth optical flow
        pred_flow (np.ndarray): Predicted optical flow
        th (int, optional): Threshold for the PEPN sqrt error metric. Defaults to 3.

    Returns:
        float: Result of the PEPN metric.
    """
    mask = gt_flow[:, :, 2] == 1 # mask of the valid points
    error_u = gt_flow[:, :, 0] - pred_flow[:, :, 0]
    error_v = gt_flow[:, :, 1] - pred_flow[:, :, 1]

    sqrt_error = np.sqrt(error_u ** 2 + error_v ** 2)
    sqrt_error_masked = sqrt_error[mask]

    return np.sum(sqrt_error_masked > th) / len(sqrt_error_masked)