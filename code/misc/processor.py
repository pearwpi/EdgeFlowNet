import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

import misc.FlowVisUtilsNP as fvu
import misc.FlowShift as fs


def generate_plus(H, W, w = 10):
    # Create a black plus
    plus = np.zeros((H, W, 1), dtype=np.uint8)

    # Calculate the step size to divide the plus into equal parts
    step_i = W // 2
    step_j = H // 2

    # Draw vertical lines for the + sign
    cv2.line(plus, (step_i, 0), (step_i, H), (1, 1, 1), thickness=w)

    # Draw horizontal lines for the + sign
    cv2.line(plus, (0, step_j), (W, step_j), (1, 1, 1), thickness=w)

    return plus

def blur_flow(predicted_flow):
    plus = generate_plus(predicted_flow.shape[0], predicted_flow.shape[1])
    predicted_flow_blurred = gaussian_filter(predicted_flow, sigma=0.5, order=0)

    flow_hada_prod_plus = np.multiply(predicted_flow_blurred,plus)
    flow_hada_prod_not_plus = np.multiply(predicted_flow, np.logical_not(plus).astype(int))
    flow_hada_prod = flow_hada_prod_plus + flow_hada_prod_not_plus
    return flow_hada_prod

def deoverlap_flow(predicted_flow, Args):
    CenterX = Args.OutputPatchSize[0]//2
    CenterY = Args.OutputPatchSize[1]//2

    p_tl = predicted_flow[0][0:CenterX, 0:CenterY]
    p_tr = predicted_flow[2][0:CenterX, -CenterY:]
    p_bl = predicted_flow[1][-CenterX:, 0:CenterY]
    p_br = predicted_flow[3][-CenterX:, -CenterY:]
    p_t = np.concatenate((p_tl, p_tr), axis=1)
    p_b = np.concatenate((p_bl, p_br), axis=1)
    predicted_flow = np.concatenate((p_t, p_b), axis=0)
    return predicted_flow

def stacking_images(predicted_flow, halves=1, axis=1):
    if halves == 1:
        predicted_flow = [np.squeeze(pred_flow) for pred_flow in predicted_flow]
        p_f = np.concatenate((predicted_flow[0], predicted_flow[2]), axis=1)
        p_h = np.concatenate((predicted_flow[1], predicted_flow[3]), axis=1)
        return np.concatenate((p_f, p_h), axis=0)
    else:
        halves -= 1
        top_left = stacking_images(predicted_flow[:4], halves, axis)
        top_right = stacking_images(predicted_flow[4:8], halves, axis)
        bottom_left = stacking_images(predicted_flow[8:12], halves, axis)
        bottom_right = stacking_images(predicted_flow[12:], halves, axis)

        p_f = np.concatenate([top_left, top_right], axis=1)
        p_h = np.concatenate([bottom_left, bottom_right], axis=1)
        return np.concatenate([p_f, p_h],axis=0)
    
def reconstruct_image(chunks):
    S = len(chunks)
    if S == 0:
        raise ValueError("Empty list of chunks.")

    chunk_size_m, chunk_size_n, _ = chunks[0].shape
    sqrt_S = int(np.sqrt(S))

    M = chunk_size_m * sqrt_S
    N = chunk_size_n * sqrt_S

    reconstructed_image = np.zeros((M, N, chunks[0].shape[2]), dtype=chunks[0].dtype)

    for i in range(sqrt_S):
        for j in range(sqrt_S):
            chunk = chunks[i * sqrt_S + j]
            reconstructed_image[i * chunk_size_m : (i + 1) * chunk_size_m, j * chunk_size_n : (j + 1) * chunk_size_n, ...] = chunk

    return reconstructed_image
# def stacking_images(predicted_flow, halves=1, axis=1):
#     if halves == 1:
#         predicted_flow = [np.squeeze(pred_flow) for pred_flow in predicted_flow]
#         stacked_flows = np.concatenate([np.concatenate(predicted_flow[:2], axis=1),
#                                         np.concatenate(predicted_flow[2:], axis=1)], axis=0)
#         return stacked_flows
#     else:
#         # Recursively process the predicted flows for N halves
#         halves -= 1
#         splits = [stacking_images(predicted_flow[i:i+2], halves, axis) for i in range(0, len(predicted_flow), 2)]

#         # Check if there are enough elements for concatenation
#         if len(splits) > 1:
#             if axis == 1:
#                 return np.concatenate(splits, axis=1)
#             elif axis == 2:
#                 return np.concatenate(splits, axis=2)
#         else:
#             # Handle the case where there are not enough elements to concatenate
#             return splits[0] if splits else None


class FlowPostProcessor():
    def __init__(self, suffix = "", is_multiscale=False):
        self.final_loss = 0.0
        self.errorEPEs = []
        self.totalTime  = 0.0
        self.counter = 0
        self.is_multiscale = is_multiscale
    
    def post_process_prediction(self, prediction, Args):
        
        predicted_flow = prediction
  
        if self.is_multiscale:
            predicted_flow = np.squeeze(prediction) # H x W x 4
            if Args.uncertainity:
                predicted_flow = np.squeeze(prediction)[..., 0:2] # H x W x 2
                
        if Args is not None:
            if Args.ShiftedFlow:
                if Args.ResizeCropStack or Args.ResizeNearestCropStack:
                    predicted_flow = [fs.unshift_flow_tf(pred_flow) for pred_flow in predicted_flow]
                else:
                    predicted_flow = fs.unshift_flow_tf(np.squeeze(prediction)) # H x W x 2
                #predicted_flow = fp.flow_from_polar_np(np.squeeze(prediction)) # H x W x 2
            if Args.ResizeToHalf:
                predicted_flow = cv2.resize(predicted_flow, (Args.OutputPatchSize[1], Args.OutputPatchSize[0]))
            if Args.ResizeCropStack or Args.ResizeNearestCropStack:
                predicted_flow = [np.squeeze(pred_flow) for pred_flow in predicted_flow]
                p_f = np.concatenate((predicted_flow[0],predicted_flow[2]),axis=1)
                p_h = np.concatenate((predicted_flow[1],predicted_flow[3]),axis=1)
                predicted_flow = np.concatenate((p_f,p_h),axis=0)
            if Args.NumberOfHalves != 0:
                # predicted_flow = stacking_images(predicted_flow, Args.NumberOfHalves)
                predicted_flow = reconstruct_image(predicted_flow)
    
            elif Args.ResizeCropStackBlur:
                p_f = np.concatenate((predicted_flow[0],predicted_flow[2]),axis=1)
                p_h = np.concatenate((predicted_flow[1],predicted_flow[3]),axis=1)
                predicted_flow = np.concatenate((p_f,p_h),axis=0)
                predicted_flow = blur_flow(predicted_flow)
            elif Args.OverlapCropStack:
                if Args.Display:
                    pred_1_full_name = f"{Args.ModelPath}/test_pred_1.png"
                    pred_2_full_name = f"{Args.ModelPath}/test_pred_2.png"
                    pred_3_full_name = f"{Args.ModelPath}/test_pred_3.png"
                    pred_4_full_name = f"{Args.ModelPath}/test_pred_4.png"
                    cv2.imwrite(pred_1_full_name, fvu.flow_viz_np(predicted_flow[0].squeeze()[:,:,0],predicted_flow[0].squeeze()[:,:,1]))
                    cv2.imwrite(pred_2_full_name, fvu.flow_viz_np(predicted_flow[1].squeeze()[:,:,0],predicted_flow[1].squeeze()[:,:,1]))
                    cv2.imwrite(pred_3_full_name, fvu.flow_viz_np(predicted_flow[2].squeeze()[:,:,0],predicted_flow[2].squeeze()[:,:,1]))
                    cv2.imwrite(pred_4_full_name, fvu.flow_viz_np(predicted_flow[3].squeeze()[:,:,0],predicted_flow[3].squeeze()[:,:,1]))

                predicted_flow = deoverlap_flow(predicted_flow, Args)
        return predicted_flow

    def update(self, label, prediction, Args = None):
        label_flow = np.squeeze(label[0])

        Args.ShiftedFlow = False
        Args.ResizeToHalf = False
        Args.ResizeCropStack = False
        Args.ResizeNearestCropStack = False
        Args.NumberOfHalves = 0
        Args.ResizeCropStackBlur = False
        Args.OverlapCropStack = False
        predicted_flow = self.post_process_prediction(prediction, Args)
        
        final_loss = np.mean(np.abs(label_flow - predicted_flow))

        errorEPE = np.sqrt(np.sum((label_flow - predicted_flow)**2, axis=2)).flatten()
        
        self.final_loss += final_loss
        self.errorEPEs.append(errorEPE)
        self.counter += 1

    def print(self):
        if self.counter == 0:
            print("Nothing to print")
            return
        errorEPE = np.concatenate(self.errorEPEs).mean()

        print("-----SUMMARY-------")
        print(f"EPE:{errorEPE}")