import numpy as np
import torch
import torchvision

def pairwise_distances(x, y=None):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between
    x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    source:
    https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
    """
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    return torch.clamp(dist, 0.0, np.inf)


class SoftTripletLossFunc(torch.autograd.Function):
    """Class of Soft triplet loss computation.
        In this project we use the exponetial triplet loss, as described in paper.
    """
    @staticmethod
    def forward(ctx, features, triplets):
        ctx.save_for_backward(features, triplets)
        triplet_count = len(triplets)
        learning_weight = 1.0

        # given feature x : pw_feature of x = (sum of element (xi)^p)^(1/p) (Euclid distance if p=2).
        ctx.distances = pairwise_distances(features).cpu().numpy()
        loss = 0.0
        triplet_count = 0.0
        correct_count = 0.0
        for anchor, positive, negative in triplets:
            triplet_count += 1.0
            # Apply exponential triplet loss.
            loss += learning_weight * np.log(1 + np.exp(ctx.distances[anchor, positive] - ctx.distances[anchor, negative])) # F(X)
            if ctx.distances[anchor, positive] < ctx.distances[anchor, negative]:
                correct_count += 1

        loss /= triplet_count
        return torch.FloatTensor((loss,))


    @staticmethod
    def backward(ctx, grad_output):
        features, triplets = ctx.saved_tensors
        triplet_count = len(triplets) if len(triplets)!= 0 else 1.0
        learning_weight = 1.0

        features_np = features.cpu().numpy()
        grad_features = features.clone() * 0.0
        grad_features_np = grad_features.cpu().numpy()

        for anchor, positive, negative in triplets:
            e_f = np.exp(ctx.distances[anchor, positive] - ctx.distances[anchor, negative])
            # Apply GRADIENT of soft triplet loss.
            f =  1.0 - (1.0 / (1.0 + e_f)) # F'(X)*X'
            # Update gradient
            grad_features_np[anchor, :] += learning_weight * f * (features_np[anchor, :] - features_np[positive, :]) / triplet_count

            grad_features_np[positive, :] += learning_weight * f * (features_np[positive, :] - features_np[anchor, :]) / triplet_count

            grad_features_np[anchor, :] += -learning_weight * f * (features_np[anchor, :] - features_np[negative, :]) / triplet_count

            grad_features_np[negative, :] += -learning_weight * f * (features_np[negative, :] - features_np[anchor, :]) / triplet_count


        for i in range(features_np.shape[0]):
            grad_features[i, :] = torch.from_numpy(grad_features_np[i, :])
        grad_features *= float(grad_output.data[0])

        if grad_features.isnan().any():
            # NaN loss occurs in the batch having no possitive pair.
            grad_features[torch.isnan(grad_features)] = 1e-6
            # raise NotImplementedError

        return grad_features, None




class SoftTripletLoss(torch.nn.Module):
    """Class for the triplet loss.
    """
    def __init__(self):
        super(SoftTripletLoss, self).__init__()

    def forward(self, x, triplets):
        loss = SoftTripletLossFunc().apply(x, triplets)
        return loss
