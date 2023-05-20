import torch as th


def distance_field_loss(distance_field_A, distance_field_B):
    vol_S = distance_field_A.numel()
    loss = th.sum(discrepancy_l2_squared(distance_field_A, distance_field_B)) / vol_S
    return loss


def discrepancy_l2_squared(distance_field_A, distance_field_B):
    return (distance_field_A - distance_field_B) ** 2
