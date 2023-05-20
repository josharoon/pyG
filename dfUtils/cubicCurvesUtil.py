import torch as th
import numpy as np
import matplotlib.pyplot as plt
from p2mUtils.viz import plotCubicSpline
from torch_geometric.data import DataLoader
from torchvision.transforms import ToPILImage


# Add the updated distance_to_curves function here
def make_safe(tensor):
    return th.where(tensor.abs() < 1e-6, tensor.new_tensor(1e-6).expand_as(tensor), tensor)


def distance_to_curves(source_points, curves,grid_size):
    source_points_swapped = source_points.clone()
    source_points_swapped[..., 0], source_points_swapped[..., 1] = source_points[..., 1], source_points[..., 0]
    source_points_swapped[..., 0] = grid_size - 1 - source_points_swapped[..., 0]

    min_distances = []
    for curve in curves:
        distances = distance_to_single_curve(source_points_swapped, curve[None])
        min_distances.append(distances)
    min_distances = th.stack(min_distances, dim=-1)
    distances, _ = th.min(min_distances, dim=-1)
    #distances_flipped = th.flip(distances, (0,))
    #distances_flipped = th.flip(distances_flipped, (0,))
    return distances





def distance_to_single_curve(source_points, curve):
    """Compute the distance from each source point to each cubic Bezier curve.

        source_points -- [n_points, 2]
        curves -- [..., 4, 2]
        """
    p0, p1, p2, p3 = th.split(curve, 1, dim=-2)  # [..., 1, 2]

    X = p1 - p0  # [..., 1, 2]
    Y = p2 - 2 * p1 + p0  # [..., 1, 2]
    Z = p3 - 3 * p2 + 3 * p1 - p0  # [..., 1, 2]
    W = p0 - source_points  # [..., n_points, 2]

    a = 3 * th.sum(Z * Z, dim=-1)  # [..., 1]
    a = make_safe(a)

    b = 6 * th.sum(Y * Z, dim=-1)  # [..., 1]
    c = 3 * th.sum(X * Z, dim=-1) + 3 * th.sum(Y * Y, dim=-1)  # [..., n_points]
    d = 3 * th.sum(X * Y, dim=-1) - 3 * th.sum(Y * W, dim=-1)  # [..., n_points]
    e = th.sum(X * W, dim=-1)  # [..., n_points]

    A = b / a
    B = c / a
    C = d / a
    D = e / a

    # Solving the cubic equation is more complex than for quadratic ones.
    # Numerical approaches such as Newton-Raphson's method can be employed.
    # Here, we use a naive sampling approach for the purpose of illustration.
    num_samples = 100
    ts = th.linspace(0, 1, num_samples).to(source_points.device)  # [num_samples]
    ts = ts[None, None, :]  # [1, 1, num_samples]

    ts_powers = ts[..., None].pow(ts.new_tensor([0, 1, 2, 3])).double()  # [1, 1, num_samples, 4]
    A = ts_powers.new_tensor([[1., 0, 0, 0],
                              [-3, 3, 0, 0],
                              [3, -6, 3, 0],
                              [-1, 3, -3, 1]]).double()
    points = ts_powers @ A @ curve.unsqueeze(-3)  # [..., n_points, num_samples, 2]

    distances, _ = th.min(th.sqrt(th.sum((points - source_points[:, None, :]) ** 2, dim=-1) + 1e-6),
                          dim=-1)  # [..., n_points]

    return distances


def distance_field_loss(distance_field_A, distance_field_B, discrepancy_fn):
    vol_S = distance_field_A.numel()
    loss = th.sum(discrepancy_fn(distance_field_A, distance_field_B)) / vol_S
    return loss

def create_ellipse_spline(center_x, center_y, radius_x, radius_y, num_segments=4):
    control_points = []
    angle = 2 * np.pi / num_segments

    for i in range(num_segments):
        start_angle = i * angle
        end_angle = (i + 1) * angle

        start_point = th.tensor([center_x + radius_x * np.cos(start_angle), center_y + radius_y * np.sin(start_angle)])
        end_point = th.tensor([center_x + radius_x * np.cos(end_angle), center_y + radius_y * np.sin(end_angle)])
        tangent_length = 4 / 3 * np.tan(angle / 4)

        tangent_vector_1 = th.tensor([radius_x * -np.sin(start_angle) * tangent_length, radius_y * np.cos(start_angle) * tangent_length])
        control_point_1 = start_point + tangent_vector_1

        tangent_vector_2 = th.tensor([radius_x * np.sin(end_angle) * tangent_length, radius_y * -np.cos(end_angle) * tangent_length])
        control_point_2 = end_point - tangent_vector_2

        curve = th.stack([start_point, control_point_1, control_point_2, end_point], dim=0)
        control_points.append(curve)

    return th.stack(control_points, dim=0)




def convert_to_cubic_control_points(tensor, device='cuda'):
    control_points = []

    for i in range(tensor.shape[1] - 1):
        main_point1_x, main_point1_y, lt1_x, lt1_y, rt1_x, rt1_y = tensor[0, i, :6]
        main_point2_x, main_point2_y, lt2_x, lt2_y, _, _ = tensor[0, i + 1, :6]

        p0 = th.tensor([main_point1_x, main_point1_y], device=device)
        #p1 = p0 + th.tensor([rt1_x, rt1_y])
        p1 = th.tensor([rt1_x, rt1_y], device=device)
        # p2 = th.tensor([main_point2_x, main_point2_y]) + th.tensor([lt2_x, lt2_y])
        # p3 = th.tensor([main_point2_x, main_point2_y])
        p2 = th.tensor([lt2_x, lt2_y], device=device)
        p3 = th.tensor([main_point2_x, main_point2_y], device=device)

        control_points.append(th.stack([p0, p1, p2, p3]))

    # Closing the shape
    main_point1_x, main_point1_y, lt1_x, lt1_y, rt1_x, rt1_y = tensor[0, -1, :6]
    main_point2_x, main_point2_y, lt2_x, lt2_y, _, _ = tensor[0, 0, :6]

    p0 = th.tensor([main_point1_x, main_point1_y], device=device)
    #p1 = p0 + th.tensor([rt1_x, rt1_y])
    p1 =  th.tensor([rt1_x, rt1_y], device=device)
    #p2 = th.tensor([main_point2_x, main_point2_y]) + th.tensor([lt2_x, lt2_y])
    p2 = th.tensor([lt2_x, lt2_y], device=device)
    p3 = th.tensor([main_point2_x, main_point2_y], device=device)

    control_points.append(th.stack([p0, p1, p2, p3]))

    return th.stack(control_points)

    # Closing the shape
    main_point1, lt1_x, lt1_y, rt1_x, rt1_y = tensor[0, -1, :5]
    main_point2, lt2_x, lt2_y, _, _ = tensor[0, 0, :5]

    p0 = main_point1
    p1 = main_point1 + th.tensor([rt1_x, rt1_y])
    p2 = main_point2 + th.tensor([lt2_x, lt2_y])
    p3 = main_point2

    control_points.append(th.stack([p0, p1, p2, p3]))

    return th.stack(control_points)


def create_grid_points(grid_size, xmin, xmax, ymin, ymax):
    x = th.linspace(xmin, xmax, grid_size, dtype=th.float64)
    y = th.linspace(ymin, ymax, grid_size, dtype=th.float64)
    xv, yv = th.meshgrid(x, y)
    source_points = th.stack([xv.flatten(), yv.flatten()], dim=-1)
    return source_points


if __name__ == '__main__':
    from simpleRotoDataset import SimpleRotoDataset
    dataset = SimpleRotoDataset(root=r'D:\pyG\data\points\120423_183451_rev',labelsJson="points120423_183451_rev.json")
    print(len(dataset))
    print(dataset[59])
    dataloader=DataLoader(dataset, batch_size=1, shuffle=True)
    dataIter=iter(dataloader)
    data=next(dataIter)
    # print(data)
    image=ToPILImage()(data[0][0])
    #plot image using matplotlib
    plt.imshow(image)
    control_points = convert_to_cubic_control_points(data[1]).to(th.float64)
    grid_size = 224
    source_points = create_grid_points(grid_size, 0, 250, 0, 250)

    plotCubicSpline(control_points)

    distance_field = distance_to_curves(source_points, control_points).view(grid_size, grid_size)
    distance_field= th.flip(distance_field, (1,))
    plt.figure(figsize=(6, 6))
    plt.imshow(distance_field, extent=(0, 250, 0, 250), origin='lower', cmap='viridis')
    plt.colorbar(label='Distance')
    plt.title('Distance Field')
    plt.show()
