import numpy as np

np.random.seed(42)
def generate_random_boxes(num_boxes):
    x = np.random.randint(0, 10, (num_boxes, 1))       # shape: (N, 1)
    y = np.random.randint(0, 10, (num_boxes, 1))       # shape: (N, 1)
    w = np.random.randint(10, 20, (num_boxes, 1))      # shape: (N, 1)
    h = np.random.randint(10, 20, (num_boxes, 1))      # shape: (N, 1)

    return np.hstack((x, y, x + w, y + h))

a = generate_random_boxes(5)  # Generate 5 random boxes for A, shape: (5, 4)    "(N,4)"
b = generate_random_boxes(10)  # Generate 10 random boxes for B, shape: (10, 4) "(M,4)"

"""
Matrixs A and B are reshaped to have an additional dimension, 
allowing for broadcasting during the comparison.
Matrix A is reshaped to (N, 1, 4) and Matrix B is reshaped to (1, M, 4). 
This enables element-wise comparison between all pairs of rows from A and B.
"""

print("Matrix A:\n", a[:, None, :], "\nMatrix A shape:", a[:, None, :].shape) 
"""Matrix A will be reshaped to N layers, each one containing 1 row and 4 columns."""

print("\nMatrix B:\n", b[None, :, :], "\nMatrix B shape:", b[None, :, :].shape)
"""Matrix B will be reshaped to 1 layer, containing M rows and 4 columns."""

# Broadcasting comparison to find boxes coordinates
x1 = np.maximum(a[:, None, 0], b[None, :, 0])
y1 = np.maximum(a[:, None, 1], b[None, :, 1])
x2 = np.minimum(a[:, None, 2], b[None, :, 2])
y2 = np.minimum(a[:, None, 3], b[None, :, 3])

print("\nIntersection Coordinates Shapes:")
print("x1 shape:", x1.shape)
print("y1 shape:", y1.shape)
print("x2 shape:", x2.shape)
print("y2 shape:", y2.shape)

widths = np.clip(x2 - x1, a_min=0, a_max=None)    # can make use of np.maximum(x2 - x1, 0) as well
heights = np.maximum(0, y2-y1)                    # can make use of np.clip(y2 - y1, 0, None) as well
print("Widths:\n", widths)
print("\nHeights:\n", heights)


# Calculate the intersection over union (IoU) for each pair of boxes
def intersection_over_union(a, b):
    intersection = widths * heights
    
    # Calculate the area of each box in A and B
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])  # tuple of shape (N,)
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])  # tuple of shape (M,)

    # Calculate the union area for each pair of boxes
    union = area_a[:, None] + area_b[None, :] - intersection  # shape: (N, M)
    if np.any(union == 0):
        raise ValueError("Union area is zero, cannot compute IoU.")
    return intersection / union  # shape: (N, M)

print("\nIntersection over Union (IoU):\n", intersection_over_union(a, b))