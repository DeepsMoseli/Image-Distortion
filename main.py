import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from noise import pnoise2
import json

def create_checkerboard(size, num_squares):
    checkerboard = np.zeros((size, size, 3), dtype=np.uint8)
    square_size = size // num_squares

    for i in range(num_squares):
        for j in range(num_squares):
            if (i + j) % 2 == 0:
                checkerboard[i * square_size:(i + 1) * square_size,
                             j * square_size:(j + 1) * square_size] = [255, 255, 255]
    return checkerboard


def generate_smooth_random_field(width, height, scale, distortion_strength, offset_x, offset_y):
    field_x = np.zeros((height, width), np.float32)
    field_y = np.zeros((height, width), np.float32)
    for y in range(height):
        for x in range(width):
            noise_x = pnoise2((x + offset_x) / scale, (y + offset_y) / scale, octaves=6, persistence=0.5)
            noise_y = pnoise2((x + offset_x + 100) / scale, (y + offset_y + 100) / scale, octaves=6, persistence=0.5)
            field_x[y, x] = distortion_strength * noise_x
            field_y[y, x] = distortion_strength * noise_y
    return field_x, field_y


def apply_vortex_transformation(image, strength):
    height, width = image.shape[:2]
    center_x, center_y = width / 2, height / 2

    x, y = np.meshgrid(np.arange(width), np.arange(height))
    dx = x - center_x
    dy = y - center_y
    distance = np.sqrt(dx**2 + dy**2) + 1e-5

    angle = strength / (distance + 1e-5)
    map_x = center_x + dx * np.cos(angle) - dy * np.sin(angle)
    map_y = center_y + dx * np.sin(angle) + dy * np.cos(angle)

    map_x = np.clip(map_x, 0, width - 1).astype(np.float32)
    map_y = np.clip(map_y, 0, height - 1).astype(np.float32)

    transformed_image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    return transformed_image


def apply_radial_distortion(image, params=None, json_path="/home/hok/Desktop/Image-Distortion-Removal/main/param/distortion_params.json"):
    if params is None:
        with open(json_path, "r") as file:
            param_ranges = json.load(file)

        params = {key: random.uniform(*param_ranges[key]) for key in param_ranges}

    k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4 = [params[key] for key in ['k1', 'k2', 'k3', 'k4', 'k5', 'k6', 'p1', 'p2', 's1', 's2', 's3', 's4']]

    print(f"Applying radial distortion with parameters: k1={k1}, k2={k2}, k3={k3}, k4={k4}, k5={k5}, k6={k6}, p1={p1}, p2={p2}, s1={s1}, s2={s2}, s3={s3}, s4={s4}")

    h, w = image.shape[:2]
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    x_normalized = (x - w / 2) / (w / 2)
    y_normalized = (y - h / 2) / (h / 2)

    r = np.sqrt(x_normalized**2 + y_normalized**2)

    radial_distortion_xfactor = 1 + k1 * r**2 + k2 * r**4 + k3 * r**6
    radial_distortion_yfactor = 1 + k4 * r**2 + k5 * r**4 + k6 * r**6

    x_tangential_distortion = 2 * p1 * x_normalized * y_normalized + p2 * (r**2 + 2 * x_normalized**2)
    y_tangential_distortion = p1 * (r**2 + 2 * y_normalized**2) + 2 * p2 * x_normalized * y_normalized

    x_distorted_sfactor = s1 * r**2 + s2 * r**4
    y_distorted_sfactor = s3 * r**2 + s4 * r**4

    x_distorted = (x_normalized * radial_distortion_xfactor / radial_distortion_yfactor + x_tangential_distortion + x_distorted_sfactor) * w / 2 + w / 2
    y_distorted = (y_normalized * radial_distortion_xfactor / radial_distortion_yfactor + y_tangential_distortion + y_distorted_sfactor) * h / 2 + h / 2

    uv_map = np.zeros((h, w, 2), dtype=np.float32)
    uv_map[..., 0] = x_distorted
    uv_map[..., 1] = y_distorted

    distorted_image = cv2.remap(image, x_distorted.astype(np.float32), y_distorted.astype(np.float32), interpolation=cv2.INTER_LINEAR)

    return distorted_image, uv_map, params


def save_uv_map(uv_map, output_dir, base_name):
    uv_output_path = os.path.join(output_dir, f"{base_name}_uv_map.npy")
    np.save(uv_output_path, uv_map)
    print(f"UV map saved: {uv_output_path}")


def process_images_in_directory(input_dir, output_dir, transformation_type):
    grid_dir = os.path.join(output_dir, 'grid')
    distorted_dir = os.path.join(output_dir, 'distorted')
    uv_dir = os.path.join(output_dir, 'uv_maps')

    os.makedirs(grid_dir, exist_ok=True)
    os.makedirs(distorted_dir, exist_ok=True)
    os.makedirs(uv_dir, exist_ok=True)
    
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not image_files:
        raise FileNotFoundError(f"No image files found in directory: {input_dir}")
    
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        try:
            base_name = os.path.splitext(image_file)[0]
            
            transformed_output_path = os.path.join(distorted_dir, f"{base_name}_{transformation_type}.png")
            grid_output_path = os.path.join(grid_dir, f"{base_name}_grid_{transformation_type}.png")
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Skipping invalid image file {image_path}")
                continue

            checkerboard = create_checkerboard(size=image.shape[1], num_squares=10)

            if transformation_type == 'random_field':
                height, width = image.shape[:2]
                offset_x = random.uniform(0, 1000)
                offset_y = random.uniform(0, 1000)
                field_x, field_y = generate_smooth_random_field(width, height, 50, 15, offset_x, offset_y)

                uv_map = np.zeros((height, width, 2), dtype=np.float32)
                uv_map[..., 0] = field_x
                uv_map[..., 1] = field_y

                map_x, map_y = np.meshgrid(np.arange(width), np.arange(height))
                distorted_map_x = (map_x + field_x).astype(np.float32)
                distorted_map_y = (map_y + field_y).astype(np.float32)
                transformed_image = cv2.remap(image, distorted_map_x, distorted_map_y, interpolation=cv2.INTER_LINEAR)
                distorted_grid = cv2.remap(checkerboard, distorted_map_x, distorted_map_y, interpolation=cv2.INTER_LINEAR)

                save_uv_map(uv_map, uv_dir, base_name)

            elif transformation_type == 'radial_distortion':
                transformed_image, uv_map, distortion_params = apply_radial_distortion(image)
                distorted_grid, _, _ = apply_radial_distortion(checkerboard, params=distortion_params)

                save_uv_map(uv_map, uv_dir, base_name)

            elif transformation_type == 'vortex_transformation':
                transformed_image = apply_vortex_transformation(image, strength=500)
                distorted_grid = apply_vortex_transformation(checkerboard, strength=500)

            else:
                raise ValueError(f"Unknown transformation type: {transformation_type}")

            cv2.imwrite(transformed_output_path, transformed_image)
            cv2.imwrite(grid_output_path, distorted_grid)
            print(f"Transformed image saved: {transformed_output_path}")
            print(f"Distorted grid saved: {grid_output_path}")
        
        except Exception as e:
            print(f"Error processing {image_file}: {e}")


if __name__ == "__main__":
    input_dir = "main/input_data"
    transformation_type = "radial_distortion"
    output_dir = "main/output_data"

    process_images_in_directory(input_dir, output_dir, transformation_type)
    