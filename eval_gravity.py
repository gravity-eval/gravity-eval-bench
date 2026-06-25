import torch
import cv2
import os
import json
import numpy as np
from matplotlib import pyplot as plt
import argparse


from sam3.model_builder import build_sam3_video_model
sam3_model = build_sam3_video_model()
predictor = sam3_model.tracker
predictor.backbone = sam3_model.detector.backbone


def get_video_properties(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()
    return fps, width, height, frame_count


def find_centroid_cv2(mask):
    # Ensure the mask is in the correct format
    mask = mask.squeeze()
    
    # Convert the tensor to a NumPy array and ensure correct type
    mask = mask.detach().cpu().numpy().astype(np.uint8)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Calculate the moments of the mask
    M = cv2.moments(mask)
    
    # Check if the total moment (area) is non-zero
    if M["m00"] == 0:
        return None
    
    # Calculate the centroid coordinates from the moments
    centroid_x = M["m10"] / M["m00"]
    centroid_y = M["m01"] / M["m00"]
    
    return (centroid_x, centroid_y)


def calculate_masks(gt_folder, folder, sample_id):
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        video_dir = f"{folder}/generated_{sample_id}"
        reference_dir = f"{gt_folder}/sample_{sample_id}"
        reference_json = json.load(open(os.path.join(reference_dir, "metadata.json")))

        ground_gt_threshold_y = reference_json["ground"]["positive_points"][0][1]
        

        print("ground_gt_threshold:", ground_gt_threshold_y)
        
        def prepare_points(label_name):
            pos = reference_json[label_name]["positive_points"]
            neg = reference_json[label_name]["negative_points"]

            points = []
            labels = []

            for p in pos:
                points.append(p)
                labels.append(1)   # positive

            for p in neg:
                points.append(p)
                labels.append(0)   # negative

            if len(points) == 0:
                raise ValueError(f"No points provided for {label_name}")

            return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)
        
        try:
            points_ball1, labels_ball1 = prepare_points("object_1")
            points_ball2, labels_ball2 = prepare_points("object_2")
        except:
            raise ValueError(f"Missing points for sample {sample_id}. Check metadata.json for object_1 and object_2.")
        
        # Read video properties
        video_path = f"{video_dir}/generated_video.mp4"
        fps, frame_width, frame_height, frame_count = get_video_properties(video_path)
        print("Video FPS:", fps)
        print("Resolution:", frame_width, "x", frame_height)
        print("GT Frame count:", frame_count)

        state = predictor.init_state(video_path=f"{folder}/generated_{sample_id}/generated_video.mp4")
    
        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 1  # give a unique id to each object we interact with (it can be any integers)
        rel_points_ball1 = np.zeros_like(points_ball1, dtype=np.float32)
        rel_points_ball1[:, 0] = points_ball1[:, 0] / frame_width
        rel_points_ball1[:, 1] = points_ball1[:, 1] / frame_height
        _, out_obj_ids, _, video_res_masks = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=rel_points_ball1,
            labels=labels_ball1,
        )

        ann_frame_idx = 0  # the frame index we interact with
        ann_obj_id = 2  # give a unique id to each object we interact with (it can be any integers)
        rel_points_ball2 = np.zeros_like(points_ball2, dtype=np.float32)
        rel_points_ball2[:, 0] = points_ball2[:, 0] / frame_width
        rel_points_ball2[:, 1] = points_ball2[:, 1] / frame_height
        _, out_obj_ids, _, video_res_masks = predictor.add_new_points_or_box(
            inference_state=state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=rel_points_ball2,
            labels=labels_ball2,
        )

        # run propagation throughout the video and collect the results in a dict
        video_segments_gen = {}  # video_segments contains the per-frame segmentation results
        generated_centroids_1, generated_centroids_2 = [], []
        for out_frame_idx, out_obj_ids, _, video_res_masks, _ in predictor.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=240, reverse=False, propagate_preflight=True):
            video_segments_gen[out_frame_idx] = {
                out_obj_id: (video_res_masks[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }
            
            generated_centroids_1.append(find_centroid_cv2(video_res_masks[0]> 0.0))
            generated_centroids_2.append(find_centroid_cv2(video_res_masks[1]> 0.0))
        
    threshold = 1
    # Find the drop frame and time of flight for generated object 1
    len_generated_centroids_1 = len(generated_centroids_1)
    for i in range(1, len_generated_centroids_1):
        try:
            if generated_centroids_1[i] is None or generated_centroids_1[i-1] is None :
                drop_frame_gen_1 = None
                time_of_flight_gen_1 = None
                break
            else:
                if generated_centroids_1[i][1] > ground_gt_threshold_y and generated_centroids_1[i][1] - generated_centroids_1[i-1][1] < threshold:
                    drop_frame_gen_1 = (i-1)
                    time_of_flight_gen_1 = (i-1) / fps
                    break
        except:
            drop_frame_gen_1 = None
            time_of_flight_gen_1 = None
            break
    else:
        time_of_flight_gen_1 = None
        drop_frame_gen_1 = None

    print("drop_frame_generated_1:", drop_frame_gen_1)
    print("time_of_flight_generated_1:", time_of_flight_gen_1)

    # Find the drop frame and time of flight for generated object 2
    len_generated_centroids_2 = len(generated_centroids_2)
    for i in range(1, len_generated_centroids_2):
        try:
            if generated_centroids_2[i] is None or generated_centroids_2[i-1] is None:
                drop_frame_gen_2 = None
                time_of_flight_gen_2 = None
                break
            else:
                if generated_centroids_2[i][1] > ground_gt_threshold_y and generated_centroids_2[i][1] - generated_centroids_2[i-1][1] < threshold:
                    drop_frame_gen_2 = (i-1)
                    time_of_flight_gen_2 = (i-1) / fps
                    break
        except:
            drop_frame_gen_2 = None
            time_of_flight_gen_2 = None
            break
    else:
        time_of_flight_gen_2 = None
        drop_frame_gen_2 = None
        
    print("drop_frame_generated_2:", drop_frame_gen_2)
    print("time_of_flight_generated_2:", time_of_flight_gen_2)

    # Compute heights
    height_gen_1 = height_gen_2 = None #drop height in pixels
    start_gen_y1 = start_gen_y2 = None #starting y value in pixels
    impact_gen_y1 = impact_gen_y2 = None #impact y value in pixels 

    if drop_frame_gen_1 is not None:
        start_gen_y1 = generated_centroids_1[0][1]
        impact_gen_y1 = generated_centroids_1[drop_frame_gen_1][1]
        if start_gen_y1 is not None and impact_gen_y1 is not None:
            height_gen_1 = impact_gen_y1 - start_gen_y1   # pixel drop distance
    print("start_gen_y1:", start_gen_y1, "impact_gen_y1:", impact_gen_y1, "height_gen_1:", height_gen_1)

    if drop_frame_gen_2 is not None:
        start_gen_y2 = generated_centroids_2[0][1]
        impact_gen_y2 = generated_centroids_2[drop_frame_gen_2][1]
        if start_gen_y2 is not None and impact_gen_y2 is not None:
            height_gen_2 = impact_gen_y2 - start_gen_y2   # pixel drop distance
    print("start_gen_y2:", start_gen_y2, "impact_gen_y2:", impact_gen_y2, "height_gen_2:", height_gen_2)

    h_ratio = None
    t_ratio = None
    if height_gen_1 is not None and height_gen_2 is not None and height_gen_1 > 0 and height_gen_2 > 0:
        h_ratio = height_gen_1 / height_gen_2 
    if time_of_flight_gen_1 is not None and time_of_flight_gen_2 is not None and time_of_flight_gen_1 > 0 and time_of_flight_gen_2 > 0:
        t_ratio = (time_of_flight_gen_1 / time_of_flight_gen_2)**2 
    
    results = {
        "height_ratio": h_ratio,
        "time_of_flight_ratio_squared": t_ratio,

        "height_gen_1": height_gen_1,
        "height_gen_2": height_gen_2,
        "start_gen_y1": start_gen_y1,
        "impact_gen_y1": impact_gen_y1,
        "start_gen_y2": start_gen_y2,
        "impact_gen_y2": impact_gen_y2,

        "time_of_flight_gen_1" : time_of_flight_gen_1,
        "drop_frame_gen_1" : drop_frame_gen_1,
        "time_of_flight_gen_2" : time_of_flight_gen_2,
        "drop_frame_gen_2" : drop_frame_gen_2,

        "centroid_positions_gen_1" : generated_centroids_1,
        "centroid_positions_gen_2" : generated_centroids_2,
    }

    
    with open(f"{folder}/generated_{sample_id}/results.json", "w") as f:
        json.dump(results, f, indent=4)
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample_folder",
        type=str,
        required=True,
        help="Folder containing generated samples"
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=1,
        help="Starting sample index"
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default = 113,
        help="Ending sample index (inclusive)"
    )
    args = parser.parse_args()

    sample_folder = args.sample_folder
    start_index = args.start_index
    end_index = args.end_index

    n = end_index + 1 - start_index

    print(f"Sample folder : {sample_folder}")
    print(f"Start index   : {start_index}")
    print(f"End index     : {end_index}")
    print(f"Num samples   : {n}")

    unsuccessful_samples = {}
    for sample_id in range(start_index, end_index + 1):
        print(f"\nProcessing sample {sample_id}...")
        try:
            calculate_masks(
                "gravity",
                sample_folder,
                sample_id
            )

        except Exception as e:

            print(
                f"Error processing sample {sample_id}: {e}"
            )

            unsuccessful_samples[
                f"sample_{sample_id}"
            ] = str(e)

    print("\nEvaluation complete.")
    print(
        f"Failed samples: {len(unsuccessful_samples)}"
    )

    print(
        "Unsuccessful samples:",
        unsuccessful_samples
    )


if __name__ == "__main__":
    main()