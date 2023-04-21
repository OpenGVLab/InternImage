# Devkit
Here we describe the API provided by the OpenLane-V2 devkit.

## openlanev2.dataset

### Collection
is a collection of frames. 
#### `Collection.get_frame_via_identifier(identifier : tuple) -> Frame` 
Returns a frame with the given identifier (split, segment_id, timestamp).
#### `Collection.get_frame_via_index(index : int) -> (tuple, Frame)`
Returns a frame with the given index.

### Frame
is a data structure containing meta data of a frame.
#### `Frame.get_camera_list() -> list` 
Retuens a list of camera names.
#### `Frame.get_pose() -> dict`
Retuens the pose of ego vehicle.
#### `Frame.get_image_path(camera : str) -> str`
Retuens the image path given a camera.
#### `Frame.get_rgb_image(camera : str) -> np.ndarray`
Retuens the RGB image given a camera.
#### `Frame.get_intrinsic(camera : str) -> dict`
Retuens the intrinsic given a camera.
#### `Frame.get_extrinsic(camera : str) -> dict`
Retuens the extrinsic given a camera.
#### `Frame.get_annotations() -> dict`
Retuens annotations of the current frame.
#### `Frame.get_annotations_lane_centerlines() -> dict`
Retuens lane centerline annotations of the current frame.
#### `Frame.get_annotations_traffic_elements() -> dict`
Retuens traffic element annotations of the current frame.
#### `Frame.get_annotations_topology_lclc() -> list`
Retuens the adjacent matrix of topology_lclc.
#### `Frame.get_annotations_topology_lcte() -> list`
Retuens the adjacent matrix of topology_lcte.

## openlanev2.evaluation

#### `evaluate(ground_truth, predictions) -> dict`
Given the ground truth and predictions, which are formatted dict or the path to pickle storing the dict that ground truth is preprocessed pickle file and predictions are formatted as described [here](./submission.md#format), this function returns a dict storing all metrics defined by our task.

## openlanev2.io
This subpackage wraps all IO operations of the OpenLane-V2 devkit.
It can be modified for different IO operations.

## openlanev2.preprocessing

#### `collect(root_path : str, data_dict : dict, collection : str, point_interval : int = 1) -> None`
Given a data_dict storing identifiers of frames, this function collects meta data the frames and stores it into a pickle file for efficient IO for the following operations.
#### `check_results(results : dict) -> None`
Check format of results.

## openlanev2.visualization
This subpackage provides tools for visualization. Please refer to the [tutorial](../tutorial.ipynb) for examples.
