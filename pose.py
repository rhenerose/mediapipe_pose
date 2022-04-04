import argparse
from typing import List, Optional, Tuple
import dataclasses

import cv2
import numpy as np
import matplotlib.pyplot as plt

import mediapipe as mp
from mediapipe.python.solutions import drawing_utils as mp_drawing
from mediapipe.framework.formats import landmark_pb2

# custom pose connections
POSE_CONNECTIONS = frozenset(
    [
        (
            mp.solutions.pose.PoseLandmark.NOSE,
            mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
        ),
        (
            mp.solutions.pose.PoseLandmark.NOSE,
            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
        ),
        (
            mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
        ),
        (
            mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
            mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
        ),
        (
            mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
            mp.solutions.pose.PoseLandmark.LEFT_WRIST,
        ),
        (
            mp.solutions.pose.PoseLandmark.LEFT_WRIST,
            mp.solutions.pose.PoseLandmark.LEFT_PINKY,
        ),
        (
            mp.solutions.pose.PoseLandmark.LEFT_WRIST,
            mp.solutions.pose.PoseLandmark.LEFT_INDEX,
        ),
        (
            mp.solutions.pose.PoseLandmark.LEFT_WRIST,
            mp.solutions.pose.PoseLandmark.LEFT_THUMB,
        ),
        (
            mp.solutions.pose.PoseLandmark.LEFT_PINKY,
            mp.solutions.pose.PoseLandmark.LEFT_INDEX,
        ),
        (
            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
            mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
        ),
        (
            mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
            mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
        ),
        (
            mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
            mp.solutions.pose.PoseLandmark.RIGHT_PINKY,
        ),
        (
            mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
            mp.solutions.pose.PoseLandmark.RIGHT_INDEX,
        ),
        (
            mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
            mp.solutions.pose.PoseLandmark.RIGHT_THUMB,
        ),
        (
            mp.solutions.pose.PoseLandmark.RIGHT_PINKY,
            mp.solutions.pose.PoseLandmark.RIGHT_INDEX,
        ),
        (
            mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
            mp.solutions.pose.PoseLandmark.LEFT_HIP,
        ),
        (
            mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
            mp.solutions.pose.PoseLandmark.RIGHT_HIP,
        ),
        (
            mp.solutions.pose.PoseLandmark.LEFT_HIP,
            mp.solutions.pose.PoseLandmark.RIGHT_HIP,
        ),
        (
            mp.solutions.pose.PoseLandmark.LEFT_HIP,
            mp.solutions.pose.PoseLandmark.LEFT_KNEE,
        ),
        (
            mp.solutions.pose.PoseLandmark.RIGHT_HIP,
            mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
        ),
        (
            mp.solutions.pose.PoseLandmark.LEFT_KNEE,
            mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
        ),
        (
            mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
            mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
        ),
        (
            mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
            mp.solutions.pose.PoseLandmark.LEFT_HEEL,
        ),
        (
            mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
            mp.solutions.pose.PoseLandmark.RIGHT_HEEL,
        ),
        (
            mp.solutions.pose.PoseLandmark.LEFT_HEEL,
            mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX,
        ),
        (
            mp.solutions.pose.PoseLandmark.RIGHT_HEEL,
            mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX,
        ),
        (
            mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
            mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX,
        ),
        (
            mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
            mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX,
        ),
    ]
)


# override DrawingSpec
@dataclasses.dataclass
class DrawingSpec(mp_drawing.DrawingSpec):
    @property
    def normalize_color(self):
        return tuple(v / 255.0 for v in self.color)


def plot_landmarks(
    landmark_list: landmark_pb2.NormalizedLandmarkList,
    connections: Optional[List[Tuple[int, int]]] = None,
    landmark_drawing_spec: DrawingSpec = DrawingSpec(
        color=mp_drawing.RED_COLOR, thickness=5
    ),
    connection_drawing_spec: DrawingSpec = DrawingSpec(
        color=mp_drawing.BLACK_COLOR, thickness=5
    ),
    elevation: int = 10,
    azimuth: int = 10,
):
    if not landmark_list:
        return

    ax.cla()
    ax.view_init(elev=elevation, azim=azimuth)
    ax.set_xlim3d(-1, 1)
    ax.set_ylim3d(-1, 1)
    ax.set_zlim3d(-1, 1)

    plotted_landmarks = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if (
            landmark.HasField("visibility")
            and landmark.visibility < mp_drawing._VISIBILITY_THRESHOLD
        ) or (
            landmark.HasField("presence")
            and landmark.presence < mp_drawing._PRESENCE_THRESHOLD
        ):
            continue

        plotted_landmarks[idx] = (-landmark.z, landmark.x, -landmark.y)

    # plot keypoints
    vals = np.array(list(plotted_landmarks.values()), dtype=float)
    ax.scatter3D(
        xs=vals[:, 0],
        ys=vals[:, 1],
        zs=vals[:, 2],
        color=landmark_drawing_spec.normalize_color[::-1],
        linewidth=landmark_drawing_spec.thickness,
    )

    if connections:
        num_landmarks = len(landmark_list.landmark)
        # Draws the connections
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                continue

            if start_idx in plotted_landmarks and end_idx in plotted_landmarks:
                landmark_pair = [
                    plotted_landmarks[start_idx],
                    plotted_landmarks[end_idx],
                ]
                ax.plot3D(
                    xs=[landmark_pair[0][0], landmark_pair[1][0]],
                    ys=[landmark_pair[0][1], landmark_pair[1][1]],
                    zs=[landmark_pair[0][2], landmark_pair[1][2]],
                    color=connection_drawing_spec.normalize_color[::-1],
                    linewidth=connection_drawing_spec.thickness,
                )

    plt.pause(0.001)


def main(args: argparse.Namespace):
    # mp_drawing = mp_drawing
    mp_pose = mp.solutions.pose

    # create video capture
    if args.device.isnumeric():
        # camera device index
        cap = cv2.VideoCapture(int(args.device))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    else:
        # video file
        cap = cv2.VideoCapture(args.device)

    with mp_pose.Pose(
        model_complexity=args.model_complexity,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        while True:
            # read frame
            ret, frame = cap.read()
            if not ret:
                break

            # detect body
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False
            results = pose.process(frame)

            # draw body
            if results.pose_landmarks is not None:
                # Recolor back to BGR
                frame.flags.writeable = True
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Render detections
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    DrawingSpec(
                        color=mp_drawing.BLUE_COLOR,
                        thickness=2,
                        circle_radius=2,
                    ),
                    DrawingSpec(
                        color=mp_drawing.GREEN_COLOR,
                        thickness=2,
                        circle_radius=2,
                    ),
                )

            # plot world landmarks
            if args.plot_landmark and results.pose_world_landmarks is not None:
                plot_landmarks(
                    results.pose_world_landmarks,
                    POSE_CONNECTIONS,
                    landmark_drawing_spec=DrawingSpec(
                        color=mp_drawing.RED_COLOR, thickness=3
                    ),
                    connection_drawing_spec=DrawingSpec(
                        color=mp_drawing.BLACK_COLOR, thickness=1
                    ),
                )

            if cv2.waitKey(1) & 0xFF == 27:
                break

            # Display the result frame
            cv2.imshow("MediaPipe Pose", frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        help="Camera device index or video file",
        default="./media/dance.mp4",
    )
    parser.add_argument("--width", help="Camera width", type=int, default=640)
    parser.add_argument("--height", help="Camera height", type=int, default=480)

    parser.add_argument(
        "--model_complexity",
        help="model complexity (0:lite, 1:full(default), 2:heavy)",
        type=int,
        default=1,
    )
    parser.add_argument("--plot_landmark", help="plot 3d landmarks",action="store_true")

    args = parser.parse_args()

    # setup matplotlib
    if args.plot_landmark:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0.0, right=1, bottom=0, top=1)

    ##############################
    # Run the main function
    ##############################
    main(args)
