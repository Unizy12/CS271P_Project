from inference import load_image
import my_utils
import cv2
import datetime
import argparse
from my_utils import load_checkpoint, build_predictor
from apex.fp16_utils import network_to_half
import threading
# from my_utils import Net as classification
import torch

class CameraBufferCleanerThread(threading.Thread):
    def __init__(self, camera, name='camera-buffer-cleaner-thread'):
        self.camera = camera
        self.last_frame = None
        self.fps = 1.
        super(CameraBufferCleanerThread, self).__init__(name=name)
        self.start()

    def run(self):
        while True:
            ret, self.last_frame = self.camera.read()

# detection_graph, sess = detector_utils.load_inference_graph()

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-sth',
        '--scorethreshold',
        dest='score_thresh',
        type=float,
        default=0.5,
        help='Score threshold for displaying bounding boxes')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        default='http://192.168.86.20:4747/video',
        help='Device index of the camera.')
    parser.add_argument(
        '-wd',
        '--width',
        dest='width',
        type=int,
        default=300,
        help='Width of the frames in the video stream.')
    parser.add_argument(
        '-ht',
        '--height',
        dest='height',
        type=int,
        default=300,
        help='Height of the frames in the video stream.')
    parser.add_argument(
        '-ds',
        '--display',
        dest='display',
        type=int,
        default=1,
        help='Display the detected images using OpenCV. This reduces FPS')
    parser.add_argument(
        '-num-w',
        '--num-workers',
        dest='num_workers',
        type=int,
        default=4,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=5,
        help='Size of the queue.')
    args = parser.parse_args()
    ssd300 = build_predictor('/CS271P_Project/model/epoch_99.pt')
    net = my_utils.creat_model()
    # net = net.cuda()
    net.load_state_dict(torch.load('/CS271P_Project/CNN_classification/ResNet18_79.pth'))
    net.eval()
    ssd300 = ssd300.cuda()
    ssd300 = network_to_half(ssd300.cuda())
    ssd300 = ssd300.eval()
    cap = cv2.VideoCapture(args.video_source)
    # cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cam_cleaner = CameraBufferCleanerThread(cap)
    start_time = datetime.datetime.now()
    im_width, im_height = (cap.get(3), cap.get(4))
    # max number of hands we want to detect/track
    num_hands_detect = 2

    cv2.namedWindow('Single-Threaded Detection', cv2.WINDOW_NORMAL)
    while True:
        start_time = datetime.datetime.now()
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        # cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # ret, image_np = cap.read()
        image_np = cam_cleaner.last_frame
        # image_np = cv2.flip(image_np, 1)

        execution_start = datetime.datetime.now()
        try:
            image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        except:
            print("Error converting to RGB")

        # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
        # while scores contains the confidence for each of these boxes.
        # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)
        exeframe = my_utils.execution(image_np)
        boxes, scores = my_utils.detect(exeframe, ssd300)

        # draw bounding boxes on frame
        my_utils.draw_box_on_image(num_hands_detect, args.score_thresh,
                                        scores, boxes, im_width, im_height,
                                        image_np, net)

        # Calculate Frames per second (FPS)
        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        fps = 1 / elapsed_time

        if (args.display > 0):
            # Display FPS on frame
            if (args.fps > 0):
                my_utils.draw_fps_on_image("FPS : " + str(int(fps)),image_np)

            cv2.imshow('Single-Threaded Detection',
                       cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
        else:
            print("frames processed: ", num_frames, "elapsed time: ",
                  elapsed_time, "fps: ", str(int(fps)))