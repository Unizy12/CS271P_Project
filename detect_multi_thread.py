from utils.webcam import WebcamVideoStream
from multiprocessing import Queue, Pool
import utils.webcam as webcam
import my_utils
import cv2
import argparse
import datetime
import sys
from my_utils import load_checkpoint, build_predictor
from inference import load_image
from apex.fp16_utils import network_to_half

def worker(input_q, output_q, cap_params, frame_processed):
    print(">> loading frozen model for worker")
    ssd300 = build_predictor('/CS271P_Project/model/epoch_119.pt')
    ssd300 = ssd300.cuda()
    ssd300 = network_to_half(ssd300.cuda())
    ssd300 = ssd300.eval()
    while True:
        print("> ===== in worker loop, frame ", frame_processed)
        frame= input_q.get()
        print(frame)
        if (frame is not None):
            # Actual detection. Variable boxes contains the bounding box cordinates for hands detected,
            # while scores contains the confidence for each of these boxes.
            # Hint: If len(boxes) > 1 , you may assume you have found atleast one hand (within your score threshold)
            exeframe = my_utils.execution(frame)
            boxes, scores = my_utils.detect(exeframe, ssd300)
            print(boxes, scores)
            # draw bounding boxes
            my_utils.draw_box_on_image(
                cap_params['num_hands_detect'], cap_params["score_thresh"],
                scores, boxes, cap_params['im_width'], cap_params['im_height'],
                frame)
            # add frame annotated with bounding box to queue
            output_q.put(frame)
            frame_processed += 1
        else:
            print('no frame')
            output_q.put(frame)

    sess.close()
if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-src',
        '--source',
        dest='video_source',
        type= str,
        default='http://192.168.86.20:4747/video',
        help='Device index of the camera.')
    parser.add_argument(
        '-nhands',
        '--num_hands',
        dest='num_hands',
        type=int,
        default=2,
        help='Max number of hands to detect.')
    parser.add_argument(
        '-fps',
        '--fps',
        dest='fps',
        type=int,
        default=1,
        help='Show FPS on detection/display visualization')
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
        default=1,
        help='Number of workers.')
    parser.add_argument(
        '-q-size',
        '--queue-size',
        dest='queue_size',
        type=int,
        default=2,
        help='Size of the queue.')
    args = parser.parse_args()

    input_q = Queue(maxsize=args.queue_size)
    output_q = Queue(maxsize=args.queue_size)

    video_capture = WebcamVideoStream(src=args.video_source, width=args.width, height=args.height).start()

    score_thresh = 0.5
    cap_params = {}
    frame_processed = 0
    cap_params['im_width'], cap_params['im_height'] = video_capture.size()
    cap_params['score_thresh'] = score_thresh

    # max number of hands we want to detect/track
    cap_params['num_hands_detect'] = args.num_hands


    # spin up workers to paralleize detection.
    pool = Pool(args.num_workers, worker,(input_q, output_q, cap_params, frame_processed))

    start_time = datetime.datetime.now()
    num_frames = 0
    fps = 0
    index = 0

    cv2.namedWindow('Multi-Threaded Detection', cv2.WINDOW_NORMAL)
    while True:
        frame = video_capture.read()
        # frame = cv2.flip(frame, 1)
        index += 1
        print(index,frame)
        input_q.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output_frame = output_q.get()
        output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

        elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
        num_frames += 1
        fps = num_frames / elapsed_time
        # print("frame ",  index, num_frames, elapsed_time, fps)

        if (output_frame is not None):
            if (args.display > 0):
                if (args.fps > 0):
                    my_utils.draw_fps_on_image("FPS : " + str(int(fps)),
                                                     output_frame)
                while True:
                    cv2.imshow('Multi-Threaded Detection', output_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                if (num_frames == 400):
                    num_frames = 0
                    start_time = datetime.datetime.now()
                else:
                    print("frames processed: ", index, "elapsed time: ",
                            elapsed_time, "fps: ", str(int(fps)))
        else:
            print("video end")
            break
    elapsed_time = (datetime.datetime.now() - start_time).total_seconds()
    fps = num_frames / elapsed_time
    print("fps", fps)
    pool.terminate()
    video_capture.stop()
    cv2.destroyAllWindows()