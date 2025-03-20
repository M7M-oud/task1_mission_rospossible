import sys
import rclpy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# Global Variables
publisher = None
cap = None
br = CvBridge()

FRAME_WIDTH = 640   # Desired frame width
FRAME_HEIGHT = 480  # Desired frame height

def timer_callback():
    """Reads frames from the camera or video file, resizes them, publishes them, and displays them."""
    global cap, publisher, br

    ret, frame = cap.read()
    if ret:
        # Resize the frame to a smaller size
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))

        # Display the feed
        cv2.imshow('Stream', frame)

        # Convert the frame to a ROS Image message
        img_msg = br.cv2_to_imgmsg(frame, encoding="bgr8")
        publisher.publish(img_msg)
        rclpy.logging.get_logger('video_stream').info('Publishing video frame')

        # Allow window to update
        cv2.waitKey(1)
    else:
        rclpy.logging.get_logger('video_stream').warn("No more frames available.")

def main(args=None):
    global publisher, cap

    rclpy.init(args=args)
    node = rclpy.create_node('video_stream_node')

    # Check if a video file path was provided as a command line argument
    video_source = 0  # default to camera
    if len(sys.argv) > 1:
        video_source = sys.argv[1]  # user-specified video file path

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        node.get_logger().error(f"Failed to open video source: {video_source}")
        return

    # Set the camera/video resolution (if supported)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    publisher = node.create_publisher(Image, '/Stream', 10)
    timer = node.create_timer(0.01, timer_callback)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt detected, shutting down gracefully...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
