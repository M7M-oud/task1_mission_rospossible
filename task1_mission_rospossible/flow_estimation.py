import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from custom_msgs.msg import TrackingObjects, ObjectSpeed, ObjectsSpeed

# Constants
PIXELS_TO_METERS = 0.01  # Adjust based on your setup (meters per pixel)
FPS = 30  # Frames per second

# Global variables
bridge = None
prev_gray = None
tracked_objects = {}  # {object_id: (x, y)}
prev_positions = {}  # {object_id: (x, y)}

def tracking_callback(msg):
    """Updates tracked objects' bounding box centers."""
    global tracked_objects
    tracked_objects = {}  
    
    for obj in msg.objects:
        # Compute center of bounding box
        center_x = (obj.minx + obj.maxx) / 2
        center_y = (obj.miny + obj.maxy) / 2
        tracked_objects[obj.object_id] = (center_x, center_y)

def stream_callback(msg):
    """Processes frames, estimates object speed, and publishes results."""
    global bridge, prev_gray, tracked_objects, prev_positions

    try:
        # Convert ROS Image to OpenCV
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_gray = gray
            return

        # Prepare messages for publishing
        speed_msg = ObjectsSpeed()
        
        for obj_id, (x, y) in tracked_objects.items():
            if obj_id in prev_positions:
                prev_x, prev_y = prev_positions[obj_id]
                distance_pixels = np.sqrt((x - prev_x) ** 2 + (y - prev_y) ** 2)
                
                # Convert to meters and compute speed in m/s
                distance_meters = distance_pixels * PIXELS_TO_METERS
                speed_mps = distance_meters * FPS

                # Create and add speed message
                obj_speed = ObjectSpeed()
                obj_speed.object_id = obj_id
                obj_speed.speed = speed_mps
                speed_msg.objects.append(obj_speed)

                # Display speed on frame
                cv2.putText(frame, f"ID {obj_id}: {speed_mps:.2f} m/s", 
                            (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)

            # Store current position for next frame
            prev_positions[obj_id] = (x, y)

        # Publish speed data
        speed_pub.publish(speed_msg)
        optical_flow_pub.publish(speed_msg) 
        # Show frame
        cv2.imshow("Speed Estimation", frame)
        cv2.waitKey(1)

        prev_gray = gray.copy()

    except Exception as e:
        node.get_logger().error(f"Error in stream_callback: {e}")

def main(args=None):
    global bridge, speed_pub, optical_flow_pub, node

    rclpy.init(args=args)
    node = Node("optical_flow_speed")

    bridge = CvBridge()

    # Subscribers
    node.create_subscription(Image, "/Stream", stream_callback, 10)
    node.create_subscription(TrackingObjects, "/tracked_objects", tracking_callback, 10)

    # Publishers
    speed_pub = node.create_publisher(ObjectsSpeed, "/object_speeds", 10)
    optical_flow_pub = node.create_publisher(ObjectsSpeed, "/optical_flow", 10)

    node.get_logger().info("Optical Flow Speed Estimation Node Started")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
