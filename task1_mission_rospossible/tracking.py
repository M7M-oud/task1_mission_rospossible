import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from custom_msgs.msg import TrackingObject, TrackingObjects
from ultralytics import YOLO

# Global variables
bridge = CvBridge()
model = YOLO("yolov8n.pt")  # YOLO model for object detection & tracking
tracked_objects = {}  # Key: YOLO track_id, Value: Assigned unique ID
available_ids = set()  # Set of available IDs for reuse
current_mask = None  # Variable to store the latest segmentation mask

def mask_callback(msg):
    """Stores the received segmentation mask."""
    global current_mask
    try:
        current_mask = bridge.imgmsg_to_cv2(msg, "mono8")
    except Exception as e:
        print(f"Error in mask_callback: {e}")

def stream_callback(msg, publisher):
    """Processes frames, tracks objects, and publishes tracking results."""
    global tracked_objects, available_ids
    try:
        # Convert ROS Image message to OpenCV frame
        frame = bridge.imgmsg_to_cv2(msg, "bgr8")

        # Run YOLO tracking on the frame
        results = model.track(frame, persist=True, show=False)

        # Prepare the message for publishing
        tracked_objects_list = TrackingObjects()

        # Reset the current frame's tracked objects
        current_frame_objects = {}

        for r in results:
            for box in r.boxes:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Get class ID and name
                class_id = int(box.cls[0])
                class_name = model.names[class_id] if class_id in model.names else "Unknown"

                # Get confidence score
                confidence = float(box.conf[0])

                # Get track ID (if available)
                track_id = int(box.id) if box.id is not None else -1

                # Assign a unique ID to the object if not already tracked
                if track_id not in tracked_objects:
                    if available_ids:
                        unique_id = min(available_ids)  # Pick the smallest available ID
                        available_ids.remove(unique_id)  # Remove from the available IDs set
                    else:
                        unique_id = len(tracked_objects) + 1  # Assign a new ID
                    tracked_objects[track_id] = unique_id

                # Get the assigned unique ID
                unique_id = tracked_objects[track_id]

                # Add the object to the current frame's tracking list
                current_frame_objects[track_id] = unique_id

                # Create TrackedObject message
                single_object = TrackingObject()
                single_object.object_id = unique_id
                single_object.class_label = class_name
                single_object.minx = float(x1)
                single_object.miny = float(y1)
                single_object.maxx = float(x2)
                single_object.maxy = float(y2)
                tracked_objects_list.objects.append(single_object)

                # Draw bounding boxes and labels on frame
                label = f"ID {unique_id} {class_name} {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Remove objects that have left the frame
        for track_id in list(tracked_objects.keys()):
            if track_id not in current_frame_objects:
                available_ids.add(tracked_objects[track_id])  # Reuse ID
                del tracked_objects[track_id]

        # Publish detected objects
        publisher.publish(tracked_objects_list)

        # Display the frame with bounding boxes and labels
        cv2.imshow("Object Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            rclpy.shutdown()

    except Exception as e:
        print(f"Error in stream_callback: {e}")

def main():
    rclpy.init()
    node = Node("object_tracking_node")

    # Create publisher
    tracked_pub = node.create_publisher(TrackingObjects, "/tracked_objects", 10)
    
    # Create subscribers
    node.create_subscription(Image, "/Stream", lambda msg: stream_callback(msg, tracked_pub), 10)
    node.create_subscription(Image, "/segmentation_mask", mask_callback, 10)
    
    print("Object Tracking Node started.")
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
