import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from ultralytics import YOLO

class SegmentationNode(Node):
    def __init__(self):
        """Initializes the ROS2 node for video segmentation."""
        super().__init__('segmentation_node')

        # Initialize OpenCV bridge
        self.bridge = CvBridge()
        
        # Load YOLO segmentation model
        try:
            self.model = YOLO("yolov8n-seg.pt")  # Load YOLOv8 segmentation model
            self.get_logger().info("YOLOv8 Segmentation Model Loaded Successfully")
        except Exception as e:
            self.get_logger().error(f"Failed to load YOLOv8 model: {e}")
            return
        
        # Set up subscriber & publisher
        self.subscription = self.create_subscription(Image, "/Stream", self.feed_callback, 10)
        self.mask_pub = self.create_publisher(Image, "/segmentation_mask", 10)

        # Color map for class-based segmentation masks
        self.color_map = {
            0:  [255, 0, 0],      # Bright Red
            1:  [0, 255, 0],      # Bright Green
            2:  [0, 0, 255],      # Bright Blue
            3:  [255, 255, 0],    # Yellow
            4:  [255, 165, 0],    # Orange
            5:  [128, 0, 128],    # Purple
            6:  [75, 0, 130],     # Indigo
            7:  [0, 255, 255],    # Cyan
            8:  [255, 20, 147],   # Deep Pink
            9:  [34, 139, 34],    # Forest Green
            10: [173, 216, 230],  # Light Blue
            11: [128, 128, 0],    # Olive
            12: [210, 105, 30],   # Chocolate
            13: [255, 105, 180],  # Hot Pink
            14: [47, 79, 79],     # Dark Slate Gray
            15: [240, 230, 140],  # Khaki
            16: [0, 128, 128],    # Teal
            17: [220, 20, 60],    # Crimson
            18: [139, 69, 19],    # Saddle Brown
            19: [255, 250, 250],  # Snow White
        }

    def feed_callback(self, msg):
        """Processes frames, performs segmentation, and publishes the segmentation mask."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")  # Convert ROS Image to OpenCV frame
            results = self.model(frame)  # Run YOLOv8 segmentation
            mask = np.zeros_like(frame, dtype=np.uint8)  # Background
            
            # Process results
            for result in results:
                if result.masks is not None:
                    for i, mask_data in enumerate(result.masks.data):
                        class_id = int(result.boxes.cls[i]) % 20  # Get class ID
                        color = self.color_map.get(class_id, [255, 255, 255])  # Assign color to each class
                        binary_mask = (mask_data.cpu().numpy() * 255).astype(np.uint8)  # Mask to binary
                        
                        # Resize mask to match frame dimensions
                        binary_mask = cv2.resize(binary_mask, (frame.shape[1], frame.shape[0]))

                        # Color mask
                        color_mask = np.zeros_like(frame, dtype=np.uint8)
                        for c in range(3):  # Apply color to each channel
                            color_mask[:, :, c] = (binary_mask / 255) * color[c]
                        
                        # Resize mask if dimensions do not match
                        if mask.shape[:2] != frame.shape[:2]:
                            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                        # Blend the mask with the original mask (for multiple detections)
                        mask = cv2.addWeighted(mask, 1.0, color_mask, 0.7, 0)
            
            # Publish the segmentation mask
            mask_msg = self.bridge.cv2_to_imgmsg(mask, encoding="bgr8")
            self.mask_pub.publish(mask_msg)
            
            # Display the segmentation mask
            cv2.imshow("Segmentation Mask", mask)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

def main(args=None):
    """Main function to start the ROS 2 node."""
    rclpy.init(args=args)
    node = SegmentationNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down segmentation node...")
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
