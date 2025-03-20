import rclpy
from rclpy.node import Node
from custom_msgs.msg import ObjectsSpeed, TrackingObjects, MultipleObjects, SingleObject

class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')
        
        # Subscribers
        self.create_subscription(ObjectsSpeed, '/object_speeds', self.speed_callback, 10)
        self.create_subscription(TrackingObjects, '/tracked_objects', self.class_callback, 10)
        
        # Publisher
        self.fused_pub = self.create_publisher(MultipleObjects, '/fused_objects', 10)
        
        # Data storage
        self.speeds = {}  # {object_id: speed}
        self.objects = {}  # {object_id: (minx, miny, maxx, maxy, class_label)}
        
    def speed_callback(self, msg):
        """Handles incoming speed messages."""
        for obj in msg.objects:
            self.speeds[obj.object_id] = obj.speed
        self.fuse_data()
    
    def class_callback(self, msg):
        """Handles incoming class messages."""
        for obj in msg.objects:
            self.objects[obj.object_id] = (obj.minx, obj.miny, obj.maxx, obj.maxy, obj.class_label)
        self.fuse_data()
    
    def fuse_data(self):
        """Fuses tracking and speed data into a single message."""
        fused_msg = MultipleObjects()
        
        object_ids = set(self.speeds.keys()).intersection(set(self.objects.keys()))
        for obj_id in object_ids:
            minx, miny, maxx, maxy, class_label = self.objects[obj_id]
            speed = self.speeds[obj_id]
            
            fused_obj = SingleObject()
            fused_obj.object_id = obj_id
            fused_obj.class_label = class_label
            fused_obj.minx = minx
            fused_obj.miny = miny
            fused_obj.maxx = maxx
            fused_obj.maxy = maxy
            fused_obj.speed = speed
            
            fused_msg.objects.append(fused_obj)
        
        self.fused_pub.publish(fused_msg)
        self.get_logger().info(f'Published fused data for {len(fused_msg.objects)} objects')


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()