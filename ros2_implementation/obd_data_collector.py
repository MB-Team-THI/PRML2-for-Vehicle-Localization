import rclpy
from rclpy.node import Node
from collections import defaultdict, deque
from std_msgs.msg import Float64MultiArray
from builtin_interfaces.msg import Time
from std_msgs.msg import Float64MultiArray, MultiArrayLayout, MultiArrayDimension
import numpy as np

BUFFER_DURATION = 2.0
#LatAcc_obd,brake_pressure_obd,speedo_obd,SW_pos_obd,VelFR_obd,VelFL_obd,VelRR_obd,VelRL_obd,Yawrate_obd
TOPICS = [f"/topic{i}" for i in range(1, 10)]  # your 10 topics


class MatrixCollector(Node):
    def __init__(self):
        super().__init__("matrix_collector")

        # store buffers per topic
        self.buffers = {t: deque() for t in TOPICS}

        # subscriptions
        self.subs = []
        for topic in TOPICS:
            sub = self.create_subscription(
                Float64MultiArray, topic, self.make_callback(topic), 10
            )
            self.subs.append(sub)

        # publisher
        self.publisher = self.create_publisher(Float64MultiArray, "matrix_out", 10)

        # timer to periodically publish (e.g., 50 Hz)
        self.timer = self.create_timer(0.02, self.publish_matrix)

    def make_callback(self, topic):
        def callback(msg):
            now = self.get_clock().now().nanoseconds / 1e9
            self.buffers[topic].append((now, msg.data))

            # drop older than BUFFER_DURATION
            while (
                self.buffers[topic]
                and now - self.buffers[topic][0][0] > BUFFER_DURATION
            ):
                self.buffers[topic].popleft()

        return callback


def publish_matrix(self):
    now = self.get_clock().now().nanoseconds / 1e9
    num_slots = 100
    slot_size = BUFFER_DURATION / num_slots  # 2.0 / 100 = 0.02s

    matrix = []

    for topic in TOPICS:
        # Extract buffer entries within the last BUFFER_DURATION
        buffer = [(t, v) for t, v in self.buffers[topic] if now - t <= BUFFER_DURATION]
        buffer.sort(key=lambda x: x[0])  # sort by timestamp

        values = []
        if buffer:
            last_val = buffer[0][1]  # initialize with first available sample
        else:
            last_val = 0.0  # fallback if no samples ever arrived

        buffer_idx = 0
        n = len(buffer)

        # Generate values for each slot
        for i in range(num_slots):
            slot_time = now - BUFFER_DURATION + i * slot_size

            # Advance pointer while there are newer samples for this slot
            while buffer_idx < n and buffer[buffer_idx][0] <= slot_time:
                last_val = buffer[buffer_idx][1]
                buffer_idx += 1

            values.append(last_val)

        matrix.append(values)

    # Convert matrix to Float64MultiArray
    msg = Float64MultiArray()
    arr = np.array(matrix, dtype=float)

    msg.data = arr.flatten().tolist()

    # Fill in layout metadata (2D: topics × samples)
    msg.layout = MultiArrayLayout()
    msg.layout.dim = [
        MultiArrayDimension(label='topics', size=len(TOPICS), stride=arr.shape[1] * len(TOPICS)),
        MultiArrayDimension(label='samples', size=num_slots, stride=num_slots),
    ]
    msg.layout.data_offset = 0

    self.publisher.publish(msg)


def main():
    rclpy.init()
    node = MatrixCollector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
