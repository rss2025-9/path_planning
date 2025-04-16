#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import matplotlib.pyplot as plt
from ackermann_msgs.msg import AckermannDriveStamped  # adjust based on your message package

class SteeringAnglePlotter(Node):
    def __init__(self):
        super().__init__('steering_angle_plotter')
        # Subscribe to your steering command topic.
        # Adjust the topic name, QoS, and message type as needed.
        self.subscription = self.create_subscription(
            AckermannDriveStamped,
            '/vesc/high_level/input/nav_0',   # change this to your topic name if different
            self.cmd_callback,
            10
        )
        self.subscription  # prevent unused variable warning

        # Initialize lists to store timestamps and steering angles.
        self.steering_data = []

        # Record the node start time.
        self.start_time = self.get_clock().now().nanoseconds / 1e9  # seconds

        # Create a Matplotlib figure and axis for plotting.
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], '-o')
        self.ax.set_xlabel('Time')
        # Turns off markers for the x-axis.
        self.ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        self.ax.set_ylabel('Steering Angle (Radians)')
        self.ax.set_title('Steering Angle over Time')
        # Uses a 1 long 2 wide aspect ratio for the figure.
        self.fig.set_size_inches(6, 3)

        # Enable interactive mode for live updates.
        plt.ion()
        plt.show()

    def cmd_callback(self, msg):
        # Store the data.
        self.steering_data.append( msg.drive.steering_angle)

    def update_plot(self):
        # Update the min and max steering angles using the data received so far.
        min_angle = min(self.steering_data)
        max_angle = max(self.steering_data)

        # Update the plot with the new data.
        self.line.set_data(range(0, len(self.steering_data)-800), self.steering_data[800:])
        self.ax.relim()          # Re-calculate limits for new data
        self.ax.autoscale_view() # Automatically adjust the view limits
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # Saves the figure to a file.
        self.fig.savefig('steering_angle_plot.png')
        self.ax.set_ylim(min_angle - 0.1, max_angle + 0.1)
        self.ax.set_xlim(0, len(self.steering_data) - 1)
        plt.pause(0.1)

        self.get_logger().info(f"Min steering angle: {min_angle}, Max steering angle: {max_angle}")

def main(args=None):
    rclpy.init(args=args)
    node = SteeringAnglePlotter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass  # Allow clean shutdown on Ctrl-C.
    finally:
        node.update_plot()  # Final update before shutdown
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
