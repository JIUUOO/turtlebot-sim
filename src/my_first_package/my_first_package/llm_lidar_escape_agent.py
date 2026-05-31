#!/usr/bin/env python3
import json
import math
import threading
from pathlib import Path

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from dotenv import load_dotenv
from openai import OpenAI


class LlmLidarEscapeAgent(Node):
    def __init__(self):
        super().__init__('llm_lidar_escape_agent')

        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('model', 'gpt-4o-mini')
        self.declare_parameter('front_stop_distance', 0.45)
        self.declare_parameter('front_warning_distance', 0.6)

        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.scan_topic = self.get_parameter('scan_topic').value
        self.model = self.get_parameter('model').value
        self.front_stop_distance = float(self.get_parameter('front_stop_distance').value)
        self.front_warning_distance = float(self.get_parameter('front_warning_distance').value)

        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.laser_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.lidar_callback,
            10,
        )

        self.current_front_distance = 10.0
        self._load_env_files()
        self.client = OpenAI()

        self.get_logger().info('=== LLM LiDAR escape agent started ===')
        self.get_logger().info(f'Publishing velocity commands to {self.cmd_vel_topic}')
        self.get_logger().info(f'Subscribing to LiDAR data from {self.scan_topic}')

    def _load_env_files(self):
        repo_root = Path(__file__).resolve().parents[3]
        load_dotenv(repo_root / '.env', override=False)
        load_dotenv(repo_root / 'dev' / '.env', override=False)

    def lidar_callback(self, msg):
        """
        Keep a local safety loop alive even while the LLM request is waiting.
        """
        front_ranges = self._front_ranges(msg)
        valid_ranges = [
            value for value in front_ranges
            if math.isfinite(value) and msg.range_min < value < msg.range_max
        ]

        if not valid_ranges:
            return

        self.current_front_distance = min(valid_ranges)

        if self.current_front_distance <= self.front_stop_distance:
            emergency_twist = Twist()
            self.cmd_vel_pub.publish(emergency_twist)
            self.get_logger().warn(
                f'Emergency stop: front obstacle at {self.current_front_distance:.2f} m'
            )

    def ask_llm_and_move(self, user_command):
        """
        Combine the user's command with current LiDAR distance before publishing velocity.
        """
        self.get_logger().info(
            f"User command: '{user_command}' | front distance: {self.current_front_distance:.2f} m"
        )

        if self.current_front_distance <= self.front_stop_distance:
            self.get_logger().warn(
                'Front obstacle is too close. Running local escape motion before using the LLM.'
            )
            escape_twist = Twist()
            escape_twist.linear.x = -0.1
            escape_twist.angular.z = 0.5
            self.cmd_vel_pub.publish(escape_twist)
            return

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "control_robot",
                    "description": "Control robot linear and angular velocity using the user command and sensor state.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "linear_velocity": {
                                "type": "number",
                                "description": "Positive moves forward in m/s, negative moves backward. Keep normal commands between -0.3 and 0.3."
                            },
                            "angular_velocity": {
                                "type": "number",
                                "description": "Positive turns left, negative turns right in rad/s. Keep normal commands between -1.0 and 1.0."
                            },
                            "ai_reasoning_message": {
                                "type": "string",
                                "description": "A short English explanation of the safety decision."
                            },
                        },
                        "required": [
                            "linear_velocity",
                            "angular_velocity",
                            "ai_reasoning_message",
                        ],
                    },
                },
            }
        ]

        system_prompt = f"""
You are a safety-first autonomous driving brain for a TurtleBot3 simulation.
The current front obstacle distance is [{self.current_front_distance:.2f} meters].

Choose exactly one mode.

[Mode A: Collision Risk]
- Condition: front distance is less than or equal to {self.front_warning_distance:.2f} meters.
- Behavior: ignore forward user commands and output a backward-and-turning escape motion.
- Example output: linear_velocity = -0.15, angular_velocity = 0.6

[Mode B: Normal Driving]
- Condition: front distance is greater than {self.front_warning_distance:.2f} meters.
- Behavior: convert the user's command into a small, practical robot velocity.
- Example output: if the user says "go 1" or "go forward", use a positive linear velocity such as 0.3.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_command},
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "control_robot"}},
            )

            tool_call = response.choices[0].message.tool_calls[0]
            arguments = json.loads(tool_call.function.arguments)

            linear_v = self._clamp(float(arguments.get("linear_velocity", 0.0)), -0.3, 0.3)
            angular_v = self._clamp(float(arguments.get("angular_velocity", 0.0)), -1.0, 1.0)
            reason = arguments.get("ai_reasoning_message", "No explanation provided.")

            if self.current_front_distance <= self.front_warning_distance and linear_v > 0.0:
                self.get_logger().warn('Forward motion blocked by local safety guard.')
                linear_v = -0.15
                angular_v = 0.6

            self.get_logger().info(f'AI safety decision: {reason}')
            self.get_logger().info(
                f'Publishing velocity -> linear: {linear_v} m/s, angular: {angular_v} rad/s'
            )

            twist_msg = Twist()
            twist_msg.linear.x = linear_v
            twist_msg.angular.z = angular_v
            self.cmd_vel_pub.publish(twist_msg)

        except Exception as e:
            self.get_logger().error(f'Agent error: {str(e)}')

    @staticmethod
    def _front_ranges(msg):
        range_count = len(msg.ranges)
        if range_count == 0:
            return []

        front_width = min(15, range_count // 12)
        return list(msg.ranges[:front_width]) + list(msg.ranges[-front_width:])

    @staticmethod
    def _clamp(value, minimum, maximum):
        return max(minimum, min(maximum, value))


def main(args=None):
    rclpy.init(args=args)
    node = LlmLidarEscapeAgent()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        while rclpy.ok():
            user_input = input("\n[Enter a command for the AI agent (q to quit)]: ")
            if user_input.lower() == 'q':
                break
            if user_input.strip() == '':
                continue

            node.ask_llm_and_move(user_input)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
