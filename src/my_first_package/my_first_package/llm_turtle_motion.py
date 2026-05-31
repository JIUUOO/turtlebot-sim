#!/usr/bin/env python3
import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from dotenv import load_dotenv
from openai import OpenAI


class LlmTurtleMotion(Node):
    def __init__(self):
        super().__init__('llm_turtle_motion')

        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('model', 'gpt-4o-mini')

        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value
        self.model = self.get_parameter('model').value

        self.cmd_vel_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

        self._load_env_files()
        self.client = OpenAI()

        self.get_logger().info('=== LLM TurtleBot motion agent node started ===')
        self.get_logger().info(f'Publishing velocity commands to {self.cmd_vel_topic}')
        self.get_logger().info('Example commands: "Move forward 2 meters", "Rotate clockwise 90 degrees"')

    def _load_env_files(self):
        repo_root = Path(__file__).resolve().parents[3]
        load_dotenv(repo_root / '.env', override=False)
        load_dotenv(repo_root / 'dev' / '.env', override=False)

    def ask_llm_and_move(self, user_command):
        """
        Analyze a natural-language command with the OpenAI API, then drive the robot.
        """
        self.get_logger().info(f"Analyzing user command: '{user_command}'")

        # Define the function schema that the LLM should fill.
        # This makes the LLM return structured JSON instead of free-form text.
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "control_robot",
                    "description": "Control the robot's linear velocity and angular velocity.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "linear_velocity": {
                                "type": "number",
                                "description": "Positive moves forward in m/s, negative moves backward, 0 stops. Keep normal commands between -0.3 and 0.3."
                            },
                            "angular_velocity": {
                                "type": "number",
                                "description": "Positive turns counterclockwise in rad/s, negative turns clockwise, 0 stops. Keep normal commands between -1.0 and 1.0."
                            }
                        },
                        "required": ["linear_velocity", "angular_velocity"]
                    }
                }
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an AI robot interface that converts human commands "
                            "into safe TurtleBot3 velocity parameters. Return small, "
                            "practical speeds for simulation."
                        ),
                    },
                    {"role": "user", "content": user_command}
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "control_robot"}}
            )

            tool_call = response.choices[0].message.tool_calls[0]
            arguments = json.loads(tool_call.function.arguments)

            linear_v = self._clamp(float(arguments.get("linear_velocity", 0.0)), -0.3, 0.3)
            angular_v = self._clamp(float(arguments.get("angular_velocity", 0.0)), -1.0, 1.0)

            self.get_logger().info(f"LLM result -> linear velocity: {linear_v} m/s, angular velocity: {angular_v} rad/s")

            twist_msg = Twist()
            twist_msg.linear.x = linear_v
            twist_msg.angular.z = angular_v

            self.cmd_vel_pub.publish(twist_msg)
            self.get_logger().info("ROS 2 topic published successfully.")

        except Exception as e:
            self.get_logger().error(f"An error occurred: {str(e)}")

    @staticmethod
    def _clamp(value, minimum, maximum):
        return max(minimum, min(maximum, value))


def main(args=None):
    rclpy.init(args=args)
    node = LlmTurtleMotion()

    try:
        while rclpy.ok():
            user_input = input("\n[Enter a command (q to quit)]: ")
            if user_input.lower() == 'q':
                break
            if user_input.strip() == '':
                continue

            node.ask_llm_and_move(user_input)
            rclpy.spin_once(node, timeout_sec=0.1)

    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
